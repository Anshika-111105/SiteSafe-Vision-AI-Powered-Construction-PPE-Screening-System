import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.transforms import get_eval_transforms
from src.models.architectures import create_resnet50_model, create_mobilenet_v3_model
from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.utils.logger import setup_logger

logger = setup_logger("predictor")


class PPEPredictor:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        if model_path is None:
            model_path = PROJECT_ROOT / "artifacts" / "models" / "production_model.pt"
        if metadata_path is None:
            metadata_path = PROJECT_ROOT / "artifacts" / "models" / "model_metadata.json"

        self.model_path: Path = Path(model_path)
        self.metadata_path: Path = Path(metadata_path)

        if device:
            self.device: torch.device = torch.device(device)
        else:
            self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[nn.Module] = None
        self.target_layer: Optional[nn.Module] = None
        self.gradcam: Optional[GradCAM] = None
        self.model_name: str = "mobilenet_v3_large"
        self.metadata: Dict[str, Any] = {}
        self.idx_to_class: Dict[int, str] = {0: "FULL_PPE", 1: "PARTIAL_PPE", 2: "NO_PPE"}
        self.class_to_idx: Dict[str, int] = {"FULL_PPE": 0, "PARTIAL_PPE": 1, "NO_PPE": 2}
        self.transforms = get_eval_transforms(image_size=224, resize_size=256)
        self._gradcam_lock: threading.Lock = threading.Lock()

        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            logger.warning(f"Production model not found at {self.model_path}. Predictor will run in uninitialized state.")
            return

        logger.info(f"Loading production model from {self.model_path} onto {self.device}...")
        ckpt = torch.load(self.model_path, map_location="cpu")
        model_name = ckpt.get("model_name", "mobilenet_v3_large")
        self.model_name = model_name

        # Load dynamic class mappings from checkpoint if available
        if "class_to_idx" in ckpt:
            self.class_to_idx = ckpt["class_to_idx"]
            self.idx_to_class = {int(v): k for k, v in self.class_to_idx.items()}
        elif "idx_to_class" in ckpt:
            self.idx_to_class = {int(k): v for k, v in ckpt["idx_to_class"].items()}
            self.class_to_idx = {v: int(k) for k, v in self.idx_to_class.items()}

        num_classes = len(self.class_to_idx)

        if model_name == "resnet50":
            model = create_resnet50_model(num_classes=num_classes, pretrained=False)
            model.load_state_dict(ckpt["state_dict"])
            self.target_layer = model.layer4[-1]
            self.model = model
        elif model_name == "mobilenet_v3_large":
            model = create_mobilenet_v3_model(num_classes=num_classes, pretrained=False)
            model.load_state_dict(ckpt["state_dict"])
            self.target_layer = model.features[-1]
            self.model = model
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Enable parameter gradients for Grad-CAM backward propagation
        for p in self.model.parameters():
            p.requires_grad = True

        self.model.to(self.device)
        self.model.eval()

        # Initialize Grad-CAM
        if self.target_layer is not None:
            self.gradcam = GradCAM(self.model, self.target_layer)

        # Load metadata if present
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read model metadata: {e}")

        logger.info(f"Model {self.model_name} initialized and ready for inference.")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Runs inference on a PIL Image and returns prediction details.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Ensure production_model.pt exists.")

        model: nn.Module = self.model
        t0 = time.perf_counter()
        img_rgb = image.convert("RGB")
        tensor = self.transforms(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        pred_idx = int(np.argmax(probs))
        pred_class = self.idx_to_class[pred_idx]
        conf = float(probs[pred_idx])

        probabilities = {
            self.idx_to_class[i]: round(float(probs[i]), 4)
            for i in range(len(probs))
        }

        return {
            "prediction": pred_class,
            "confidence": round(conf, 4),
            "probabilities": probabilities,
            "inference_latency_ms": round(latency_ms, 2),
            "model_version": self.metadata.get("model_version", "1.0.0"),
            "model_name": self.model_name,
        }

    def predict_with_gradcam(self, image: Image.Image) -> Tuple[Dict[str, Any], Image.Image]:
        """
        Runs inference and generates a Grad-CAM overlay PIL Image with thread-safe backward locking.
        """
        if self.model is None or self.gradcam is None:
            raise RuntimeError("Model is not loaded or Grad-CAM is not initialized.")

        model: nn.Module = self.model
        gradcam: GradCAM = self.gradcam
        img_rgb = image.convert("RGB")
        tensor = self.transforms(img_rgb).unsqueeze(0).to(self.device)

        with self._gradcam_lock:
            t0 = time.perf_counter()
            # 1. Forward pass for prediction probabilities
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            pred_idx = int(np.argmax(probs))
            pred_class = self.idx_to_class[pred_idx]
            conf = float(probs[pred_idx])

            # 2. Generate Grad-CAM activation heatmap for predicted class
            heatmap = gradcam.generate_heatmap(tensor, class_idx=pred_idx)
            overlay = overlay_heatmap(img_rgb, heatmap)

        probabilities = {
            self.idx_to_class[i]: round(float(probs[i]), 4)
            for i in range(len(probs))
        }

        res = {
            "prediction": pred_class,
            "confidence": round(conf, 4),
            "probabilities": probabilities,
            "inference_latency_ms": round(latency_ms, 2),
            "model_version": self.metadata.get("model_version", "1.0.0"),
            "model_name": self.model_name,
        }
        return res, overlay


# Global singleton instance
_predictor_instance: Optional[PPEPredictor] = None


def get_predictor() -> PPEPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = PPEPredictor()
    return _predictor_instance
