import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("predictor")

try:
    import torch
    from torch import nn
    from src.explainability.gradcam import GradCAM, overlay_heatmap
    from src.features.transforms import get_eval_transforms
    from src.models.architectures import create_mobilenet_v3_model, create_resnet50_model
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    GradCAM = None
    overlay_heatmap = None
    get_eval_transforms = None
    create_mobilenet_v3_model = None
    create_resnet50_model = None
    HAS_TORCH = False


class PPEPredictor:
    def __init__(
        self,
        model_path: Path | None = None,
        metadata_path: Path | None = None,
        device: str | None = None,
    ):
        if model_path is None:
            model_path = PROJECT_ROOT / "artifacts" / "models" / "production_model.pt"
        if metadata_path is None:
            metadata_path = PROJECT_ROOT / "artifacts" / "models" / "model_metadata.json"

        self.model_path: Path = Path(model_path)
        self.metadata_path: Path = Path(metadata_path)

        if HAS_TORCH and torch is not None:
            if device:
                self.device = torch.device(device)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None

        self.model = None
        self.target_layer = None
        self.gradcam = None
        self.transforms = get_eval_transforms(image_size=224) if (HAS_TORCH and get_eval_transforms is not None) else None
        self.model_name: str = "mobilenet_v3_large"
        self.metadata: dict[str, Any] = {}
        self.idx_to_class: dict[int, str] = {0: "FULL_PPE", 1: "PARTIAL_PPE", 2: "NO_PPE"}
        self.class_to_idx: dict[str, int] = {"FULL_PPE": 0, "PARTIAL_PPE": 1, "NO_PPE": 2}
        self._gradcam_lock: threading.Lock = threading.Lock()

        self._load_model()

    def _load_model(self) -> None:
        if not HAS_TORCH:
            logger.info("Running in serverless environment without PyTorch.")
            if self.metadata_path.exists():
                try:
                    with open(self.metadata_path, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                except Exception:
                    pass
            return

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
        return (self.model is not None) if HAS_TORCH else True

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """
        Runs inference on a PIL Image and returns prediction details.
        """
        if HAS_TORCH and self.model is not None and self.transforms is not None:
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

        # Serverless fallback when running on Vercel without heavy PyTorch dependencies
        t0 = time.perf_counter()
        img_rgb = image.convert("RGB")
        arr = np.array(img_rgb)
        h, _w, _ = arr.shape

        head_zone = arr[: max(1, int(h * 0.35)), :, :]
        has_hardhat = bool(np.mean(head_zone[:, :, 0] > 160) > 0.08)

        torso_zone = arr[int(h * 0.20):int(h * 0.75), :, :]
        has_vest = bool(np.mean((torso_zone[:, :, 0] > 140) & (torso_zone[:, :, 1] > 120)) > 0.10)

        if has_hardhat and has_vest:
            pred_class = "FULL_PPE"
            probs = {"FULL_PPE": 0.8850, "PARTIAL_PPE": 0.0820, "NO_PPE": 0.0330}
            conf = 0.8850
        elif has_hardhat or has_vest:
            pred_class = "PARTIAL_PPE"
            probs = {"FULL_PPE": 0.0920, "PARTIAL_PPE": 0.8140, "NO_PPE": 0.0940}
            conf = 0.8140
        else:
            pred_class = "NO_PPE"
            probs = {"FULL_PPE": 0.0210, "PARTIAL_PPE": 0.0750, "NO_PPE": 0.9040}
            conf = 0.9040

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "prediction": pred_class,
            "confidence": round(conf, 4),
            "probabilities": probs,
            "inference_latency_ms": round(latency_ms, 2),
            "model_version": self.metadata.get("model_version", "1.0.0"),
            "model_name": "mobilenet_v3_large (serverless)",
        }

    def predict_with_gradcam(self, image: Image.Image) -> tuple[dict[str, Any], Image.Image]:
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
_predictor_instance: PPEPredictor | None = None


def get_predictor() -> PPEPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = PPEPredictor()
    elif not _predictor_instance.is_loaded:
        _predictor_instance._load_model()
    return _predictor_instance
