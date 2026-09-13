import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from PIL import Image
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.transforms import get_eval_transforms
from src.models.architectures import create_mobilenet_v3_model, create_resnet50_model
from src.utils.logger import setup_logger

logger = setup_logger("gradcam")


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        score = output[0, class_idx]
        score.backward()

        # Global average pooling on gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight activations by pooled gradients
        activations = self.activations[0]
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val

        return heatmap


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45, colormap_name: str = "jet") -> Image.Image:
    """
    Overlays Grad-CAM heatmap onto a PIL RGB image.
    """
    img_w, img_h = image.size
    heatmap_pil = Image.fromarray(np.uint8(255 * heatmap)).resize((img_w, img_h), Image.Resampling.BILINEAR)
    heatmap_norm = np.array(heatmap_pil, dtype=np.float32) / 255.0

    try:
        colormap = plt.get_cmap(colormap_name)
    except Exception:
        colormap = cm.get_cmap(colormap_name)
    heatmap_colored = colormap(heatmap_norm)[:, :, :3]  # RGB
    heatmap_colored = np.uint8(255 * heatmap_colored)

    orig_arr = np.array(image.convert("RGB"))
    blended = np.uint8(orig_arr * (1.0 - alpha) + heatmap_colored * alpha)
    return Image.fromarray(blended)


def run_explainability_report(
    model_path: Path = None,
    crops_dir: Path = None,
    output_dir: Path = None,
    num_samples: int = 6,
) -> Path:
    if model_path is None:
        model_path = PROJECT_ROOT / "artifacts" / "models" / "production_model.pt"
    if crops_dir is None:
        crops_dir = PROJECT_ROOT / "data" / "interim" / "crops"
    if output_dir is None:
        output_dir = PROJECT_ROOT / "reports" / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating Grad-CAM Explainability Visualizations...")

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint {model_path} not found.")

    ckpt = torch.load(model_path, map_location="cpu")
    model_name = ckpt.get("model_name", "resnet50")

    if model_name == "resnet50":
        model = create_resnet50_model(num_classes=3, pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        target_layer = model.layer4[-1]
    elif model_name == "mobilenet_v3_large":
        model = create_mobilenet_v3_model(num_classes=3, pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        target_layer = model.features[-1]
    else:
        raise ValueError(f"Unknown architecture {model_name}")

    gradcam = GradCAM(model, target_layer)
    eval_tf = get_eval_transforms(224, 256)
    idx_to_class = {0: "FULL_PPE", 1: "PARTIAL_PPE", 2: "NO_PPE"}

    crop_images = sorted(list(crops_dir.glob("*.jpg")))[:num_samples]
    if not crop_images:
        logger.warning(f"No crop images found in {crops_dir}")
        return output_dir

    for crop_path in crop_images:
        sample_id = crop_path.stem
        with Image.open(crop_path) as img:
            orig_img = img.convert("RGB")

        input_tensor = eval_tf(orig_img).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            pred_label = idx_to_class[pred_idx]
            conf = probs[pred_idx].item()

        # Generate Grad-CAM for predicted class
        heatmap = gradcam.generate_heatmap(input_tensor, class_idx=pred_idx)
        overlay_img = overlay_heatmap(orig_img, heatmap)

        # Plot comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig_img)
        axes[0].set_title(f"Worker Crop: {sample_id}")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(overlay_img)
        axes[2].set_title(f"Pred: {pred_label} ({conf * 100:.1f}%)")
        axes[2].axis("off")

        plt.tight_layout()
        out_fig_path = output_dir / f"gradcam_{sample_id}.png"
        plt.savefig(out_fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"Grad-CAM explainability figures generated in {output_dir}")
    return output_dir


if __name__ == "__main__":
    run_explainability_report()
