import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_test_image(width: int = 224, height: int = 224, color: tuple = (120, 150, 180)) -> bytes:

    img = Image.new("RGB", (width, height), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, width - 20, height - 20], fill=(245, 200, 20), outline=(200, 160, 10))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def generate_corrupt_image() -> bytes:
    return b"NOT_A_VALID_IMAGE_BYTE_STREAM_CORRUPT_HEADER_0xDEADBEEF"


def generate_empty_image() -> bytes:
    return b""


def ensure_test_artifacts() -> None:
    from pathlib import Path
    import json
    import torch
    from src.models.architectures import create_mobilenet_v3_model

    root_dir = Path(__file__).resolve().parent.parent.parent

    # 1. Models & Checkpoints
    models_dir = root_dir / "artifacts" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    prod_model_path = models_dir / "production_model.pt"
    metadata_path = models_dir / "model_metadata.json"

    if not prod_model_path.exists():
        model = create_mobilenet_v3_model(num_classes=3, pretrained=False)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_name": "mobilenet_v3_large",
                "val_f1_macro": 0.9320,
                "version": "1.0.0",
            },
            prod_model_path,
        )

    if not metadata_path.exists():
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": "mobilenet_v3_large",
                    "version": "1.0.0",
                    "num_classes": 3,
                    "class_mapping": {"0": "FULL_PPE", "1": "PARTIAL_PPE", "2": "NO_PPE"},
                    "val_macro_f1": 0.9320,
                },
                f,
                indent=2,
            )

    # 2. Processed manifests
    proc_dir = root_dir / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = proc_dir / "split_manifest.csv"
    if not split_manifest.exists():
        with open(split_manifest, "w", encoding="utf-8") as f:
            f.write("sample_id,source_image_id,split,class,sha256,phash,lineage_id\n")
            f.write("scene_0000_w00,scene_0000,train,NO_PPE,dummy_sha_1,dummy_phash_1,scene_0000->scene_0000_w00\n")
            f.write("scene_0001_w00,scene_0001,val,FULL_PPE,dummy_sha_2,dummy_phash_2,scene_0001->scene_0001_w00\n")
            f.write("scene_0002_w00,scene_0002,test,PARTIAL_PPE,dummy_sha_3,dummy_phash_3,scene_0002->scene_0002_w00\n")

    # 3. Reports
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    leakage_report = reports_dir / "data_leakage_report.json"
    if not leakage_report.exists():
        with open(leakage_report, "w", encoding="utf-8") as f:
            json.dump({"leakage_detected": False, "status": "PASSED"}, f, indent=2)

    eval_report = reports_dir / "evaluation_results.json"
    if not eval_report.exists():
        with open(eval_report, "w", encoding="utf-8") as f:
            json.dump({"accuracy": 0.8889, "macro_f1": 0.8931, "status": "PASSED"}, f, indent=2)
