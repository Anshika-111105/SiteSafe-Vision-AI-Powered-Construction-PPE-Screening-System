import csv
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.utils.metrics import compute_classification_metrics
from src.features.transforms import get_eval_transforms
from src.models.architectures import create_resnet50_model, create_mobilenet_v3_model
from src.models.dataset import PPEDataset

logger = setup_logger("evaluate")


def load_model_from_checkpoint(checkpoint_path: Path, model_name: str, num_classes: int = 3) -> nn.Module:
    if model_name == "resnet50":
        model = create_resnet50_model(num_classes=num_classes, pretrained=False)
    elif model_name == "mobilenet_v3_large":
        model = create_mobilenet_v3_model(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def select_best_model(
    artifacts_dir: Path,
    selection_config_path: Path,
) -> Dict[str, Any]:
    """
    Applies formal model selection rules using validation metrics only.
    """
    logger.info("Executing Formal Model Selection Gate (Validation Metrics Only)...")
    with open(selection_config_path, "r", encoding="utf-8") as f:
        selection_cfg = yaml.safe_load(f)

    meta_files = list(artifacts_dir.glob("metadata_*.json"))
    if not meta_files:
        raise FileNotFoundError(f"No metadata files found in {artifacts_dir}")

    candidates = {}
    for mf in meta_files:
        with open(mf, "r", encoding="utf-8") as f:
            meta = json.load(f)
            mname = meta["model_name"]
            candidates[mname] = meta

    logger.info(f"Loaded {len(candidates)} candidate models: {list(candidates.keys())}")

    # Evaluate decision criteria
    champion_name = None
    best_score = -1.0
    evaluation_log = []

    for name, meta in candidates.items():
        val_metrics = meta["validation_metrics"]
        macro_f1 = val_metrics["macro_f1"]
        unsafe_recall = val_metrics["safety_audit"]["unsafe_recall_min"]
        val_loss = val_metrics["loss"]

        # Check Gates
        gate1_pass = macro_f1 >= selection_cfg["decision_rules"]["gate_1_macro_f1"]["minimum_threshold"]
        gate2_pass = unsafe_recall >= selection_cfg["decision_rules"]["gate_2_unsafe_recall"]["minimum_threshold"]

        # Composite score prioritizing Macro F1 and Unsafe Recall
        composite_score = macro_f1 * 0.6 + unsafe_recall * 0.4

        evaluation_log.append({
            "model_name": name,
            "val_macro_f1": macro_f1,
            "val_unsafe_recall_min": unsafe_recall,
            "val_loss": val_loss,
            "gate_1_passed": gate1_pass,
            "gate_2_passed": gate2_pass,
            "composite_score": round(composite_score, 4),
        })

        if composite_score > best_score:
            best_score = composite_score
            champion_name = name

    logger.info(f"Champion Model Selected: {champion_name} (Score: {best_score:.4f})")

    # Copy champion model to best_model.pt and production_model.pt
    champion_ckpt = artifacts_dir / f"best_{champion_name}.pt"
    champion_meta = artifacts_dir / f"metadata_{champion_name}.json"

    best_dest = artifacts_dir / "best_model.pt"
    prod_dest = artifacts_dir / "production_model.pt"
    meta_dest = artifacts_dir / "model_metadata.json"

    shutil.copy2(champion_ckpt, best_dest)
    shutil.copy2(champion_ckpt, prod_dest)
    shutil.copy2(champion_meta, meta_dest)

    selection_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "champion_model": champion_name,
        "champion_score": round(best_score, 4),
        "decision_rationale": selection_cfg["rationale"],
        "candidates_evaluated": evaluation_log,
    }

    selection_out = PROJECT_ROOT / "reports" / "model_selection.json"
    with open(selection_out, "w", encoding="utf-8") as f:
        json.dump(selection_summary, f, indent=2)

    return selection_summary


def evaluate_test_set(
    config_path: Path = None,
    artifacts_dir: Path = None,
    reports_dir: Path = None,
) -> Dict[str, Any]:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    if artifacts_dir is None:
        artifacts_dir = PROJECT_ROOT / "artifacts" / "models"
    if reports_dir is None:
        reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Run Formal Model Selection First
    selection_cfg_path = PROJECT_ROOT / "configs" / "model_selection.yaml"
    selection_res = select_best_model(artifacts_dir, selection_cfg_path)
    champion_name = selection_res["champion_model"]

    # 2. Evaluate on Untouched Test Set
    logger.info(f"Evaluating Champion Model '{champion_name}' on Untouched Test Set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_manifest_path = PROJECT_ROOT / config["data"]["split_manifest_path"]
    crops_dir = PROJECT_ROOT / config["data"]["interim_dir"] / "crops"
    eval_tf = get_eval_transforms(
        image_size=config["preprocessing"]["image_size"],
        resize_size=config["preprocessing"]["resize_size"],
    )

    test_dataset = PPEDataset(split_manifest_path, crops_dir, split="test", transform=eval_tf)
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    champion_model = load_model_from_checkpoint(artifacts_dir / "production_model.pt", champion_name)
    champion_model.to(device)

    all_preds = []
    all_targets = []
    all_confidences = []
    all_sample_ids = []

    t0 = time.time()
    with torch.no_grad():
        for images, targets, sample_ids in test_loader:
            images = images.to(device)
            outputs = champion_model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_confidences.extend(confs.cpu().numpy().tolist())
            all_sample_ids.extend(sample_ids)

    latency_total = time.time() - t0
    latency_per_sample_ms = (latency_total / len(test_dataset)) * 1000.0

    class_names = config["data"]["target_classes"]
    metrics = compute_classification_metrics(all_targets, all_preds, class_names=class_names)
    metrics["latency_per_sample_ms"] = round(latency_per_sample_ms, 2)

    # 3. Misclassification & Error Analysis
    misclassified_rows = []
    for sid, true_idx, pred_idx, conf in zip(all_sample_ids, all_targets, all_preds, all_confidences):
        if true_idx != pred_idx:
            misclassified_rows.append({
                "sample_id": sid,
                "true_label": class_names[true_idx],
                "predicted_label": class_names[pred_idx],
                "confidence": round(float(conf), 4),
                "source_image_id": sid.split("_w")[0],
                "split": "test",
                "image_path": str(crops_dir / f"{sid}.jpg"),
            })

    # Save Misclassified Samples CSV
    misclassified_csv_path = reports_dir / "misclassified_samples.csv"
    with open(misclassified_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sample_id",
            "true_label",
            "predicted_label",
            "confidence",
            "source_image_id",
            "split",
            "image_path",
        ])
        writer.writeheader()
        writer.writerows(misclassified_rows)

    # Save Final Evaluation JSON
    eval_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": champion_name,
        "evaluation_split": "test",
        "total_test_samples": len(test_dataset),
        "misclassified_count": len(misclassified_rows),
        "metrics": metrics,
    }

    eval_json_path = reports_dir / "evaluation_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    # Save Markdown Evaluation Report
    eval_md_path = reports_dir / "evaluation_results.md"
    with open(eval_md_path, "w", encoding="utf-8") as f:
        f.write("# SiteSafe Vision: Model Evaluation & Benchmark Report\n\n")
        f.write(f"- **Champion Model**: `{champion_name}`\n")
        f.write(f"- **Evaluation Split**: `test` (Untouched during training and selection)\n")
        f.write(f"- **Test Set Size**: `{len(test_dataset)}` worker crops\n")
        f.write(f"- **Single Image Latency**: `{latency_per_sample_ms:.2f} ms`\n\n")
        f.write("## Overall Performance\n\n")
        f.write(f"- **Accuracy**: `{metrics['accuracy'] * 100:.2f}%`\n")
        f.write(f"- **Macro F1**: `{metrics['macro_f1']:.4f}`\n")
        f.write(f"- **Weighted F1**: `{metrics['weighted_f1']:.4f}`\n")
        f.write(f"- **Macro Recall**: `{metrics['macro_recall']:.4f}`\n\n")
        f.write("## Safety-Critical Audit\n\n")
        s_audit = metrics["safety_audit"]
        f.write(f"- **Unsafe Class Minimum Recall**: `{s_audit['unsafe_recall_min']:.4f}`\n")
        f.write(f"- **Critical NO_PPE False Negatives**: `{s_audit['no_ppe_critical_fn_count']}` (Rate: `{s_audit['no_ppe_critical_fn_rate'] * 100:.2f}%`)\n")
        f.write(f"- **PARTIAL_PPE False Negatives**: `{s_audit['partial_ppe_fn_count']}` (Rate: `{s_audit['partial_ppe_fn_rate'] * 100:.2f}%`)\n\n")
        f.write("## Per-Class Breakdown\n\n")
        f.write("| Class | Precision | Recall | F1 Score | Support |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for cls_name, pstats in metrics["per_class"].items():
            f.write(f"| `{cls_name}` | {pstats['precision']:.4f} | {pstats['recall']:.4f} | {pstats['f1']:.4f} | {pstats['support']} |\n")
        f.write("\n## Confusion Matrix\n\n")
        f.write("Rows = Ground Truth, Columns = Prediction (`[FULL_PPE, PARTIAL_PPE, NO_PPE]`)\n\n")
        f.write("```\n")
        for r in metrics["confusion_matrix"]:
            f.write(f"{r}\n")
        f.write("```\n\n")
        f.write(f"## Error Analysis Summary\n\n")
        f.write(f"- **Total Misclassified Samples**: `{len(misclassified_rows)}`\n")
        f.write(f"- Detailed error trace saved to [`reports/misclassified_samples.csv`](file:///{misclassified_csv_path.as_posix()})\n")

    logger.info(f"Evaluation complete. Accuracy: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}")
    logger.info(f"Reports saved to {eval_json_path} and {eval_md_path}")
    return eval_summary


if __name__ == "__main__":
    evaluate_test_set()
