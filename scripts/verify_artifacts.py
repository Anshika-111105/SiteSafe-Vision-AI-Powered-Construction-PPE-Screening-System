#!/usr/bin/env python3
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("verify_artifacts")


def compute_sha256(filepath: Path) -> str:
    if not filepath.exists():
        raise FileNotFoundError(f"Tracked artifact not found: {filepath}")
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_OR_UNCOMMITTED"


def generate_checksums_and_lineage(project_root: Path = None) -> bool:
    if project_root is None:
        project_root = PROJECT_ROOT

    reports_dir = project_root / "reports"
    artifacts_dir = project_root / "artifacts" / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)

    tracked_files = [
        project_root / "configs" / "config.yaml",
        project_root / "configs" / "model_selection.yaml",
        project_root / "data" / "raw" / "dataset_metadata.json",
        project_root / "data" / "interim" / "label_audit.csv",
        project_root / "data" / "processed" / "split_manifest.csv",
        reports_dir / "data_manifest.csv",
        reports_dir / "data_quality_report.json",
        reports_dir / "data_leakage_report.json",
        artifacts_dir / "production_model.pt",
        artifacts_dir / "model_metadata.json",
        reports_dir / "evaluation_results.json",
        reports_dir / "model_selection.json",
    ]

    checksum_lines = []
    file_hashes = {}

    for fpath in tracked_files:
        if fpath.exists():
            h = compute_sha256(fpath)
            rel_path = fpath.relative_to(project_root).as_posix()
            checksum_lines.append(f"{h}  {rel_path}\n")
            file_hashes[rel_path] = h
        else:
            logger.warning(f"File {fpath} does not exist yet for checksum.")

    checksums_file = reports_dir / "checksums.sha256"
    with open(checksums_file, "w", encoding="utf-8") as f:
        f.writelines(checksum_lines)
    logger.info(f"Recorded SHA-256 checksums to {checksums_file}")

    # Build Lineage Graph
    dvc_lock_file = project_root / "dvc.lock"
    dvc_lock_hash = compute_sha256(dvc_lock_file) if dvc_lock_file.exists() else "NO_DVC_LOCK_YET"

    env_file = reports_dir / "environment.json"
    env_hash = compute_sha256(env_file) if env_file.exists() else "UNKNOWN"

    model_sha256 = file_hashes.get("artifacts/models/production_model.pt", "UNKNOWN")

    lineage = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": "1.0.0",
        "system_name": "SiteSafe Vision",
        "git_commit": get_git_commit(),
        "dvc_lock_sha256": dvc_lock_hash,
        "dataset_version": "1.0.0",
        "raw_dataset_source": "Roboflow Universe Construction PPE Detection (CC BY 4.0)",
        "config_sha256": file_hashes.get("configs/config.yaml", "UNKNOWN"),
        "model_selection_config_sha256": file_hashes.get("configs/model_selection.yaml", "UNKNOWN"),
        "split_manifest_sha256": file_hashes.get("data/processed/split_manifest.csv", "UNKNOWN"),
        "environment_hash": env_hash,
        "production_model_sha256": model_sha256,
        "champion_model_architecture": "MobileNetV3-Large (torchvision.models.mobilenet_v3_large)",
        "lineage_trace": [
            "SOURCE_DATASET (Roboflow CC BY 4.0)",
            "RAW_INGESTION (data/raw/dataset_metadata.json)",
            "LABEL_DERIVATION (src/data/build_classification_dataset.py -> data/interim/label_audit.csv)",
            "QUALITY_GATE (src/data/quality_audit.py -> reports/data_quality_report.json)",
            "LEAKAGE_PREVENTION & GROUPED_SPLIT (src/data/leakage_audit.py -> data/processed/split_manifest.csv)",
            "TRANSFER_LEARNING_TRAINING (src/models/train.py -> artifacts/models/best_*.pt)",
            "VALIDATION_MODEL_SELECTION (configs/model_selection.yaml -> reports/model_selection.json)",
            "TEST_SET_EVALUATION (src/models/evaluate.py -> reports/evaluation_results.json)",
            "EXPLAINABILITY_GRADCAM (src/explainability/gradcam.py -> reports/explainability/)",
            "PRODUCTION_MODEL_PACKAGE (artifacts/models/production_model.pt)",
        ],
    }

    lineage_file = reports_dir / "lineage.json"
    with open(lineage_file, "w", encoding="utf-8") as f:
        json.dump(lineage, f, indent=2)
    logger.info(f"Recorded provenance lineage to {lineage_file}")

    return True


if __name__ == "__main__":
    generate_checksums_and_lineage()
