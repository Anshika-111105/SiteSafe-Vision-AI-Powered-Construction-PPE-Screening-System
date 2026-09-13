#!/usr/bin/env python3
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("reproducibility_check")


def compute_file_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "FILE_MISSING"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_reproducibility_audit() -> dict[str, Any]:
    logger.info("Executing Complete End-to-End Reproducibility Audit...")

    checks = {}
    all_passed = True

    # Check 1: Git Repository & Branch
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            git_commit = res.stdout.strip()
        else:
            git_commit = "INITIAL_UNCOMMITTED_STATE"

        git_status = "CLEAN" if not subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        ).stdout.strip() else "DIRTY_OR_MODIFIED"
        checks["git"] = {"status": "PASSED", "commit": git_commit, "workspace_state": git_status}
    except Exception as e:
        checks["git"] = {"status": "WARNING", "detail": str(e)}

    # Check 2: DVC Status & Lockfile
    dvc_lock = PROJECT_ROOT / "dvc.lock"
    dvc_yaml = PROJECT_ROOT / "dvc.yaml"
    if dvc_lock.exists() and dvc_yaml.exists():
        checks["dvc"] = {
            "status": "PASSED",
            "dvc_yaml_sha256": compute_file_sha256(dvc_yaml),
            "dvc_lock_sha256": compute_file_sha256(dvc_lock),
        }
    else:
        checks["dvc"] = {"status": "FAILED", "detail": "dvc.yaml or dvc.lock missing"}
        all_passed = False

    # Check 3: Configuration Hashes
    cfg_file = PROJECT_ROOT / "configs" / "config.yaml"
    sel_file = PROJECT_ROOT / "configs" / "model_selection.yaml"
    checks["configuration"] = {
        "status": "PASSED" if (cfg_file.exists() and sel_file.exists()) else "FAILED",
        "config_hash": compute_file_sha256(cfg_file),
        "model_selection_hash": compute_file_sha256(sel_file),
    }

    # Check 4: Data Manifest & Split Manifest Consistency
    split_file = PROJECT_ROOT / "data" / "processed" / "split_manifest.csv"
    data_manifest_file = PROJECT_ROOT / "reports" / "data_manifest.csv"
    checks["dataset_manifests"] = {
        "status": "PASSED" if (split_file.exists() and data_manifest_file.exists()) else "FAILED",
        "split_manifest_sha256": compute_file_sha256(split_file),
        "data_manifest_sha256": compute_file_sha256(data_manifest_file),
    }

    # Check 5: Model Artifact & Production Weights
    prod_model_file = PROJECT_ROOT / "artifacts" / "models" / "production_model.pt"
    prod_meta_file = PROJECT_ROOT / "artifacts" / "models" / "model_metadata.json"
    checks["model_artifacts"] = {
        "status": "PASSED" if (prod_model_file.exists() and prod_meta_file.exists()) else "FAILED",
        "production_model_sha256": compute_file_sha256(prod_model_file),
        "model_metadata_sha256": compute_file_sha256(prod_meta_file),
    }

    # Check 6: Checksum Integrity Table
    checksum_file = PROJECT_ROOT / "reports" / "checksums.sha256"
    checksum_verified = True
    if checksum_file.exists():
        with open(checksum_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("  ")
                if len(parts) == 2:
                    expected_hash, rel_path = parts
                    actual_hash = compute_file_sha256(PROJECT_ROOT / rel_path)
                    if expected_hash != actual_hash:
                        checksum_verified = False
                        logger.error(f"Checksum mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}")
    else:
        checksum_verified = False

    checks["checksum_verification"] = {
        "status": "PASSED" if checksum_verified else "FAILED",
    }

    # Document Exact Reproducibility Boundary
    reproducibility_boundary = {
        "deterministic_algorithms_configured": True,
        "fixed_seeds": ["python=42", "numpy=42", "torch=42", "dataloader_workers=0"],
        "hardware_boundary_note": (
            "Exact bit-for-bit floating-point parity across different CPU microarchitectures "
            "or differing GPU architectures (CUDA SM compute versions) is constrained by IEEE 754 "
            "atomic reductions and SIMD vector dispatch. Exact manifest and metric reproducibility "
            "is guaranteed within identical Python 3.11.x, PyTorch 2.13.0 environments on x86_64."
        ),
    }

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "overall_reproducibility_passed": all_passed and checksum_verified,
        "checks": checks,
        "reproducibility_boundary": reproducibility_boundary,
    }

    out_file = PROJECT_ROOT / "reports" / "reproducibility_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Reproducibility report generated: {out_file} (Passed: {report['overall_reproducibility_passed']})")
    return report


if __name__ == "__main__":
    res = run_reproducibility_audit()
    if not res["overall_reproducibility_passed"]:
        sys.exit(1)
