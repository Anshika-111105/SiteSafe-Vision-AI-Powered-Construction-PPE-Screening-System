#!/usr/bin/env python3
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def get_git_commit(cwd: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_OR_UNCOMMITTED"


def get_git_branch(cwd: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def get_dvc_version(cwd: Path) -> str:
    try:
        res = subprocess.run(
            ["dvc", "version"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=True,
        )
        lines = res.stdout.strip().splitlines()
        return lines[0] if lines else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def compute_config_hash(config_path: Path) -> str:
    if not config_path.exists():
        return "NO_CONFIG_FOUND"
    hasher = hashlib.sha256()
    with open(config_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_environment_report(project_root: Path = None) -> dict:
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    # Check imported library versions
    lib_versions = {}
    for lib in [
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "sklearn",
        "PIL",
        "scipy",
        "fastapi",
        "uvicorn",
        "gradio",
        "pydantic",
        "yaml",
        "pytest",
    ]:
        try:
            mod = __import__(lib)
            lib_versions[lib] = getattr(mod, "__version__", "AVAILABLE")
        except ImportError:
            lib_versions[lib] = "NOT_INSTALLED"

    # PyTorch and CUDA specs
    cuda_available = False
    cuda_version = "None"
    gpu_devices = []
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda or "UNKNOWN"
            gpu_devices = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except Exception:
        pass

    config_path = project_root / "configs" / "config.yaml"
    config_hash = compute_config_hash(config_path)

    # Check dataset metadata if present
    dataset_meta_path = project_root / "data" / "raw" / "dataset_metadata.json"
    dataset_version = "NOT_INGESTED"
    if dataset_meta_path.exists():
        try:
            with open(dataset_meta_path, "r", encoding="utf-8") as f:
                dmeta = json.load(f)
                dataset_version = dmeta.get("dataset_version", "UNKNOWN")
        except Exception:
            dataset_version = "METADATA_UNREADABLE"

    # Check model metadata if present
    model_meta_path = project_root / "artifacts" / "models" / "model_metadata.json"
    model_version = "NOT_TRAINED"
    if model_meta_path.exists():
        try:
            with open(model_meta_path, "r", encoding="utf-8") as f:
                mmeta = json.load(f)
                model_version = mmeta.get("model_version", "UNKNOWN")
        except Exception:
            model_version = "METADATA_UNREADABLE"

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "project_name": "SiteSafe Vision",
        "system": {
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
        },
        "hardware": {
            "cpu_count_logical": os.cpu_count(),
            "cuda_available": cuda_available,
            "cuda_version": cuda_version,
            "gpu_device_count": len(gpu_devices),
            "gpu_devices": gpu_devices,
        },
        "version_control": {
            "git_commit": get_git_commit(project_root),
            "git_branch": get_git_branch(project_root),
            "dvc_version": get_dvc_version(project_root),
        },
        "governance_hashes": {
            "config_hash_sha256": config_hash,
            "dataset_version": dataset_version,
            "model_version": model_version,
        },
        "dependencies": lib_versions,
    }

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "environment.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[SUCCESS] Environment report saved to {report_file}")
    return report


if __name__ == "__main__":
    generate_environment_report()
