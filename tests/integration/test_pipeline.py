import json
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_manifest_files_exist():
    raw_meta = PROJECT_ROOT / "data" / "raw" / "dataset_metadata.json"
    label_audit = PROJECT_ROOT / "data" / "interim" / "label_audit.csv"
    split_manifest = PROJECT_ROOT / "data" / "processed" / "split_manifest.csv"
    quality_rep = PROJECT_ROOT / "reports" / "data_quality_report.json"
    leakage_rep = PROJECT_ROOT / "reports" / "data_leakage_report.json"

    assert raw_meta.exists()
    assert label_audit.exists()
    assert split_manifest.exists()
    assert quality_rep.exists()
    assert leakage_rep.exists()


def test_leakage_report_status():
    leakage_rep = PROJECT_ROOT / "reports" / "data_leakage_report.json"
    with open(leakage_rep, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["leakage_gate_passed"] is True
    assert len(data["leakage_violations"]) == 0


def test_model_artifact_and_metadata():
    model_path = PROJECT_ROOT / "artifacts" / "models" / "production_model.pt"
    meta_path = PROJECT_ROOT / "artifacts" / "models" / "model_metadata.json"

    assert model_path.exists()
    assert meta_path.exists()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert "model_name" in meta
    assert "validation_metrics" in meta
