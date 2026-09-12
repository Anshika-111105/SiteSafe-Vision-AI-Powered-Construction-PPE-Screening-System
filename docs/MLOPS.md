# MLOps Architecture & Pipeline Specification

## 1. MLOps Lifecycle Overview

SiteSafe Vision integrates Data Version Control (DVC), Git, PyTorch, FastAPI, and Docker into a unified production pipeline [^dvc_doc, ^fastapi_docker].

```
SOURCE DATA (Roboflow CC BY 4.0)
    ↓
DVC PIPELINE (dvc.yaml 10 Stages)
    ↓
VALIDATION SELECTION GATE (configs/model_selection.yaml)
    ↓
PRODUCTION PACKAGE (artifacts/models/production_model.pt)
    ↓
FASTAPI MICROSERVICE & DOCKER CONTAINER
    ↓
MONITORING & GRADIO USER INTERFACE
```

---

## 2. Git & DVC Separation of Concerns

- **Git Tracks**: Source code, test suites, documentation, pipeline DAG (`dvc.yaml`), lockfiles (`dvc.lock`), dependency configurations (`pyproject.toml`), and CI workflows (`.github/workflows/ci.yml`).
- **DVC Tracks**: Raw images, extracted worker crops, model binaries (`.pt`), and cached evaluation metrics [^dvc_doc].

---

## 3. The 10-Stage DVC Pipeline

| Stage | Command | Key Dependencies | Primary Outputs |
| :--- | :--- | :--- | :--- |
| `data_ingest` | `src/data/ingest_dataset.py` | `configs/config.yaml` | `data/raw/dataset_metadata.json` |
| `build_classification_dataset` | `src/data/build_classification_dataset.py` | `data/raw/` | `data/interim/label_audit.csv`, `data/interim/crops/` |
| `data_validate` | `src/data/quality_audit.py` | `data/interim/crops/` | `reports/data_manifest.csv`, `reports/data_quality_report.json` |
| `data_deduplicate_and_leakage_audit` | `src/data/leakage_audit.py` | `reports/data_manifest.csv` | `data/processed/split_manifest.csv`, `reports/data_leakage_report.json` |
| `train_resnet` | `src/models/train.py --model resnet50` | `data/processed/split_manifest.csv` | `artifacts/models/best_resnet50.pt` |
| `train_mobilenet` | `src/models/train.py --model mobilenet_v3_large` | `data/processed/split_manifest.csv` | `artifacts/models/best_mobilenet_v3_large.pt` |
| `evaluate_and_select` | `src/models/evaluate.py` | `configs/model_selection.yaml` | `artifacts/models/production_model.pt`, `reports/evaluation_results.json` |
| `explainability` | `src/explainability/gradcam.py` | `artifacts/models/production_model.pt` | `reports/explainability/` |
| `package_and_lineage` | `scripts/verify_artifacts.py` | `artifacts/models/production_model.pt` | `reports/lineage.json`, `reports/checksums.sha256` |
| `environment_report` | `scripts/environment_report.py` | `configs/config.yaml` | `reports/environment.json` |

---

## 4. Lineage Graph (`reports/lineage.json`)

For every released model, SiteSafe Vision records an immutable provenance chain linking:
$$\text{Model Version} \rightarrow \text{Git Commit} \rightarrow \text{DVC Lockfile} \rightarrow \text{Data Manifest} \rightarrow \text{Config SHA-256} \rightarrow \text{Artifact SHA-256}$$

---

## 5. References & Citations

[^dvc_doc]: DVC (Data Version Control) Pipeline and Artifact Versioning Guide. URL: [https://dvc.org/doc](https://dvc.org/doc)
[^fastapi_docker]: FastAPI Docker and Production Deployment Guide. URL: [https://fastapi.tiangolo.com/deployment/docker/](https://fastapi.tiangolo.com/deployment/docker/)
