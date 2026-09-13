# SiteSafe Vision
### AI-Powered Construction PPE Compliance Screening System

[![CI Pipeline](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=github-actions&logoColor=white)](.github/workflows/ci.yml)
[![DVC Tracked](https://img.shields.io/badge/Data_Version_Control-DVC-945DD6?logo=dvc&logoColor=white)](dvc.yaml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)

---

> **SAFETY & REGULATORY DISCLAIMER**: **SiteSafe Vision is an AI-based image screening aid.** It does **not** perform certified safety inspections, autonomous entry gating, or legally binding regulatory safety auditing. It is designed to assist site safety supervisors through standardized, explainable compliance screening.

---

## 1. Project Overview

Construction environments require strict adherence to **Personal Protective Equipment (PPE)** standards. SiteSafe Vision classifies worker-centered image crops into three discrete compliance categories:

1. **`FULL_PPE`**: Worker is wearing both a hardhat and a high-visibility safety vest.
2. **`PARTIAL_PPE`**: Worker is missing either a hardhat or a high-visibility safety vest.
3. **`NO_PPE`**: Worker has neither mandatory item present.

In addition to classification, the system computes:
- **Calibrated Confidence**: Multi-class softmax distribution.
- **Risk Level**: `LOW`, `MEDIUM`, or `HIGH` risk rating.
- **Actionable Recommendation**: Standard operating safety instructions for site supervisors.
- **Grad-CAM Saliency Maps**: Visual heatmaps explaining model focus on protective apparel [^gradcam_paper].

---

## 2. System Architecture

```
                                  [ Roboflow Universe Source (CC BY 4.0) ]
                                                     │
                                                     ▼
                                       [ src/data/ingest_dataset.py ]
                                                     │
                                                     ▼
                                 [ src/data/build_classification_dataset.py ]
                                (Bounding Box Containment & Label Derivation)
                                                     │
                                                     ▼
                                         [ src/data/quality_audit.py ]
                                         (Zero-Corruption Quality Gate)
                                                     │
                                                     ▼
                                         [ src/data/leakage_audit.py ]
                                    (Grouped Split by Source Image ID)
                                                     │
                                                     ▼
                                            [ 10-Stage DVC Pipeline ]
                                  (ResNet50 vs. MobileNetV3 Transfer Learning)
                                                     │
                                                     ▼
                                       [ configs/model_selection.yaml ]
                                  (Validation-Based Champion Selection)
                                                     │
                                                     ▼
                                         [ src/models/evaluate.py ]
                                    (Untouched Test Set Evaluation)
                                                     │
                                                     ▼
                      ┌──────────────────────────────┴──────────────────────────────┐
                      ▼                                                             ▼
           [ FastAPI Microservice ]                                      [ Gradio Web Demo ]
         (GET /health, POST /predict)                                      (demo/app.py)
```

---

## 3. Primary Dataset & Data Governance

- **Source**: Construction PPE Detection Dataset (Roboflow Universe) [^roboflow_source].
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0) [^cc_by_40].
- **Original Classes**: `human`, `helmet`, `vest`, `boots`, `gloves`.
- **Derived Classes**: `FULL_PPE`, `PARTIAL_PPE`, `NO_PPE`.
- **Transformation Logic**: Auditable worker-centered cropping with relative vertical spatial bands (Hardhat: top 40%; Vest: 20%-85% torso). Full decision log saved to `data/interim/label_audit.csv`.
- **Data Leakage Guarantee**: Strict grouped splitting by `source_image_id`. All crops from the same scene remain exclusively in the same split (`reports/data_leakage_report.json`).

---

## 4. Model Benchmark & Safety Evaluation

Evaluated on the **untouched test split** ($N = 27$ worker crops) [^scikit_learn_eval]:

| Metric | Champion: MobileNetV3-Large | Baseline: ResNet50 | Target Gate | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Test Accuracy** | **88.89%** | 81.48% | $\ge 80.0\%$ | ✅ PASSED |
| **Macro F1 Score** | **0.8931** | 0.8124 | $\ge 0.750$ | ✅ PASSED |
| **Unsafe Class Min Recall** | **0.8571** | 0.7143 | $\ge 0.600$ | ✅ PASSED |
| **NO_PPE Critical FN Count** | **0** (0.0%) | 0 (0.0%) | $0$ | ✅ PASSED |
| **Inference Latency (CPU)** | **14.8 ms** | 54.7 ms | $\le 50.0\text{ ms}$ | ✅ PASSED |
| **Model Size** | **17.0 MB** | 90.1 MB | $\le 100\text{ MB}$ | ✅ PASSED |

---

## 5. Quickstart & Installation

### Option A: Local Virtual Environment
```bash
# 1. Clone repository
git clone <repo_url> sitesafe-vision
cd sitesafe-vision

# 2. Bootstrap system (Linux/macOS)
chmod +x scripts/bootstrap.sh
./scripts/bootstrap.sh

# Or on Windows PowerShell:
.\scripts\bootstrap.ps1
```

### Option B: Manual Setup with Pip / uv
```bash
# Install pinned dependencies
pip install -e ".[dev]"

# Reproduce entire DVC pipeline
dvc repro

# Run test suite
pytest tests/unit tests/integration -v
```

---

## 6. FastAPI Microservice

Start the production REST API:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Endpoints
- `GET /health`: Service health status, active model, and timestamp.
- `GET /metadata`: Model architecture, training commit, and validation metrics.
- `GET /version`: Semantic versioning and Git commit hash.
- `POST /predict`: Upload a single worker crop image.
- `POST /predict/batch`: Upload multiple worker crop images simultaneously.

### Prediction Request Example
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/interim/crops/scene_0000_w00.jpg"
```

### JSON Response
```json
{
  "prediction": "FULL_PPE",
  "confidence": 0.9412,
  "risk_level": "LOW",
  "recommendation": "COMPLIANT: Standard PPE detected (Hardhat and High-Visibility Vest). Authorized for site entry under standard safety protocols.",
  "model_version": "1.0.0",
  "request_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "probabilities": {
    "FULL_PPE": 0.9412,
    "PARTIAL_PPE": 0.0451,
    "NO_PPE": 0.0137
  },
  "inference_latency_ms": 14.82
}
```

---

## 7. Interactive Gradio Web Demo

Launch the local interactive Gradio screening dashboard:
```bash
python demo/app.py
```
Open [http://localhost:7860](http://localhost:7860) in your browser.

---

## 8. Production Docker Container

Build and execute the secured, non-root OCI container [^fastapi_docker, ^docker_security]:
```bash
# Build container image
docker build -t sitesafe-vision:1.0.0 .

# Run container with port forwarding
docker run -p 8000:8000 --name sitesafe-vision-app sitesafe-vision:1.0.0

# Verify health probe
curl http://localhost:8000/health
```

---

## 9. CI/CD & Automated Quality Gates

GitHub Actions workflow (`.github/workflows/ci.yml`):
1. **Linting & Code Quality**: Ruff and Flake8 syntax checks.
2. **Automated Testing**: 26 unit and integration tests (100% passing).
3. **Artifact Integrity**: SHA-256 manifest and lineage graph validation.
4. **Citation Audit**: 100% verification of external factual citations.
5. **Docker Smoke Test**: Container build, container startup, `/health`, `/metadata`, and `/predict` smoke tests.

---

## 10. Complete Documentation Index

- [System Architecture](docs/ARCHITECTURE.md)
- [Data Card & Governance](docs/DATA_CARD.md)
- [Model Card](docs/MODEL_CARD.md)
- [Reproducibility Specification](docs/REPRODUCIBILITY.md)
- [MLOps Pipeline](docs/MLOPS.md)
- [Security & Privacy](docs/SECURITY.md)
- [Testing Guide](docs/TESTING.md)
- [System Limitations](docs/LIMITATIONS.md)

---

## 11. Official References & Citations

[^roboflow_source]: Roboflow Universe Construction PPE Detection Dataset (CC BY 4.0). URL: [https://universe.roboflow.com/new-project-ds9wg/construction-ppe-detection-vqbc0](https://universe.roboflow.com/new-project-ds9wg/construction-ppe-detection-vqbc0)
[^cc_by_40]: Creative Commons Attribution 4.0 International License (CC BY 4.0). URL: [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)
[^torchvision_models]: PyTorch Torchvision Model Zoo Documentation. URL: [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)
[^pytorch_transfer_learning]: PyTorch Transfer Learning Tutorial. URL: [https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
[^pytorch_randomness]: PyTorch Reproducibility & Randomness Notes. URL: [https://pytorch.org/docs/stable/notes/randomness.html](https://pytorch.org/docs/stable/notes/randomness.html)
[^dvc_doc]: DVC (Data Version Control) Documentation. URL: [https://dvc.org/doc](https://dvc.org/doc)
[^fastapi_docker]: FastAPI Container Deployment Guidance. URL: [https://fastapi.tiangolo.com/deployment/docker/](https://fastapi.tiangolo.com/deployment/docker/)
[^gradcam_paper]: Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization." IEEE ICCV. URL: [https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)
[^scikit_learn_eval]: Scikit-Learn Model Evaluation Guidance. URL: [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)
[^docker_security]: Docker OCI Security Best Practices. URL: [https://docs.docker.com/develop/security-best-practices/](https://docs.docker.com/develop/security-best-practices/)
