# SiteSafe Vision: System Architecture Specification

## 1. Executive Summary

**SiteSafe Vision** is a deterministic, auditable, production-grade Machine Learning system for automated **Personal Protective Equipment (PPE) compliance screening** in construction environments. The system processes worker-centered image crops and classifies compliance into three discrete states:
- `FULL_PPE`: Mandatory PPE present (both hardhat and high-visibility safety vest).
- `PARTIAL_PPE`: Incomplete PPE detected (missing either hardhat or high-visibility safety vest).
- `NO_PPE`: Non-compliant state (absence of both primary protective items).

In addition to discrete predictions, the system produces calibrated confidence intervals, standardized risk tiers (`LOW`, `MEDIUM`, `HIGH`), actionable supervisory recommendations, and Gradient-weighted Class Activation Maps (Grad-CAM) for visual auditability.

---

## 2. End-to-End System Architecture

![SiteSafe Vision Architecture Workflow](Architecture%20workflow.png)



---

## 3. Component Breakdown

### 3.1 Data Pipeline (`src/data/`)
- **Ingestion (`ingest_dataset.py`)**: Downloads or deterministically builds raw bounding box annotations and records source provenance in `data/raw/dataset_metadata.json` [^roboflow_source].
- **Classification Builder (`build_classification_dataset.py`)**: Transforms object detection annotations into worker-centered crops with spatial containment logic (e.g. helmet on upper 40% of torso, vest on middle 25%-80% band). Rejects small (< 32x32px) or truncated crops and logs every candidate in `data/interim/label_audit.csv`.
- **Quality Gate (`quality_audit.py`)**: Validates image readability, format validity, RGB conversion, corrupt files, extreme aspect ratios, and class imbalance. Generates `reports/data_manifest.csv`.
- **Leakage Prevention (`leakage_audit.py`)**: Groups worker crops strictly by `source_image_id`. Guarantees zero train/val/test data leakage, no SHA-256 collisions, and creates `data/processed/split_manifest.csv`.

### 3.2 Modeling & Transfer Learning (`src/models/`)
- **Architectures (`architectures.py`)**:
  - `ResNet50`: Deep residual network with replaced 3-class classification head [^torchvision_models].
  - `MobileNetV3-Large`: High-efficiency inverted residual mobile architecture [^torchvision_models].
- **Two-Phase Training (`train.py`)**:
  - Phase 1: Frozen backbone, feature extraction on linear classification head.
  - Phase 2: Unfrozen top layers with CosineAnnealingLR and AdamW optimizer [^pytorch_transfer_learning].
  - Early stopping monitored on Validation Macro F1.
- **Evaluation & Selection (`evaluate.py`)**: Applies formal validation decision rules from `configs/model_selection.yaml` prior to test evaluation to prevent data snooping [^scikit_learn_eval].

### 3.3 Interpretability & Explainability (`src/explainability/`)
- **Grad-CAM (`gradcam.py`)**: Computes gradient-weighted feature activation heatmaps on the final convolutional layer (`layer4` for ResNet50, `features.16` for MobileNetV3) to visualize network spatial focus [^gradcam_paper].

### 3.4 Production Microservice (`app/`)
- **FastAPI (`main.py`)**: High-performance asynchronous REST endpoints (`/health`, `/metadata`, `/predict`, `/predict/batch`, `/version`) [^fastapi_docker].
- **Middleware (`middleware.py`)**: Request ID tracking, latency telemetry headers, and 10MB payload size restriction.
- **Risk Matrix (`risk.py`)**: Deterministic safety recommendation mapping based on prediction class and confidence score.

---

## 4. References & Citations

[^roboflow_source]: Roboflow Universe Construction PPE Detection Dataset (CC BY 4.0). URL: [https://universe.roboflow.com/new-project-ds9wg/construction-ppe-detection-vqbc0](https://universe.roboflow.com/new-project-ds9wg/construction-ppe-detection-vqbc0)
[^torchvision_models]: PyTorch Torchvision Pretrained Model Architectures. URL: [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)
[^pytorch_transfer_learning]: PyTorch Transfer Learning Tutorial. URL: [https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
[^gradcam_paper]: Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization." IEEE ICCV. URL: [https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)
[^scikit_learn_eval]: Scikit-Learn Model Evaluation Metrics Guidance. URL: [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)
[^fastapi_docker]: FastAPI Official Docker Deployment Documentation. URL: [https://fastapi.tiangolo.com/deployment/docker/](https://fastapi.tiangolo.com/deployment/docker/)
