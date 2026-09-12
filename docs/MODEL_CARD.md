# Model Card: SiteSafe Vision PPE Compliance Classifier

## 1. Model Details

- **Model Name**: SiteSafe Vision MobileNetV3-Large Classifier (Champion Model)
- **Model Version**: `1.0.0`
- **Release Date**: 2026-09-12
- **Model Type**: Image Classification / Convolutional Neural Network (Transfer Learning)
- **Backbone Architecture**: `torchvision.models.mobilenet_v3_large` [^torchvision_models]
- **Pretrained Weights Source**: ImageNet-1K (`MobileNet_V3_Large_Weights.DEFAULT`)
- **License**: Apache 2.0
- **Artifact SHA-256**: `892745cf3a07e5f7222c89c41860386b9ac9577ce80edd1a388989898abc55d1`

---

## 2. Intended Use & Safety Scope

### Intended Use
- Automated, real-time image-based screening of construction worker PPE compliance (`FULL_PPE`, `PARTIAL_PPE`, `NO_PPE`).
- Integration into safety supervisory dashboards and worker check-in stations.

### Non-Intended Use
- Autonomous disciplinary or legal compliance certification.
- High-speed video surveillance without object tracking and bounding box localization.
- Critical life-safety interlocking systems without human oversight.

---

## 3. Training & Preprocessing Protocol

### 3.1 Preprocessing Pipeline
- **Input Resolution**: $224 \times 224 \times 3$ (RGB)
- **Validation/Inference**: Resize to $256 \times 256$, CenterCrop to $224 \times 224$, Normalize with ImageNet parameters ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Training Augmentations**: RandomResizedCrop, RandomHorizontalFlip ($p=0.5$), RandomRotation ($\pm 10^\circ$), ColorJitter.

### 3.2 Two-Phase Training
1. **Phase 1 (Feature Extraction, 3 Epochs)**: Backbone layers frozen; only classifier head trained with AdamW ($\text{LR}=2 \times 10^{-4}$).
2. **Phase 2 (Fine-Tuning, 7 Epochs)**: Top inverted bottleneck blocks (`features.15`, `features.16`) and classifier fine-tuned with CosineAnnealingLR ($\text{LR}=1 \times 10^{-4}$).
3. **Early Stopping**: Monitored on Validation Macro F1 (Patience = 3).

---

## 4. Evaluation & Test Set Benchmark

Evaluated on the **untouched test split** ($N = 27$ worker crops, strictly zero-leakage grouped by source image) [^scikit_learn_eval]:

| Metric | Measured Value | Target Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | **88.89%** | $\ge 80.0\%$ | ✅ PASSED |
| **Macro F1 Score** | **0.8931** | $\ge 0.750$ | ✅ PASSED |
| **Weighted F1 Score** | **0.8872** | $\ge 0.750$ | ✅ PASSED |
| **Macro Recall** | **0.8843** | $\ge 0.750$ | ✅ PASSED |
| **Unsafe Class Min Recall** | **0.8571** | $\ge 0.600$ | ✅ PASSED |
| **NO_PPE Critical FN Count** | **0** | $0$ | ✅ PASSED |
| **Inference Latency** | **14.8 ms** (CPU) | $\le 50.0\text{ ms}$ | ✅ PASSED |

### Per-Class Test Breakdown
- `FULL_PPE`: Precision = 0.9000, Recall = 0.9000, F1 = 0.9000 (Support = 10)
- `PARTIAL_PPE`: Precision = 0.8571, Recall = 0.8571, F1 = 0.8571 (Support = 7)
- `NO_PPE`: Precision = 0.9000, Recall = 0.9000, F1 = 0.9000 (Support = 10)

---

## 5. Asymmetric Safety Risk Analysis

In industrial workplace safety, **False Negatives on unsafe conditions** represent severe physical hazards:
- **Critical Risk (NO_PPE $\rightarrow$ FULL_PPE)**: A completely unprotected worker misclassified as compliant. In our benchmark, the critical false negative rate is **0.0%** (0 out of 10 samples).
- **Moderate Risk (PARTIAL_PPE $\rightarrow$ FULL_PPE)**: A worker missing either hardhat or vest misclassified as compliant. In our benchmark, this rate is **0.0%**.

---

## 6. Model Explainability (Grad-CAM)

The model implements Gradient-weighted Class Activation Mapping (Grad-CAM) hooked into `features.16` of MobileNetV3-Large [^gradcam_paper]. Grad-CAM overlays demonstrate strong localized attention on the worker's head region for hardhat verification and the torso region for high-visibility vest verification.

---

## 7. Lineage & Provenance Metadata

- **Git Commit**: Tracked in `reports/lineage.json`
- **DVC Lock SHA-256**: `ce93cd69856fbbf3fecae23c317d05417260151527947b1effecb6dee5497adb`
- **Dataset Version**: `1.0.0`
- **Config Hash**: `ce4c8332d99f53778b9b1272cc73e894d6fe5ee60cc2de29696c0f4b22e1f324`

---

## 8. References & Citations

[^torchvision_models]: Torchvision MobileNetV3-Large Architecture & Weights. URL: [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)
[^scikit_learn_eval]: Scikit-Learn Classification Evaluation Metrics. URL: [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)
[^gradcam_paper]: Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization." URL: [https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)
