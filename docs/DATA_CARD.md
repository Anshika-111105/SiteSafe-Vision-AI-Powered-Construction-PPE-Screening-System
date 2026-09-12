# Data Card: SiteSafe Vision Construction PPE Dataset

## 1. Dataset Overview & Provenance

- **Dataset Name**: Construction PPE Detection Dataset [^roboflow_source]
- **Primary Source Repository**: Roboflow Universe
- **Dataset Source URL**: [https://universe.roboflow.com/new-project-ds9wg/construction-ppe-detection-vqbc0](https://universe.roboflow.com/new-project-ds9wg/construction-ppe-detection-vqbc0)
- **Publisher / Maintainer**: Roboflow Universe Community (`new-project-ds9wg`)
- **Published License**: Creative Commons Attribution 4.0 International (CC BY 4.0) [^cc_by_40]
- **Allowed Usage**: Commercial and non-commercial reproduction, transformation, and adaptation with appropriate attribution.
- **Dataset Version Tracked**: `1.0.0`
- **Transformation Engine Version**: `1.0.0` (`src/data/build_classification_dataset.py`)

---

## 2. Annotation & Label Governance

### 2.1 Original Annotation Classes
The raw primary dataset contains object-detection bounding boxes across 5 classes:
1. `human` (Worker bounding box)
2. `helmet` (Hardhat / protective headwear)
3. `vest` (High-visibility reflective safety vest)
4. `boots` (Work / protective footwear)
5. `gloves` (Hand protective equipment)

### 2.2 Derived Classification Classes
Because SiteSafe Vision is an image-based screening classifier, raw object detections are systematically converted into worker-centered classification examples using deterministic business logic:
- `FULL_PPE`: Worker crop where both `helmet` AND `vest` are present.
- `PARTIAL_PPE`: Worker crop where either `helmet` OR `vest` is present, but not both.
- `NO_PPE`: Worker crop where neither `helmet` nor `vest` is present.

### 2.3 Transformation Rules & Ambiguity Rejection
1. **Spatial Overlap**:
   - `helmet` is associated with a worker if its center is within the worker bounding box and located in the top 40% vertical band.
   - `vest` is associated with a worker if its center is within the worker bounding box and located between 20% and 85% vertical band.
2. **Rejection Thresholds**:
   - Crops with height or width $< 32\text{px}$ are rejected (`WORKER_TOO_SMALL`).
   - Crops extending more than 50% outside image boundaries are rejected (`SEVERE_BOUNDARY_TRUNCATION`).
3. **Auditability**:
   - Every candidate is recorded in `data/interim/label_audit.csv` with columns: `source_image_id`, `crop_id`, `worker_bbox`, `helmet_present`, `vest_present`, `boots_present`, `gloves_present`, `derived_label`, `label_confidence`, `rejection_reason`, `transformation_version`.

---

## 3. Data Leakage Prevention & Splitting

- **Grouping Strategy**: `group_by_source_image_id`. All worker crops extracted from the exact same source image are strictly assigned to the same split.
- **Ratios**: Train: 70%, Validation: 15%, Test: 15% (Deterministic Random Seed: `42`).
- **Integrity Validation**: SHA-256 hash collision checks and perceptual difference hash (dHash) audits are executed prior to training (`src/data/leakage_audit.py`).
- **Artifact**: Tracked in `data/processed/split_manifest.csv` and audited in `reports/data_leakage_report.json`.

---

## 4. Known Limitations & Domain Biases

- **Geographic & Environmental Bias**: Imagery predominantly features daylight outdoor building construction scenes. Night shifts, heavy precipitation, and subterranean tunneling are underrepresented.
- **Occlusion**: Workers partially hidden behind scaffolding or machinery may have obscured PPE resulting in classification uncertainty.
- **Apparel Variety**: Extreme specialized PPE (e.g. welding masks, chemical suits, harnesses) are not modeled in this 3-class baseline.
- **Resolution Boundaries**: Workers distant from the camera (< 32px height) cannot be reliably classified and must be filtered out.

---

## 5. Intended vs. Non-Intended Use

### Intended Use
- Automated secondary screening of worker-centered images to flag potential PPE oversights.
- Continuous audit logging of PPE compliance trends on job sites.

### Non-Intended Use
- **NOT** an autonomous safety inspector.
- **NOT** a certified OSHA or ISO workplace compliance auditing tool.
- **NOT** an autonomous entry barrier controller without human-in-the-loop verification.

---

## 6. References & Citations

[^roboflow_source]: Roboflow Universe Construction PPE Detection Dataset (CC BY 4.0). URL: [https://universe.roboflow.com/new-project-ds9wg/construction-ppe-detection-vqbc0](https://universe.roboflow.com/new-project-ds9wg/construction-ppe-detection-vqbc0)
[^cc_by_40]: Creative Commons Attribution 4.0 International Public License (CC BY 4.0). URL: [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)
