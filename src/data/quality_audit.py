import csv
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("quality_audit")


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_phash(img: Image.Image, hash_size: int = 8) -> str:
    """
    Computes perceptual difference hash (dHash) for near-duplicate image detection.
    """
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(resized, dtype=np.float32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return "".join(["1" if b else "0" for b in diff.flatten()])


def run_quality_audit(
    raw_dir: Path = None,
    interim_dir: Path = None,
    reports_dir: Path = None,
) -> dict[str, Any]:
    if raw_dir is None:
        raw_dir = PROJECT_ROOT / "data" / "raw"
    if interim_dir is None:
        interim_dir = PROJECT_ROOT / "data" / "interim"
    if reports_dir is None:
        reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    crops_dir = interim_dir / "crops"
    label_audit_path = interim_dir / "label_audit.csv"

    logger.info("Executing formal Data Quality Audit...")

    target_classes = {"FULL_PPE", "PARTIAL_PPE", "NO_PPE"}
    issues_found = []
    manifest_rows = []

    class_counts = {c: 0 for c in target_classes}
    total_crops_inspected = 0
    blank_images_count = 0
    corrupt_images_count = 0
    extreme_aspect_ratio_count = 0
    invalid_dimensions_count = 0
    duplicate_sha256_count = 0

    seen_hashes = {}

    if not label_audit_path.exists():
        msg = f"Label audit CSV {label_audit_path} missing. Run build_classification_dataset.py first."
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Read label audit
    audit_rows = []
    with open(label_audit_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            audit_rows.append(r)

    now_iso = datetime.now(UTC).isoformat()

    for row in audit_rows:
        derived_label = row["derived_label"]
        if derived_label == "REJECTED":
            continue

        crop_id = row["crop_id"]
        source_id = row["source_image_id"]
        crop_file = crops_dir / f"{crop_id}.jpg"

        if not crop_file.exists():
            issues_found.append(f"Missing crop file for accepted candidate {crop_id}")
            continue

        total_crops_inspected += 1
        file_size = crop_file.stat().st_size

        if file_size == 0:
            corrupt_images_count += 1
            issues_found.append(f"Zero byte empty file: {crop_id}")
            continue

        try:
            with Image.open(crop_file) as img:
                img.verify()
            with Image.open(crop_file) as img:
                img = img.convert("RGB")
                w, h = img.size
                arr = np.array(img, dtype=np.float32)
                phash_val = compute_phash(img)
        except Exception as e:
            corrupt_images_count += 1
            issues_found.append(f"Corrupt image {crop_id}: {e!s}")
            continue

        # Check dimension validity
        if w < 16 or h < 16:
            invalid_dimensions_count += 1
            issues_found.append(f"Image {crop_id} invalid tiny dimensions ({w}x{h})")

        aspect_ratio = w / float(h) if h > 0 else 0
        if aspect_ratio > 8.0 or aspect_ratio < 0.12:
            extreme_aspect_ratio_count += 1
            issues_found.append(f"Extreme aspect ratio for {crop_id}: {aspect_ratio:.2f}")

        # Check blank / uniform image
        std_dev = np.std(arr)
        if std_dev < 1.0:
            blank_images_count += 1
            issues_found.append(f"Blank/uniform solid color crop {crop_id} (std={std_dev:.2f})")

        # Check SHA-256 duplicate
        file_sha256 = compute_sha256(crop_file)
        if file_sha256 in seen_hashes:
            duplicate_sha256_count += 1
            issues_found.append(f"Exact duplicate detected between {crop_id} and {seen_hashes[file_sha256]}")
        else:
            seen_hashes[file_sha256] = crop_id

        # Check label validity
        if derived_label not in target_classes:
            issues_found.append(f"Invalid class label '{derived_label}' in {crop_id}")
        else:
            class_counts[derived_label] += 1

        manifest_rows.append({
            "sample_id": crop_id,
            "source_image_id": source_id,
            "sha256": file_sha256,
            "phash": phash_val,
            "width": w,
            "height": h,
            "format": "JPEG",
            "file_size": file_size,
            "class": derived_label,
            "split": "UNASSIGNED",
            "dataset_version": "1.0.0",
            "transformation_version": row.get("transformation_version", "1.0.0"),
            "created_at": now_iso,
        })

    # Imbalance metric calculations
    counts_list = list(class_counts.values())
    total_valid = sum(counts_list)
    imbalance_ratio = max(counts_list) / float(min(counts_list)) if min(counts_list) > 0 else 0.0

    passed = len(issues_found) == 0 and total_valid >= 50

    report = {
        "timestamp_utc": now_iso,
        "quality_gate_passed": passed,
        "total_candidates_processed": len(audit_rows),
        "total_valid_crops": total_valid,
        "class_distribution": class_counts,
        "imbalance_ratio": round(imbalance_ratio, 3),
        "corrupt_images": corrupt_images_count,
        "blank_images": blank_images_count,
        "extreme_aspect_ratios": extreme_aspect_ratio_count,
        "invalid_dimensions": invalid_dimensions_count,
        "duplicate_sha256_count": duplicate_sha256_count,
        "issues_detected": issues_found,
    }

    # Save JSON report
    report_json_path = reports_dir / "data_quality_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save Markdown report
    report_md_path = reports_dir / "data_quality_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# SiteSafe Vision: Data Quality Audit Report\n\n")
        f.write(f"- **Audit Status**: `{'PASSED' if passed else 'FAILED'}`\n")
        f.write(f"- **Audit Timestamp (UTC)**: `{now_iso}`\n")
        f.write(f"- **Total Valid Examples**: `{total_valid}`\n")
        f.write(f"- **Imbalance Ratio**: `{imbalance_ratio:.2f}`\n\n")
        f.write("## Class Distribution\n\n")
        f.write("| Class Name | Count | Percentage |\n")
        f.write("| :--- | :--- | :--- |\n")
        for cls_name, cnt in class_counts.items():
            pct = (cnt / total_valid * 100) if total_valid > 0 else 0
            f.write(f"| `{cls_name}` | {cnt} | {pct:.1f}% |\n")
        f.write("\n## Integrity Metrics\n\n")
        f.write(f"- Corrupt Images: `{corrupt_images_count}`\n")
        f.write(f"- Blank/Constant Images: `{blank_images_count}`\n")
        f.write(f"- Extreme Aspect Ratios: `{extreme_aspect_ratio_count}`\n")
        f.write(f"- Invalid Dimensions: `{invalid_dimensions_count}`\n")
        f.write(f"- Exact Duplicates (SHA-256): `{duplicate_sha256_count}`\n\n")
        if issues_found:
            f.write("## Quality Violations Logged\n\n")
            f.writelines(f"- ⚠️ {issue}\n" for issue in issues_found[:20])
        else:
            f.write("✅ **Zero quality gate violations detected.**\n")

    # Save Data Manifest CSV
    manifest_csv_path = reports_dir / "data_manifest.csv"
    manifest_fields = [
        "sample_id",
        "source_image_id",
        "sha256",
        "phash",
        "width",
        "height",
        "format",
        "file_size",
        "class",
        "split",
        "dataset_version",
        "transformation_version",
        "created_at",
    ]
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info(f"Quality audit finished. Passed: {passed}. Issues: {len(issues_found)}")
    logger.info(f"Saved reports to {report_json_path} and {manifest_csv_path}")

    if not passed:
        logger.error("Quality audit gate FAILED. Please review reports/data_quality_report.json.")
        sys.exit(1)

    return report


if __name__ == "__main__":
    run_quality_audit()
