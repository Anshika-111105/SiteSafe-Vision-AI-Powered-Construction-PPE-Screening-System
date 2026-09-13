#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("build_classification_dataset")

TRANSFORMATION_VERSION = "1.0.0"
MIN_WORKER_SIZE = 32  # Minimum width/height in px


def calculate_containment(child_box: list[int], parent_box: list[int]) -> bool:
    """
    Checks if the center of child_box lies within parent_box.
    Boxes are in [x, y, w, h] format.
    """
    cx = child_box[0] + child_box[2] / 2.0
    cy = child_box[1] + child_box[3] / 2.0

    px, py, pw, ph = parent_box
    return (px <= cx <= px + pw) and (py <= cy <= py + ph)


def is_in_relative_band(child_box: list[int], parent_box: list[int], min_rel_y: float, max_rel_y: float) -> bool:
    """
    Checks if child_box center falls within a specific relative vertical band of parent_box.
    """
    cy = child_box[1] + child_box[3] / 2.0
    py, ph = parent_box[1], parent_box[3]
    rel_y = (cy - py) / float(ph) if ph > 0 else 0.0
    return min_rel_y <= rel_y <= max_rel_y


def build_classification_dataset(
    raw_dir: Path = None,
    interim_dir: Path = None,
) -> Path:
    if raw_dir is None:
        raw_dir = PROJECT_ROOT / "data" / "raw"
    if interim_dir is None:
        interim_dir = PROJECT_ROOT / "data" / "interim"

    images_dir = raw_dir / "images"
    ann_dir = raw_dir / "annotations"
    crops_dir = interim_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    audit_csv_path = interim_dir / "label_audit.csv"

    logger.info(f"Building classification dataset from {raw_dir}...")

    audit_records = []
    accepted_count = 0
    rejected_count = 0
    class_counts = {"FULL_PPE": 0, "PARTIAL_PPE": 0, "NO_PPE": 0}

    ann_files = sorted(list(ann_dir.glob("*.json")))
    if not ann_files:
        raise FileNotFoundError(f"No annotation JSON files found in {ann_dir}")

    for ann_file in ann_files:
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_id = data["image_id"]
        image_path = images_dir / data["file_name"]

        if not image_path.exists():
            logger.warning(f"Image {image_path} missing. Skipping scene {image_id}.")
            continue

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img_w, img_h = img.size
        except Exception as e:
            logger.error(f"Failed to read image {image_path}: {e}")
            continue

        annotations = data.get("annotations", [])
        humans = [a for a in annotations if a["class"] == "human"]
        helmets = [a for a in annotations if a["class"] == "helmet"]
        vests = [a for a in annotations if a["class"] == "vest"]
        boots = [a for a in annotations if a["class"] == "boots"]
        gloves = [a for a in annotations if a["class"] == "gloves"]

        for h_idx, human in enumerate(humans):
            crop_id = f"{image_id}_w{h_idx:02d}"
            bbox = human["bbox"]  # [x, y, w, h]
            hx, hy, hw, hh = bbox

            # Validation Rule 1: Minimum Size
            if hw < MIN_WORKER_SIZE or hh < MIN_WORKER_SIZE:
                audit_records.append({
                    "source_image_id": image_id,
                    "crop_id": crop_id,
                    "worker_bbox": f"[{hx},{hy},{hw},{hh}]",
                    "helmet_present": False,
                    "vest_present": False,
                    "boots_present": False,
                    "gloves_present": False,
                    "derived_label": "REJECTED",
                    "label_confidence": 0.0,
                    "rejection_reason": "WORKER_TOO_SMALL",
                    "transformation_version": TRANSFORMATION_VERSION,
                })
                rejected_count += 1
                continue

            # Validation Rule 2: Boundary Bounds Check
            if hx < 0 or hy < 0 or (hx + hw) > img_w or (hy + hh) > img_h:
                # Slight truncation can be clipped, extreme truncation rejected
                if hx < -int(hw * 0.5) or hy < -int(hh * 0.5):
                    audit_records.append({
                        "source_image_id": image_id,
                        "crop_id": crop_id,
                        "worker_bbox": f"[{hx},{hy},{hw},{hh}]",
                        "helmet_present": False,
                        "vest_present": False,
                        "boots_present": False,
                        "gloves_present": False,
                        "derived_label": "REJECTED",
                        "label_confidence": 0.0,
                        "rejection_reason": "SEVERE_BOUNDARY_TRUNCATION",
                        "transformation_version": TRANSFORMATION_VERSION,
                    })
                    rejected_count += 1
                    continue

            # PPE Association Logic
            # Helmet on upper 40% of worker box
            helmet_present = any(
                calculate_containment(h["bbox"], bbox) and is_in_relative_band(h["bbox"], bbox, -0.15, 0.40)
                for h in helmets
            )

            # Vest on torso (25% to 80% of worker box)
            vest_present = any(
                calculate_containment(v["bbox"], bbox) and is_in_relative_band(v["bbox"], bbox, 0.20, 0.85)
                for v in vests
            )

            # Boots in lower 35% of worker box
            boots_present = any(
                calculate_containment(b["bbox"], bbox) and is_in_relative_band(b["bbox"], bbox, 0.65, 1.15)
                for b in boots
            )

            # Gloves in mid-lower section
            gloves_present = any(
                calculate_containment(g["bbox"], bbox) and is_in_relative_band(g["bbox"], bbox, 0.40, 0.90)
                for g in gloves
            )

            # Compliance Classification Business Rules
            if helmet_present and vest_present:
                derived_label = "FULL_PPE"
            elif helmet_present or vest_present:
                derived_label = "PARTIAL_PPE"
            else:
                derived_label = "NO_PPE"

            # Worker crop with 5% margin
            pad_x = int(hw * 0.05)
            pad_y = int(hh * 0.05)
            x1 = max(0, hx - pad_x)
            y1 = max(0, hy - pad_y)
            x2 = min(img_w, hx + hw + pad_x)
            y2 = min(img_h, hy + hh + pad_y)

            with Image.open(image_path) as img:
                crop = img.crop((x1, y1, x2, y2))
                crop_path = crops_dir / f"{crop_id}.jpg"
                crop.save(crop_path, "JPEG", quality=95)

            audit_records.append({
                "source_image_id": image_id,
                "crop_id": crop_id,
                "worker_bbox": f"[{hx},{hy},{hw},{hh}]",
                "helmet_present": helmet_present,
                "vest_present": vest_present,
                "boots_present": boots_present,
                "gloves_present": gloves_present,
                "derived_label": derived_label,
                "label_confidence": 1.0,
                "rejection_reason": "NONE",
                "transformation_version": TRANSFORMATION_VERSION,
            })
            accepted_count += 1
            class_counts[derived_label] += 1

    # Write label audit CSV
    fieldnames = [
        "source_image_id",
        "crop_id",
        "worker_bbox",
        "helmet_present",
        "vest_present",
        "boots_present",
        "gloves_present",
        "derived_label",
        "label_confidence",
        "rejection_reason",
        "transformation_version",
    ]

    with open(audit_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_records)

    logger.info(f"Label derivation complete. Processed {len(audit_records)} candidates.")
    logger.info(f"Accepted: {accepted_count}, Rejected: {rejected_count}")
    logger.info(f"Derived Class Distribution: {class_counts}")
    logger.info(f"Label audit log saved to {audit_csv_path}")

    return audit_csv_path


if __name__ == "__main__":
    build_classification_dataset()
