import csv
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Dict
from PIL import Image
from torch.utils.data import Dataset

CLASS_TO_IDX = {
    "FULL_PPE": 0,
    "PARTIAL_PPE": 1,
    "NO_PPE": 2,
}


class PPEDataset(Dataset):
    def __init__(
        self,
        split_manifest_path: Path,
        crops_dir: Path,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        self.split = split
        self.crops_dir = crops_dir
        self.transform = transform
        self.samples: List[Dict[str, str]] = []

        if not split_manifest_path.exists():
            raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")

        with open(split_manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.samples.append(row)

        if not self.samples:
            raise ValueError(f"No samples found for split '{split}' in {split_manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[any, int, str]:
        item = self.samples[idx]
        sample_id = item["sample_id"]
        class_name = item["class"]
        label = CLASS_TO_IDX[class_name]

        img_path = self.crops_dir / f"{sample_id}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"Crop image missing: {img_path}")

        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, sample_id
