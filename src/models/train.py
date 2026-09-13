import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.transforms import get_eval_transforms, get_train_transforms
from src.models.architectures import (
    create_mobilenet_v3_model,
    create_resnet50_model,
    set_trainable_layers,
)
from src.models.dataset import PPEDataset
from src.utils.logger import setup_logger
from src.utils.metrics import compute_classification_metrics
from src.utils.seed import seed_worker, set_seed

logger = setup_logger("train")


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_OR_UNCOMMITTED"


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_classification_metrics(all_targets, all_preds)
    return {
        "loss": round(epoch_loss, 4),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    val_loss = running_loss / len(dataloader.dataset)
    metrics = compute_classification_metrics(all_targets, all_preds)
    metrics["loss"] = round(val_loss, 4)
    return metrics


def train_model(
    model_name: str = "resnet50",
    config_path: Path = None,
    output_dir: Path = None,
) -> dict[str, Any]:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    if output_dir is None:
        output_dir = PROJECT_ROOT / "artifacts" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config["reproducibility"]["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting training for {model_name} on device: {device} (Seed={seed})")

    # DataLoaders
    split_manifest_path = PROJECT_ROOT / config["data"]["split_manifest_path"]
    crops_dir = PROJECT_ROOT / config["data"]["interim_dir"] / "crops"

    train_tf = get_train_transforms(
        image_size=config["preprocessing"]["image_size"],
        resize_size=config["preprocessing"]["resize_size"],
    )
    eval_tf = get_eval_transforms(
        image_size=config["preprocessing"]["image_size"],
        resize_size=config["preprocessing"]["resize_size"],
    )

    train_dataset = PPEDataset(split_manifest_path, crops_dir, split="train", transform=train_tf)
    val_dataset = PPEDataset(split_manifest_path, crops_dir, split="val", transform=eval_tf)

    batch_size = config["training"]["batch_size"]
    num_workers = config["reproducibility"]["dataloader_workers"]

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    # Instantiate Model
    num_classes = len(config["data"]["target_classes"])
    if model_name == "resnet50":
        model = create_resnet50_model(num_classes=num_classes, pretrained=True)
    elif model_name == "mobilenet_v3_large":
        model = create_mobilenet_v3_model(num_classes=num_classes, pretrained=True)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.to(device)

    # Class Weights for Asymmetric Safety Loss (Higher penalty on missing PPE)
    criterion = nn.CrossEntropyLoss()

    total_epochs = config["training"]["num_epochs"]
    phase1_epochs = config["training"]["phase1_freeze_backbone_epochs"]
    patience = config["training"]["early_stopping_patience"]

    best_val_f1 = -1.0
    best_model_weights = None
    patience_counter = 0
    history = []

    # Phase 1: Feature Extractor
    logger.info(f"--- Phase 1: Feature Extraction Training ({phase1_epochs} epochs) ---")
    trainable_p1 = set_trainable_layers(model, model_name, phase=1)
    logger.info(f"Trainable layers in Phase 1: {len(trainable_p1)} tensors")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"] * 2.0,
        weight_decay=config["training"]["weight_decay"],
    )

    for epoch in range(1, phase1_epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_res = evaluate(model, val_loader, criterion, device)
        dur = time.time() - t0

        logger.info(
            f"Epoch {epoch:02d}/{total_epochs:02d} [P1] (took {dur:.1f}s) - "
            f"Train Loss: {train_res['loss']:.4f}, Train F1: {train_res['macro_f1']:.4f} | "
            f"Val Loss: {val_res['loss']:.4f}, Val F1: {val_res['macro_f1']:.4f}, Val Acc: {val_res['accuracy']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "phase": 1,
            "train": train_res,
            "val": val_res,
        })

        if val_res["macro_f1"] > best_val_f1:
            best_val_f1 = val_res["macro_f1"]
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

    # Phase 2: Fine-Tuning
    logger.info(f"--- Phase 2: Fine-Tuning ({total_epochs - phase1_epochs} epochs) ---")
    trainable_p2 = set_trainable_layers(model, model_name, phase=2)
    logger.info(f"Trainable layers in Phase 2: {len(trainable_p2)} tensors")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(total_epochs - phase1_epochs),
        eta_min=config["training"]["min_lr"],
    )

    for epoch in range(phase1_epochs + 1, total_epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_res = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        dur = time.time() - t0

        logger.info(
            f"Epoch {epoch:02d}/{total_epochs:02d} [P2] (took {dur:.1f}s) - "
            f"Train Loss: {train_res['loss']:.4f}, Train F1: {train_res['macro_f1']:.4f} | "
            f"Val Loss: {val_res['loss']:.4f}, Val F1: {val_res['macro_f1']:.4f}, Val Acc: {val_res['accuracy']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "phase": 2,
            "train": train_res,
            "val": val_res,
        })

        if val_res["macro_f1"] > best_val_f1:
            best_val_f1 = val_res["macro_f1"]
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            logger.info(f"--> Saved new best model checkpoint (Val Macro F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs (patience={patience}).")
                break

    # Load best weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # Save model artifact
    model_save_path = output_dir / f"best_{model_name}.pt"
    torch.save(
        {
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "class_to_idx": config["data"]["class_to_idx"],
            "idx_to_class": config["data"]["idx_to_class"],
            "input_size": config["preprocessing"]["image_size"],
            "normalization": {
                "mean": config["preprocessing"]["mean"],
                "std": config["preprocessing"]["std"],
            },
        },
        model_save_path,
    )
    logger.info(f"Saved {model_name} checkpoint to {model_save_path}")

    # Compute final validation score with best weights
    final_val_eval = evaluate(model, val_loader, criterion, device)

    # Compute config hash
    with open(config_path, "rb") as f:
        cfg_hash = hashlib.sha256(f.read()).hexdigest()

    metadata = {
        "model_name": model_name,
        "model_version": "1.0.0",
        "architecture": f"torchvision.models.{model_name}",
        "weights_source": "torchvision.models.DEFAULT",
        "training_commit": get_git_commit(),
        "dataset_version": "1.0.0",
        "config_hash": cfg_hash,
        "random_seed": seed,
        "training_timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "class_mapping": config["data"]["class_to_idx"],
        "input_size": config["preprocessing"]["image_size"],
        "normalization": {
            "mean": config["preprocessing"]["mean"],
            "std": config["preprocessing"]["std"],
        },
        "artifact_sha256": compute_sha256(model_save_path),
        "validation_metrics": final_val_eval,
        "training_history": history,
    }

    meta_save_path = output_dir / f"metadata_{model_name}.json"
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {meta_save_path}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SiteSafe Vision Transfer Learning Model")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "mobilenet_v3_large"])
    args = parser.parse_args()
    train_model(model_name=args.model)
