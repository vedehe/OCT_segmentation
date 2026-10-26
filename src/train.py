import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    PROCESSED_DIR,
    MASK_DIR,
    CHECKPOINT_DIR,
    SELECTED_VOLUMES,
    TRAIN_VOLUMES,
    VAL_VOLUMES,
    NUM_INPUT_CHANNELS,
    NUM_CLASSES,
    BASE_CHANNELS,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    RANDOM_SEED,
    NUM_WORKERS,
    create_output_dirs,
)
from src.dataset import OCTDataset
from src.model import UNet
from src.losses import get_loss_function
from src.metrics import (
    predictions_from_logits,
    dice_score,
    iou_score,
)
from src.utils import save_checkpoint


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataset(volumes):
    datasets = []

    for volume in volumes:
        image_dir = PROCESSED_DIR / volume

        dataset = OCTDataset(
            image_dir=image_dir,
            mask_dir=MASK_DIR / volume,
            require_masks=True,
        )

        datasets.append(dataset)
        print(f"Volume {volume}: {len(dataset)} labelled stacks")

    if not datasets:
        raise ValueError("No volumes supplied for this dataset split.")

    return ConcatDataset(datasets)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device).long()

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

        predictions = predictions_from_logits(logits)

        total_loss += loss.item()
        total_dice += dice_score(
            predictions, masks, NUM_CLASSES
        ).item()
        total_iou += iou_score(
            predictions, masks, NUM_CLASSES
        ).item()

        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("The data loader produced no batches.")

    return {
        "loss": total_loss / num_batches,
        "dice": total_dice / num_batches,
        "iou": total_iou / num_batches,
    }


def main():
    create_output_dirs()
    set_seed(RANDOM_SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    if not TRAIN_VOLUMES or not VAL_VOLUMES:
        raise ValueError(
            "Training and validation volumes are not configured. "
            "Annotate masks and set TRAIN_VOLUMES and VAL_VOLUMES "
            "in configs/config.py before training."
        )

    if set(TRAIN_VOLUMES) & set(VAL_VOLUMES):
        raise ValueError(
            "Training and validation volumes overlap. "
            "Use separate volumes to avoid data leakage."
        )

    unknown = (
        set(TRAIN_VOLUMES + VAL_VOLUMES) - set(SELECTED_VOLUMES)
    )
    if unknown:
        raise ValueError(
            f"Unknown volume IDs in split configuration: {sorted(unknown)}"
        )

    train_dataset = build_dataset(TRAIN_VOLUMES)
    val_dataset = build_dataset(VAL_VOLUMES)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = UNet(
        in_channels=NUM_INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
    ).to(device)

    criterion = get_loss_function(num_classes=NUM_CLASSES)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, criterion, device
            )

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f}"
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_metrics["loss"],
            CHECKPOINT_DIR / f"epoch_{epoch:03d}.pth",
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_val_loss,
                CHECKPOINT_DIR / "best_model.pth",
            )


if __name__ == "__main__":
    main()
