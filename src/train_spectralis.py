
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import OCTDataset
from src.model import UNet
from src.losses import get_loss_function
from src.metrics import predictions_from_logits, dice_score, iou_score


# --------------------------------------------------
# Configuration: Spectralis only
# --------------------------------------------------

DATA_ROOT = PROJECT_ROOT / "data" / "spectralis_processed"
IMAGE_ROOT = DATA_ROOT / "images"
MASK_ROOT = DATA_ROOT / "masks"

TRAIN_SUBJECTS = [
    "hc01_spectralis_macula_v1_s1_R",
]
VAL_SUBJECTS = [
    "hc02_spectralis_macula_v1_s1_R",
]

NUM_CLASSES = 9
IN_CHANNELS = 5
BASE_CHANNELS = 8

BATCH_SIZE = 1
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 42
NUM_WORKERS = 0

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "spectralis"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
LOG_PATH = OUTPUT_ROOT / "training_log.csv"


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------
# Dataset construction
# --------------------------------------------------

def build_dataset(subjects):
    datasets = []

    for subject in subjects:
        image_dir = IMAGE_ROOT / subject
        mask_dir = MASK_ROOT / subject

        dataset = OCTDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            require_masks=True,
            num_classes=NUM_CLASSES,
        )

        if len(dataset) == 0:
            raise ValueError(f"No samples found for {subject}")

        datasets.append(dataset)
        print(f"{subject}: {len(dataset)} labelled scans")

    return torch.utils.data.ConcatDataset(datasets)


# --------------------------------------------------
# One training or validation epoch
# --------------------------------------------------

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

        if masks.min().item() < 0 or masks.max().item() >= NUM_CLASSES:
            raise ValueError("Invalid mask class encountered.")

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)

            if logits.shape[1] != NUM_CLASSES:
                raise ValueError(
                    f"Expected {NUM_CLASSES} output classes, "
                    f"got {logits.shape[1]}"
                )

            if logits.shape[-2:] != masks.shape[-2:]:
                raise ValueError(
                    f"Prediction size {logits.shape[-2:]} "
                    f"does not match mask size {masks.shape[-2:]}"
                )

            loss = criterion(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

        predictions = predictions_from_logits(logits)

        total_loss += loss.item()
        total_dice += dice_score(
            predictions, masks, NUM_CLASSES, include_background=False
        ).item()
        total_iou += iou_score(
            predictions, masks, NUM_CLASSES, include_background=False
        ).item()

        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("DataLoader produced no batches.")

    return {
        "loss": total_loss / num_batches,
        "dice": total_dice / num_batches,
        "iou": total_iou / num_batches,
    }


# --------------------------------------------------
# Main training procedure
# --------------------------------------------------

def main():
    set_seed(RANDOM_SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if set(TRAIN_SUBJECTS) & set(VAL_SUBJECTS):
        raise ValueError("Training and validation subjects must be separate.")

    for subject in TRAIN_SUBJECTS + VAL_SUBJECTS:
        if not (IMAGE_ROOT / subject / "stacks").exists():
            raise FileNotFoundError(
                f"Missing image stacks for subject {subject}"
            )
        if not (MASK_ROOT / subject).exists():
            raise FileNotFoundError(
                f"Missing masks for subject {subject}"
            )

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Device:", device)

    train_dataset = build_dataset(TRAIN_SUBJECTS)
    val_dataset = build_dataset(VAL_SUBJECTS)

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
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
    ).to(device)

    # Verify the expected spatial and class dimensions before training.
    model.eval()
    with torch.no_grad():
        test_input = torch.zeros(
            1, IN_CHANNELS, 496, 1024, device=device
        )
        test_output = model(test_input)

    print("Test input shape:", (1, IN_CHANNELS, 496, 1024))
    print("Test output shape:", tuple(test_output.shape))

    if tuple(test_output.shape) != (1, NUM_CLASSES, 496, 1024):
        raise ValueError("Unexpected U-Net output shape.")

    del test_input, test_output

    criterion = get_loss_function(num_classes=NUM_CLASSES)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")

    with LOG_PATH.open("w", newline="") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_dice",
                "train_iou",
                "val_loss",
                "val_dice",
                "val_iou",
            ],
        )
        writer.writeheader()

        for epoch in range(1, NUM_EPOCHS + 1):
            train_metrics = run_epoch(
                model, train_loader, criterion, device, optimizer
            )

            with torch.no_grad():
                val_metrics = run_epoch(
                    model, val_loader, criterion, device
                )

            print(
                f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                f"Train loss {train_metrics['loss']:.4f} | "
                f"Val loss {val_metrics['loss']:.4f} | "
                f"Val Dice {val_metrics['dice']:.4f} | "
                f"Val IoU {val_metrics['iou']:.4f}"
            )

            writer.writerow({
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "train_iou": train_metrics["iou"],
                "val_loss": val_metrics["loss"],
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
            })
            log_file.flush()

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "num_classes": NUM_CLASSES,
                        "in_channels": IN_CHANNELS,
                        "base_channels": BASE_CHANNELS,
                        "train_subjects": TRAIN_SUBJECTS,
                        "val_subjects": VAL_SUBJECTS,
                    },
                    CHECKPOINT_DIR / "best_model.pth",
                )

                print("Saved new best Spectralis checkpoint.")

    print("\nTraining finished.")
    print("Best checkpoint:", CHECKPOINT_DIR / "best_model.pth")
    print("Training log:", LOG_PATH)


if __name__ == "__main__":
    main()
