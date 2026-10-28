
from pathlib import Path
import csv
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import UNet


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SUBJECT = "hc03_spectralis_macula_v1_s1_R"
NUM_CLASSES = 9
IN_CHANNELS = 5

DATA_ROOT = PROJECT_ROOT / "data" / "spectralis_processed"
IMAGE_DIR = DATA_ROOT / "images" / SUBJECT / "stacks"
MASK_DIR = DATA_ROOT / "masks" / SUBJECT

CHECKPOINT_PATH = (
    PROJECT_ROOT / "outputs" / "spectralis"
    / "checkpoints" / "best_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "spectralis"
    / f"evaluation_{SUBJECT}"
)
PREDICTION_DIR = OUTPUT_DIR / "predictions"
FIGURE_DIR = OUTPUT_DIR / "figures"


def dice_iou_per_class(pred, target, num_classes):
    """Calculate global per-class Dice and IoU across all evaluated pixels."""
    results = []

    for class_id in range(num_classes):
        pred_class = pred == class_id
        target_class = target == class_id

        intersection = np.logical_and(
            pred_class, target_class
        ).sum(dtype=np.int64)

        pred_count = pred_class.sum(dtype=np.int64)
        target_count = target_class.sum(dtype=np.int64)
        union = np.logical_or(pred_class, target_class).sum(dtype=np.int64)

        dice_denominator = pred_count + target_count

        dice = (
            2.0 * intersection / dice_denominator
            if dice_denominator > 0
            else float("nan")
        )

        iou = (
            intersection / union
            if union > 0
            else float("nan")
        )

        results.append({
            "class_id": class_id,
            "dice": dice,
            "iou": iou,
            "target_pixels": int(target_count),
            "predicted_pixels": int(pred_count),
        })

    return results



def save_comparison(image, target, prediction, scan_number):
    from matplotlib.colors import ListedColormap

    # Distinct colors for classes 0-8.
    # colors = plt.get_cmap("tab10")(np.arange(9))
    # colors[0] = [0, 0, 0, 1]
    # class_cmap = ListedColormap(colors)
    class_cmap = "viridis"

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Original OCT")

    axes[0, 1].imshow(
        target, cmap=class_cmap, vmin=0, vmax=8,
        interpolation="nearest"
    )
    axes[0, 1].set_title("Ground-truth mask")

    axes[1, 0].imshow(
        prediction, cmap=class_cmap, vmin=0, vmax=8,
        interpolation="nearest"
    )
    axes[1, 0].set_title("Predicted mask")

    # Overlay: ground-truth boundaries in green, predictions in red.
    axes[1, 1].imshow(image, cmap="gray", vmin=0, vmax=1)

    for class_id in range(1, NUM_CLASSES):
        gt_boundary = target == class_id
        pred_boundary = prediction == class_id

        axes[1, 1].contour(
            gt_boundary.astype(float),
            levels=[0.5],
            colors=["lime"],
            linewidths=0.6,
        )
        axes[1, 1].contour(
            pred_boundary.astype(float),
            levels=[0.5],
            colors=["red"],
            linewidths=0.6,
        )

    axes[1, 1].set_title("Overlay: ground truth (green), prediction (red)")

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(f"{SUBJECT} — scan {scan_number:03d}")
    fig.tight_layout()

    output_path = FIGURE_DIR / f"scan_{scan_number:03d}_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    image_paths = sorted(IMAGE_DIR.glob("scan_*.npy"))

    if not image_paths:
        raise FileNotFoundError(f"No image stacks found in {IMAGE_DIR}")

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Device:", device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    base_channels = checkpoint["base_channels"]
    model = UNet(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=base_channels,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_predictions = []
    all_targets = []
    preview_scans = {1, 23, 49}

    with torch.no_grad():
        for image_path in image_paths:
            scan_number = int(image_path.stem.split("_")[-1])
            mask_path = MASK_DIR / image_path.name

            if not mask_path.exists():
                raise FileNotFoundError(mask_path)

            stack = np.load(image_path).astype(np.float32)
            target = np.load(mask_path).astype(np.int64)

            if stack.shape != (5, 496, 1024):
                raise ValueError(
                    f"Unexpected input shape in {image_path}: {stack.shape}"
                )

            if target.shape != (496, 1024):
                raise ValueError(
                    f"Unexpected target shape in {mask_path}: {target.shape}"
                )

            input_tensor = torch.from_numpy(
                np.ascontiguousarray(stack)
            ).unsqueeze(0).to(device)

            logits = model(input_tensor)
            prediction = torch.argmax(logits, dim=1)[0].cpu().numpy()

            if prediction.min() < 0 or prediction.max() >= NUM_CLASSES:
                raise ValueError("Model produced invalid class IDs.")

            np.save(
                PREDICTION_DIR / f"scan_{scan_number:03d}.npy",
                prediction.astype(np.uint8),
            )

            all_predictions.append(prediction.reshape(-1))
            all_targets.append(target.reshape(-1))

            if scan_number in preview_scans:
                # The middle channel is the scan being segmented.
                image = stack[2]
                save_comparison(image, target, prediction, scan_number)

            print(f"Predicted scan {scan_number:03d}/{len(image_paths):03d}")

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    results = dice_iou_per_class(
        predictions, targets, NUM_CLASSES
    )

    csv_path = OUTPUT_DIR / "per_class_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id", "dice", "iou",
                "target_pixels", "predicted_pixels",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    valid_dice = [
        row["dice"] for row in results[1:]
        if np.isfinite(row["dice"])
    ]
    valid_iou = [
        row["iou"] for row in results[1:]
        if np.isfinite(row["iou"])
    ]

    print("\nPer-class metrics on hc02:")
    for row in results:
        print(
            f"Class {row['class_id']}: "
            f"Dice={row['dice']:.4f}, "
            f"IoU={row['iou']:.4f}, "
            f"target pixels={row['target_pixels']}"
        )

    print("\nMean foreground Dice:", np.mean(valid_dice))
    print("Mean foreground IoU:", np.mean(valid_iou))
    print("Metrics saved to:", csv_path)
    print("Comparison figures saved to:", FIGURE_DIR)


if __name__ == "__main__":
    main()
