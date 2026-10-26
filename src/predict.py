import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    PROCESSED_DIR,
    CHECKPOINT_DIR,
    PREDICTION_DIR,
    NUM_INPUT_CHANNELS,
    NUM_CLASSES,
    BASE_CHANNELS,
    create_output_dirs,
)
from src.model import UNet


def predict(stack_path, model_path, output_path=None):
    """
    Predict eight segmentation classes for a preprocessed OCT stack.

    Stack shape: (5, 640, 304)
    Output shape: (640, 304)
    """

    stack_path = Path(stack_path)
    model_path = Path(model_path)

    if not stack_path.is_file():
        raise FileNotFoundError(f"Stack not found: {stack_path}")

    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    image = np.load(stack_path).astype(np.float32)

    if image.ndim != 3 or image.shape[0] != NUM_INPUT_CHANNELS:
        raise ValueError(
            f"Expected stack shape (5, H, W), got {image.shape}"
        )

    # Our preprocessing already normalizes these stacks to [0, 1].
    if image.max() > 1.0:
        image /= 255.0

    image = np.clip(image, 0.0, 1.0)
    image_tensor = torch.from_numpy(
        np.ascontiguousarray(image)
    ).unsqueeze(0).to(device)

    model = UNet(
        in_channels=NUM_INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
    ).to(device)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(image_tensor)
        prediction = logits.argmax(dim=1)[0].cpu().numpy()

    # Display the central B-scan from the five-slice stack.
    central_image = image[NUM_INPUT_CHANNELS // 2]

    if output_path is None:
        create_output_dirs()
        output_path = PREDICTION_DIR / f"{stack_path.stem}_prediction.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(central_image, cmap="gray")
    axes[0].set_title("Central OCT B-scan")
    axes[0].axis("off")

    axes[1].imshow(prediction, cmap="tab10", vmin=0, vmax=NUM_CLASSES - 1)
    axes[1].set_title("Predicted Class Map")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Prediction shape:", prediction.shape)
    print("Output saved to:", output_path)

    return prediction


if __name__ == "__main__":
    stack_path = (
        PROCESSED_DIR / "10401" / "stacks" / "stack_001.npy"
    )
    model_path = CHECKPOINT_DIR / "best_model.pth"

    predict(stack_path, model_path)
