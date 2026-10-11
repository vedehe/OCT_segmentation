import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from configs.config import (
    DEVICE,
    NUM_CLASSES,
    IN_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PREDICTION_DIR
)

from src.model import UNet


def predict(
    image_path,
    model_path
):

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image = Image.open(
        image_path
    ).convert("L")

    original_image = image.copy()


    image = image.resize(
        (IMAGE_WIDTH, IMAGE_HEIGHT)
    )


    image = np.array(
        image,
        dtype=np.float32
    ) / 255.0


    image_tensor = torch.from_numpy(
        image
    ).unsqueeze(0).unsqueeze(0)


    image_tensor = image_tensor.to(
        DEVICE
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = UNet(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()


    # ========================================================
    # PREDICTION
    # ========================================================

    with torch.no_grad():

        output = model(
            image_tensor
        )

        prediction = torch.argmax(
            output,
            dim=1
        )

    prediction = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    plt.figure(
        figsize=(12, 5)
    )


    plt.subplot(1, 2, 1)

    plt.imshow(
        original_image,
        cmap="gray"
    )

    plt.title("Original OCT")

    plt.axis("off")


    plt.subplot(1, 2, 2)

    plt.imshow(
        prediction,
        cmap="tab20"
    )

    plt.title("Predicted Segmentation")

    plt.axis("off")


    plt.tight_layout()


    output_path = (
        PREDICTION_DIR
        /
        "prediction.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.show()


if __name__ == "__main__":

    image_path = (
        PROJECT_ROOT
        /
        "data"
        /
        "images"
        /
        "001.png"
    )

    model_path = (
        PROJECT_ROOT
        /
        "outputs"
        /
        "checkpoints"
        /
        "epoch_50.pth"
    )

    predict(
        image_path,
        model_path
    )