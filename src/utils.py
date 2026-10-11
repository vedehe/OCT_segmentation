from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    path
):

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        },
        path
    )


def save_prediction(
    image,
    prediction,
    path
):

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.subplot(1, 2, 1)

    plt.imshow(
        image,
        cmap="gray"
    )

    plt.title("OCT Image")
    plt.axis("off")


    plt.subplot(1, 2, 2)

    plt.imshow(
        prediction,
        cmap="tab20"
    )

    plt.title("Segmentation")
    plt.axis("off")


    plt.tight_layout()

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()