import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from configs.config import (
    IMAGE_DIR,
    MASK_DIR,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    NUM_CLASSES,
    IN_CHANNELS,
    DEVICE,
    NUM_WORKERS
)

from src.dataset import OCTDataset
from src.model import UNet
from src.losses import get_loss_function
from src.utils import save_checkpoint


def main():

    print("Using device:", DEVICE)


    # ========================================================
    # DATASET
    # ========================================================

    dataset = OCTDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR
    )

    print(
        "Number of training images:",
        len(dataset)
    )


    # ========================================================
    # DATALOADER
    # ========================================================

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = UNet(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)


    # ========================================================
    # LOSS
    # ========================================================

    criterion = get_loss_function()


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(NUM_EPOCHS):

        model.train()

        total_loss = 0.0


        for images, masks in loader:

            images = images.to(DEVICE)

            masks = masks.to(DEVICE)


            # Forward
            outputs = model(images)


            # Loss
            loss = criterion(
                outputs,
                masks
            )


            # Backpropagation
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            total_loss += loss.item()


        average_loss = (
            total_loss / len(loader)
        )


        print(
            f"Epoch "
            f"{epoch + 1}/{NUM_EPOCHS} "
            f"- Loss: "
            f"{average_loss:.4f}"
        )


        # Save checkpoint
        checkpoint_path = (
            CHECKPOINT_DIR
            /
            f"epoch_{epoch + 1}.pth"
        )

        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            average_loss,
            checkpoint_path
        )


if __name__ == "__main__":

    main()