from pathlib import Path

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset


class OCTDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        image_size=(256, 512)
    ):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.image_size = image_size

        self.images = sorted(
            self.image_dir.glob("*.png")
        )

        self.masks = sorted(
            self.mask_dir.glob("*.png")
        )

        if len(self.images) != len(self.masks):
            raise ValueError(
                f"Number of images ({len(self.images)}) "
                f"does not match number of masks "
                f"({len(self.masks)})"
            )

        if len(self.images) == 0:
            raise ValueError(
                "No PNG images found."
            )

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image_path = self.images[index]

        image = Image.open(
            image_path
        ).convert("L")


        # ----------------------------------------------------
        # Load mask
        # ----------------------------------------------------

        mask_path = self.masks[index]

        mask = Image.open(
            mask_path
        )


        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        width = self.image_size[1]
        height = self.image_size[0]

        image = image.resize(
            (width, height),
            Image.Resampling.BILINEAR
        )

        # VERY IMPORTANT:
        # masks must use nearest-neighbor interpolation

        mask = mask.resize(
            (width, height),
            Image.Resampling.NEAREST
        )


        # ----------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------

        image = np.array(
            image,
            dtype=np.float32
        )

        mask = np.array(
            mask,
            dtype=np.int64
        )


        # ----------------------------------------------------
        # Normalize image
        # ----------------------------------------------------

        image = image / 255.0


        # ----------------------------------------------------
        # Convert to PyTorch tensor
        # ----------------------------------------------------

        image = torch.from_numpy(
            image
        )

        mask = torch.from_numpy(
            mask
        )


        # ----------------------------------------------------
        # Add channel dimension
        #
        # [H, W]
        #     ↓
        # [1, H, W]
        # ----------------------------------------------------

        image = image.unsqueeze(0)


        return image, mask