
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class OCTDataset(Dataset):
    """
    Dataset for 2.5D OCT retinal-layer segmentation.

    Each input contains five neighbouring B-scans.
    Each target mask contains one class ID per pixel.

    Expected image shape: (5, 640, 304)
    Expected mask shape:  (640, 304)
    """

    def __init__(
        self,
        image_dir,
        mask_dir=None,
        require_masks=True,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.require_masks = require_masks

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        # Find the preprocessed five-slice stacks.
        self.images = sorted(
            self.image_dir.glob("stacks/*.npy")
        )

        if not self.images:
            raise FileNotFoundError(
                f"No .npy stacks found in {self.image_dir / 'stacks'}"
            )

        self.masks = []

        if self.mask_dir is not None and self.mask_dir.exists():
            for image_path in self.images:
                mask_path = self.mask_dir / f"{image_path.stem}.npy"
                self.masks.append(
                    mask_path if mask_path.exists() else None
                )

        if self.require_masks:
            missing = [
                image_path.name
                for image_path, mask_path in zip(
                    self.images, self.masks
                )
                if mask_path is None
            ]

            if self.mask_dir is None or len(self.masks) != len(self.images):
                raise FileNotFoundError(
                    "Ground-truth masks are not configured for every image. "
                    "Annotate masks before supervised training."
                )

            if missing:
                raise FileNotFoundError(
                    f"Missing masks for {len(missing)} images. "
                    f"First missing image: {missing[0]}"
                )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]

        image = np.load(image_path).astype(np.float32)

        # Require (channels, height, width).
        if image.ndim != 3 or image.shape[0] != 5:
            raise ValueError(
                f"Expected stack shape (5, H, W), "
                f"got {image.shape} for {image_path}"
            )

        # The preprocessed stacks are expected to be normalized.
        # Handle either [0, 1] or [0, 255] image values.
        if image.max() > 1.0:
            image = image / 255.0

        image = np.clip(image, 0.0, 1.0)
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image)
        )

        if not self.require_masks:
            return image_tensor, image_path.stem

        mask_path = self.masks[index]
        mask = np.load(mask_path)

        if mask.ndim != 2:
            raise ValueError(
                f"Expected a 2D mask, got {mask.shape} "
                f"for {mask_path}"
            )

        if mask.shape != image.shape[1:]:
            raise ValueError(
                f"Image/mask size mismatch: "
                f"{image.shape[1:]} versus {mask.shape}"
            )

        mask = mask.astype(np.int64)

        if mask.min() < 0 or mask.max() >= 8:
            raise ValueError(
                f"Mask labels must be integers from 0 to 7; "
                f"found range {mask.min()} to {mask.max()} "
                f"in {mask_path}"
            )

        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask)
        )

        return image_tensor, mask_tensor
