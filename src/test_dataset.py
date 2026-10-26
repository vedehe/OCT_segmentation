
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import OCTDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "images"

for volume in ["10401", "10402", "10403"]:
    dataset = OCTDataset(
        image_dir=PROCESSED_ROOT / volume,
        require_masks=False,
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    images, filenames = next(iter(loader))

    print(f"\nVolume: {volume}")
    print("Number of stacks:", len(dataset))
    print("Batch shape:", tuple(images.shape))
    print("Data type:", images.dtype)
    print("Value range:", images.min().item(), images.max().item())
    print("Example filenames:", filenames)
    
    assert images.shape == (2, 5, 640, 304)
    assert images.dtype == torch.float32

print("\nDataset loading test passed.")
