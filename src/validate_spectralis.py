
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "data" / "spectralis_processed"

image_root = ROOT / "images"
mask_root = ROOT / "masks"

for subject_dir in sorted(image_root.iterdir()):
    if not subject_dir.is_dir():
        continue

    subject = subject_dir.name
    stack_files = sorted((subject_dir / "stacks").glob("*.npy"))
    subject_mask_dir = mask_root / subject
    mask_files = sorted(subject_mask_dir.glob("*.npy"))

    print(f"\nSubject: {subject}")
    print(f"Image stacks: {len(stack_files)}")
    print(f"Masks: {len(mask_files)}")

    assert len(stack_files) == 49, "Expected 49 image stacks"
    assert len(mask_files) == 49, "Expected 49 masks"

    for stack_path in stack_files:
        mask_path = subject_mask_dir / stack_path.name

        assert mask_path.exists(), f"Missing mask: {mask_path.name}"

        stack = np.load(stack_path)
        mask = np.load(mask_path)

        assert stack.shape == (5, 496, 1024), (
            f"Unexpected stack shape: {stack_path.name}, {stack.shape}"
        )
        assert mask.shape == (496, 1024), (
            f"Unexpected mask shape: {mask_path.name}, {mask.shape}"
        )
        assert np.isfinite(stack).all(), f"Non-finite image values: {stack_path.name}"
        assert stack.min() >= 0 and stack.max() <= 1, (
            f"Image values outside [0, 1]: {stack_path.name}"
        )
        assert mask.min() >= 0 and mask.max() <= 8, (
            f"Invalid class IDs: {mask_path.name}"
        )

    first_mask = np.load(mask_files[0])
    print("First mask class counts:", np.bincount(first_mask.ravel(), minlength=9))
    print("Validation passed for this subject.")

print("\nAll selected subjects validated.")

