
from pathlib import Path
import csv

import numpy as np
from PIL import Image


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Your existing patient folders contain the original BMP images.
DATA_ROOT = PROJECT_ROOT / "data" / "images"

# Save processed outputs separately to avoid overwriting original BMPs.
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "images"

SELECTED_VOLUMES = ["10401", "10402", "10403"]

NEIGHBORS = 2
P_LOW = 1
P_HIGH = 99


# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# DATA_ROOT = (
#     PROJECT_ROOT
#     / "CSANet"
#     / "datasets"
#     / "OCTA_3mm 2"
#     / "OCT"
# )

# OUTPUT_ROOT = PROJECT_ROOT / "data" / "images"

# Use only these three volumes for now.
# SELECTED_VOLUMES = ["10401", "10402", "10403"]



# "per_slice": normalize each B-scan independently.
# "per_volume": use common intensity limits for a whole volume.
NORMALIZATION_MODE = "per_slice"

# Save original and normalized volumes for traceability.
SAVE_ORIGINAL_VOLUME = True
SAVE_NORMALIZED_VOLUME = True

# Set True to save stacks as individual .npy files.
SAVE_STACKS = True


# ============================================================
# IMAGE LOADING
# ============================================================

def load_bscan(path):
    """Load a grayscale B-scan without resizing."""
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def get_ordered_slices(volume_dir):
    """Find BMP files and sort them by numeric slice index."""
    paths = list(volume_dir.glob("*.bmp"))

    if not paths:
        return []

    for path in paths:
        if not path.stem.isdigit():
            raise ValueError(
                f"Unexpected slice filename in {volume_dir}: "
                f"{path.name}. Expected numeric BMP filenames."
            )

    paths.sort(key=lambda p: int(p.stem))
    indices = [int(p.stem) for p in paths]

    if len(indices) != len(set(indices)):
        raise ValueError(
            f"Duplicate slice indices in {volume_dir}"
        )

    missing = sorted(
        set(range(indices[0], indices[-1] + 1)) - set(indices)
    )

    if missing:
        print(
            f"WARNING: Volume {volume_dir.name}: "
            f"{len(missing)} missing slice indices. "
            f"First missing indices: {missing[:10]}"
        )

    return paths


# ============================================================
# INTENSITY NORMALIZATION
# ============================================================

def normalize_with_limits(image, low, high):
    """
    Clip intensities to [low, high] and scale to [0, 1].
    """
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)

    image = np.clip(image, low, high)
    image = (image - low) / (high - low)

    return image.astype(np.float32)


def normalize_bscan(image):
    """Normalize one B-scan using its own percentiles."""
    low, high = np.percentile(image, [P_LOW, P_HIGH])
    return normalize_with_limits(image, low, high)


def normalize_volume(volume):
    """
    Normalize a volume according to NORMALIZATION_MODE.

    Input shape: (number_of_slices, height, width)
    Output shape: same as input, with values in [0, 1].
    """
    if NORMALIZATION_MODE == "per_slice":
        normalized = np.stack(
            [normalize_bscan(image) for image in volume],
            axis=0,
        )

    elif NORMALIZATION_MODE == "per_volume":
        low, high = np.percentile(
            volume, [P_LOW, P_HIGH]
        )
        normalized = normalize_with_limits(
            volume, low, high
        )

    else:
        raise ValueError(
            f"Unknown normalization mode: {NORMALIZATION_MODE}"
        )

    return normalized.astype(np.float32)


# ============================================================
# 2.5D STACK CONSTRUCTION
# ============================================================

def construct_stack(normalized, slice_indices, center_index):
    """
    Construct a five-slice stack around a central B-scan.

    Replicate the nearest slice at the volume boundaries.
    Skip stacks that cross missing internal slice indices.

    Returns:
        stack: (5, H, W), or None if the stack crosses a gap.
    """
    index_to_position = {
        slice_id: position
        for position, slice_id in enumerate(slice_indices)
    }

    center_position = index_to_position[center_index]
    stack_slices = []

    for offset in range(-NEIGHBORS, NEIGHBORS + 1):
        requested_index = center_index + offset

        # Replicate the boundary slice outside the volume range.
        if requested_index < slice_indices[0]:
            position = 0

        elif requested_index > slice_indices[-1]:
            position = len(slice_indices) - 1

        # Do not cross a missing internal slice.
        elif requested_index not in index_to_position:
            return None

        else:
            position = index_to_position[requested_index]

        stack_slices.append(normalized[position])

    stack = np.stack(stack_slices, axis=0)

    if stack.shape[0] != 2 * NEIGHBORS + 1:
        raise RuntimeError("Incorrect number of stack channels.")

    return stack.astype(np.float32)


# ============================================================
# PROCESS ONE VOLUME
# ============================================================

def process_volume(volume_dir, writer):
    paths = get_ordered_slices(volume_dir)

    if not paths:
        print(f"Skipping empty volume: {volume_dir.name}")
        return 0

    print(f"\nProcessing volume: {volume_dir.name}")

    images = [load_bscan(path) for path in paths]

    shapes = {image.shape for image in images}

    if len(shapes) != 1:
        raise ValueError(
            f"Inconsistent image dimensions in volume "
            f"{volume_dir.name}: {shapes}"
        )

    original = np.stack(images, axis=0)
    slice_indices = [int(path.stem) for path in paths]

    normalized = normalize_volume(original)

    volume_id = volume_dir.name
    volume_output = OUTPUT_ROOT / volume_id
    volume_output.mkdir(parents=True, exist_ok=True)

    # Preserve the original image values when requested.
    if SAVE_ORIGINAL_VOLUME:
        np.save(
            volume_output / "original_bscans.npy",
            original,
        )

    # Save the normalized volume once.
    if SAVE_NORMALIZED_VOLUME:
        np.save(
            volume_output / "normalized_bscans.npy",
            normalized,
        )

    stack_dir = volume_output / "stacks"

    if SAVE_STACKS:
        stack_dir.mkdir(parents=True, exist_ok=True)

    valid_stacks = 0
    skipped_stacks = 0

    for position, center_index in enumerate(slice_indices):
        stack = construct_stack(
            normalized,
            slice_indices,
            center_index,
        )

        if stack is None:
            skipped_stacks += 1
            continue

        stack_name = f"stack_{center_index:03d}.npy"

        if SAVE_STACKS:
            np.save(stack_dir / stack_name, stack)

        # Each row identifies one training example.
        writer.writerow({
            "volume_id": volume_id,
            "center_slice_id": center_index,
            "center_position": position,
            "original_image": str(
                paths[position].relative_to(PROJECT_ROOT)
            ),
            "normalized_volume": str(
                (
                    volume_output / "normalized_bscans.npy"
                ).relative_to(PROJECT_ROOT)
            ),
            "stack_path": (
                str((stack_dir / stack_name).relative_to(PROJECT_ROOT))
                if SAVE_STACKS else ""
            ),
            "annotation_path": "",
            "height": stack.shape[1],
            "width": stack.shape[2],
            "channels": stack.shape[0],
        })

        valid_stacks += 1

    print(f"  Number of B-scans: {len(paths)}")
    print(f"  Image dimensions: {original.shape[1:]}")
    print(f"  Valid stacks: {valid_stacks}")
    print(f"  Skipped stacks due to gaps: {skipped_stacks}")

    return valid_stacks


# ============================================================
# MAIN
# ============================================================


def main():
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(
            f"OCT directory not found:\n{DATA_ROOT}\n\n"
            "Check PROJECT_ROOT and DATA_ROOT."
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Process only the three selected volumes.
    selected_volumes = ["10401", "10402", "10403"]

    volume_dirs = [
        DATA_ROOT / volume_id
        for volume_id in selected_volumes
    ]

    # Check that every selected volume exists.
    missing = [
        path for path in volume_dirs
        if not path.is_dir()
    ]

    if missing:
        raise FileNotFoundError(
            "The following volume folders were not found:\n"
            + "\n".join(str(path) for path in missing)
        )

    manifest_path = OUTPUT_ROOT / "manifest.csv"
    total_stacks = 0
    processed_volumes = 0

    fields = [
        "volume_id",
        "center_slice_id",
        "center_position",
        "original_image",
        "normalized_volume",
        "stack_path",
        "annotation_path",
        "height",
        "width",
        "channels",
    ]

    with manifest_path.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fields,
        )
        writer.writeheader()

        for volume_dir in volume_dirs:
            if not list(volume_dir.glob("*.bmp")):
                print(
                    f"Skipping folder without BMP images: "
                    f"{volume_dir.name}"
                )
                continue

            count = process_volume(volume_dir, writer)
            total_stacks += count
            processed_volumes += 1

    print("\nPreprocessing completed.")
    print(f"Selected volumes: {selected_volumes}")
    print(f"Volumes processed: {processed_volumes}")
    print(f"Total valid stacks: {total_stacks}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"Manifest: {manifest_path}")
    print("No segmentation labels were generated.")


if __name__ == "__main__":
    main()


