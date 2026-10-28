from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import heyexReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "OCT_Manual_Delineations-2018_June_29"
)

VOLUME_DIR = DATA_ROOT / "vol"
ANNOTATION_DIR = DATA_ROOT / "delineation"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "spectralis_processed"
IMAGE_DIR = OUTPUT_ROOT / "images"
MASK_DIR = OUTPUT_ROOT / "masks"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

NUM_CLASSES = 9
NUM_INPUT_CHANNELS = 5
NEIGHBORS = 2

# The nine consistently nonempty slots in control_pts.
CONTROL_SLOTS = [0, 1, 3, 4, 5, 6, 7, 8, 9]


def load_boundaries(mat_path):
    """Return boundaries with shape (49, 9, 1024)."""
    data = sio.loadmat(mat_path)

    if "control_pts" in data:
        control_pts = data["control_pts"]
        num_scans = control_pts.shape[0]
        boundaries = []

        for scan_idx in range(num_scans):
            scan_boundaries = []

            for slot in CONTROL_SLOTS:
                points = np.asarray(
                    control_pts[scan_idx, slot],
                    dtype=np.float64,
                )

                if points.size == 0:
                    raise ValueError(
                        f"Missing boundary {slot} in "
                        f"{mat_path.name}, scan {scan_idx + 1}"
                    )

                x = points[:, 0] - 1.0
                y = points[:, 1]

                # Interpolate control points across the image width.
                x_full = np.arange(1024, dtype=np.float64)
                y_full = np.interp(x_full, x, y)

                scan_boundaries.append(y_full)

            boundaries.append(scan_boundaries)

        return np.asarray(boundaries, dtype=np.float32)

    if "bd_pts" in data:
        # Source shape: (width, scans, boundaries).
        bd_pts = np.asarray(data["bd_pts"], dtype=np.float32)
        return np.transpose(bd_pts, (1, 2, 0))

    raise ValueError(
        f"No supported annotation variable in {mat_path.name}"
    )


def boundaries_to_mask(boundaries, height, width):
    """
    Convert nine ordered boundaries into labels:
    0 = background; 1..8 = regions between boundaries.
    """
    if boundaries.shape != (9, width):
        raise ValueError(
            f"Expected boundaries shape (9, {width}), "
            f"got {boundaries.shape}"
        )

    # Clip to image limits and check boundary ordering.
    boundaries = np.clip(boundaries, 0, height - 1)

    if np.any(np.diff(boundaries, axis=0) < -1.0):
        raise ValueError("Boundaries cross or are out of order.")

    y = np.arange(height)[:, None]
    mask = np.zeros((height, width), dtype=np.uint8)

    for region in range(8):
        upper = boundaries[region]
        lower = boundaries[region + 1]

        rows = y[:, 0]
        for x in range(width):
            top = int(np.ceil(upper[x]))
            bottom = int(np.floor(lower[x]))

            if bottom >= top:
                mask[top:bottom + 1, x] = region + 1

    return mask


def make_stack(volume, scan_idx):
    """Create a five-channel stack with replicated edge slices."""
    indices = [
        min(max(scan_idx + offset, 0), len(volume) - 1)
        for offset in range(-NEIGHBORS, NEIGHBORS + 1)
    ]

    stack = np.stack([volume[i] for i in indices], axis=0)
    return stack.astype(np.float32) / 255.0


def save_preview(image, mask, boundaries, output_path):
    """Save an OCT/mask/overlay validation figure."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("OCT B-scan")

    axes[1].imshow(mask, cmap="tab10", vmin=0, vmax=8)
    axes[1].set_title("Generated mask")

    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(mask, cmap="tab10", vmin=0, vmax=8, alpha=0.4)
    for boundary in boundaries:
        axes[2].plot(
            np.arange(image.shape[1]),
            boundary,
            linewidth=0.5,
        )
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(ANNOTATION_DIR.glob("*.mat"))
    print("Annotation files:", len(mat_files))

    total_scans = 0

    for mat_path in mat_files:
        volume_path = VOLUME_DIR / f"{mat_path.stem}.vol"

        if not volume_path.is_file():
            print("Skipping missing volume:", volume_path.name)
            continue

        print("Processing:", mat_path.stem)

        volume = np.asarray(
            heyexReader.volFile(str(volume_path)).oct
        )

        if volume.ndim != 3:
            raise ValueError(
                f"Unexpected volume shape {volume.shape}: "
                f"{volume_path.name}"
            )

        num_scans, height, width = volume.shape
        boundaries_by_scan = load_boundaries(mat_path)

        if len(boundaries_by_scan) != num_scans:
            raise ValueError(
                f"Scan count mismatch for {mat_path.stem}: "
                f"volume={num_scans}, "
                f"annotations={len(boundaries_by_scan)}"
            )

        subject_image_dir = IMAGE_DIR / mat_path.stem
        subject_mask_dir = MASK_DIR / mat_path.stem

        subject_image_dir.mkdir(parents=True, exist_ok=True)
        subject_mask_dir.mkdir(parents=True, exist_ok=True)

        for scan_idx in range(num_scans):
            image = volume[scan_idx]
            boundaries = boundaries_by_scan[scan_idx]

            mask = boundaries_to_mask(
                boundaries,
                height,
                width,
            )

            stack = make_stack(volume, scan_idx)
            stem = f"scan_{scan_idx + 1:03d}"

            np.save(subject_image_dir / f"{stem}.npy", stack)
            np.save(subject_mask_dir / f"{stem}.npy", mask)

            if scan_idx == 0:
                save_preview(
                    image,
                    mask,
                    boundaries,
                    FIGURE_DIR / f"{mat_path.stem}_preview.png",
                )

        total_scans += num_scans
        print(f"  Saved {num_scans} scans ({height} x {width})")

    print("\nConversion complete.")
    print("Total scans:", total_scans)
    print("Images:", IMAGE_DIR)
    print("Masks:", MASK_DIR)


if __name__ == "__main__":
    main()