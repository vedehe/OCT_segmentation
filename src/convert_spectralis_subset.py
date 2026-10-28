
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# heyexReader is already installed and patched in your environment.
import heyexReader


# 1. Paths and settings


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "OCT_Manual_Delineations-2018_June_29"
)

VOLUME_DIR = DATASET_ROOT / "vol"
ANNOTATION_DIR = DATASET_ROOT / "delineation"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "spectralis_processed"
IMAGE_OUTPUT_DIR = OUTPUT_ROOT / "images"
MASK_OUTPUT_DIR = OUTPUT_ROOT / "masks"
PREVIEW_OUTPUT_DIR = OUTPUT_ROOT / "previews"

SUBJECTS = [
    # "hc01_spectralis_macula_v1_s1_R",
    # "hc02_spectralis_macula_v1_s1_R",
    "hc03_spectralis_macula_v1_s1_R",
]

HEIGHT = 496
WIDTH = 1024
NUM_SCANS = 49
NUM_BOUNDARIES = 9
NUM_CLASSES = 9
NEIGHBORS = 2



# 2. Load the Spectralis volume



def load_volume(path):
    volume_reader = heyexReader.volFile(str(path))
    volume = np.asarray(volume_reader.oct)

    # Expected layout: (scans, height, width)
    if volume.shape != (NUM_SCANS, HEIGHT, WIDTH):
        raise ValueError(
            f"Unexpected volume shape for {path.name}: {volume.shape}"
        )

    return volume




# 3. Load and interpolate manual boundaries


def load_boundaries(mat_path):
    data = sio.loadmat(mat_path)

    if "control_pts" not in data:
        raise ValueError(f"No control_pts found in {mat_path.name}")

    control_pts = data["control_pts"]

    if control_pts.shape[0] != NUM_SCANS:
        raise ValueError(
            f"Unexpected annotation shape: {control_pts.shape}"
        )

    # These are the nine nonempty slots found during inspection.
    slots = [0, 1, 3, 4, 5, 6, 7, 8, 9]

    all_boundaries = np.zeros(
        (NUM_SCANS, NUM_BOUNDARIES, WIDTH),
        dtype=np.float32,
    )

    x_full = np.arange(WIDTH, dtype=np.float32)

    for scan_idx in range(NUM_SCANS):
        for boundary_idx, slot in enumerate(slots):
            points = control_pts[scan_idx, slot]

            if points is None or np.asarray(points).size == 0:
                raise ValueError(
                    f"Missing boundary: scan {scan_idx + 1}, slot {slot}"
                )

            points = np.asarray(points, dtype=np.float32)

            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError(
                    f"Unexpected point shape: {points.shape}"
                )

            # Annotation coordinates are (x, y), with x starting at 1.
            x = points[:, 0] - 1.0
            y = points[:, 1]

            # Sort points by x before interpolation.
            order = np.argsort(x)
            x = x[order]
            y = y[order]

            if np.any(np.diff(x) <= 0):
                raise ValueError(
                    f"Duplicate or invalid x coordinates in "
                    f"scan {scan_idx + 1}, boundary {boundary_idx + 1}"
                )

            all_boundaries[scan_idx, boundary_idx] = np.interp(
                x_full, x, y
            )

    return all_boundaries


def inspect_boundary_order(boundaries, subject, scan_number):
    differences = np.diff(boundaries, axis=0)

    print(f"\nBoundary check: {subject}, scan {scan_number}")

    for i in range(NUM_BOUNDARIES - 1):
        diff = differences[i]
        bad_x = np.where(diff < 0)[0]

        if len(bad_x) > 0:
            print(
                f"Boundary {i + 1} and {i + 2}: "
                f"{len(bad_x)} crossing positions"
            )

            for x in bad_x[:10]:
                print(
                    f"  x={x}: "
                    f"upper={boundaries[i, x]:.2f}, "
                    f"lower={boundaries[i + 1, x]:.2f}"
                )

def save_crossing_debug_plot(image, boundaries, subject, scan_number):
    x = np.arange(WIDTH)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    for i, boundary in enumerate(boundaries):
        axes[0].plot(x, boundary, linewidth=0.8, label=f"B{i + 1}")

    axes[0].set_xlim(470, 540)
    axes[0].set_title(f"{subject}: scan {scan_number}")
    axes[0].legend(ncol=5)

    axes[1].plot(x, boundaries[1], label="Boundary 2")
    axes[1].plot(x, boundaries[2], label="Boundary 3")
    axes[1].set_xlim(470, 540)
    axes[1].set_xlabel("Horizontal pixel x")
    axes[1].set_ylabel("Vertical pixel y")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()

    output_path = (
        PREVIEW_OUTPUT_DIR
        / f"{subject}_scan_{scan_number:03d}_crossing_debug.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved crossing diagnostic: {output_path}")



# 4. Convert boundaries into a pixel mask



def boundaries_to_mask(boundaries):
    """
    Class 0: background.
    Classes 1-8: regions between the nine annotated boundaries.

    If two boundaries overlap or cross slightly, use their midpoint
    to define a valid pixel assignment for that column. The original
    boundary coordinates are not modified.
    """
    if boundaries.shape != (NUM_BOUNDARIES, WIDTH):
        raise ValueError(
            f"Expected boundaries shape "
            f"({NUM_BOUNDARIES}, {WIDTH}), got {boundaries.shape}"
        )

    if not np.isfinite(boundaries).all():
        raise ValueError("Boundaries contain non-finite values.")

    if np.any(boundaries < 0) or np.any(boundaries > HEIGHT - 1):
        raise ValueError("A boundary lies outside the image.")

    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    for x in range(WIDTH):
        column_boundaries = boundaries[:, x].copy()

        # Enforce nondecreasing vertical positions for rasterization
        # only. The source annotations remain unchanged.
        for i in range(1, NUM_BOUNDARIES):
            if column_boundaries[i] < column_boundaries[i - 1]:
                midpoint = (
                    column_boundaries[i] + column_boundaries[i - 1]
                ) / 2.0
                column_boundaries[i - 1] = midpoint
                column_boundaries[i] = midpoint

        # Rasterize the eight regions using half-open intervals.
        for region in range(8):
            top = int(np.ceil(column_boundaries[region]))
            bottom = int(np.ceil(column_boundaries[region + 1])) - 1

            top = max(top, 0)
            bottom = min(bottom, HEIGHT - 1)

            if bottom >= top:
                mask[top:bottom + 1, x] = region + 1

    return mask



# 5. Make five neighbouring B-scans


def make_stack(volume, scan_idx):
    indices = [
        min(max(scan_idx + offset, 0), len(volume) - 1)
        for offset in range(-NEIGHBORS, NEIGHBORS + 1)
    ]

    stack = volume[indices].astype(np.float32) / 255.0

    # Output shape: (5, 496, 1024)
    return stack



# 6. Save a preview for visual verification


def save_preview(image, boundaries, mask, subject, scan_number):
    PREVIEW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original OCT B-scan")

    axes[1].imshow(image, cmap="gray", vmin=0, vmax=255)
    for boundary in boundaries:
        axes[1].plot(
            np.arange(WIDTH),
            boundary,
            linewidth=0.8,
        )
    axes[1].set_title("Manual boundaries")

    axes[2].imshow(mask, vmin=0, vmax=8, interpolation="nearest")
    axes[2].set_title("Generated mask (classes 0–8)")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()

    output_path = PREVIEW_OUTPUT_DIR / f"{subject}_scan_{scan_number:03d}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------
# 7. Process one subject
# --------------------------------------------------


def process_subject(subject):
    volume_path = VOLUME_DIR / f"{subject}.vol"
    annotation_path = ANNOTATION_DIR / f"{subject}.mat"

    if not volume_path.exists():
        raise FileNotFoundError(volume_path)

    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)

    print(f"\nProcessing {subject}...")

    volume = load_volume(volume_path)
    boundaries = load_boundaries(annotation_path)

    image_subject_dir = IMAGE_OUTPUT_DIR / subject / "stacks"
    mask_subject_dir = MASK_OUTPUT_DIR / subject

    image_subject_dir.mkdir(parents=True, exist_ok=True)
    mask_subject_dir.mkdir(parents=True, exist_ok=True)

    for scan_idx in range(NUM_SCANS):
        scan_number = scan_idx + 1
        image = volume[scan_idx]
        scan_boundaries = boundaries[scan_idx]

        # differences = np.diff(scan_boundaries, axis=0)

        # if np.any(differences < 0):
        #     save_crossing_debug_plot(
        #         image,
        #         scan_boundaries,
        #         subject,
        #         scan_number,
        #     )
        #     raise ValueError(
        #         f"Crossing detected in {subject}, scan {scan_number}. "
        #         "Inspect the saved diagnostic plot."
        #     )

        mask = boundaries_to_mask(scan_boundaries)
        # stack = make_stack(volume, scan_idx)
        stack = make_stack(volume, scan_idx)

        stack_path = image_subject_dir / f"scan_{scan_number:03d}.npy"
        mask_path = mask_subject_dir / f"scan_{scan_number:03d}.npy"

        np.save(stack_path, stack)
        np.save(mask_path, mask)

        if scan_idx == 0:
            save_preview(
                image,
                scan_boundaries,
                mask,
                subject,
                scan_number,
            )

        if np.unique(mask).size < 2:
            print(
                f"Warning: scan {scan_number} has fewer than "
                "two label values."
            )

    print(f"Saved {NUM_SCANS} stacks and masks for {subject}.")




# 8. Main: process only the two selected subjects

def main():
    for directory in (
        IMAGE_OUTPUT_DIR,
        MASK_OUTPUT_DIR,
        PREVIEW_OUTPUT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    for subject in SUBJECTS:
        process_subject(subject)

    print("\nFinished processing the two selected healthy subjects.")
    print(f"Output directory: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
