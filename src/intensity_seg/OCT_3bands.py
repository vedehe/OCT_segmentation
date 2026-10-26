
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = PROJECT_ROOT / "data/raw/OCTA_3mm/116.bmp"
PROCESSED_PATH = PROJECT_ROOT / "data/processed/images/116.bmp"

OUTPUT_DIR = PROJECT_ROOT / "outputs/intensity_3bands_116"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Settings

ROI_THRESHOLD = 10
THICKNESS_FRACTION = 0.35
N_BANDS = 3

# Colours ordered from darkest to brightest intensity (RGB)
PALETTE = np.array([
    [30, 30, 150],     # Band 1: deep blue
    [255, 80, 130],    # Band 2: pink
    [255, 225, 40],    # Band 3: yellow
], dtype=np.uint8)


# 2. Load images

image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
processed = cv2.imread(str(PROCESSED_PATH), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Original image not found: {IMAGE_PATH}")

if processed is None:
    raise FileNotFoundError(f"Processed image not found: {PROCESSED_PATH}")

if image.shape != processed.shape:
    raise ValueError(
        "Original and processed images must have the same dimensions."
    )

H, W = image.shape


# 3. Estimate retinal ROI

roi = (processed > ROI_THRESHOLD).astype(np.uint8)

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (5, 5)
)

roi = cv2.morphologyEx(
    roi, cv2.MORPH_CLOSE, kernel
)

n, component_map, stats, _ = cv2.connectedComponentsWithStats(
    roi, connectivity=8
)

if n <= 1:
    raise ValueError("No retinal region detected in the processed image.")

# Select the largest connected component, excluding the background
largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
roi = (component_map == largest).astype(np.uint8)

top = np.zeros(W, dtype=np.int32)
bottom = np.full(W, H - 1, dtype=np.int32)
valid_cols = []

for x in range(W):
    ys = np.flatnonzero(roi[:, x])

    if len(ys) > 0:
        top[x] = ys[0]
        bottom[x] = ys[-1]
        valid_cols.append(x)

if len(valid_cols) < 2:
    raise ValueError("Could not determine the retinal ROI.")

# Interpolate missing columns
valid_cols = np.asarray(valid_cols)
xs = np.arange(W)

top = np.rint(
    np.interp(xs, valid_cols, top[valid_cols])
).astype(int)

bottom = np.rint(
    np.interp(xs, valid_cols, bottom[valid_cols])
).astype(int)


# Restrict the ROI to the upper retinal region,
# excluding as much of the lower choroidal region as possible.


# Restrict ROI to the retina, ending near the bright outer retinal band.
# This is an initial boundary estimate and should be checked visually.

from scipy.ndimage import gaussian_filter1d

retinal_roi = np.zeros((H, W), dtype=np.uint8)

# Smooth the original image vertically to reduce pixel-level noise
smoothed = gaussian_filter1d(
    image.astype(np.float32),
    sigma=2.0,
    axis=0
)

for x in range(W):
    if roi[:, x].sum() == 0:
        continue

    upper = int(top[x])
    full_thickness = max(1, int(bottom[x]) - upper)

    # Search below the upper retinal boundary.
    # Adjust these limits after inspecting the detected boundary.
    search_start = min(H - 1, upper + int(0.35 * full_thickness))
    search_end = min(H, upper + int(0.90 * full_thickness) + 1)

    if search_end <= search_start:
        continue

    # Find the brightest smoothed location in the lower retinal region.
    profile = smoothed[search_start:search_end, x]
    boundary = search_start + int(np.argmax(profile))

    # Retain the region through the detected bright boundary.
    retinal_roi[upper:boundary + 1, x] = 1

if retinal_roi.sum() == 0:
    raise ValueError("The estimated retinal ROI is empty.")


# Verify that the ROI contains pixels
if retinal_roi.sum() == 0:
    raise ValueError("The estimated retinal ROI is empty.")

# Restrict the ROI to the upper fraction of the detected retina

# retinal_roi = np.zeros((H, W), dtype=np.uint8)

# for x in range(W):
#     thickness = max(1, bottom[x] - top[x])

#     lower = int(top[x] + THICKNESS_FRACTION * thickness)
#     lower = max(top[x] + 1, min(lower, H - 1))

#     retinal_roi[top[x]:lower + 1, x] = 1


# 4. Split original intensities into 3 quantile bands

intensity = image.astype(np.float32)
roi_values = intensity[retinal_roi == 1]

if roi_values.size == 0:
    raise ValueError("The estimated retinal ROI is empty.")

# Two thresholds divide the ROI intensities into 3 groups
thresholds = np.quantile(
    roi_values,
    np.arange(1, N_BANDS) / N_BANDS
)

# Labels: 1 = darkest, 2 = middle, 3 = brightest
# 0 = outside ROI
labels = np.zeros((H, W), dtype=np.uint8)

labels[retinal_roi == 1] = (
    np.searchsorted(
        thresholds,
        intensity[retinal_roi == 1],
        side="left"
    ) + 1
).astype(np.uint8)


# 5. Colour map

segmented_rgb = np.zeros((H, W, 3), dtype=np.uint8)

for band_id in range(1, N_BANDS + 1):
    segmented_rgb[labels == band_id] = PALETTE[band_id - 1]


# 6. Measure area and vertical span by band

rows = []

for band_id in range(1, N_BANDS + 1):
    band_mask = labels == band_id
    pixel_count = int(band_mask.sum())

    # Vertical span in each column containing this band
    spans = []

    for x in range(W):
        ys = np.flatnonzero(band_mask[:, x])

        if ys.size:
            spans.append(int(ys[-1] - ys[0] + 1))

    # Intensity range for each band
    if band_id == 1:
        intensity_range = f"<= {thresholds[0]:.1f}"
    elif band_id == N_BANDS:
        intensity_range = f"> {thresholds[-1]:.1f}"
    else:
        intensity_range = (
            f"> {thresholds[band_id - 2]:.1f} "
            f"to <= {thresholds[band_id - 1]:.1f}"
        )

    rows.append({
        "Band": band_id,
        "Intensity range": intensity_range,
        "Pixels": pixel_count,
        "ROI area (%)": (
            100 * pixel_count / max(1, int(retinal_roi.sum()))
        ),
        "Columns present": len(spans),
        "Mean vertical span (px)": (
            float(np.mean(spans)) if spans else 0.0
        ),
        "Median vertical span (px)": (
            float(np.median(spans)) if spans else 0.0
        ),
        "Std vertical span (px)": (
            float(np.std(spans)) if spans else 0.0
        ),
    })

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_DIR / "band_measurements.csv",
    index=False
)


# 7. Visualize

fig, ax = plt.subplots(1, 3, figsize=(19, 6))

ax[0].imshow(image, cmap="gray", vmin=0, vmax=255)
ax[0].set_title("Original OCT")

ax[1].imshow(retinal_roi, cmap="gray")
ax[1].set_title("Estimated retinal ROI")

ax[2].imshow(segmented_rgb)
ax[2].set_title("3 intensity bands (1 = darkest, 3 = brightest)")

legend_handles = [
    Patch(
        facecolor=PALETTE[i] / 255.0,
        label=f"Band {i + 1}"
    )
    for i in range(N_BANDS)
]

ax[2].legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=3,
    frameon=False
)

for a in ax:
    a.axis("off")

plt.tight_layout()

# Save the complete three-panel visualization
plt.savefig(
    OUTPUT_DIR / "intensity_3bands_116.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# Save the segmented RGB image separately
cv2.imwrite(
    str(OUTPUT_DIR / "segmented_3bands_116.png"),
    cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR)
)

print("Intensity thresholds:", np.round(thresholds, 2))
print(df.to_string(index=False))
print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")
print(
    "Note: intensity-band spans are not anatomical layer thicknesses."
)
