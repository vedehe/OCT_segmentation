# ```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. Load images
# --------------------------------------------------

image_path = "data/raw/OCTA_3mm/116.bmp"
processed_path = "data/processed/images/116.bmp"

output_path = Path("outputs/intensity_colours_116.png")
output_path.parent.mkdir(parents=True, exist_ok=True)

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
processed = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)

if image is None or processed is None:
    raise FileNotFoundError("Could not load OCT images.")

if image.shape != processed.shape:
    raise ValueError("Image dimensions do not match.")

H, W = image.shape

# --------------------------------------------------
# 2. Estimate retinal ROI
# --------------------------------------------------

# Initial ROI from the processed image.
roi = (processed > 10).astype(np.uint8)

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (5, 5)
)

roi = cv2.morphologyEx(
    roi, cv2.MORPH_CLOSE, kernel
)

# Keep the largest connected component.
n, component_map, stats, _ = (
    cv2.connectedComponentsWithStats(roi, 8)
)

if n <= 1:
    raise ValueError("No retinal region detected.")

largest = 1 + np.argmax(
    stats[1:, cv2.CC_STAT_AREA]
)

roi = (component_map == largest).astype(np.uint8)

# --------------------------------------------------
# 3. Restrict ROI to the retinal band
# --------------------------------------------------

# Find top and bottom ROI pixels per column.
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
    raise ValueError("Could not determine retinal ROI.")

valid_cols = np.asarray(valid_cols)
xs = np.arange(W)

top = np.rint(
    np.interp(xs, valid_cols, top[valid_cols])
).astype(int)

bottom = np.rint(
    np.interp(xs, valid_cols, bottom[valid_cols])
).astype(int)

# IMPORTANT:
# The ROI derived above may extend below the actual retina.
# Limit the lower edge using a manually adjustable fraction.
# Inspect the overlay and tune this value for image 116.
THICKNESS_FRACTION = 0.55

retinal_roi = np.zeros((H, W), dtype=np.uint8)

for x in range(W):
    thickness = bottom[x] - top[x]
    lower = int(top[x] + THICKNESS_FRACTION * thickness)
    lower = max(top[x] + 1, min(lower, H - 1))

    retinal_roi[top[x]:lower + 1, x] = 1

# --------------------------------------------------
# 4. Map grayscale intensity to colour
# --------------------------------------------------

# Original OCT grayscale values: 0 = dark, 255 = bright.
intensity = image.astype(np.float32)

# Use quantile bins within the retinal ROI.
values = intensity[retinal_roi == 1]

if len(values) == 0:
    raise ValueError("Retinal ROI is empty.")

q = np.quantile(values, [0.2, 0.4, 0.6, 0.8])

# Five colours from dark to bright.
palette = np.array([
    [25,  25, 120],   # Dark blue: darkest
    [0,   90, 220],   # Blue
    [0,  200, 220],   # Cyan
    [0,  220, 100],   # Green
    [255, 220,   0],  # Yellow: brightest
], dtype=np.uint8)

segmented = np.zeros((H, W, 3), dtype=np.uint8)

for x in range(W):
    for y in range(H):
        if retinal_roi[y, x] == 0:
            continue

        value = intensity[y, x]

        if value <= q[0]:
            idx = 0
        elif value <= q[1]:
            idx = 1
        elif value <= q[2]:
            idx = 2
        elif value <= q[3]:
            idx = 3
        else:
            idx = 4

        segmented[y, x] = palette[idx]

# --------------------------------------------------
# 5. Display
# --------------------------------------------------

fig, ax = plt.subplots(1, 3, figsize=(17, 6))

ax[0].imshow(image, cmap="gray")
ax[0].set_title("Original OCT")

ax[1].imshow(retinal_roi, cmap="gray")
ax[1].set_title("Estimated retinal ROI")

ax[2].imshow(segmented)
ax[2].set_title("Intensity-based colour mapping")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()

cv2.imwrite(
    str(output_path),
    cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)
)

print(f"Saved to: {output_path.resolve()}")
print("Intensity quantiles:", q)
# ```