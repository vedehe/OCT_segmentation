import cv2
import numpy as np
import matplotlib.pyplot as plt

# Original OCT image
image_path = (
    "data/raw/OCTA_3mm/116.bmp"
)

processed_path = (
    "data/processed/images/116.bmp"
)

# Parameters to experiment with
E = 12
MIN_REGION_SIZE = 15
ROI_THRESHOLD = 10

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
processed = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Original image not found: {image_path}")

if processed is None:
    raise FileNotFoundError(
        f"Processed image not found: {processed_path}"
    )

if image.shape != processed.shape:
    raise ValueError("Original and processed images must have the same size.")

# --------------------------------------------------
# 1. Identify the retinal region using processed image
# --------------------------------------------------

roi = (processed > ROI_THRESHOLD).astype(np.uint8)

# Fill small gaps in the retinal region
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (5, 5)
)

roi = cv2.morphologyEx(
    roi, cv2.MORPH_CLOSE, kernel
)

# Keep the largest connected region
num_labels, component_map, stats, _ = (
    cv2.connectedComponentsWithStats(roi, 8)
)

if num_labels > 1:
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    roi = (component_map == largest).astype(np.uint8)

# --------------------------------------------------
# 2. Divide retinal pixels into intensity bands
# --------------------------------------------------

intensity = image.astype(np.int32)
bands = intensity // E

# --------------------------------------------------
# 3. Find connected regions within each band
# --------------------------------------------------

labels = np.zeros(image.shape, dtype=np.int32)
next_label = 1

for band in np.unique(bands[roi == 1]):

    binary = (
        (bands == band) & (roi == 1)
    ).astype(np.uint8)

    n, component_map, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
    )

    for component in range(1, n):

        area = stats[component, cv2.CC_STAT_AREA]

        if area < MIN_REGION_SIZE:
            continue

        region = component_map == component
        labels[region] = next_label
        next_label += 1

# --------------------------------------------------
# 4. Assign a distinct colour to every region
# --------------------------------------------------

palette = np.array([
    [255,   0,   0],  # Red
    [  0, 255,   0],  # Green
    [255, 255,   0],  # Yellow
    [255,   0, 255],  # Magenta
    [  0, 255, 255],  # Cyan
    [255, 128,   0],  # Orange
    [128,   0, 255],  # Purple
    [255, 255, 255],  # White
    [  0, 128, 255],  # Orange-red
    [128, 255,   0],  # Lime
], dtype=np.uint8)

colours = np.zeros((next_label, 3), dtype=np.uint8)

for label in range(1, next_label):
    colours[label] = palette[(label - 1) % len(palette)]

segmented = colours[labels]

# Pixels outside the retina remain black
segmented[roi == 0] = [0, 0, 0]

# --------------------------------------------------
# 5. Display the results
# --------------------------------------------------

fig, ax = plt.subplots(1, 3, figsize=(16, 7))

ax[0].imshow(image, cmap="gray")
ax[0].set_title("Original OCT")

ax[1].imshow(roi, cmap="gray")
ax[1].set_title("Retinal region mask")

ax[2].imshow(segmented)
ax[2].set_title("Distinct connected regions")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.show()

# Save the result
output_path = (
    "/Users/vaideheedongre/OCT_Segmentation/"
    "outputs/intensity_regions_116.png"
)

cv2.imwrite(
    output_path,
    cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR)
)

print(f"Saved result to: {output_path}")