
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUBJECT = "hc01_spectralis_macula_v1_s1_R"
SCAN_NUMBER = 1

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "OCT_Manual_Delineations-2018_June_29"
)
PROCESSED_ROOT = PROJECT_ROOT / "data" / "spectralis_processed"

# Load the original OCT volume.
import heyexReader

volume_path = DATASET_ROOT / "vol" / f"{SUBJECT}.vol"
volume_reader = heyexReader.volFile(str(volume_path))
volume = np.asarray(volume_reader.oct)

image = volume[SCAN_NUMBER - 1]

# Load the generated mask.
mask_path = (
    PROCESSED_ROOT
    / "masks"
    / SUBJECT
    / f"scan_{SCAN_NUMBER:03d}.npy"
)
mask = np.load(mask_path)

# Display the mask with transparency over the OCT image.
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Original OCT")

axes[1].imshow(image, cmap="gray", vmin=0, vmax=255)
axes[1].imshow(mask, alpha=0.45, vmin=0, vmax=8, interpolation="nearest")
axes[1].set_title("OCT with generated mask overlay")

for ax in axes:
    ax.axis("off")

fig.tight_layout()

output_path = (
    PROCESSED_ROOT / "previews" / f"{SUBJECT}_scan_{SCAN_NUMBER:03d}_overlay.png"
)
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"Saved overlay: {output_path}")
