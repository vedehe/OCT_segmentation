from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import heyexReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "OCT_Manual_Delineations-2018_June_29"
)

volume_path = DATA_ROOT / "vol" / "hc01_spectralis_macula_v1_s1_R.vol"
annotation_path = (
    DATA_ROOT
    / "delineation"
    / "hc01_spectralis_macula_v1_s1_R.mat"
)

volume = heyexReader.volFile(str(volume_path)).oct
annotations = sio.loadmat(annotation_path)["control_pts"]

# Visualize the first B-scan.
scan_index = 0
image = volume[scan_index]

fig, ax = plt.subplots(figsize=(14, 6))
ax.imshow(image, cmap="gray", aspect="auto")

for boundary_index, points in enumerate(annotations[scan_index]):
    points = np.asarray(points)

    if points.size == 0:
        continue

    # Each row is (x, y): horizontal position, vertical position.
    x = points[:, 0] - 1  # Convert 1-based x coordinates to 0-based.
    y = points[:, 1]

    ax.plot(x, y, linewidth=1, label=f"Boundary {boundary_index}")

ax.set_title(f"hc01 — B-scan {scan_index + 1}: manual boundaries")
ax.set_xlabel("Image column")
ax.set_ylabel("Image row")
ax.set_xlim(0, image.shape[1] - 1)
ax.set_ylim(image.shape[0] - 1, 0)
ax.legend(loc="upper right", fontsize=7, ncol=2)
fig.tight_layout()

output_path = PROJECT_ROOT / "outputs" / "figures" / "hc01_annotation_overlay.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=180, bbox_inches="tight")
plt.show()
plt.close(fig)

print("Saved overlay to:", output_path)