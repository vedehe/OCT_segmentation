from pathlib import Path
import numpy as np
import scipy.io as sio

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANNOTATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "OCT_Manual_Delineations-2018_June_29"
    / "delineation"
)

mat_path = ANNOTATION_DIR / "hc01_spectralis_macula_v1_s1_R.mat"
data = sio.loadmat(mat_path)

control_pts = data["control_pts"]
# import numpy as np

for scan_index, scan_annotations in enumerate(control_pts):
    valid = [
        i for i, points in enumerate(scan_annotations)
        if np.asarray(points).size > 0
    ]
    print(
        f"B-scan {scan_index + 1:02d}: "
        f"{len(valid)} valid boundaries — {valid}"
    )

print("control_pts shape:", control_pts.shape)
print("control_pts dtype:", control_pts.dtype)
print("segmode:", data["segmode"])

print("\nFirst B-scan annotation:")
for i, item in enumerate(control_pts[0]):
    print(f"Entry {i}: type={type(item)}")

    if isinstance(item, np.ndarray):
        print("  shape:", item.shape)
        print("  dtype:", item.dtype)
        print("  values:", item)
    else:
        print("  value:", item)

print("\nAll annotation entries per B-scan:", control_pts.shape[1])