
from pathlib import Path
import scipy.io as sio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANNOTATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "OCT_Manual_Delineations-2018_June_29"
    / "delineation"
)

mat_files = sorted(ANNOTATION_DIR.glob("*.mat"))
print("Annotation files:", len(mat_files))

for mat_path in mat_files:
    data = sio.loadmat(mat_path)

    if "control_pts" not in data:
        print(f"\nMISSING control_pts: {mat_path.name}")
        print("Available variables:", [
            key for key in data if not key.startswith("__")
        ])
        continue

    control_pts = data["control_pts"]
    inconsistent = []

    for scan_idx, scan in enumerate(control_pts):
        valid_slots = [
            i for i, points in enumerate(scan)
            if np.asarray(points).size > 0
        ]

        if len(valid_slots) != 9:
            inconsistent.append((scan_idx + 1, valid_slots))

    first_scan_slots = [
        i for i, points in enumerate(control_pts[0])
        if np.asarray(points).size > 0
    ]

    print(
        f"{mat_path.stem}: {len(control_pts)} scans, "
        f"first-scan slots={first_scan_slots}, "
        f"inconsistent scans={len(inconsistent)}"
    )

    for scan_idx, slots in inconsistent[:5]:
        print(f"  B-scan {scan_idx}: valid slots={slots}")
