
from pathlib import Path
import numpy as np
import scipy.io as sio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = (
    PROJECT_ROOT
    / "data/images/OCT_Manual_Delineations-2018_June_29/delineation"
)

for filename in [
    "ms14_spectralis_macula_v1_s1_R.mat",
    "ms16_spectralis_macula_v1_s1_R.mat",
    "ms17_spectralis_macula_v1_s1_R.mat",
]:
    data = sio.loadmat(ANNOTATION_DIR / filename)
    bd = data["bd_pts"]

    print(f"\n{filename}: shape={bd.shape}")
    print("Global range:", np.nanmin(bd), np.nanmax(bd))

    # One B-scan: (horizontal position, boundary)
    scan = bd[:, 0, :]
    for boundary in range(scan.shape[1]):
        values = scan[:, boundary]
        valid = np.isfinite(values) & (values > 0)
        print(
            f"Boundary {boundary}: "
            f"valid={valid.sum()}/{len(values)}, "
            f"range=({values[valid].min() if valid.any() else 'NA'}, "
            f"{values[valid].max() if valid.any() else 'NA'})"
        )

    print("First 10 columns of first B-scan:")
    print(bd[:10, 0, :])
