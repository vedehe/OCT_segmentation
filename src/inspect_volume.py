from pathlib import Path
import numpy as np
import heyexReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "OCT_Manual_Delineations-2018_June_29"
)

vol_path = DATA_ROOT / "vol" / "hc01_spectralis_macula_v1_s1_R.vol"

volume = heyexReader.volFile(str(vol_path))
oct_data = np.asarray(volume.oct)

print("Volume:", vol_path.name)
print("Volume shape:", oct_data.shape)
print("Data type:", oct_data.dtype)
print("Intensity range:", oct_data.min(), oct_data.max())

print("\nAnnotation file:")
print(DATA_ROOT / "delineation" / "hc01_spectralis_macula_v1_s1_R.mat")

