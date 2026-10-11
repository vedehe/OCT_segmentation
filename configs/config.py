from pathlib import Path
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
IMAGE_DIR = DATA_DIR / "images"
MASK_DIR = DATA_DIR / "masks"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
FIGURE_DIR = OUTPUT_DIR / "figures"


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512

IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)


# ============================================================
# MODEL SETTINGS
# ============================================================

# 0 = background
# 1, 2, ... = retinal layers

NUM_CLASSES = 9

IN_CHANNELS = 1


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 4

NUM_EPOCHS = 50

LEARNING_RATE = 1e-3

NUM_WORKERS = 2


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)