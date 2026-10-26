
from pathlib import Path


# PROJECT PATHS

NUM_WORKERS = 0
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"

RAW_IMAGE_DIR = DATA_ROOT / "images"
PROCESSED_DIR = DATA_ROOT / "processed" / "images"
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"
MASK_DIR = DATA_ROOT / "masks"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

PREDICTION_DIR = OUTPUT_DIR / "predictions"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"



# DATASET


SELECTED_VOLUMES = ["10401", "10402", "10403"]

IMAGE_HEIGHT = 640
IMAGE_WIDTH = 304

NUM_INPUT_CHANNELS = 5
NUM_CLASSES = 8
NEIGHBORS = 2



# MODEL


BASE_CHANNELS = 32



# TRAINING


BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

RANDOM_SEED = 42

# Set these when annotation files and subject splits are ready.
TRAIN_VOLUMES = []
VAL_VOLUMES = []
TEST_VOLUMES = []



# INITIALIZATION
 

def create_output_dirs():
    """Create directories used for model outputs."""
    for directory in (
        CHECKPOINT_DIR,
        PREDICTION_DIR,
        FIGURE_DIR,
        LOG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_output_dirs()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Processed images: {PROCESSED_DIR}")
    print(f"Manifest exists: {MANIFEST_PATH.is_file()}")
    print(f"Mask directory exists: {MASK_DIR.is_dir()}")
    print("Output directories initialized.")
