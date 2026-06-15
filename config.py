# config.py
# Configuration file for FMCW Sparse MIMO Radar Imaging project

from pathlib import Path

# ============================================================
# 1. Dataset paths
# ============================================================

# Change this to your downloaded dataset path
DATA_ROOT = Path("./Dataset")

# Processed data will be saved here
PROCESSED_ROOT = Path("./Processed_Dataset")

# Output folders
CHECKPOINT_DIR = Path("./checkpoints")
RESULTS_DIR = Path("./results")
FIGURE_DIR = RESULTS_DIR / "figures"
LOG_DIR = RESULTS_DIR / "logs"


# ============================================================
# 2. Dataset structure
# ============================================================

RADAR_FOLDER_NAME = "radar_raw_frame"
IMAGE_FOLDER_NAME = "images_0"
LABEL_FOLDER_NAME = "text_labels"

RADAR_FILE_EXT = ".mat"
IMAGE_FILE_EXT = ".jpg"
LABEL_FILE_EXT = ".csv"


# ============================================================
# 3. Radar configuration
# ============================================================

# Raw ADC shape from dataset:
# samples × chirps × receivers × transmitters
NUM_ADC_SAMPLES = 128
NUM_CHIRPS = 255
NUM_RX = 4
NUM_TX = 2
NUM_VIRTUAL_CHANNELS = NUM_TX * NUM_RX   # 8 virtual MIMO channels

# Radar carrier frequency
FC = 77e9  # 77 GHz
C = 3e8
LAMBDA = C / FC


# ============================================================
# 4. FFT settings
# ============================================================

RANGE_FFT_SIZE = 128

# 255 chirps, so using 256 for Doppler FFT is convenient
DOPPLER_FFT_SIZE = 256

# Angle FFT size. Larger than 8 for smoother angle image visualization.
ANGLE_FFT_SIZE = 64


# ============================================================
# 5. Radar image settings
# ============================================================

# Final radar image size will usually be:
# range_bins × angle_bins
IMAGE_HEIGHT = RANGE_FFT_SIZE
IMAGE_WIDTH = ANGLE_FFT_SIZE

# Use dB scale for saving radar images
USE_DB_SCALE = True

# Small value to avoid log(0)
EPS = 1e-8

# Normalize each radar image before saving/training
NORMALIZE_IMAGES = True


# ============================================================
# 6. Sparse MIMO settings
# ============================================================

# Ratios of virtual channels to keep
# 1.0 means full 8 channels
# 0.75 means 6 channels
# 0.50 means 4 channels
# 0.25 means 2 channels
SPARSE_RATIOS = [1.0, 0.75, 0.5, 0.25]

# For the first version, use fixed masks.
# Later, we can add random masks.
MASK_MODE = "fixed"   # options: "fixed", "random"

# Fixed sparse masks for 8 virtual channels
# 1 = channel used, 0 = channel removed
FIXED_MASKS = {
    1.0:  [1, 1, 1, 1, 1, 1, 1, 1],
    0.75: [1, 1, 1, 0, 1, 1, 1, 0],
    0.50: [1, 0, 1, 0, 1, 0, 1, 0],
    0.25: [1, 0, 0, 0, 1, 0, 0, 0],
}


# ============================================================
# 7. Train / validation / test split
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# 8. Training settings
# ============================================================

BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

NUM_WORKERS = 4

DEVICE = "cuda"  # change to "cpu" if needed


# ============================================================
# 9. Model settings
# ============================================================

# Input channels:
# sparse radar image = 1 channel
# mask can be added as extra channels later if needed
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1

BASE_CHANNELS = 32


# ============================================================
# 10. Evaluation settings
# ============================================================

SAVE_EXAMPLE_FIGURES = True
NUM_EXAMPLE_FIGURES = 20

METRICS = [
    "nmse",
    "psnr",
    "ssim",
]


# ============================================================
# 11. Utility function
# ============================================================

def create_dirs():
    """Create required project directories."""
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_dirs()
    print("Configuration loaded successfully.")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"PROCESSED_ROOT: {PROCESSED_ROOT}")
    print(f"Virtual MIMO channels: {NUM_VIRTUAL_CHANNELS}")