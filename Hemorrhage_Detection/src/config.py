"""Central configuration for hemorrhage detection."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
DATA_ROOT = PROJECT_ROOT / "Dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "best_model.pt"
V2_CHECKPOINT = CHECKPOINT_DIR / "best_model_v2.pt"

# Classes (order = label index)
CLASS_NAMES = ["normal", "hemorrhage"]
NUM_CLASSES = len(CLASS_NAMES)

# Decision threshold on P(hemorrhage). Default 0.5 = argmax.
# Lower favors recall (catch more bleeds) at cost of precision — matters for medical triage.
DECISION_THRESHOLD = 0.5

# Model
MODEL_NAME = "efficientnet_b0"
IMAGE_SIZE = 224

# Training
BATCH_SIZE = 32
NUM_WORKERS = 0  # 0 = main process only; avoids worker spawn issues on Windows with user-installed packages
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 7
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
CHECKPOINT_METRIC = "f2"  # one of "f2" | "auc"

# Splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ImageNet normalization (EfficientNet expects this)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Reproducibility
SEED = 42
