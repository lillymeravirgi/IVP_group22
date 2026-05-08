from pathlib import Path


RANDOM_STATE = 42
TRAIN_VAL_SPLIT = 0.3
NUM_CLASSES = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV_PATH = DATA_DIR / "train.csv"
TEST_CSV_PATH = DATA_DIR / "test.csv"
TRAIN_IMAGE_DIR = DATA_DIR / "train" / "train"
TEST_IMAGE_DIR = DATA_DIR / "test" / "test"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"

CNN_MODEL_PATH = MODEL_DIR / "cnn.keras"
CNN_V2_MODEL_PATH = MODEL_DIR / "cnn_v2.keras"
SVM_MODEL_PATH = MODEL_DIR / "svm.joblib"
SVM_HOG_MODEL_PATH = MODEL_DIR / "svm_hog.joblib"

CNN_WEIGHT = 0.90
HOG_WEIGHT = 0.10


def ensure_output_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
