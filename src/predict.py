import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from src.config import (
    CNN_V2_MODEL_PATH,
    CNN_WEIGHT,
    HOG_WEIGHT,
    PREDICTIONS_PATH,
    RANDOM_STATE,
    SVM_HOG_MODEL_PATH,
    TEST_CSV_PATH,
    TEST_IMAGE_DIR,
    TRAIN_VAL_SPLIT,
    ensure_output_dirs,
)
from src.features import extract_features, extract_hog_features, preprocess_image


def load_test_data():
    test_df = pd.read_csv(TEST_CSV_PATH)
    X_pix, X_hog, ids = [], [], []

    for _, row in test_df.iterrows():
        img_id = str(row["Id"])
        img_path = TEST_IMAGE_DIR / f"{img_id}.png"
        try:
            img = preprocess_image(img_path)
            X_pix.append(extract_features(img))
            X_hog.append(extract_hog_features(img))
            ids.append(img_id)
        except Exception as exc:
            print(f"Error loading image {img_id}: {exc}")

    return np.array(X_pix), np.array(X_hog), ids


def generate_predictions():
    ensure_output_dirs()
    cnn = keras.models.load_model(CNN_V2_MODEL_PATH)
    svm_hog = joblib.load(SVM_HOG_MODEL_PATH)
    print(f"Loaded CNN v2 from {CNN_V2_MODEL_PATH}")
    print(f"Loaded HOG-SVM from {SVM_HOG_MODEL_PATH}")

    X_pix, X_hog, test_ids = load_test_data()
    print(f"Loaded {len(X_pix)} test images")

    print("\nGenerating predictions...")
    cnn_probs = cnn.predict(X_pix, verbose=1)
    hog_probs = svm_hog.predict_proba(X_hog)

    ensemble_probs = CNN_WEIGHT * cnn_probs + HOG_WEIGHT * hog_probs
    y_pred = ensemble_probs.argmax(axis=1)

    results_df = pd.DataFrame({"Id": test_ids, "Category": y_pred})
    results_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\nPredictions saved to: {PREDICTIONS_PATH}")
    return results_df


def grid_search_weights():
    """Grid search over CNN/SVM ensemble weights on the validation set."""
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    from src.train import load_training_data, load_training_data_hog

    X_pix, y = load_training_data()
    X_hog, _ = load_training_data_hog()
    _, X_pix_val, _, y_val = train_test_split(
        X_pix,
        y,
        test_size=TRAIN_VAL_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    _, X_hog_val, _, _ = train_test_split(
        X_hog,
        y,
        test_size=TRAIN_VAL_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cnn = keras.models.load_model(CNN_V2_MODEL_PATH)
    svm_hog = joblib.load(SVM_HOG_MODEL_PATH)

    p_cnn = cnn.predict(X_pix_val, verbose=0)
    p_hog = svm_hog.predict_proba(X_hog_val)

    print(f"{'CNN weight':>12} {'HOG weight':>12} {'Val accuracy':>14}")
    print("-" * 42)
    best_acc, best_w = 0, 0
    for w in [0.70, 0.80, 0.85, 0.90, 0.95, 1.00]:
        acc = accuracy_score(y_val, (w * p_cnn + (1 - w) * p_hog).argmax(1))
        print(f"{w:>12.2f} {1 - w:>12.2f} {acc * 100:>13.4f}%")
        if acc > best_acc:
            best_acc, best_w = acc, w

    print(f"\nBest: CNN={best_w:.2f}, HOG={1 - best_w:.2f} -> {best_acc * 100:.4f}%")


if __name__ == "__main__":
    generate_predictions()
