import os
import joblib
import pandas as pd
import numpy as np
from tensorflow import keras

from src.features import preprocess_image, extract_features, extract_hog_features

TEST_PATH     = "data/test/test"
TEST_CSV_PATH = "data/test.csv"
OUTPUT_PATH   = "outputs/predictions.csv"

CNN_PATH     = "outputs/models/cnn_v2.keras"
SVM_HOG_PATH = "outputs/models/svm_hog.joblib"

#Optimised on validation set
CNN_WEIGHT = 0.90
HOG_WEIGHT = 0.10


def load_test_data():
    test_df = pd.read_csv(TEST_CSV_PATH)
    X_pix, X_hog, ids = [], [], []
    for _, row in test_df.iterrows():
        img_id   = str(row["Id"])
        img_path = os.path.join(TEST_PATH, f"{img_id}.png")
        try:
            img = preprocess_image(img_path)
            X_pix.append(extract_features(img))
            X_hog.append(extract_hog_features(img))
            ids.append(img_id)
        except Exception as e:
            print(f"Error loading image {img_id}: {e}")
    return np.array(X_pix), np.array(X_hog), ids


def generate_predictions():
    cnn     = keras.models.load_model(CNN_PATH)
    svm_hog = joblib.load(SVM_HOG_PATH)
    print(f"Loaded CNN v2 from {CNN_PATH}")
    print(f"Loaded HOG-SVM from {SVM_HOG_PATH}")

    X_pix, X_hog, test_ids = load_test_data()
    print(f"Loaded {len(X_pix)} test images")

    print("\nGenerating predictions...")
    cnn_probs = cnn.predict(X_pix, verbose=1)
    hog_probs = svm_hog.predict_proba(X_hog)

    ensemble_probs = CNN_WEIGHT * cnn_probs + HOG_WEIGHT * hog_probs
    y_pred = ensemble_probs.argmax(axis=1)

    results_df = pd.DataFrame({'Id': test_ids, 'Category': y_pred})
    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nPredictions saved to: {OUTPUT_PATH}")
    return results_df


if __name__ == "__main__":
    generate_predictions()
