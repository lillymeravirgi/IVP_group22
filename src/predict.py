import os
import joblib
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.features import preprocess_image, extract_features


TEST_PATH = "data/test/test"
TEST_CSV_PATH = "data/test.csv"
TRAIN_CSV_PATH = "data/train.csv"
OUTPUT_PATH = "outputs/predictions.csv"

CNN_PATH = "outputs/models/cnn.keras"


def load_test_data():    
    test_df = pd.read_csv(TEST_CSV_PATH)

    X = []
    ids = []

    for _, row in test_df.iterrows():

        img_id = str(row["Id"])
        img_path = os.path.join(TEST_PATH, f"{img_id}.png")

        try:
            img = preprocess_image(img_path)
            features = extract_features(img)
            X.append(features)
            ids.append(img_id)
        except Exception as e:
            print(f"Error loading image {img_id}: {e}")

    return np.array(X), ids




def generate_predictions():
    cnn = keras.models.load_model(CNN_PATH)
    print(f"CNN loaded from: {CNN_PATH}")


    X_test, test_ids = load_test_data()
    print(f"Loaded {len(X_test)} test images")


    print("\nGenerating predictions...")
    predictions = cnn.predict(X_test, verbose=1)
    y_pred = predictions.argmax(axis=1)

  
    results_df = pd.DataFrame({
        'Id': test_ids,
        'Category': y_pred
    })

    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n Predictions saved to: {OUTPUT_PATH}")

    return results_df




if __name__ == "__main__":
    generate_predictions()
    
