import os
import joblib
import pandas as pd
import numpy as np

from src.features import preprocess_image, extract_features

TEST_PATH = "data/test/test"
CSV_PATH = "data/test.csv"

MODEL_PATH = "outputs/models/svm_model.pkl"

OUTPUT_PATH = "outputs/submissions/submission.csv"


def generate_predictions():

    model = joblib.load(MODEL_PATH)

    test_df = pd.read_csv(CSV_PATH)

    predictions = []

    for _, row in test_df.iterrows():

        img_id = str(row["Id"])

        img_path = os.path.join(TEST_PATH, f"{img_id}.png")

        img = preprocess_image(img_path)

        features = extract_features(img)

        pred = model.predict([features])[0]

        predictions.append(pred)

    submission = pd.DataFrame({
        "Id": test_df["Id"],
        "Category": predictions
    })

    submission.to_csv(OUTPUT_PATH, index=False)

    print("Submission saved to:", OUTPUT_PATH)