import os
import joblib
import pandas as pd
import numpy as np

from src.features import preprocess_image, extract_features
from src.model import create_model

TRAIN_PATH = "data/train/train"
CSV_PATH = "data/train.csv"

MODEL_PATH = "outputs/models/svm_model.pkl"


def load_training_data():

    train_df = pd.read_csv(CSV_PATH)

    X = []
    y = []

    for _, row in train_df.iterrows():

        img_id = str(row["Id"])
        label = int(row["Category"])

        img_path = os.path.join(TRAIN_PATH, str(label), f"{img_id}.png")

        img = preprocess_image(img_path)

        features = extract_features(img)

        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


def train_model():

    print("Loading training data...")

    X, y = load_training_data()

    print("Training model...")

    model = create_model()

    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)

    print("Model saved to:", MODEL_PATH)

    return model