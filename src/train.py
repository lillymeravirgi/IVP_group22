import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras

from src.config import (
    CNN_MODEL_PATH,
    CNN_V2_MODEL_PATH,
    NUM_CLASSES,
    RANDOM_STATE,
    SVM_HOG_MODEL_PATH,
    SVM_MODEL_PATH,
    TRAIN_CSV_PATH,
    TRAIN_IMAGE_DIR,
    TRAIN_VAL_SPLIT,
    ensure_output_dirs,
)
from src.features import extract_features, extract_hog_features, preprocess_image
from src.model import create_cnn, create_cnn_v2, create_model


def _image_path_for_row(row):
    img_id = str(row["Id"])
    label = int(row["Category"])
    return TRAIN_IMAGE_DIR / str(label) / f"{img_id}.png"


def _load_training_data_with_extractor(extractor):
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    X, y = [], []

    for _, row in train_df.iterrows():
        img = preprocess_image(_image_path_for_row(row))
        X.append(extractor(img))
        y.append(int(row["Category"]))

    return np.array(X), np.array(y)


def load_training_data():
    """Load flattened pixel features and labels from the training set."""
    return _load_training_data_with_extractor(extract_features)


def load_training_data_hog():
    """Load HOG features and labels from the training set."""
    return _load_training_data_with_extractor(extract_hog_features)


def _train_val_split(X, y):
    return train_test_split(
        X,
        y,
        test_size=TRAIN_VAL_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def train_cnn():
    """Train the baseline CNN model."""
    print("Training CNN...")
    ensure_output_dirs()
    X, y = load_training_data()
    X_train, X_val, y_train, y_val = _train_val_split(X, y)

    model = create_cnn(num_classes=NUM_CLASSES)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=256,
        callbacks=[early_stop],
        verbose=1,
    )

    y_pred = model.predict(X_val).argmax(axis=1)
    print(f"VALIDATION ACCURACY: {accuracy_score(y_val, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    model.save(CNN_MODEL_PATH)
    print(f"\nCNN model saved to: {CNN_MODEL_PATH}")
    return model, history


def train_svm():
    """Train an SVM on raw pixel features."""
    print("Training SVM...")
    ensure_output_dirs()
    X, y = load_training_data()
    X_train, X_val, y_train, y_val = _train_val_split(X, y)

    model = create_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    print(f"SVM VALIDATION ACCURACY: {accuracy_score(y_val, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    joblib.dump(model, SVM_MODEL_PATH)
    print(f"SVM saved to: {SVM_MODEL_PATH}")
    return model


def train_svm_hog():
    """Train the SVM on HOG features."""
    print("Training HOG-SVM...")
    ensure_output_dirs()
    X, y = load_training_data_hog()
    X_train, X_val, y_train, y_val = _train_val_split(X, y)

    model = create_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    print(f"HOG-SVM VALIDATION ACCURACY: {accuracy_score(y_val, y_pred) * 100:.2f}%")
    print(classification_report(y_val, y_pred))

    joblib.dump(model, SVM_HOG_MODEL_PATH)
    print(f"HOG-SVM saved to: {SVM_HOG_MODEL_PATH}")
    return model


def train_cnn_v2():
    """Train the improved CNN with batch normalization."""
    print("Training CNN v2 (BatchNorm)...")
    ensure_output_dirs()
    X, y = load_training_data()
    X_train, X_val, y_train, y_val = _train_val_split(X, y)

    model = create_cnn_v2(num_classes=NUM_CLASSES)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        restore_best_weights=True,
    )
    lr_schedule = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=256,
        callbacks=[early_stop, lr_schedule],
        verbose=1,
    )

    y_pred = model.predict(X_val).argmax(axis=1)
    print(f"CNN v2 VALIDATION ACCURACY: {accuracy_score(y_val, y_pred) * 100:.2f}%")
    print(classification_report(y_val, y_pred))

    model.save(CNN_V2_MODEL_PATH)
    print(f"CNN v2 saved to: {CNN_V2_MODEL_PATH}")
    return model, history


if __name__ == "__main__":
    train_cnn()
