import os
import joblib
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.features import preprocess_image, extract_features, extract_hog_features, extract_combined_features
from src.model import create_cnn, create_cnn_v2, create_model

TRAIN_PATH = "data/train/train"
CSV_PATH = "data/train.csv"

TRAIN_VAL_SPLIT = 0.3  #30% validation - 70% training
RANDOM_STATE = 42


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

    X = np.array(X)
    y = np.array(y)

    return X, y

def train_cnn():
    """
    Train CNN model
    """

    print("Training cnn.............................")
  
    
  
    X, y = load_training_data()


    X_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # X_val, x_test, y_val, y_test = train_test_split(
    #     x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    # )

    

    model = create_cnn(num_classes=10)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    

    #model.summary() #prints model architecture
    

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    print("\nTraining CNN...")
    history = model.fit(
        X_train, y_train,
        validation_data=(x_val, y_val),
        epochs=30,
        batch_size=256,
        callbacks=[early_stop],
        verbose=1
    )
    


    y_pred = model.predict(x_val).argmax(axis=1)
    test_accuracy = accuracy_score(y_val, y_pred)
    
    print(f" VALIDATION ACCURACY: {test_accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    
    # Save model
    os.makedirs("outputs/models", exist_ok=True)
    model.save("outputs/models/cnn.keras")
    print(f"\nNN model saved to: outputs/models/cnn.keras")
    
    return model, history



def train_svm():
    print("training SVM.......")

    X, y = load_training_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    model = create_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    print(f"SVM VALIDATION ACCURACY: {accuracy_score(y_val, y_pred)*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    os.makedirs("outputs/models", exist_ok=True)
    joblib.dump(model, "outputs/models/svm.joblib")
    print("SVM saved to: outputs/models/svm.joblib")

    return model


def load_training_data_hog():
    #Load training data with HOG features instead of raw pixels.
    train_df = pd.read_csv(CSV_PATH)
    X, y = [], []
    for _, row in train_df.iterrows():
        img_path = os.path.join(TRAIN_PATH, str(int(row["Category"])), f"{str(row['Id'])}.png")
        img = preprocess_image(img_path)
        X.append(extract_hog_features(img))
        y.append(int(row["Category"]))
    return np.array(X), np.array(y)


def train_svm_hog():
    #Train svm on HOG features.
    print("Training HOG-SVM...")
    X, y = load_training_data_hog()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    model = create_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"HOG-SVM VALIDATION ACCURACY: {accuracy_score(y_val, y_pred)*100:.2f}%")
    print(classification_report(y_val, y_pred))
    os.makedirs("outputs/models", exist_ok=True)
    joblib.dump(model, "outputs/models/svm_hog.joblib")
    print("HOG-SVM saved to: outputs/models/svm_hog.joblib")
    return model


def train_cnn_v2():
    #Train improved cnn with batchnormalization.
    print("Training CNN v2 (BatchNorm)...")
    X, y = load_training_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = create_cnn_v2(num_classes=10)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=7, restore_best_weights=True
    )
    lr_schedule = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=256,
        callbacks=[early_stop, lr_schedule],
        verbose=1,
    )
    y_pred = model.predict(X_val).argmax(axis=1)
    print(f"CNN v2 VALIDATION ACCURACY: {accuracy_score(y_val, y_pred)*100:.2f}%")
    print(classification_report(y_val, y_pred))
    model.save("outputs/models/cnn_v2.keras")
    print("CNN v2 saved to: outputs/models/cnn_v2.keras")
    return model, history


if __name__ == "__main__":
    train_cnn()