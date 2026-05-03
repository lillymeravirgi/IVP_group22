import os
import joblib
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.features import preprocess_image, extract_features
from src.model import create_cnn

TRAIN_PATH = "data/train/train"
CSV_PATH = "data/train.csv"

TRAIN_VAL_SPLIT = 0.2  # 20% validation- 80% training
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



if __name__ == "__main__":
    train_cnn()