from sklearn.svm import SVC
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBClassifier
import numpy as np



def create_model():
    model = SVC(
        kernel="rbf",
        gamma="scale",
        C=5
    )

    return model

def create_cnn(num_classes=10):
    """
    CNN classifier for digit recognition
    """
    model = keras.Sequential([
        layers.Reshape((28, 28, 1), input_shape=(784,)),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Dropout(0.2),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model



def create_classifier():
    """
    Create a XGBClassifier that works on encoded features
    """
    classifier = XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
  
    return classifier



    