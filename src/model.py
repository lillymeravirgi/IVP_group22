from sklearn.ensemble import RandomForestClassifier
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
    Create a RandomForestClassifier that works on encoded features
    """
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,  # Limit tree depth
        min_samples_split=10,  # Require more samples to split
        min_samples_leaf=5,  # Require more samples at leaf
        max_features='sqrt',  # Reduce feature space
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    return classifier
  
    return classifier



    