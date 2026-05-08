from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow import keras
from tensorflow.keras import layers


def create_model():
    return SVC(
        kernel="rbf",
        gamma="scale",
        C=5,
        probability=True,
    )


def create_cnn_v2(num_classes=10):
    """CNN classifier with augmentation and batch normalization."""
    return keras.Sequential([
        layers.Reshape((28, 28, 1), input_shape=(784,)),

        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),

        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.2),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])


def create_cnn(num_classes=10):
    """Baseline CNN classifier for digit recognition."""
    return keras.Sequential([
        layers.Reshape((28, 28, 1), input_shape=(784,)),

        # Augmentation layers — only active during training
        layers.RandomRotation(0.05),          # ±18 degrees
        layers.RandomTranslation(0.1, 0.1),   # ±10% shift
        layers.RandomZoom(0.1),               # ±10% zoom

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Dropout(0.2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])


def create_classifier():
    """Create a RandomForestClassifier that works on encoded features."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=15,  # Limit tree depth
        min_samples_split=10,  # Require more samples to split
        min_samples_leaf=5,  # Require more samples at leaf
        max_features='sqrt',  # Reduce feature space
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
