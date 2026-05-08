import cv2
import numpy as np
from skimage.feature import hog


IMG_SIZE = 28


def preprocess_image(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not load image: {img_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img.astype(np.float32) / 255.0


def extract_features(img):
    return img.flatten()


def extract_hog_features(img):
    """Extract stroke-orientation features for the SVM model."""
    return hog(
        img,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )


def extract_combined_features(img):
    """Concatenate raw pixels and HOG features."""
    return np.concatenate([img.flatten(), extract_hog_features(img)])
