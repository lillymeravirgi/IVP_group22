import cv2
import numpy as np

IMG_SIZE = 28


def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not load image: {img_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = cv2.GaussianBlur(img, (3, 3), 0)

    img = img.astype(np.float32) / 255.0

    return img


def extract_features(img):
    return img.flatten()