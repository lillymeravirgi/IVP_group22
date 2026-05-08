from concurrent.futures import ThreadPoolExecutor

from src.predict import generate_predictions
from src.train import train_cnn_v2, train_svm_hog


def main():
    """Train both models, then generate ensemble predictions."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_cnn = executor.submit(train_cnn_v2)
        future_svm = executor.submit(train_svm_hog)

        future_cnn.result()
        future_svm.result()

    generate_predictions()


if __name__ == "__main__":
    main()
