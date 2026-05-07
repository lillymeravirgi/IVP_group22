from concurrent.futures import ThreadPoolExecutor
from src.train import train_cnn_v2, train_svm_hog
from src.predict import generate_predictions


def main():
    """
    pipeline:
    1.Train cnn v2 (Batchnorm) and svm with hog in parallel
    2.Generate ensemble predictions: 0.90 * cnn v2 + 0.10 * svm (to get a higher score)
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_cnn = executor.submit(train_cnn_v2)
        future_svm = executor.submit(train_svm_hog)

        model_cnn, history = future_cnn.result()
        model_svm = future_svm.result()

    results = generate_predictions()


if __name__ == "__main__":
    main()