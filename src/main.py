from src.train import train_cnn
from src.predict import generate_predictions


def main():
    """
    Main pipeline:
    1. Train cnn
    2. Generate predictions on test set
    """
    #model, history = train_model()

    model, history = train_cnn()
    


    results = generate_predictions()


if __name__ == "__main__":
    main()
