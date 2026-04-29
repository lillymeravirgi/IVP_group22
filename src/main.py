from src.train import train_model
from src.predict import generate_predictions


def main():

    train_model()

    generate_predictions()


if __name__ == "__main__":
    main()