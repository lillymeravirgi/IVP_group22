# IVP Group 22

## Image and Video Processing Challenge 2026

This project trains digit classifiers for the KEN3238 Image and Video Processing Challenge 2026

The current pipeline trains two complementary models:

- a CNN with data augmentation and batch normalization
- an SVM trained on HOG features

Predictions are generated with a weighted ensemble of both models.

## Setup

Create an environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Place the challenge data in this structure:

```text
data/
  train.csv
  test.csv
  sample_submission.csv
  train/train/0/*.png
  train/train/1/*.png
  ...
  train/train/9/*.png
  test/test/*.png
```

## Run

Run the full training and prediction pipeline:

```bash
python run.py
```

This will:

- load and preprocess the training images
- train the CNN and HOG-SVM models
- save trained models in `outputs/models/`
- generate `outputs/predictions.csv`

## Project Structure

```text
IVP_group22/
  data/                  # Local dataset, ignored by Git
  notebooks/             # Exploration and training notebooks
  outputs/               # Generated models, logs, and predictions
  src/
    config.py            # Shared paths and pipeline settings
    features.py          # Image preprocessing and feature extraction
    model.py             # Model definitions
    train.py             # Training utilities
    predict.py           # Prediction and ensembling
    main.py              # End-to-end pipeline
  run.py                 # Pipeline entry point
```

## Notes

- The data and generated outputs are ignored by Git.
- Paths are resolved from the project root, so `python run.py` works from any current working directory as long as the project files are intact.
