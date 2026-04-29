# IVP Group22

## Introduction to Image and Video Processing – Challenge 2026

This project contains our implementation for the KEN3238 Image and Video Processing Challenge 2026.

The goal of the challenge is to classify digit images using image processing and machine learning techniques.

---

# Setup

Install all required dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python run.py
```

This will:
- load the training data
- train the model
- generate predictions
- create a submission file

---

# Project Structure

```text
IVP_group22/
│
├── data/                           # Dataset folder (provided by teachers)
│   ├── test/test/                  # Test images used for prediction
│   │
│   └── train/train/                # Training dataset
│       ├── 0/                      # Images of digit 0
│       ├── 1/                      # Images of digit 1
│       ├── 2/                      # Images of digit 2
│       ├── 3/                      # Images of digit 3
│       ├── 4/                      # Images of digit 4
│       ├── 5/                      # Images of digit 5
│       ├── 6/                      # Images of digit 6
│       ├── 7/                      # Images of digit 7
│       ├── 8/                      # Images of digit 8
│       ├── 9/                      # Images of digit 9
│       │
│       ├── train.csv               # Training labels and image IDs
│       ├── test.csv                # Test image IDs
│       └── sample_submission.csv   # Example Kaggle submission format
│
├── src/                            # Main source code
│   ├── features.py                 # Image preprocessing and feature extraction
│   ├── model.py                    # Machine learning model definition
│   ├── train.py                    # Training pipeline
│   ├── predict.py                  # Prediction and submission generation
│   └── main.py                     # Main execution pipeline
│
├── notebooks/                      # Jupyter notebooks for experiments
│   ├── exploration.ipynb           # Data visualization and exploration
│   └── training.ipynb              # Model experimentation and testing
│
├── outputs/                        # Generated outputs
│   ├── submissions/                # Kaggle submission CSV files
│   ├── models/                     # Saved trained models
│   └── logs/                       # Training logs and experiment notes
│
├── requirements.txt                # Python dependencies
├── .gitignore                      # Ignored files for Git
├── README.md                       # Project documentation
└── run.py                          # Runs the complete pipeline
```

---

# Workflow

1. Load and preprocess images
2. Extract image features
3. Train the classifier
4. Predict test labels
5. Generate Kaggle submission file

---

# Technologies Used

- Python
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

# Output

Generated submission files are saved in:

```text
outputs/submissions/
```

Saved trained models are stored in:

```text
outputs/models/
```