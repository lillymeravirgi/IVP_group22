from sklearn.svm import SVC


def create_model():
    model = SVC(
        kernel="rbf",
        gamma="scale",
        C=5
    )

    return model