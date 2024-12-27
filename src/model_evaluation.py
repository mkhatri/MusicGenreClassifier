import pandas as pd
import os
from pathlib import Path
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Local Data and Model Directories
DATA_DIR = "data"
PROCESSED_DATA_DIR = os.path.join(Path(__file__).parent.parent, os.path.join(DATA_DIR, "processed"))
MODEL_DIR = os.path.join(Path(__file__).parent.parent, "models")



def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)


def load_model(file_path):
    """Loads a trained model from a file."""
    return joblib.load(file_path)



def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model and generates a classification report and confusion matrix."""

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))


    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()



if __name__ == "__main__":

    X_test = load_data(os.path.join(PROCESSED_DATA_DIR, "X_test.csv"))
    y_test = load_data(os.path.join(PROCESSED_DATA_DIR, "y_test.csv"))

    # Load the trained model
    model = load_model(os.path.join(MODEL_DIR, "music_genre_classifier.pkl"))

    # Evaluate the model
    evaluate_model(model, X_test, y_test.values.ravel())

