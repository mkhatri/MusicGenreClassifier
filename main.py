import pandas as pd
from pathlib import Path
import json
import os
from sklearn.model_selection import train_test_split
from src.model_evaluation import evaluate_model, load_model, load_data

# Local Data Directory (adjust as needed)
DATA_DIR = "data"  # Relative path to your data directory
RAW_DATA_DIR = os.path.join(Path(__file__).parent.parent, os.path.join(DATA_DIR, "raw"))
PROCESSED_DATA_DIR = os.path.join(Path(__file__).parent.parent, os.path.join(DATA_DIR, "processed"))
MODEL_DIR = os.path.join(Path(__file__).parent.parent, "models")


if __name__ == "__main__":
    model_filepath = os.path.join(MODEL_DIR, "music_genre_classifier.pkl")


    # 1. Load Model
    try:  # Check if the model file exists
        loaded_model = load_model(model_filepath)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_filepath}. Please train and save a model first.")
        exit() # Or handle the error appropriately

    # Load the test data
    X_test = load_data(os.path.join(PROCESSED_DATA_DIR, "X_test.csv"))
    y_test = load_data(os.path.join(PROCESSED_DATA_DIR, "y_test.csv"))

    # 2. Evaluate Model
    evaluate_model(loaded_model, X_test, y_test)