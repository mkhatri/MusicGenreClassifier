import pandas as pd
from pathlib import Path
import json
import os
from sklearn.model_selection import train_test_split


# Local Data Directory (adjust as needed)
DATA_DIR = "data"  # Relative path to your data directory
RAW_DATA_DIR = os.path.join(Path(__file__).parent.parent, os.path.join(DATA_DIR, "raw"))
PROCESSED_DATA_DIR = os.path.join(Path(__file__).parent.parent, os.path.join(DATA_DIR, "processed"))


def load_data(csv_file, json_file):
    """Loads data from local CSV and JSON files."""

    df_csv = pd.read_csv(csv_file)
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    df_json = pd.DataFrame(json_data)

    return df_csv, df_json




def preprocess_data(df_csv, df_json):
    """Combines, cleans, and preprocesses the data."""

    merged_df = pd.merge(df_csv, df_json, on='track_id', how='inner')  # Adjust merge 'on' if needed


    # Example preprocessing (fill missing numeric values with the mean)
    for col in merged_df.select_dtypes(include=['number']):
        merged_df[col] = merged_df[col].fillna(merged_df[col].mean())

    # Add other preprocessing steps as needed (feature scaling, one-hot encoding, etc.)

    return merged_df


def split_data(df, test_size=0.2, random_state=42):
    """Splits data into training and testing sets."""

    X = df.drop('genre_top', axis=1)  # Assuming 'genre_top' is the target
    y = df['genre_top']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test



def save_data(file_path, data):
    """Saves the processed DataFrame to a local CSV file."""
    data.to_csv(file_path, index=False)




if __name__ == "__main__":

    csv_file = os.path.join(RAW_DATA_DIR, "fma-rock-vs-hiphop.csv")
    json_file = os.path.join(RAW_DATA_DIR, "echonest-metrics.json")

    # 1. Load Data
    df_csv, df_json = load_data(csv_file, json_file)


    # 2. Preprocess Data
    processed_df = preprocess_data(df_csv, df_json)

    # 3. Split Data
    X_train, X_test, y_train, y_test = split_data(processed_df)

    # 4. Save Processed and Split Data
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)


    save_data(os.path.join(PROCESSED_DATA_DIR, "combined_data.csv"), processed_df)
    save_data(os.path.join(PROCESSED_DATA_DIR, "X_train.csv"), X_train)
    save_data(os.path.join(PROCESSED_DATA_DIR, "X_test.csv"), X_test)
    save_data(os.path.join(PROCESSED_DATA_DIR, "y_train.csv"), y_train.to_frame()) # y needs to be a DataFrame
    save_data(os.path.join(PROCESSED_DATA_DIR, "y_test.csv"), y_test.to_frame())
