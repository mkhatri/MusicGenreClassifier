import pandas as pd
import os
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report  # Or other relevant metrics
import joblib # For saving the model

# Local Data and Model Directories
DATA_DIR = "data"
PROCESSED_DATA_DIR = os.path.join(Path(__file__).parent.parent, os.path.join(DATA_DIR, "processed"))
MODEL_DIR = os.path.join(Path(__file__).parent.parent, "models")


def load_data(file_path):
    return pd.read_csv(file_path)




def create_preprocessor(numerical_features, categorical_features):
    """Creates a preprocessing pipeline for numerical and categorical features."""

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor




def train_and_evaluate(X_train, y_train, X_test, y_test, preprocessor, model, param_grid=None, cv=5):
    """
    Trains a model with an optional hyperparameter search using cross-validation.

    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    if param_grid:  # Perform hyperparameter tuning if param_grid is provided
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1_macro')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

    else:           # Train without hyperparameter tuning
        best_model = pipeline.fit(X_train, y_train)


    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return best_model



def save_model(model, file_path):
    """Saves the trained model to a file."""

    joblib.dump(model, file_path)
    print(f"Model saved to: {file_path}")






if __name__ == "__main__":
    X_train = load_data(os.path.join(PROCESSED_DATA_DIR, "X_train.csv"))
    y_train = load_data(os.path.join(PROCESSED_DATA_DIR, "y_train.csv"))
    X_test = load_data(os.path.join(PROCESSED_DATA_DIR, "X_test.csv"))
    y_test = load_data(os.path.join(PROCESSED_DATA_DIR, "y_test.csv"))



    # Feature lists (You'll need to determine these from your data exploration)
    numerical_features = ['bit_rate', 'comments', 'duration', 'favorites', 'interest', 'listens', 'number',
                       'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness',
                       'tempo', 'valence']
    categorical_features = [] # Add your categorical features, if any

    # Create preprocessor
    preprocessor = create_preprocessor(numerical_features, categorical_features)


    # Choose a model and define a parameter grid (if tuning)
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],  # Access parameters within the pipeline
        'classifier__max_depth': [None, 10, 20]
    }

    # Train and evaluate

    trained_model = train_and_evaluate(X_train, y_train.values.ravel(), X_test, y_test.values.ravel(), preprocessor, model, param_grid=param_grid)



    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    # Save the trained model
    save_model(trained_model, os.path.join(MODEL_DIR, "music_genre_classifier.pkl"))
