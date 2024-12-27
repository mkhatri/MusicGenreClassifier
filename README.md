# Music Genre Classification Project

This project aims to build a machine learning model that can classify music tracks into different genres based on their audio features.  It uses a dataset combining track metadata and audio features extracted using the Echo Nest API. The [Free Music Archive (FMA)](add fma website link here if you have access to the internet) dataset is also integrated for richer genre information.



## Project Structure
music-genre-classification/  
├── data/ │ 
    ├── raw/ <-- Raw data files (CSV, JSON) 
    │ └── processed/ <-- Processed and split data (CSV) 
├── src/ <-- Source code for data processing, training, evaluation │ 
    ├── data_processing.py 
    ├── model_training.py 
    └── model_evaluation.py 
├── notebooks/ <-- Jupyter notebooks for exploration and experimentation │ 
    ├── data_exploration.ipynb 
    └── model_experimentation.ipynb 
├── models/ <-- Saved trained models 
├── requirements.txt <-- Project dependencies 
├── main.py <-- Main script to run the project 
└── README.md <-- This file

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   ```

2. **Create and activate a virtual environment:**
   ```bash
    python -m venv .venv
    .venv\Scripts\activate  #(Windows)
    source .venv/bin/activate (macOS/Linux)
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place raw data:** Put `fma-rock-vs-hiphop.csv` and `echonest-metrics.json` in the `data/raw` directory.

5. **Run the main script:**
   ```bash
   python main.py
   ```

## Data

The project uses two main data sources:

*   **`fma-rock-vs-hiphop.csv`:** Contains track metadata and genre labels.
*   **`echonest-metrics.json`:** Contains pre-computed audio features from the Echo Nest API.

These are combined and preprocessed using the `data_processing.py` script.  The processed data is split into training and testing sets and saved in the `data/processed` directory.

## Model Training and Evaluation

The `model_training.py` script trains a machine learning model (currently a Random Forest classifier, but you can experiment with others in `model_experimentation.ipynb`) to classify music genres. It uses a pipeline that includes preprocessing steps (e.g., scaling, one-hot encoding) and hyperparameter tuning using cross-validation.

The trained model is saved to the `models` directory.  The `model_evaluation.py` script evaluates the trained model on the test set and generates performance metrics (classification report, confusion matrix).

## Experimentation and Exploration

The `notebooks` directory contains Jupyter notebooks for:

*   **`data_exploration.ipynb`:**  Exploratory Data Analysis (EDA) to understand the data and identify potential features.
*   **`model_experimentation.ipynb`:** Trying out different models, preprocessing techniques, and hyperparameter settings to find the best performing model.

## Dependencies

The project dependencies are listed in `requirements.txt`.  It is recommended to use a virtual environment to manage the project's dependencies.

## Future Work

*   Explore more advanced feature engineering techniques.
*   Experiment with other classification models (e.g., SVM, Gradient Boosting).
*   Implement a cloud-based deployment strategy for serving predictions.
*   Potentially incorporate more data from the Free Music Archive (FMA) dataset.

