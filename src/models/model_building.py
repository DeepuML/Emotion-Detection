# ==========================================================
# Model Building Script
# This script trains a Gradient Boosting Classifier model
# using TF-IDF features from the processed dataset and saves
# the trained model as a pickle file for later use.
# ==========================================================

# Import Required Libraries
import numpy as np                                # For handling numerical arrays
import pandas as pd                               # For data loading and manipulation
import pickle                                     # For saving the trained model to disk
from sklearn.ensemble import GradientBoostingClassifier  # ML model for classification
import yaml                                       # For loading hyperparameters from params.yaml
import logging                                    # For logging debug info and errors

# ----------------------------------------------------------
# Logging Configuration
# ----------------------------------------------------------

# Create a logger for this script
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')  # Capture all logs at DEBUG level and above

# Console handler → show logs on terminal
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File handler → save only ERROR logs into a log file
file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

# Define log format: timestamp - module - level - message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Apply formatter to handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ----------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------

def load_params(params_path: str) -> dict:
    """
    Load parameters from a YAML file (params.yaml).
    Example:
    model_building:
        n_estimators: 100
        learning_rate: 0.1
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)  # Read YAML as Python dict
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file into a pandas DataFrame.
    Used for loading TF-IDF features (train/test data).
    """
    try:
        df = pd.read_csv(file_path)   # Read CSV file into DataFrame
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> GradientBoostingClassifier:
    """
    Train a Gradient Boosting Classifier.
    Args:
        X_train: Training feature matrix (TF-IDF vectors).
        y_train: Training labels.
        params: Dictionary of hyperparameters from params.yaml.
    Returns:
        Trained GradientBoostingClassifier object.
    """
    try:
        # Initialize model with hyperparameters
        clf = GradientBoostingClassifier(
            n_estimators=params['n_estimators'],       # Number of boosting stages
            learning_rate=params['learning_rate']      # Learning rate for shrinkage
        )
        
        # Train the model on training data
        clf.fit(X_train, y_train)
        logger.debug('Model training completed successfully')
        return clf
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise


import os

def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file using pickle.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create folder if it doesn't exist
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise



# ----------------------------------------------------------
# Main Function
# ----------------------------------------------------------

def main():
    """
    Main pipeline for model building:
    1. Load model hyperparameters from params.yaml.
    2. Load processed training dataset (TF-IDF features).
    3. Separate features (X) and labels (y).
    4. Train Gradient Boosting Classifier.
    5. Save trained model into 'models/model.pkl'.
    """
    try:
        # Load model parameters from params.yaml
        params = load_params('params.yaml')['model_building']

        # Load processed training data
        train_data = load_data('./data/processed/train_tfidf.csv')

        # Separate features (all columns except last) and labels (last column)
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        # Train the Gradient Boosting model
        clf = train_model(X_train, y_train, params)

        # Save the trained model in 'models' directory
        save_model(clf, 'models/model.pkl')
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


# ----------------------------------------------------------
# Script Execution Entry Point
# ----------------------------------------------------------
if __name__ == '__main__':
    main()
