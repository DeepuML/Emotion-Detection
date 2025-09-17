# ==========================================================
# Model Evaluation Script
# This script evaluates a trained machine learning model
# using the processed test dataset and saves evaluation
# metrics (accuracy, precision, recall, AUC) in JSON format.
# ==========================================================

# ------------------------------
# Import Required Libraries
# ------------------------------

import numpy as np                                # For handling numerical arrays
import pandas as pd                               # For reading CSV files and handling tabular data
import pickle                                     # For loading the saved trained model
import json                                       # For saving evaluation metrics in JSON format
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score  # Evaluation metrics
import logging                                    # For structured logging of debug and error information

# ------------------------------
# Logging Configuration
# ------------------------------

# Create a logger for this script
logger = logging.getLogger('model_evaluation')   
logger.setLevel('DEBUG')  # Capture all messages at DEBUG level or higher

# Console handler → outputs logs to terminal
console_handler = logging.StreamHandler()       
console_handler.setLevel('DEBUG')  # Show all debug messages in console

# File handler → outputs only ERROR logs into a file
file_handler = logging.FileHandler('model_evaluation_errors.log')  
file_handler.setLevel('ERROR')      # Only log errors to file

# Define log message format: timestamp - logger name - level - message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ------------------------------
# Utility Functions
# ------------------------------

def load_model(file_path: str):
    """
    Load the trained machine learning model from a pickle file.
    
    Args:
        file_path (str): Path to the saved model file.
    
    Returns:
        model: Loaded model object.
    """
    try:
        with open(file_path, 'rb') as file:   # Open file in binary read mode
            model = pickle.load(file)         # Deserialize the model
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:                 # Specific error if file is missing
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:                    # Catch-all for unexpected errors
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load test dataset from a CSV file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded test dataset.
    """
    try:
        df = pd.read_csv(file_path)           # Read CSV file
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:        # Error if CSV parsing fails
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:                    # Catch-all for unexpected errors
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the trained model on the test dataset.
    
    Args:
        clf: Trained classifier model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): True labels for the test set.
    
    Returns:
        dict: Dictionary containing evaluation metrics: accuracy, precision, recall, AUC.
    """
    try:
        # Predict labels for test set
        y_pred = clf.predict(X_test)
        
        # Predict probabilities for positive class (used for AUC)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Compute evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)       # Fraction of correctly classified samples
        precision = precision_score(y_test, y_pred)    # True positives / predicted positives
        recall = recall_score(y_test, y_pred)          # True positives / actual positives
        auc = roc_auc_score(y_test, y_pred_proba)      # Area Under ROC Curve

        # Store metrics in a dictionary
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:                             # Catch errors during evaluation
        logger.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """
    Save evaluation metrics to a JSON file.
    
    Args:
        metrics (dict): Dictionary of evaluation metrics.
        file_path (str): Path to save JSON file.
    """
    try:
        # Ensure the directory exists
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write metrics to file with pretty formatting
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

# ------------------------------
# Main Function
# ------------------------------

def main():
    """
    Main execution flow:
    1. Load trained model from disk.
    2. Load test dataset.
    3. Separate features and labels.
    4. Evaluate model performance.
    5. Save metrics to JSON file.
    """
    try:
        # Load the trained model
        clf = load_model('./models/model.pkl')
        
        # Load processed test data
        test_data = load_data('./data/processed/test_tfidf.csv')
        
        # Separate features (all columns except last) and labels (last column)
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        # Evaluate the model
        metrics = evaluate_model(clf, X_test, y_test)
        
        # Save evaluation metrics
        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

# ------------------------------
# Script Execution Entry Point
# ------------------------------

if __name__ == '__main__':
    main()
