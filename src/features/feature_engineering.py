# Feature Engineering Script
# This script performs feature engineering on preprocessed text data.
# It applies TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical vectors,
# and saves the transformed training and testing datasets for model training and evaluation.

# Import Required Libraries
import numpy as np                                # For numerical operations (optional, used with pandas/scikit-learn outputs)
import pandas as pd                               # For data loading, manipulation, and saving
import os                                         # For directory/file handling
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text data into TF-IDF vectors
import yaml                                       # For loading hyperparameters and configuration from YAML file
import logging                                    # For logging debug info and errors

# Logging Configuration
# Create a logger for this module
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')  # Capture all logs at DEBUG level and above

# Create console handler → display logs on terminal
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Create file handler → log only errors into a file for debugging later
file_handler = logging.FileHandler('feature_engineering_errors.log')
file_handler.setLevel('ERROR')

# Define the format of log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Attach the format to both console and file handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """
    Load parameters from a YAML file (params.yaml).
    Typically contains hyperparameters such as max_features for TF-IDF.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)  # Read YAML file into Python dict
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
    Replaces missing values with empty strings to avoid issues in TF-IDF.
    """
    try:
        df = pd.read_csv(file_path)        # Load CSV file
        df.fillna('', inplace=True)        # Replace NaN with empty strings
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """
    Apply TF-IDF vectorization to convert text into numerical features.
    - max_features: limits the vocabulary size (e.g., top 5000 words).
    - Fits on training data and transforms both train/test sets.
    - Returns transformed train and test DataFrames with labels.
    """
    try:
        # Initialize TF-IDF vectorizer with max_features
        vectorizer = TfidfVectorizer(max_features=max_features)

        # Separate features (content) and labels (sentiment) from train and test data
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        # Fit TF-IDF on training data and transform both train and test sets
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        # Convert sparse TF-IDF matrices into DataFrames
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train   # Add labels column for training data

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test     # Add labels column for test data

        logger.debug('Bag of Words (TF-IDF) applied and data transformed')
        return train_df, test_df
    except Exception as e:
        logger.error('Error during Bag of Words transformation: %s', e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save the given DataFrame into a CSV file.
    Ensures the target directory exists before saving.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create folder if not exists
        df.to_csv(file_path, index=False)                       # Save DataFrame as CSV
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    """
    Main pipeline for feature engineering:
    1. Load parameters (max_features for TF-IDF).
    2. Load preprocessed training and testing datasets.
    3. Apply TF-IDF transformation.
    4. Save the transformed datasets into data/processed.
    """
    try:
        # Load parameters from params.yaml
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']

        # Load preprocessed data from data/interim
        train_data = load_data('./data/interim/train.csv')
        test_data = load_data('./data/interim/test.csv')

        # Apply TF-IDF vectorization
        train_df, test_df = apply_bow(train_data, test_data, max_features)

        # Save transformed datasets into data/processed directory
        save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


# Script Execution Entry Point
# Run main() only if this file is executed directly
if __name__ == '__main__':
    main()
