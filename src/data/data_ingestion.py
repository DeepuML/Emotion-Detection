# Data Ingestion Script

# This script performs the data ingestion process for an NLP pipeline.
# The process includes:
#   1. Reading parameters from a YAML configuration file.
#   2. Loading a dataset from a CSV URL.
#   3. Preprocessing the dataset (removing unnecessary columns, filtering, encoding labels).
#   4. Splitting the dataset into training and test sets.
#   5. Saving the processed data to the 'data/raw' directory for downstream steps.

# Import Required Libraries

import numpy as np                       # Used for numerical operations
import pandas as pd                      # Used for handling CSV files and tabular data
import os                                # For creating directories and saving files
from sklearn.model_selection import train_test_split  # For splitting dataset into train/test
import yaml                              # To read parameters from params.yaml file
import logging                           # For logging progress and errors


# Logging Configuration

# Create a logger specifically for this module
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')  # Capture all messages from DEBUG upwards

# Console handler → logs will be shown in the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')  # Console shows detailed debugging logs

# File handler → logs errors into a file named errors.log
file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')  # File will only capture ERROR logs

# Define a formatter for log messages (timestamp, module name, level, message)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Attach formatter to handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Function: Load Params

def load_params(params_path: str) -> dict:
    """
    Load configuration parameters from a YAML file.
    
    Args:
        params_path (str): Path to params.yaml file.
    Returns:
        dict: Dictionary containing all loaded parameters.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)  # Parse YAML content into Python dictionary
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        # Raised if params.yaml file is missing
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        # Raised if YAML formatting is invalid
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        # Catch-all for any unexpected issue
        logger.error('Unexpected error: %s', e)
        raise

# Function: Load Data

def load_data(data_url: str) -> pd.DataFrame:
    """
    Load dataset from a given CSV file or URL.
    
    Args:
        data_url (str): Path or URL to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataset as a DataFrame.
    """
    try:
        df = pd.read_csv(data_url)  # Load dataset into pandas DataFrame
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        # Raised if CSV file is corrupted or has formatting issues
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        # Catch-all for unexpected issues (e.g., internet connection errors)
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


# Function: Preprocess Data

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by:
      - Removing unnecessary columns.
      - Filtering specific sentiments (happiness, sadness).
      - Encoding categorical sentiment labels into numeric values.
    
    Args:
        df (pd.DataFrame): Input raw dataset.
    Returns:
        pd.DataFrame: Preprocessed dataset ready for splitting.
    """
    try:
        # Remove the 'tweet_id' column since it's not useful for training
        df.drop(columns=['tweet_id'], inplace=True)
        
        # Keep only rows where sentiment is either "happiness" or "sadness"
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        
        # Encode labels → happiness=1, sadness=0
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        
        logger.debug('Data preprocessing completed')
        return final_df
    except KeyError as e:
        # Raised if expected columns (tweet_id, sentiment) are missing
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        # Catch-all for any unexpected issue
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


# Function: Save Data

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Save training and testing datasets to CSV files.

    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        data_path (str): Directory path where datasets will be saved.
    """
    try:
        # ✅ Ensure the directory (data_path) exists, create if not
        os.makedirs(data_path, exist_ok=True)

        # ✅ Save train dataset directly inside data_path
        train_file = os.path.join(data_path, "train.csv")
        train_data.to_csv(train_file, index=False)

        # ✅ Save test dataset directly inside data_path
        test_file = os.path.join(data_path, "test.csv")
        test_data.to_csv(test_file, index=False)

        # ✅ Log confirmation message
        logger.debug('Train and test data saved to %s', data_path)

    except Exception as e:
        # If saving fails (e.g., permission denied, disk full), log error
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise



# Main Function

def main():
    """
    Main workflow of the data ingestion process:
      1. Load configuration parameters.
      2. Load dataset from a given URL.
      3. Preprocess dataset.
      4. Split dataset into train and test sets.
      5. Save datasets to 'data/interim'.
    """
    try:
        # Step 1: Load parameters (e.g., test_size from params.yaml)
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        
        # Step 2: Load dataset from given CSV URL
        df = load_data(data_url='https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        
        # Step 3: Preprocess dataset (drop columns, filter sentiments, encode labels)
        final_df = preprocess_data(df)
        
        # Step 4: Split into train and test sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        # Step 5: Save datasets into data/interim folder
        save_data(train_data, test_data, data_path='./data/interim')
    except Exception as e:
        # Log and print error if ingestion process fails
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")
        


# Script Entry Point

if __name__ == '__main__':
    # Run the main() function only if this script is executed directly
    main()
