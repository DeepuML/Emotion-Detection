# Data Preprocessing Script

# This script performs text preprocessing (cleaning and normalization) 
# on raw training and testing datasets for NLP tasks. 
# It includes steps such as lowercasing, removing stopwords, 
# lemmatization, removing punctuation, numbers, and URLs, 
# and finally saving the cleaned dataset for further use.


# Import Required Libraries

import numpy as np                     # For handling numerical operations and NaN values
import pandas as pd                    # For data manipulation (loading, cleaning, saving CSV files)
import os                              # For handling file paths and directories
import re                              # For regex-based operations (removing URLs, punctuation, etc.)
import nltk                            # Natural Language Toolkit (used for stopwords, lemmatization)
import string                          # Provides a list of punctuation characters
from nltk.corpus import stopwords       # Import stopwords list
from nltk.stem import WordNetLemmatizer # Lemmatizer to reduce words to their base form
import logging                         # For logging progress and errors


# Logging Configuration

# Create a logger object with name "data_transformation"
logger = logging.getLogger('data_transformation')
logger.setLevel('DEBUG')  # Set logging level to DEBUG (captures all levels of logs)

# Create console handler → shows logs on terminal
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')  # Console will show all DEBUG logs and above

# Create file handler → logs errors to a file named "transformation_errors.log"
file_handler = logging.FileHandler('transformation_errors.log')
file_handler.setLevel('ERROR')  # Only ERROR-level logs will be saved in the file

# Define a common log message format (timestamp, logger name, log level, message)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Apply the formatter to both handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Download Required NLTK Data
# Download necessary NLTK resources if not already present
# 'wordnet' → Required for lemmatization
# 'stopwords' → Required for stopword removal
nltk.download('wordnet')
nltk.download('stopwords')


# Text Preprocessing Functions


def lemmatization(text):
    """Reduce words to their base form (lemmatization)."""
    lemmatizer = WordNetLemmatizer()       # Initialize lemmatizer
    text = text.split()                    # Tokenize sentence into words
    text = [lemmatizer.lemmatize(word) for word in text]  # Lemmatize each word
    return " ".join(text)                  # Join back into a string

def remove_stop_words(text):
    """Remove common stopwords like 'the', 'is', 'in'."""
    stop_words = set(stopwords.words("english"))   # Load English stopwords
    text = [word for word in str(text).split() if word not in stop_words]  # Remove stopwords
    return " ".join(text)

def removing_numbers(text):
    """Remove all numeric characters from the text."""
    text = ''.join([char for char in text if not char.isdigit()])  # Exclude digits
    return text

def lower_case(text):
    """Convert all text to lowercase."""
    text = text.split()                    # Tokenize text
    text = [word.lower() for word in text] # Convert each word to lowercase
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuation marks from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)  # Replace punctuation with space
    text = text.replace('؛', "")            # Remove special punctuation (Arabic semicolon)
    text = re.sub('\s+', ' ', text).strip() # Remove extra spaces
    return text

def removing_urls(text):
    """Remove URLs from the text (http, https, www)."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')  # Regex for URLs
    return url_pattern.sub(r'', text)                   # Replace with empty string

def remove_small_sentences(df):
    """Remove rows from dataframe where 'text' column has less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:   # Check if sentence has fewer than 3 words
            df.text.iloc[i] = np.nan           # Replace with NaN (can be dropped later)


# Text Normalization Pipeline

def normalize_text(df):
    """Perform full text normalization on the 'content' column of the dataframe."""
    try:
        # Step 1: Convert all text to lowercase
        df['content'] = df['content'].apply(lower_case)
        logger.debug('converted to lower case')

        # Step 2: Remove stopwords
        df['content'] = df['content'].apply(remove_stop_words)
        logger.debug('stop words removed')

        # Step 3: Remove numbers
        df['content'] = df['content'].apply(removing_numbers)
        logger.debug('numbers removed')

        # Step 4: Remove punctuation
        df['content'] = df['content'].apply(removing_punctuations)
        logger.debug('punctuations removed')

        # Step 5: Remove URLs
        df['content'] = df['content'].apply(removing_urls)
        logger.debug('urls removed')

        # Step 6: Lemmatization
        df['content'] = df['content'].apply(lemmatization)
        logger.debug('lemmatization performed')

        # Final Log
        logger.debug('Text normalization completed')
        return df

    except Exception as e:
        # Log and raise error if any step fails
        logger.error('Error during text normalization: %s', e)
        raise


# Main Function

def main():
    try:
        # Load training and testing datasets from "data/raw" directory
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        # Apply normalization pipeline on both datasets
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save processed data inside "data/interim" directory
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.debug('Processed data saved to %s', data_path)


    except Exception as e:
        # If data transformation fails, log the error and print it
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")


# Script Execution Entry Point

if __name__ == '__main__':
    # Run main function only if script is executed directly
    main()
