import pandas as pd
import os

def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")

def fill_missing_with_median(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
    return df

def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}: {e}")

def main():
    try:
        # Update paths to match your actual file structure
        raw_data_path = "./src/data/raw/"
        processed_data_path = "./src/data/processed"

        # Load raw train and test data
        train_data = load_data(os.path.join(raw_data_path, "train.csv"))
        test_data = load_data(os.path.join(raw_data_path, "test.csv"))

        # Fill missing values with median
        train_processed_data = fill_missing_with_median(train_data)
        test_processed_data = fill_missing_with_median(test_data)

        # Ensure processed data directory exists
        os.makedirs(processed_data_path, exist_ok=True)

        # Save the processed data
        save_data(train_processed_data, os.path.join(processed_data_path, "train_processed.csv"))
        save_data(test_processed_data, os.path.join(processed_data_path, "test_processed.csv"))

    except Exception as e:
        raise Exception(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
