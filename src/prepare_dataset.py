import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import shutil # Import shutil for directory operations

# --- CONFIGURATION ---
# Define Google Drive base path for raw data inputs and processed data output
# IMPORTANT: Update this if your Google Drive structure is different or you are running locally
GOOGLE_DRIVE_DATA_BASE_PATH = "/content/drive/MyDrive/AgentAI_Data"

# --- CORRECTED RAW DATA INPUT PATHS ---
# These now point to the actual nested directories after unzipping into /tmp
RAW_CLEAN_DIR = "/tmp/raw_clean_extracted/processed/train" # Point to the nested 'train' folder
RAW_VULNERABLE_DIR = "/tmp/raw_vulnerable_extracted/SATE-IV" # Point to the nested 'SATE-IV' folder

# Processed data output path (where train.jsonl, val.jsonl, test.jsonl will be saved)
PROCESSED_DATA_OUTPUT_DIR = os.path.join(GOOGLE_DRIVE_DATA_BASE_PATH, "processed_jsonl")

# Define the split ratio
TEST_SIZE = 0.2  # 20% for testing
VAL_SIZE = 0.1   # 10% for validation (of the remaining 80%)

# --- NEW CONFIGURATION FOR LIMITING DATASET SIZE (FOR BALANCED TRAINING) ---
# This will apply to the *raw* data loading before splitting.
# Set to None to use full available data for that type, or an integer to limit files.
MAX_RAW_CLEAN_FILES = 2000 # Limit raw clean files for faster processing
MAX_RAW_VULNERABLE_FILES = 50 # Limit raw vulnerable files for faster processing

def load_code_files(directory_path, label, max_files=None):
    """
    Loads all .py files from a directory and assigns them a label.
    Can limit the number of files loaded using `max_files`.
    """
    data = []
    count = 0
    if not os.path.exists(directory_path):
        print(f"Warning: Raw data directory not found: {directory_path}")
        return data

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        code = f.read()
                    data.append({"code": code, "label": label})
                    count += 1
                    if max_files is not None and count >= max_files:
                        print(f"  Reached max_files limit ({max_files}) in {directory_path}")
                        break
                except Exception as e:
                    print(f"Could not read file {filepath}: {e}")
        if max_files is not None and count >= max_files:
            break
    return data

def main():
    """
    Main function to process raw data and create train/val/test splits.
    """
    print("--- Starting Dataset Preparation ---")

    # 1. Load Raw Data
    print(f"Loading clean files from {RAW_CLEAN_DIR}...")
    clean_data = load_code_files(RAW_CLEAN_DIR, 0, max_files=MAX_RAW_CLEAN_FILES)
    print(f"Loaded {len(clean_data)} clean files.")

    print(f"Loading vulnerable files from {RAW_VULNERABLE_DIR}...")
    vulnerable_data = load_code_files(RAW_VULNERABLE_DIR, 1, max_files=MAX_RAW_VULNERABLE_FILES)
    print(f"Loaded {len(vulnerable_data)} vulnerable files.")

    if not clean_data and not vulnerable_data:
        print("Error: No raw data found to create a dataset. Exiting.")
        return
    elif not clean_data:
        print("Warning: Only vulnerable data found. Creating a dataset with only vulnerable samples.")
        all_data = vulnerable_data
    elif not vulnerable_data:
        print("Warning: Only clean data found. Creating a dataset with only clean samples.")
        all_data = clean_data
    else:
        all_data = clean_data + vulnerable_data

    df = pd.DataFrame(all_data)
    print(f"Total examples: {len(df)}")
    if not df.empty:
        print(f"Value counts:\n{df['label'].value_counts()}")
    else:
        print("DataFrame is empty after loading data.")
        return

    # 3. Perform Stratified Split (only if both labels are present)
    print("Performing train-test-validation split...")
    if len(df['label'].unique()) > 1:
        # First split: separate out the test set (e.g., 20%)
        train_val_df, test_df = train_test_split(df, test_size=TEST_SIZE, stratify=df['label'], random_state=42)
        
        # Second split: split the remainder into train and validation
        relative_val_size = VAL_SIZE / (1 - TEST_SIZE)
        train_df, val_df = train_test_split(train_val_df, test_size=relative_val_size, stratify=train_val_df['label'], random_state=42)
    else:
        print("Warning: Only one class found, skipping stratified split. Using simple split.")
        train_val_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42)
        relative_val_size = VAL_SIZE / (1 - TEST_SIZE)
        train_df, val_df = train_test_split(train_val_df, test_size=relative_val_size, random_state=42)

    print(f"Train set size: {len(train_df)}\n{train_df['label'].value_counts()}\n")
    print(f"Validation set size: {len(val_df)}\n{val_df['label'].value_counts()}\n")
    print(f"Test set size: {len(test_df)}\n{test_df['label'].value_counts()}\n")

    # 4. Save to JSONL files (in Google Drive)
    print(f"Saving processed files to {PROCESSED_DATA_OUTPUT_DIR}...")
    os.makedirs(PROCESSED_DATA_OUTPUT_DIR, exist_ok=True)
    train_df.to_json(os.path.join(PROCESSED_DATA_OUTPUT_DIR, "train.jsonl"), orient='records', lines=True)
    val_df.to_json(os.path.join(PROCESSED_DATA_OUTPUT_DIR, "val.jsonl"), orient='records', lines=True)
    test_df.to_json(os.path.join(PROCESSED_DATA_OUTPUT_DIR, "test.jsonl"), orient='records', lines=True)

    print("--- Dataset Preparation Complete ---")

if __name__ == "__main__":
    main()
