from datasets import load_dataset, DatasetDict
import os

# Assuming your raw splits are JSON Lines files in the data/processed directory
DATA_DIR = "data/processed"
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl") # Adjust if your files have different names or extensions
VAL_FILE = os.path.join(DATA_DIR, "val.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")

# Output directories for the new format
OUTPUT_TRAIN_DIR = os.path.join(DATA_DIR, "train")
OUTPUT_VAL_DIR = os.path.join(DATA_DIR, "val")
OUTPUT_TEST_DIR = os.path.join(DATA_DIR, "test")

def convert_and_save_dataset():
    print(f"--- Converting dataset format for HuggingFace Trainer ---")

    # 1. Load each split from your current JSONL files
    print(f"Loading train data from {TRAIN_FILE}...")
    train_dataset = load_dataset('json', data_files=TRAIN_FILE, split='train')
    print(f"Loading validation data from {VAL_FILE}...")
    val_dataset = load_dataset('json', data_files=VAL_FILE, split='train') # 'train' split name is default for single file
    print(f"Loading test data from {TEST_FILE}...")
    test_dataset = load_dataset('json', data_files=TEST_FILE, split='train')

    # Optional: If your dataset might have an 'id' column or similar that's not 'code' or 'label'
    # You might want to remove it here if it causes issues later or is not needed.

    print("Preview of train dataset first entry:")
    print(train_dataset[0])

    # 2. Save each dataset to a directory using save_to_disk()
    print(f"Saving train dataset to {OUTPUT_TRAIN_DIR}...")
    train_dataset.save_to_disk(OUTPUT_TRAIN_DIR)
    print(f"Saving validation dataset to {OUTPUT_VAL_DIR}...")
    val_dataset.save_to_disk(OUTPUT_VAL_DIR)
    print(f"Saving test dataset to {OUTPUT_TEST_DIR}...")
    test_dataset.save_to_disk(OUTPUT_TEST_DIR)

    print("Conversion complete. You can now run fine_tune_codebert.py")

if __name__ == "__main__":
    convert_and_save_dataset()
