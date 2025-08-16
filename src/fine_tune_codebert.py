import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import numpy as np
import os

# Configuration
MODEL_NAME = "microsoft/codebert-base"
DATA_PATH = "data/processed"
OUTPUT_DIR = "./results_codebert"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3 # As per your plan, 3-5 epochs. Let's start with 3.

# --- NEW CONFIGURATION FOR LIMITING DATASET SIZE ---
# Set to None to use full dataset, or an integer to limit the number of files loaded per split.
# Adjust these values based on your CPU and time constraints.
MAX_TRAIN_FILES = 1000  # Limit training to 1000 files for quick testing
MAX_EVAL_FILES = 100    # Limit evaluation to 100 files
MAX_TEST_FILES = 100    # Limit test to 100 files

def load_code_files_as_dataset(directory_path, label=0, max_files=None):
    """
    Loads .py files from a directory, reads their content as 'code',
    and assigns a given 'label'. Returns a HuggingFace Dataset.
    Can limit the number of files loaded using `max_files`.
    """
    print(f"  Scanning directory: {directory_path}")
    data = {"code": [], "label": []}
    count = 0
    if not os.path.exists(directory_path):
        print(f"  Warning: Directory not found - {directory_path}")
        return Dataset.from_dict(data)
        
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                    data["code"].append(code_content)
                    data["label"].append(label)
                    count += 1
                    if max_files is not None and count >= max_files:
                        print(f"  Reached max_files limit ({max_files}) in {directory_path}")
                        break # Stop loading if max_files is reached
                except Exception as e:
                    print(f"Could not read file {filepath}: {e}")
        if max_files is not None and count >= max_files:
            break
    print(f"  Loaded {len(data["code"])} files from {directory_path}")
    return Dataset.from_dict(data)

def tokenize_function(examples):
    global tokenizer # Access the tokenizer defined in the main block
    return tokenizer(examples["code"], truncation=True, max_length=512)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    acc = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm.tolist() # Convert numpy array to list for easier logging
    }

if __name__ == "__main__":
    print("--- Starting CodeBERT Fine-tuning ---")

    # 1. Load Dataset from individual .py files
    print(f"Loading dataset from {DATA_PATH}...")
    
    # This script assumes 'clean' code (label 0) is in data/processed/ and SATE-IV 'vulnerable' code (label 1) is in data/SATE-IV/
    # In a real scenario, you'd have more robust data organization.
    try:
        clean_train_dataset = load_code_files_as_dataset(os.path.join(DATA_PATH, 'train'), label=0, max_files=MAX_TRAIN_FILES)
        clean_val_dataset = load_code_files_as_dataset(os.path.join(DATA_PATH, 'val'), label=0, max_files=MAX_EVAL_FILES)
        clean_test_dataset = load_code_files_as_dataset(os.path.join(DATA_PATH, 'test'), label=0, max_files=MAX_TEST_FILES)

        # For this example, we'll assume SATE-IV data is all for training.
        # A more robust pipeline would split SATE-IV into train/val/test as well.
        vulnerable_dataset = load_code_files_as_dataset('data/SATE-IV', label=1, max_files=MAX_TRAIN_FILES) # Also limit vulnerable files

        # Combine clean and vulnerable datasets
        # Note: This is a simplified merge. For better results, you'd typically shuffle and then split.
        train_dataset = Dataset.from_dict({
            "code": clean_train_dataset["code"] + vulnerable_dataset["code"],
            "label": clean_train_dataset["label"] + vulnerable_dataset["label"]
        }).shuffle(seed=42)

        raw_datasets = DatasetDict({
            'train': train_dataset,
            'validation': clean_val_dataset, # Using clean data for validation for now
            'test': clean_test_dataset
        })

    except Exception as e:
        print(f"Could not load dataset. Error: {e}")
        print("Exiting.")
        exit()
    
    if len(raw_datasets["train"]) == 0:
        print("Training dataset is empty. Please ensure data exists in data/processed/train/ and/or data/SATE-IV/")
        exit()

    print(f"Total training examples: {len(raw_datasets['train'])}")
    print(f"Total validation examples: {len(raw_datasets['validation'])}")
    print("Dataset loaded. Example entry from training set:")
    print(raw_datasets["train"][0])

    # 2. Load CodeBERT Tokenizer and Model
    print("Loading tokenizer and model: {}...".format(MODEL_NAME))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2) # 2 labels: vulnerable (1) or not (0)

    # 3. Preprocessing
    print("Tokenizing datasets...")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    # The 'code' column is no longer needed after tokenization
    tokenized_datasets = tokenized_datasets.remove_columns(["code"])
    # The 'label' column should be renamed to 'labels' for the Trainer
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    # 4. Training Arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1", # Save model based on F1 score
        report_to="none" # Disable integrations like wandb for simplicity
    )

    # 5. Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6. Train and Evaluate
    print("Starting training...")
    trainer.train()

    print("Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print("--- Test Results ---")
    print(test_results)
    
    # 7. Save Model
    print(f"Saving model to {OUTPUT_DIR}/final_model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    print("Model saved.")

    print("--- CodeBERT Fine-tuning Complete ---")