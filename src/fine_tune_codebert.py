import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import numpy as np
import os
import shutil
import subprocess # NEW: Import subprocess module

# Try to import files for Colab download, and set a flag
IS_COLAB = False
try:
    from google.colab import files
    IS_COLAB = True
except ImportError:
    print("Running outside Google Colab. File download functionality will be skipped.")

# Configuration
MODEL_NAME = "microsoft/codebert-base"

# --- NEW: GOOGLE DRIVE BASE PATH ---
# IMPORTANT: Update this if your Google Drive structure is different
GOOGLE_DRIVE_DATA_BASE_PATH = "/content/drive/MyDrive/AgentAI_Data" # Assuming raw/processed data is here
PROCESSED_DATA_OUTPUT_DIR = os.path.join(GOOGLE_DRIVE_DATA_BASE_PATH, "processed_jsonl") # Where prepare_dataset saves

OUTPUT_DIR = "./results_codebert" # Stored locally in Colab session initially
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10 # Reduced epochs for faster testing

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

    # 1. Load Pre-processed Dataset from JSONL files (from Google Drive)
    print(f"Loading pre-processed dataset from {PROCESSED_DATA_OUTPUT_DIR}...")
    
    try:
        train_path = os.path.join(PROCESSED_DATA_OUTPUT_DIR, 'train.jsonl')
        val_path = os.path.join(PROCESSED_DATA_OUTPUT_DIR, 'val.jsonl')
        test_path = os.path.join(PROCESSED_DATA_OUTPUT_DIR, 'test.jsonl')

        # Check if files exist before loading
        if not os.path.exists(train_path):
            print(f"Error: Training data not found at {train_path}. Please run src/prepare_dataset.py in Colab first.")
            exit()
        
        # Load datasets. load_dataset can handle missing files by returning empty, but we'll check.
        raw_datasets_dict = {}
        raw_datasets_dict['train'] = load_dataset('json', data_files=train_path, split='train')

        if os.path.exists(val_path):
            raw_datasets_dict['validation'] = load_dataset('json', data_files=val_path, split='train')
        else:
            print(f"Warning: Validation data not found at {val_path}.")
            
        if os.path.exists(test_path):
            raw_datasets_dict['test'] = load_dataset('json', data_files=test_path, split='train')
        else:
            print(f"Warning: Test data not found at {test_path}.")

        raw_datasets = DatasetDict(raw_datasets_dict)

    except Exception as e:
        print(f"Could not load dataset from {PROCESSED_DATA_OUTPUT_DIR}. Error: {e}")
        print("Please ensure you have run 'src/prepare_dataset.py' in Colab first to create train.jsonl, val.jsonl, and test.jsonl.")
        exit()
    
    if len(raw_datasets["train"]) == 0:
        print("Training dataset is empty. Please ensure data exists in the processed JSONL files.")
        exit()

    print(f"Total training examples: {len(raw_datasets['train'])}")
    if 'validation' in raw_datasets:
        print(f"Total validation examples: {len(raw_datasets['validation'])}")
    if 'test' in raw_datasets:
        print(f"Total test examples: {len(raw_datasets['test'])}")
    print("Dataset loaded. Example entry from training set:")
    print(raw_datasets["train"][0])

    # 2. Load CodeBERT Tokenizer and Model
    print("Loading tokenizer and model: {}...".format(MODEL_NAME))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2) # 2 labels: vulnerable (1) or not (0)

    # 3. Preprocessing
    print("Tokenizing datasets...")
    # Use remove_columns directly in map for efficiency
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["code"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    # 4. Training Arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
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
        eval_dataset=tokenized_datasets["validation"] if "validation" in tokenized_datasets else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6. Train and Evaluate
    print("Starting training...")
    trainer.train()

    if "test" in tokenized_datasets and tokenized_datasets["test"] is not None:
        print("Evaluating on test set...")
        test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
        print("--- Test Results ---")
        print(test_results)
    else:
        print("No test dataset available for evaluation.")
    
    # 7. Save Model locally
    print(f"Saving model to {OUTPUT_DIR}/final_model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    print("Model saved locally.")

    print("--- Starting Model Download and Persistence ---")
    # Define Google Drive path for trained models
    DRIVE_TRAINED_MODELS_PATH = os.path.join(GOOGLE_DRIVE_DATA_BASE_PATH, "trained_models")
    os.makedirs(DRIVE_TRAINED_MODELS_PATH, exist_ok=True) # Ensure target dir in Drive exists

    # --- CodeBERT Model ---
    codebert_model_local_path = os.path.join(OUTPUT_DIR, "final_model") # Using OUTPUT_DIR for robustness
    codebert_zip_name = "codebert_final_model.zip"
    codebert_drive_path = os.path.join(DRIVE_TRAINED_MODELS_PATH, "final_model") # Copy as unzipped folder to Drive

    if os.path.exists(codebert_model_local_path):
        print(f"Zipping CodeBERT model from {codebert_model_local_path}...")
        
        # Use subprocess.run for shell commands instead of !magic_command
        codebert_zip_successful = False
        try:
            subprocess.run(["zip", "-r", codebert_zip_name, codebert_model_local_path], check=True)
            print(f"CodeBERT model zipped as {codebert_zip_name}")
            codebert_zip_successful = True
        except FileNotFoundError:
            print("Error: 'zip' command not found. Please ensure zip is installed and in your system PATH.")
            print("Skipping zipping and local download for CodeBERT model.")
        except subprocess.CalledProcessError as e:
            print(f"Error zipping CodeBERT model: {e}")
            print("Skipping local download for CodeBERT model.")
        except Exception as e:
            print(f"An unexpected error occurred during zipping CodeBERT model: {e}")
            print("Skipping local download for CodeBERT model.")

        # Offer for local download only if running in Colab and zipping was successful
        if IS_COLAB and codebert_zip_successful:
            try:
                files.download(codebert_zip_name)
                print(f"'{codebert_zip_name}' offered for download to local machine.")
            except Exception as e:
                print(f"Warning: Could not offer '{codebert_zip_name}' for local download: {e}")

        # Copy to Google Drive for persistence (as an unzipped folder)
        try:
            print(f"Copying CodeBERT model to Google Drive at {codebert_drive_path}...")
            shutil.copytree(codebert_model_local_path, codebert_drive_path, dirs_exist_ok=True)
            print(f"CodeBERT model saved to Google Drive at {codebert_drive_path}")
        except Exception as e:
            print(f"ERROR: Could not save CodeBERT model to Google Drive: {e}")
            print("This model is only saved locally in the Colab session and will be deleted after session ends.")
    else:
        print(f"CodeBERT model directory not found at {codebert_model_local_path}. Was training successful?")

    # --- RL Model ---
    rl_model_local_path = "rl_model.zip" # This assumes rl_model.zip is created elsewhere
    rl_model_drive_path = os.path.join(DRIVE_TRAINED_MODELS_PATH, rl_model_local_path)

    if os.path.exists(rl_model_local_path):
        print(f"RL model found at {rl_model_local_path}")
        # Offer for local download only if running in Colab
        if IS_COLAB:
            try:
                files.download(rl_model_local_path)
                print(f"'{rl_model_local_path}' offered for download to local machine.")
            except Exception as e:
                print(f"Warning: Could not offer '{rl_model_local_path}' for local download: {e}")

        # Copy to Google Drive for persistence
        try:
            print(f"Copying RL model to Google Drive at {rl_model_drive_path}...")
            shutil.copy(rl_model_local_path, rl_model_drive_path)
            print(f"RL model saved to Google Drive at {rl_model_drive_path}")
        except Exception as e:
            print(f"ERROR: Could not save RL model to Google Drive: {e}")
            print("This model is only saved locally in the Colab session and will be deleted after session ends.")
    else:
        print(f"RL model '{rl_model_local_path}' not found. Was training successful?")

    print("--- Model Download and Persistence Complete ---")
