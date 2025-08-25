import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets # Import concatenate_datasets and load_dataset
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import numpy as np
import os
import shutil # Added for saving to Google Drive

# Configuration
MODEL_NAME = "microsoft/codebert-base"

# --- NEW: GOOGLE DRIVE BASE PATH ---
# IMPORTANT: Update this if your Google Drive structure is different
GOOGLE_DRIVE_DATA_BASE_PATH = "/content/drive/MyDrive/AgentAI_Data" # Assuming processed data is here
PROCESSED_DATA_OUTPUT_DIR = os.path.join(GOOGLE_DRIVE_DATA_BASE_PATH, "processed_jsonl") # Where prepare_dataset saves

OUTPUT_DIR = "./results_codebert" # Stored locally in Colab session initially
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1 # Reduced epochs for faster testing

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
    
        raw_datasets = load_dataset('json', data_files={
            'train': train_path,
            'validation': val_path, # Ensure val.jsonl exists or handle its absence
            'test': test_path       # Ensure test.jsonl exists or handle its absence
        })
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
        evaluation_strategy="epoch", # Use evaluation_strategy as recommended
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
    
    # 7. Save Model (locally in Colab session) and then potentially to Drive
    print(f"Saving model to {OUTPUT_DIR}/final_model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    print("Model saved.")

    print("--- CodeBERT Fine-tuning Complete ---")

    # Optional: Save trained model to Google Drive for persistence
    try:
        print("Attempting to save trained model to Google Drive...")
        drive_model_path = os.path.join(GOOGLE_DRIVE_DATA_BASE_PATH, "trained_models", "final_model")
        os.makedirs(drive_model_path, exist_ok=True)
        # Use shutil.copytree for directories, dirs_exist_ok=True for Python 3.8+
        shutil.copytree(os.path.join(OUTPUT_DIR, "final_model"), drive_model_path, dirs_exist_ok=True)
        print(f"Model saved to Google Drive at {drive_model_path}")
    except Exception as e:
        print(f"Warning: Could not save model to Google Drive. Error: {e}")
        print("Model is only saved locally in the Colab session and will be deleted after session ends.")
```

