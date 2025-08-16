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

def load_code_files_as_dataset(directory_path, label=0):
    """
    Loads all .py files from a directory, reads their content as 'code',
    and assigns a given 'label'. Returns a HuggingFace Dataset.
    """
    data = {"code": [], "label": []}
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                    data["code"].append(code_content)
                    data["label"].append(label)
                except Exception as e:
                    print(f"Could not read file {filepath}: {e}")
    return Dataset.from_dict(data)

def tokenize_function(examples):
    global tokenizer # Access the tokenizer defined in the main block
    return tokenizer(examples["code"], truncation=True, max_length=512)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
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
    print(f"--- Starting CodeBERT Fine-tuning ---")

    # 1. Load Dataset from individual .py files
    print(f"Loading dataset from {DATA_PATH}...\n")
    
    # Assuming all .py files in data/processed/train, val, test are *clean* (label=0).
    # If you have vulnerable samples, they need to be added here with label=1.
    # For a real scenario, you'd merge your vulnerable and non-vulnerable datasets here.
    try:
        train_dataset = load_code_files_as_dataset(os.path.join(DATA_PATH, 'train'), label=0)
        val_dataset = load_code_files_as_dataset(os.path.join(DATA_PATH, 'val'), label=0)
        test_dataset = load_code_files_as_dataset(os.path.join(DATA_PATH, 'test'), label=0)
        
        # If you have vulnerable samples from a different source (e.g., 'data/vulnerable_snippets/')
        # you would load and concatenate them here:
        # vulnerable_train = load_code_files_as_dataset('data/vulnerable_snippets/train/', label=1)
        # train_dataset = Dataset.from_dict(train_dataset[:] + vulnerable_train[:]) # Merging example

        raw_datasets = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })

    except Exception as e:
        print(f"Could not load dataset from {DATA_PATH}. Error: {e}")
        print("Please ensure your data/processed/train, val, test directories contain .py files.")
        print("Exiting.")
        exit()
    
    print("Dataset loaded. Example entry:")
    print(raw_datasets["train"][0])

    # 2. Load CodeBERT Tokenizer and Model
    print(f"Loading tokenizer and model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2) # 2 labels: vulnerable (1) or not (0)

    # 3. Preprocessing
    print("Tokenizing datasets...\n")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["code"]) # Remove original code column
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels") # Rename label to labels for Trainer

    # Set format for PyTorch
    tokenized_datasets.set_format("torch")

    # 4. Training Arguments
    print("Setting up training arguments...\n")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,\n        evaluation_strategy=\"epoch\",\n        learning_rate=LEARNING_RATE,\n        per_device_train_batch_size=BATCH_SIZE,\n        per_device_eval_batch_size=BATCH_SIZE,\n        num_train_epochs=NUM_EPOCHS,\n        weight_decay=0.01,\n        logging_dir='./logs',\n        logging_steps=100,\n        save_strategy=\"epoch\",\n        load_best_model_at_end=True,\n        metric_for_best_model=\"f1\", # Save model based on F1 score\n        report_to=\"none\" # Disable integrations like wandb for simplicity\n    )

    # 5. Trainer
    print("Initializing Trainer...\n")
    trainer = Trainer(
        model=model,\n        args=training_args,\n        train_dataset=tokenized_datasets["train"],\n        eval_dataset=tokenized_datasets["validation"],\n        tokenizer=tokenizer,\n        compute_metrics=compute_metrics,\n    )

    # 6. Train and Evaluate
    print("Starting training...\n")
    trainer.train()

    print("Evaluating on test set...\n")
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])\n    print("--- Test Results ---")\n    print(test_results)\n
    # 7. Save Model
    print(f"Saving model to {OUTPUT_DIR}/final_model...\n")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    print("Model saved.")

    print("--- CodeBERT Fine-tuning Complete ---\n")