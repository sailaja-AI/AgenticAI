import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

# --- CONFIGURATION ---
# Point this to your directory of test files
# For this example, we assume you have a 'benchmark' folder with vulnerable and clean subfolders.
BENCHMARK_DIR = "tests/benchmark/"
FINE_TUNED_MODEL_PATH = "./results_codebert/final_model/"

def load_files_from_dir(directory):
    """Loads all .py files from a directory."""
    code_samples = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    code_samples.append(f.read())
    return code_samples
def evaluate_model():
    """
    Evaluates the fine-tuned CodeBERT model against a benchmark dataset.
    """
    print(f"--- Evaluating Model Performance on Benchmark Data ---")

    if not os.path.exists(BENCHMARK_DIR):
        print(f"Benchmark directory not found: {BENCHMARK_DIR}")
        print("Please create this directory with 'vulnerable' and 'clean' subdirectories containing .py files.")
        return

    # 1. Load Model and Tokenizer
    print(f"Loading model from {FINE_TUNED_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_PATH)
    model.eval() # Set model to evaluation mode
    # 2. Load and Prepare Benchmark Data
    vulnerable_files = load_files_from_dir(os.path.join(BENCHMARK_DIR, 'vulnerable'))
    clean_files = load_files_from_dir(os.path.join(BENCHMARK_DIR, 'clean'))

    if not vulnerable_files or not clean_files:
        print("Benchmark directories must contain .py files.")
        return

    # Create labels: 1 for vulnerable, 0 for clean
    true_labels = [1] * len(vulnerable_files) + [0] * len(clean_files)
    all_code = vulnerable_files + clean_files

    # 3. Make Predictions
    print("Making predictions on the benchmark dataset...")
    predictions = []
    with torch.no_grad():
        for code in all_code:
            inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512, padding=True)
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=1).item()
            predictions.append(predicted_class_id)
    # 4. Calculate and Print Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary', zero_division=0)
    accuracy = accuracy_score(true_labels, predictions)
    conf_matrix = confusion_matrix(true_labels, predictions)

    print("\n--- Benchmark Results for Fine-Tuned CodeBERT ---")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print("  Confusion Matrix:")
    print(f"    {conf_matrix}")
    print("---------------------------------------------------")
if __name__ == "__main__":
    evaluate_model()

