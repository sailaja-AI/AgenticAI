from transformers import AutoTokenizer, AutoModel
import torch
import os

# --- UPDATED TO LOAD THE FINE-TUNED MODEL ---
# The CodeAnalyzer will now use the model we trained in the previous step.
FINE_TUNED_MODEL_PATH = "./results_codebert/final_model/"

class CodeAnalyzer:
    """
    Analyzes source code using transformer-based models for semantic understanding.
    """
    def __init__(self):
        model_path = FINE_TUNED_MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fine-tuned model not found at {model_path}. Please run fine_tune_codebert.py first.")

        print(f"Loading fine-tuned model from: {model_path}...")

        # --- MODIFIED LOADING LOGIC ---
        # Check for model.safetensors first, then pytorch_model.bin
        model_weights_file = None
        if os.path.exists(os.path.join(model_path, "model.safetensors")):
            model_weights_file = "model.safetensors"
        elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
            model_weights_file = "pytorch_model.bin"
        else:
            raise FileNotFoundError(f"No model weights found (neither model.safetensors nor pytorch_model.bin) in {model_path}.")

        print(f"  Found model weights: {model_weights_file}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Load the base model. AutoModel should correctly pick up .safetensors
        self.model = AutoModel.from_pretrained(model_path)
        print("Model loaded.")

    def analyze_snippet(self, code_snippet):
        """
        Performs semantic analysis on a code snippet using the transformer model.
        Returns a vector embedding of the code.
        """
        print(f"Analyzing snippet: {code_snippet[:40]}...")
        # Tokenize the code and pass it to the model
        inputs = self.tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad(): # Inference mode, no need to compute gradients
            # We get the embeddings from the base model's last hidden state.
            outputs = self.model(**inputs)

        # The last hidden state of the [CLS] token is often used as a sentence embedding
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()

        print(f"Generated embedding of shape: {embedding.shape}")
        return {"embedding": embedding}

