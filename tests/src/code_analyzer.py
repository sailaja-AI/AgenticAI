from transformers import AutoTokenizer, AutoModel
import yaml
import torch

class CodeAnalyzer:
    """
    Analyzes source code using transformer-based models for semantic understanding.
    """
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_name = config.get('code_analysis_model')
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
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
            outputs = self.model(**inputs)

        # The last hidden state of the [CLS] token is often used as a sentence embedding
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()

        print(f"Generated embedding of shape: {embedding.shape}")
        return {"embedding": embedding}

