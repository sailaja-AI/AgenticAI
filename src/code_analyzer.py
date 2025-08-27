import yaml
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import os

class CodeAnalyzer:
    """
    Analyzes source code using transformer-based models for semantic understanding
    and can also be configured for vulnerability classification.
    """
    def __init__(self, config_path='config.yaml', is_classifier=False):
        # Dynamically determine project root for config path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        full_config_path = os.path.join(project_root, config_path)

        if not os.path.exists(full_config_path):
            raise FileNotFoundError(f"Config file not found at: {full_config_path}")
        with open(full_config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.is_classifier = is_classifier
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.is_classifier:
            model_path_from_config = config.get('fine_tuned_vulnerability_model_path')
            if not model_path_from_config:
                raise ValueError("fine_tuned_vulnerability_model_path not specified in config.yaml for classifier mode.")

            # Ensure model_path is absolute and uses correct separators
            # It might be relative to config.yaml if provided that way, or absolute
            if not os.path.isabs(model_path_from_config):
                 # Assume it's relative to the project root if not absolute
                model_path = os.path.join(project_root, model_path_from_config)
            else:
                model_path = model_path_from_config

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Fine-tuned classifier model not found at: {model_path}")

            print(f"Loading fine-tuned classifier model from: {model_path} on device: {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            model_name = config.get('code_analysis_model', "microsoft/codebert-base") # Default if not in config
            print(f"Loading base model for analysis: {model_name} on device: {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode
        print("Model loaded.")

    def analyze_snippet(self, code_snippet):
        """
        Performs semantic analysis on a code snippet (embedding) or classifies its vulnerability.
        Returns a vector embedding or a prediction (0/1) and confidence scores.
        """
        if not code_snippet or not code_snippet.strip():
            # Handle empty/whitespace snippets gracefully
            if self.is_classifier:
                # Assuming 2 classes: 0 for not vulnerable, 1 for vulnerable
                # If an empty snippet is considered not vulnerable, return [1.0, 0.0]
                # If there's an ambiguity or need for an 'unknown' state, it would be handled differently.
                # For now, default to 'not vulnerable' with high confidence.
                return {"prediction": 0, "confidence_scores": [1.0, 0.0]}
            else:
                # For embedding, return a zero vector of the expected hidden size
                # This requires knowing the model's hidden size, which is available after loading.
                # If model is not loaded (e.g., in __init__ for some reason), this might fail.
                # Assuming model is always loaded before analyze_snippet is called.
                return {"embedding": torch.zeros(self.model.config.hidden_size).to(self.device)}

        print(f"Analyzing snippet: {code_snippet[:60].replace('\n', ' ')}...")
        inputs = self.tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.is_classifier:
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            confidence_scores = logits.softmax(dim=1).tolist()[0]
            # print(f"Generated prediction: {prediction}, Confidence: {confidence_scores}")
            return {"prediction": prediction, "confidence_scores": confidence_scores}
        else:
            embedding = outputs.last_hidden_state[:, 0, :.squeeze()
            # print(f"Generated embedding of shape: {embedding.shape}")
        return {"embedding": embedding}

