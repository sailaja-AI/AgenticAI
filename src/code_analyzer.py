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

            # --- CORRECTED PATH RESOLUTION LOGIC HERE ---
            # Always join the path from config with the project_root for local files
            model_path = os.path.join(project_root, model_path_from_config)
            # --- END CORRECTED LOGIC ---
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Fine-tuned classifier model not found at: {model_path}")

            print(f"Loading fine-tuned classifier model from: {model_path} on device: {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            model_name = config.get('code_analysis_model', "microsoft/codebert-base")
            print(f"Loading base model for analysis: {model_name} on device: {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()
        print("Model loaded.")

    def analyze_snippet(self, code_snippet):
        """
        Performs semantic analysis on a code snippet (embedding) or classifies its vulnerability.
        Returns a vector embedding or a prediction (0/1) and confidence scores.
        """
        if not code_snippet or not code_snippet.strip():
            if self.is_classifier:
                return {"prediction": 0, "confidence_scores": [1.0, 0.0]}
            else:
                return {"embedding": torch.zeros(self.model.config.hidden_size).to(self.device)}

        display_snippet = code_snippet[:60].replace('\n', ' ')
        print(f"Analyzing snippet: {display_snippet}...")

        inputs = self.tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.is_classifier:
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            confidence_scores = logits.softmax(dim=1).tolist()[0]
            return {"prediction": prediction, "confidence_scores": confidence_scores}
        else:
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        return {"embedding": embedding}

```

