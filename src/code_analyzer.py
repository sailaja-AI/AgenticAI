import yaml
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import os
import requests
import zipfile
import io

class CodeAnalyzer:
    """
    Analyzes source code using transformer-based models for semantic understanding
    and can also be configured for vulnerability classification.
    """
    def __init__(self, config_path='config.yaml', is_classifier=False):
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
            repo_owner = config.get('github_repo_owner')
            repo_name = config.get('github_repo_name')
            release_tag = config.get('github_release_tag')
            asset_name = config.get('github_release_asset_name')
            local_cache_relative_path = config.get('local_model_cache_dir')

            if not all([repo_owner, repo_name, release_tag, asset_name, local_cache_relative_path]):
                raise ValueError("GitHub Release model configuration (github_repo_owner, github_repo_name, github_release_tag, github_release_asset_name, local_model_cache_dir) must be specified in config.yaml for classifier mode.")
            
            model_path = os.path.join(project_root, local_cache_relative_path)

            if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
                print(f"Fine-tuned classifier model not found locally at: {model_path}.")
                print(f"Attempting to download and extract from GitHub Release {release_tag}...")
                self._download_and_extract_github_release_asset(
                    repo_owner, repo_name, release_tag, asset_name, model_path
                )
            else:
                print(f"Fine-tuned classifier model found locally at: {model_path}. Skipping download.")
            
            print(f"Loading fine-tuned classifier model from local cache: {model_path} on device: {self.device}...")
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

    def _download_and_extract_github_release_asset(self, owner, repo, tag, asset_name, target_dir):
        release_api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
        
        headers = {"Accept": "application/vnd.github.v3+json"}
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        print(f"Fetching release information from: {release_api_url}")
        response = requests.get(release_api_url, headers=headers)
        response.raise_for_status()
        release_info = response.json()

        asset_download_url = None
        for asset in release_info.get("assets", []):
            if asset["name"] == asset_name:
                asset_download_url = asset["browser_download_url"]
                break
        
        if not asset_download_url:
            raise FileNotFoundError(f"Asset '{asset_name}' not found in GitHub Release '{tag}' for {owner}/{repo}.")

        print(f"Downloading asset from: {asset_download_url}")
        asset_response = requests.get(asset_download_url, stream=True, headers=headers)
        asset_response.raise_for_status()

        os.makedirs(target_dir, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(asset_response.content)) as zf:
            print(f"Extracting '{asset_name}' to '{target_dir}'...")
            zf.extractall(target_dir)
            print("Extraction complete.")
            
    def analyze_snippet(self, code_snippet):
        """
        Performs semantic analysis on a code snippet (embedding) or classifies its vulnerability.
        Returns a vector embedding or a prediction (0/1) and confidence scores.
        """
        # Ensure the entire method is consistently indented
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

        # --- CORRECTED INDENTATION AND MISSING BRACKET HERE ---
        if self.is_classifier:
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            confidence_scores = logits.softmax(dim=1).tolist()[0] # Added missing ']'
            return {"prediction": prediction, "confidence_scores": confidence_scores}
        else:
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
            return {"embedding": embedding}
        # --- END CORRECTED SECTION ---