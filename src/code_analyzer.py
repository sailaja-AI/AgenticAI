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
            # --- CHANGE HERE: LOAD FROM GITHUB RELEASE ---
            repo_owner = config.get('github_repo_owner')
            repo_name = config.get('github_repo_name')
            release_tag = config.get('github_release_tag')
            asset_name = config.get('github_release_asset_name')
            local_cache_relative_path = config.get('local_model_cache_dir')

            if not all([repo_owner, repo_name, release_tag, asset_name, local_cache_relative_path]):
                raise ValueError("GitHub Release model configuration (github_repo_owner, github_repo_name, github_release_tag, github_release_asset_name, local_model_cache_dir) must be specified in config.yaml for classifier mode.")

            # Construct the absolute local cache path
            model_path = os.path.join(project_root, local_cache_relative_path)

            # Check if the model is already downloaded and extracted
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
            # --- END CHANGE ---
        else:
            model_name = config.get('code_analysis_model', "microsoft/codebert-base")
            print(f"Loading base model for analysis: {model_name} on device: {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()
        print("Model loaded.")

    def _download_and_extract_github_release_asset(self, owner, repo, tag, asset_name, target_dir):
        """
        Downloads a specific asset from a GitHub Release and extracts it.
        Assumes the asset is a .zip file containing the model files.
        """
        # Construct API URL to get release details
        release_api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"

        headers = {"Accept": "application/vnd.github.v3+json"}
        # Use a GitHub token if available (for higher rate limits or private repos)
        # GITHUB_TOKEN environment variable can be set in CI/CD environments.
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        print(f"Fetching release information from: {release_api_url}")
        response = requests.get(release_api_url, headers=headers)
        response.raise_for_status() # Raise an exception for HTTP errors
        release_info = response.json()

        asset_download_url = None
        for asset in release_info.get("assets", []):
            if asset["name"] == asset_name:
                # This is the direct download URL for the asset
                asset_download_url = asset["browser_download_url"]
                break

        if not asset_download_url:
            raise FileNotFoundError(f"Asset '{asset_name}' not found in GitHub Release '{tag}' for {owner}/{repo}.")

        print(f"Downloading asset from: {asset_download_url}")
        asset_response = requests.get(asset_download_url, stream=True, headers=headers)
        asset_response.raise_for_status()

        # Create the target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)

        # Extract the zip file contents directly to the target_dir
        with zipfile.ZipFile(io.BytesIO(asset_response.content)) as zf:
            print(f"Extracting '{asset_name}' to '{target_dir}'...")
            zf.extractall(target_dir)
            print("Extraction complete.")

        # IMPORTANT: After extraction, the model files (config.json, etc.) should be
        # directly in `target_dir` (e.g., `AgentAI_Data/trained_models/final_model_from_release/config.json`).
        # If your zip file contains an *extra* folder inside (e.g., `final_model.zip` contains `final_model/config.json`),
        # you might need to adjust the extraction path or move files.
        # For simplicity, we assume `config.json` is at the top level of the zip's contents.


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

