from stable_baselines3 import PPO
from .environment import PatchSuggestionEnv
from .static_analysis_agent import StaticAnalysisAgent
# Import CodeAnalyzer and necessary transformers components
from .code_analyzer import CodeAnalyzer
from transformers import AutoTokenizer, AutoModel # ADDED AutoTokenizer, AutoModel
import torch 
import os
import shutil # Added shutil for copying to Drive

# --- CONFIGURATION ---
FILE_TO_SCAN = "vulnerable_app.py" # Using a known vulnerable file for training environment
MODEL_SAVE_PATH = "./rl_model.zip" # RL model saved locally, then copied to Drive
TOTAL_TIMESTEPS = 500 # REDUCED for faster CPU training

# --- NEW: GOOGLE DRIVE PATH FOR FINE-TUNED CODEBERT MODEL ---
# IMPORTANT: This must match where fine_tune_codebert.py saves the model to Drive
GOOGLE_DRIVE_DATA_BASE_PATH = "/content/drive/MyDrive/AgentAI_Data"
FINE_TUNED_CODEBERT_DRIVE_PATH = os.path.join(GOOGLE_DRIVE_DATA_BASE_PATH, "trained_models", "final_model")

# Update CodeAnalyzer to use the Drive path for its model
class CodeAnalyzerForRL(CodeAnalyzer):
    def __init__(self):
        # Override the FINE_TUNED_MODEL_PATH for this specific instance
        # to point to the persistent Google Drive location
        model_path = FINE_TUNED_CODEBERT_DRIVE_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fine-tuned CodeBERT model not found at {model_path}. Please ensure it was saved to Drive.")

        print(f"Loading fine-tuned CodeBERT model from Google Drive: {model_path}...")
        
        model_weights_file = None
        if os.path.exists(os.path.join(model_path, "model.safetensors")):
            model_weights_file = "model.safetensors"
        elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
            model_weights_file = "pytorch_model.bin"
        else:
            raise FileNotFoundError(f"No model weights found (neither model.safetensors nor pytorch_model.bin) in {model_path}.")

        print(f"  Found model weights: {model_weights_file}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path) # AutoTokenizer is now imported
        self.model = AutoModel.from_pretrained(model_path) # AutoModel is now imported
        print("CodeBERT Model loaded for RL agent.")

def train_agent():
    """
    Trains a PPO agent to suggest security patches.
    """
    print("--- Starting RL Agent Training ---")

    # 1. Generate a (small) dataset of vulnerabilities
    print(f"Scanning {FILE_TO_SCAN} to create training environments...")
    static_analyzer = StaticAnalysisAgent()
    # Use our custom CodeAnalyzer that loads from Drive
    semantic_analyzer = CodeAnalyzerForRL() # Use the custom class
    potential_flaws = static_analyzer.find_potential_flaws(FILE_TO_SCAN)

    if not potential_flaws:
        print("No vulnerabilities found to train on. Exiting.")
        return

    # We will use the first vulnerability found to create our training environment
    vulnerability_context = potential_flaws[0]
    # We need to add the embedding for the environment using the semantic_analyzer
    analysis = semantic_analyzer.analyze_snippet(vulnerability_context['code'])
    vulnerability_context['embedding'] = analysis['embedding']
    
    # 2. Create the Gym Environment
    print("Creating training environment...")
    env = PatchSuggestionEnv(vulnerability_context)

    # 3. Instantiate the PPO model
    # 'MlpPolicy' is a standard feed-forward neural network policy.
    # The model will learn to map the 768-dim embedding to the best action.
    # verbose=1 shows training progress.
    model = PPO("MlpPolicy", env, verbose=1)

    # 4. Train the model
    print(f"Training PPO model for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # 5. Save the trained model locally
    print(f"Training complete. Saving RL model to {MODEL_SAVE_PATH} (locally)...")
    model.save(MODEL_SAVE_PATH)
    print("RL Model saved locally.")

    # Optional: Save RL model to Google Drive for persistence
    try:
        print("Attempting to save RL model to Google Drive...")
        DRIVE_TRAINED_MODELS_PATH = os.path.join(GOOGLE_DRIVE_DATA_BASE_PATH, "trained_models")
        os.makedirs(DRIVE_TRAINED_MODELS_PATH, exist_ok=True)
        shutil.copy(MODEL_SAVE_PATH, os.path.join(DRIVE_TRAINED_MODELS_PATH, os.path.basename(MODEL_SAVE_PATH)))
        print(f"RL model saved to Google Drive at {os.path.join(DRIVE_TRAINED_MODELS_PATH, os.path.basename(MODEL_SAVE_PATH))}")
    except Exception as e:
        print(f"Warning: Could not save RL model to Google Drive. Error: {e}")
        print("RL Model is only saved locally in the Colab session and will be deleted after session ends.")

if __name__ == "__main__":
    train_agent()
