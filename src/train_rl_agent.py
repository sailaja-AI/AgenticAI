from stable_baselines3 import PPO
from .environment import PatchSuggestionEnv
from .static_analysis_agent import StaticAnalysisAgent
from .code_analyzer import CodeAnalyzer
import torch # Make sure torch is imported for tensor operations

# Configuration for training
FILE_TO_SCAN = "vulnerable_app.py" # Using a known vulnerable file for training environment
MODEL_SAVE_PATH = "./rl_model.zip" # Consistent with PatchSuggesterAgent
TOTAL_TIMESTEPS = 500 # REDUCED for faster CPU training

def train_agent():
    """
    Trains a PPO agent to suggest security patches.
    """
    print("--- Starting RL Agent Training ---")

    # 1. Generate a (small) dataset of vulnerabilities
    print(f"Scanning {FILE_TO_SCAN} to create training environments...")
    static_analyzer = StaticAnalysisAgent()
    semantic_analyzer = CodeAnalyzer() # Ensure CodeAnalyzer is initialized
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

    # 5. Save the trained model
    print(f"Training complete. Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("Model saved.")

if __name__ == "__main__":
    train_agent()
