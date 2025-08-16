from stable_baselines3 import PPO
import os
import gymnasium as gym
from .environment import PatchSuggestionEnv # Import the updated environment
from .code_analyzer import CodeAnalyzer # Needed for vulnerability embedding in dummy data
import torch
import numpy as np

# Define the path where the trained RL model will be saved
MODEL_SAVE_PATH = "./rl_model.zip"

class PatchSuggesterAgent:
    """
    An agent that uses a Reinforcement Learning model to suggest security patches.
    """
    def __init__(self):
            self.trained = False
        self.agent = None

        # Define the mapping of actions to human-readable suggestions
        self.actions_map = {
            0: "Modify 'subprocess.call' to use 'shlex.quote' for command arguments.",
            1: "Consider alternative secure API or sanitize input (general suggestion)."
        }

        if os.path.exists(MODEL_SAVE_PATH):
            try:
                # A dummy environment is needed to load the model if the real one isn't ready
                # For now, we'll create a mock vulnerability context for loading purposes
                # In a real scenario, you might pass a dummy env or ensure the env is fully defined.

                # Create a minimal mock vulnerability context for loading the model
                # This requires a dummy embedding of the correct shape (768 for CodeBERT-base)
                # and a 'type'.
                # Note: This is a temporary workaround for loading the model without a real env setup.
                # A better approach for deployment is to ensure the environment can be initialized minimally.
                mock_vulnerability = {
                    "type": "INSECURE_SUBPROCESS",
                    "code": "subprocess.call(cmd, shell=True)",
                    "embedding": torch.zeros(768) # Dummy embedding, actual value doesn't matter for loading
                }
                dummy_env = PatchSuggestionEnv(mock_vulnerability)

                self.agent = PPO.load(MODEL_SAVE_PATH, env=dummy_env) # Pass env to PPO.load
                self.trained = True
                print(f"Successfully loaded trained RL agent from {MODEL_SAVE_PATH}")
            except Exception as e:
                print(f"Error loading trained RL agent: {e}. Proceeding with dummy agent.")
        else:
            print(f"No trained RL agent found at {MODEL_SAVE_PATH}. Proceeding with dummy agent.")

    def suggest_patch(self, vulnerability):
        """
        Creates an RL environment for the vulnerability and decides on a patch.
        """
        print("\n--- Initializing Reinforcement Learning Agent for Patch Suggestion ---")

        # Ensure the vulnerability context has an embedding. This should be added by KnowledgeBaseAgent.
        if 'embedding' not in vulnerability or vulnerability['embedding'] is None:
            print("Error: Vulnerability context missing embedding. Cannot proceed with RL. Falling back to dummy.")
            # Fallback for missing embedding: use dummy agent behavior
            if vulnerability['type'] == 'INSECURE_SUBPROCESS':
                action = 0 # The best action
            else:
                action = 1 # Default to other/no patch
            print(f"Dummy agent chose action {action}")
        return self.actions_map.get(action, "Invalid action.")

        # Initialize the environment with the actual vulnerability context
        env = PatchSuggestionEnv(vulnerability)
        obs, _ = env.reset()

        if self.trained:
            # Use the trained agent to predict the best action
            print("Using trained RL agent to predict action...")
            action, _states = self.agent.predict(obs, deterministic=True)
            # Convert numpy array to a standard integer if necessary
            action = int(action) if isinstance(action, np.ndarray) else action
        else:
            # Fallback to the hardcoded logic if no model is found or loaded
            print("Using dummy agent to select action...")
            if vulnerability['type'] == 'INSECURE_SUBPROCESS':
                action = 0 # The best action
            else:
                action = env.action_space.sample() # Random action for other types

        # Simulate a step in the environment to get reward and info
        _obs, reward, _terminated, _truncated, info = env.step(action)
        print(f"Agent chose action {action}, received simulated reward: {reward}")

        return self.actions_map.get(action, "Invalid action.")
```

