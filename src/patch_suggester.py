from stable_baselines3 import PPO
import os
import torch
import numpy as np
from .environment import PatchSuggestionEnv

# Define the path where the trained RL model is saved
MODEL_SAVE_PATH = "./rl_model.zip"

class PatchSuggesterAgent:
    """
    An agent that uses a Reinforcement Learning model to suggest security patches.
    """
    def __init__(self):
        self.trained = False
        self.agent = None
        
        self.actions_map = {
            0: "Modify 'subprocess.call' to use 'shlex.quote' for command arguments.",
            1: "Consider alternative secure API or sanitize input (general suggestion)."
        }

        if os.path.exists(MODEL_SAVE_PATH):
            try:
                # A dummy environment is needed to load the PPO model.
                # We create a mock vulnerability context for loading purposes.
                mock_vulnerability = {
                    "type": "INSECURE_SUBPROCESS",
                    "code": "subprocess.call(cmd, shell=True)",
                    "embedding": torch.zeros(768) # Dummy embedding
                }
                dummy_env = PatchSuggestionEnv(mock_vulnerability)
                self.agent = PPO.load(MODEL_SAVE_PATH, env=dummy_env)
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
        
        if 'embedding' not in vulnerability or vulnerability['embedding'] is None:
            print("Error: Vulnerability context missing embedding. Cannot proceed with RL. Falling back to dummy.")
            action = 0 if vulnerability['type'] == 'INSECURE_SUBPROCESS' else 1
            print(f"Dummy agent chose action {action}")
            return self.actions_map.get(action, "Invalid action.")

        env = PatchSuggestionEnv(vulnerability)
        obs, _ = env.reset()

        if self.trained and self.agent:
            print("Using trained RL agent to predict action...")
            action, _ = self.agent.predict(obs, deterministic=True)
            action = int(action) if isinstance(action, np.ndarray) else action
        else:
            print("Using dummy agent to select action...")
            action = 0 if vulnerability['type'] == 'INSECURE_SUBPROCESS' else env.action_space.sample()

        _obs, reward, _terminated, _truncated, _info = env.step(action)
        print(f"Agent chose action {action}, received simulated reward: {reward}")

        return self.actions_map.get(action, "Invalid action.")
