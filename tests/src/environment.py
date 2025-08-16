import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PatchSuggestionEnv(gym.Env):
    """
    A custom Gym environment for suggesting security patches.
    - State: The vulnerability context (e.g., type, code snippet, embedding).
    - Action: The type of patch to apply.
    - Reward: 1 if the patch fixes the vulnerability, -1 otherwise (simplified for now).
    """
    def __init__(self, vulnerability_context):
        super().__init__()
        self.vulnerability_context = vulnerability_context
        
        # The observation space will be the embedding from the code analyzer (768 dimensions for CodeBERT-base)
        # We use a Box space for continuous values, covering the possible range of embeddings.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32)

        # Action space:
        # 0: Apply secure subprocess patch (e.g., using shlex.quote)
        # 1: No patch / Other (for other vulnerability types or if no specific patch applies)
        self.action_space = spaces.Discrete(2) # Two possible actions

        self.actions_map = {
            0: "Apply secure subprocess patch (using shlex.quote)",
            1: "No patch / Other action"
        }

        # Current state will be the embedding of the vulnerability
        self._current_state = self.vulnerability_context['embedding'.cpu().numpy().astype(np.float32)
    def _get_state(self):
        # Return the embedding as a numpy array for the gym environment
        return self._current_state
    def step(self, action):
        # In a real system, this would apply the patch and re-run analysis.
        # Here, we simulate it with a simple reward for the 'INSECURE_SUBPROCESS' type.
        reward = 0
        terminated = False
        truncated = False
        info = {'message': self.actions_map.get(action, "Invalid action.")}

        # Simplified Reward Logic:
        # If the vulnerability is INSECURE_SUBPROCESS and action is to apply a secure patch
        if self.vulnerability_context['type'] == 'INSECURE_SUBPROCESS' and action == 0:
            reward = 1  # Positive reward for attempting to fix it correctly
            info['message'] += " - Attempted to fix INSECURE_SUBPROCESS."
            terminated = True # Episode ends after attempting a patch
        else:
            reward = -1 # Negative reward for wrong action or unhandled vulnerability type
            info['message'] += " - Incorrect action or unhandled vulnerability type."
            terminated = True # Episode ends
        return self._get_state(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset environment to the initial vulnerability context
        self._current_state = self.vulnerability_context['embedding'].cpu().numpy().astype(np.float32)
        info = {}
        return self._get_state(), info
