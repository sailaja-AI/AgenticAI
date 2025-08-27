import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import shutil
import ast
import tempfile
import torch # ADDED: Missing import for torch usage in CodeAnalyzerForRL

# Corrected Imports: Use absolute imports for modules within the 'src' directory
# (Assuming 'src' is directly under your project's Python path or current working directory)
from src.code_analyzer import CodeAnalyzer
from src.static_analysis_agent import StaticAnalysisAgent
from src.knowledge_base_agent import KnowledgeBaseAgent # Import KnowledgeBaseAgent

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification # Also ensure AutoModelForSequenceClassification is imported here if CodeAnalyzerForRL uses it
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# --- Configuration ---
FILE_TO_SCAN = "src/vulnerability_examples/sql_injection_vulnerable.py" # Example file to train on
TOTAL_TIMESTEPS = 1000 # Reduced for faster testing, increase for real training

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
AGENT_AI_DATA_ROOT = os.path.join(project_root, "AgentAI_Data")
FINE_TUNED_CODEBERT_DRIVE_PATH = os.path.join(AGENT_AI_DATA_ROOT, "trained_models", "final_model")
RL_MODEL_SAVE_DIR = os.path.join(AGENT_AI_DATA_ROOT, "trained_rl_models")
os.makedirs(RL_MODEL_SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(RL_MODEL_SAVE_DIR, "ppo_patch_agent.zip")

# --- Custom CodeAnalyzer for RL (loads fine-tuned model for patching evaluation) ---
class CodeAnalyzerForRL:
    def __init__(self):
        model_path = FINE_TUNED_CODEBERT_DRIVE_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fine-tuned CodeBERT model not found at {model_path}. Please ensure it was saved to Drive and copied locally.")

        print(f"Loading fine-tuned CodeBERT classifier model from: {model_path} on device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print("CodeBERT Classifier Model loaded for RL agent's environment.")

    def classify_snippet(self, code_snippet):
        if not code_snippet.strip():
            return {"prediction": 0, "confidence_scores": [1.0, 0.0]}

        inputs = self.tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        prediction = torch.argmax(logits, dim=1).item()
        confidence_scores = logits.softmax(dim=1).tolist()[0]
        return {"prediction": prediction, "confidence_scores": confidence_scores}

# --- Gym Environment for Patch Suggestion (Conceptual) ---
class PatchSuggestionEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, initial_vulnerability_context):
        super().__init__()
        self.vulnerability_context = initial_vulnerability_context
        self.original_code = initial_vulnerability_context['code']
        self.current_code = self.original_code
        self.code_analyzer = CodeAnalyzerForRL() # Use the classifier to evaluate patches
        self.static_analyzer = StaticAnalysisAgent() # For static rule checking

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32)
        self.action_space = spaces.Discrete(5) # Example: 5 discrete actions

        self.current_step = 0
        self.max_steps = 10

        print(f"PatchSuggestionEnv initialized for vulnerability: {self.vulnerability_context['type']}")
        base_code_analyzer = CodeAnalyzer(is_classifier=False)
        initial_analysis = base_code_analyzer.analyze_snippet(self.original_code)
        self._current_observation = initial_analysis['embedding'].cpu().numpy()

    def _get_obs(self):
        base_code_analyzer = CodeAnalyzer(is_classifier=False)
        analysis = base_code_analyzer.analyze_snippet(self.current_code)
        return analysis['embedding'].cpu().numpy()

    def _get_info(self):
        return {
            "vulnerability_type": self.vulnerability_context['type'],
            "current_code_hash": hash(self.current_code)
        }

    def step(self, action):
        self.current_step += 1
        reward = 0
        terminated = False
        info = self._get_info()

        print(f"Step {self.current_step}: Agent chose action {action}")

        patched_code = self.current_code
        if self.vulnerability_context['type'] == "COMMAND_INJECTION_SHELL_TRUE":
            if action == 0:
                patched_code = self.current_code.replace("shell=True", "shell=False")
                print("  (Simulated) Attempted to remove shell=True.")
            elif action == 1:
                patched_code = "import re\n" + patched_code
                patched_code = patched_code.replace(
                    f"def runCommand(",
                    f"def runCommand(cmd_args):\n    if re.search(r'[;&|`]', cmd_args):\n        raise ValueError('Dangerous characters detected')\n"
                )
                print("  (Simulated) Attempted to add input validation.")
            else:
                 print(f"  (Simulated) Applying default patch for COMMAND_INJECTION_SHELL_TRUE with action {action}")
                 patched_code = self.current_code
        elif self.vulnerability_context['type'] == "POTENTIAL_SQL_INJECTION":
            if action == 0:
                patched_code = self.current_code.replace(f"query = f\"SELECT", f"query = \"SELECT")
                # This string manipulation for argument replacement is highly fragile for a real scenario
                # It assumes a very specific format of the f-string variable for replacement
                sql_var = self.vulnerability_context['code'].split('{')[1].split('}')[0] if '{' in self.vulnerability_context['code'] else 'some_id'
                patched_code = patched_code.replace(f"WHERE id = {{{sql_var}}}", "WHERE id = ?")
                patched_code = patched_code.replace(f"cursor.execute(query)", f"cursor.execute(query, ({sql_var},))")
                print("  (Simulated) Attempted SQL parameterization.")
            else:
                 print(f"  (Simulated) Applying default patch for POTENTIAL_SQL_INJECTION with action {action}")
                 patched_code = self.current_code
        else:
            print(f"  (Simulated) No specific patching logic for {self.vulnerability_context['type']} for action {action}. No change.")


        try:
            ast.parse(patched_code)
        except SyntaxError:
            print("  Patch introduced a syntax error!")
            reward -= 5
            self.current_code = patched_code
            terminated = True
            return self._get_obs(), reward, terminated, False, info


        dl_result = self.code_analyzer.classify_snippet(patched_code)
        is_still_vulnerable_dl = (dl_result["prediction"] == 1)

        static_flaws = self.static_analyzer.find_potential_flaws_in_string(patched_code)
        is_still_vulnerable_static = any(f['type'] == self.vulnerability_context['type'] for f in static_flaws)


        if not is_still_vulnerable_dl and not is_still_vulnerable_static:
            reward = 10
            terminated = True
            print("  Vulnerability successfully patched!")
        elif not is_still_vulnerable_dl and is_still_vulnerable_static:
            reward = 5
            print("  DL model thinks it's fixed, but static analysis still flags it.")
        elif is_still_vulnerable_dl and not is_still_vulnerable_static:
            reward = -5
            print("  Static analysis thinks it's fixed, but DL model still flags it.")
        else:
            reward = -1
            print("  Vulnerability persists.")
        
        self.current_code = patched_code
        obs = self._get_obs()

        if self.current_step >= self.max_steps:
            terminated = True
            print("  Max steps reached. Episode terminated.")
        return obs, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_code = self.original_code
        self.current_step = 0
        base_code_analyzer = CodeAnalyzer(is_classifier=False)
        initial_analysis = base_code_analyzer.analyze_snippet(self.original_code)
        self._current_observation = initial_analysis['embedding'].cpu().numpy()
        info = self._get_info()
        print(f"\n--- Environment Reset: Starting new episode for {self.vulnerability_context['type']} ---")
        return self._current_observation, info

    def render(self):
        pass

    def close(self):
        pass

def train_agent():
    print("--- Starting RL Agent Training ---")

    print(f"Scanning {FILE_TO_SCAN} to create training environments...")
    static_analyzer = StaticAnalysisAgent()
    base_semantic_analyzer = CodeAnalyzer(is_classifier=False) 
    
    potential_flaws = static_analyzer.find_potential_flaws(FILE_TO_SCAN)

    if not potential_flaws:
        print("No vulnerabilities found to train on. Exiting.")
        return

    vulnerability_context = potential_flaws[0]
    
    analysis = base_semantic_analyzer.analyze_snippet(vulnerability_context['code'])
    vulnerability_context['embedding'] = analysis['embedding']
    
    print("Creating training environment...")
    env = make_vec_env(PatchSuggestionEnv, n_envs=1, env_kwargs={'initial_vulnerability_context': vulnerability_context})

    model = PPO("MlpPolicy", env, verbose=1)

    print(f"Training PPO model for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    print(f"Training complete. Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("Model saved.")

if __name__ == "__main__":
    train_agent()

