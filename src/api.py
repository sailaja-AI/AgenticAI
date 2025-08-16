from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import tempfile

# Import your agents
from .static_analysis_agent import StaticAnalysisAgent
from .code_analyzer import CodeAnalyzer
from .patch_suggester import PatchSuggesterAgent

# Initialize the FastAPI app
app = FastAPI()

# --- Agent Initialization ---
# We initialize the agents once when the API starts up.
print("Initializing agents...")
static_analyzer = StaticAnalysisAgent()
code_analyzer = CodeAnalyzer() # This will load the fine-tuned CodeBERT model
patch_suggester = PatchSuggesterAgent() # This will load the trained RL model
print("Agents initialized.")

# --- Pydantic Models for Request/Response ---
class CodeAnalysisRequest(BaseModel):
    code: str

class Vulnerability(BaseModel):
    type: str
    line: int
    code: str
    suggestion: str

class CodeAnalysisResponse(BaseModel):
    vulnerabilities: list[Vulnerability]

# --- API Endpoints ---
@app.get("/status")
def get_status():
    """A simple health check endpoint."""
    return {"status": "ok"}

@app.post("/analyze", response_model=CodeAnalysisResponse)
def analyze_code(request: CodeAnalysisRequest):
    """
    Analyzes a given code snippet for vulnerabilities and suggests patches.
    """
    # Create a temporary file to store the code for analysis
    # This is necessary because our agents are designed to work with file paths.
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py') as temp_file:
        temp_file.write(request.code)
        temp_filepath = temp_file.name
    
    try:
        print(f"Analyzing code in temporary file: {temp_filepath}")
        
        # 1. Static Analysis Agent
        potential_flaws = static_analyzer.find_potential_flaws(temp_filepath)
        if not potential_flaws:
            return {"vulnerabilities": []}

        response_vulnerabilities = []

        for flaw in potential_flaws:
            print(f"Found potential flaw: {flaw['type']} at line {flaw['line']}")
            
            # 2. Semantic Analysis (Code Embedding)
            analysis = code_analyzer.analyze_snippet(flaw['code'])
            flaw['embedding'] = analysis['embedding']
            
            # 3. RL-based Patch Suggestion
            suggestion = patch_suggester.suggest_patch(flaw)
            
            response_vulnerabilities.append(
                Vulnerability(
                    type=flaw['type'],
                    line=flaw['line'],
                    code=flaw['code'],
                    suggestion=suggestion
                )
            )
        
        return {"vulnerabilities": response_vulnerabilities}

    except Exception as e:
        # In case of any error during analysis, return an HTTP 500
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            print(f"Cleaned up temporary file: {temp_filepath}")

# To run this API:
# 1. Install dependencies: pip install -r requirements.txt
# 2. Run from the project root: uvicorn src.api:app --reload
