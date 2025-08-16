
import ast

class StaticAnalysisAgent:
    """
    An agent that detects potential security flaws using rule-based static analysis.
    This agent identifies code patterns that *might* be vulnerabilities.
    """
    def __init__(self):
        print("Static Analysis Agent initialized.")

    def find_potential_flaws(self, file_path):
        """
        Scans a Python file for potential vulnerabilities based on AST patterns.
        """
        with open(file_path, 'r') as f:
            source_code = f.read()
        
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            print(f"Could not parse {file_path}: {e}")
            return []

        potential_flaws = []

        for node in ast.walk(tree):
            # Example Rule: Detect `subprocess.call(..., shell=True)`
            if (isinstance(node, ast.Call) and
                getattr(node.func, 'attr', '') == 'call' and
                isinstance(getattr(node.func, 'value', None), ast.Name) and
                getattr(node.func, 'value', None).id == 'subprocess'):
                
                for keyword in node.keywords:
                    if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                        flaw = {
                            "type": "INSECURE_SUBPROCESS",
                            "file": file_path,
                            "line": node.lineno,
                            "code": ast.unparse(node)
                        }
                        potential_flaws.append(flaw)
                        
        return potential_flaws
