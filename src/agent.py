import ast
from .code_analyzer import CodeAnalyzer

class VulnerabilityDetectorAgent:
    """
    An agent that detects security flaws in source code.
    """
    def __init__(self):
        self.analyzer = CodeAnalyzer()

    def find_vulnerabilities(self, file_path):
        """
        Scans a Python file for potential vulnerabilities.
        """
        with open(file_path, 'r') as f:
            source_code = f.read()
        
        tree = ast.parse(source_code)
        vulnerabilities = []

        for node in ast.walk(tree):
            # Example: Detect `subprocess.call(..., shell=True)`
            if (isinstance(node, ast.Call) and
                getattr(node.func, 'attr', '') == 'call' and
                isinstance(getattr(node.func, 'value', None), ast.Name) and
                getattr(node.func, 'value', None).id == 'subprocess'):
                
                for keyword in node.keywords:
                    if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                        vulnerability = {
                            "type": "INSECURE_SUBPROCESS",
                            "file": file_path,
                            "line": node.lineno,
                            "code": ast.unparse(node)
                        }
                        # Use the deep learning model to get more context
                        analysis_result = self.analyzer.analyze_snippet(vulnerability["code"])
                        vulnerability["embedding"] = analysis_result["embedding"]
                        vulnerabilities.append(vulnerability)
                        
        return vulnerabilities

