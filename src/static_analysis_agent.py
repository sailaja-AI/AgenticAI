import ast
import os 
import re

class StaticAnalysisAgent:
    """
    An agent that detects potential security flaws using rule-based static analysis.
    This agent identifies code patterns that *might* be vulnerabilities.
    """
    def __init__(self):
        print("Static Analysis Agent initialized.")

    def _analyze_code_ast(self, source_code, file_path="<string_input>"):
        """Internal helper to parse code and find flaws, reusable for file or string."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            print(f"Could not parse code from {file_path}: {e}")
            return []

        potential_flaws = []

        for node in ast.walk(tree):
            # 1. Rule: Detect `subprocess.call(..., shell=True)`
            if (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Attribute) and
                node.func.attr == 'call' and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'subprocess'):
                
                for keyword in node.keywords:
                    if keyword.arg == 'shell' and isinstance(keyword.value, (ast.Constant, ast.NameConstant)) and keyword.value.value is True:
                        potential_flaws.append({
                            "type": "COMMAND_INJECTION_SHELL_TRUE",
                            "file": file_path,
                            "line": node.lineno,
                            "code": ast.unparse(node).strip()
                        })
            
            # 2. Rule: Detect potential SQL Injection (sqlite3.Cursor.execute with f-string or concat)
            if (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Attribute) and
                node.func.attr == 'execute' and
                isinstance(node.func.value, ast.Name) and
                (node.func.value.id.lower() == 'cursor' or node.func.value.id.lower() == 'cur')):
                
                if node.args:
                    query_arg = node.args[0]
                    if (isinstance(query_arg, ast.JoinedStr) or
                        (isinstance(query_arg, ast.BinOp) and isinstance(query_arg.op, ast.Add) and
                         (isinstance(query_arg.left, (ast.Str, ast.Constant, ast.FormattedValue)) or
                          isinstance(query_arg.right, (ast.Str, ast.Constant, ast.FormattedValue))))):
                        potential_flaws.append({
                            "type": "POTENTIAL_SQL_INJECTION",
                            "file": file_path,
                            "line": node.lineno,
                            "code": ast.unparse(node).strip()
                        })

            # 3. Rule: Detect potential Path Traversal (os.path.join with dynamic input)
            if (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Attribute) and
                node.func.attr == 'join' and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'os'):
                
                if any(not isinstance(arg, (ast.Str, ast.Constant)) for arg in node.args):
                    potential_flaws.append({
                        "type": "POTENTIAL_PATH_TRAVERSAL",
                        "file": file_path,
                        "line": node.lineno,
                        "code": ast.unparse(node).strip()
                    })

            # 4. Rule: Detect `eval()` or `exec()` misuse
            if (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and
                node.func.id in ['eval', 'exec']):
                
                potential_flaws.append({
                    "type": "CODE_EXECUTION_EVAL_EXEC",
                    "file": file_path,
                    "line": node.lineno,
                    "code": ast.unparse(node).strip()
                })

            # 5. Rule: Basic Hardcoded Password detection (variable assignment)
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and re.match(r'.*(passw(or)?d|api_key|secret|token).*', target.id, re.IGNORECASE):
                        if isinstance(node.value, (ast.Str, ast.Constant)):
                            if len(node.value.value) > 4 and "your_password" not in node.value.value.lower():
                                potential_flaws.append({
                                    "type": "HARDCODED_CREDENTIALS",
                                    "file": file_path,
                                    "line": node.lineno,
                                    "code": ast.unparse(node).strip()
                                })
                                
        return potential_flaws

    def find_potential_flaws(self, file_path):
        """Scans a Python file for potential vulnerabilities based on AST patterns."""
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        return self._analyze_code_ast(source_code, file_path)

    def find_potential_flaws_in_string(self, code_string):
        """Scans a Python code string for potential vulnerabilities based on AST patterns."""
        return self._analyze_code_ast(code_string, "<string_input>") # Use a generic filename for string input
