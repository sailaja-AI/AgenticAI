from .static_analysis_agent import StaticAnalysisAgent
from .code_analyzer import CodeAnalyzer
from .patch_suggester import PatchSuggesterAgent

class KnowledgeBaseAgent:
    """
    The central orchestrator for command-line scans.
    It coordinates the other agents and stores findings.
    """
    def __init__(self):
        print("Initializing agents for CLI scan...")
        self.static_analyzer = StaticAnalysisAgent()
        self.semantic_analyzer = CodeAnalyzer()
        self.mitigation_agent = PatchSuggesterAgent()
        self.knowledge_base = {}
        print("Agents initialized.")

    def scan_and_mitigate(self, file_path):
        """
        Orchestrates the full scan-to-mitigation workflow for a given file.
        """
        print(f"\n--- Orchestrating scan for {file_path} ---")
        
        potential_flaws = self.static_analyzer.find_potential_flaws(file_path)
        if not potential_flaws:
            print("Static analysis found no potential flaws.")
            return

        print(f"Found {len(potential_flaws)} potential flaw(s). Handing over to Semantic Analyzer and Patch Suggester.")
        confirmed_vulnerabilities = []

        for flaw in potential_flaws:
            # Add embedding to the flaw context
            analysis = self.semantic_analyzer.analyze_snippet(flaw['code'])
            flaw['embedding'] = analysis['embedding']
            
            # Store the confirmed vulnerability
            confirmed_vulnerabilities.append(flaw)

            # Get a patch suggestion for the vulnerability
            suggestion = self.mitigation_agent.suggest_patch(flaw)
            
            print(f"\n--- Mitigation Suggestion for line {flaw['line']} ---")
            print(f"  Code: {flaw['code']}")
            print(f"  Suggestion: {suggestion}")
            print("-----------------------------------------")

        if not confirmed_vulnerabilities:
            print("Analysis did not confirm any vulnerabilities.")
            return

        # Store results in the knowledge base
        self.knowledge_base[file_path] = confirmed_vulnerabilities
        print(f"\nStored {len(confirmed_vulnerabilities)} confirmed vulnerabilities in the knowledge base for {file_path}.")
