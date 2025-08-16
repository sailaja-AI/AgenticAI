
from .static_analysis_agent import StaticAnalysisAgent
from .code_analyzer import CodeAnalyzer 
from .patch_suggester import PatchSuggester 

class KnowledgeBaseAgent:
    """
    The central orchestrator and knowledge repository.
    It coordinates the other agents and stores findings.
    """
    def __init__(self):
        print("--- Knowledge Base Agent Initializing ---")
        self.static_analyzer = StaticAnalysisAgent()
        self.semantic_analyzer = CodeAnalyzer()
        self.mitigation_agent = PatchSuggester()
        
        self.knowledge_base = {}
        print("--- All agents initialized. ---")

    def scan_and_mitigate(self, file_path):
        """
        Orchestrates the full scan-to-mitigation workflow.
        """
        print(f"\n--- Orchestrating scan for {file_path} ---")
        
        potential_flaws = self.static_analyzer.find_potential_flaws(file_path)
        if not potential_flaws:
            print("Static analysis found no potential flaws.")
            return

        print(f"Found {len(potential_flaws)} potential flaw(s). Handing over to Semantic Analyzer.")
        confirmed_vulnerabilities = []

        for flaw in potential_flaws:
            analysis = self.semantic_analyzer.analyze_snippet(flaw['code'])
            flaw['embedding'] = analysis['embedding']
            
            confirmed_vulnerabilities.append(flaw)

        if not confirmed_vulnerabilities:
            print("Semantic analysis did not confirm any vulnerabilities.")
            return

        self.knowledge_base[file_path] = confirmed_vulnerabilities
        print(f"Stored {len(confirmed_vulnerabilities)} confirmed vulnerabilities in the knowledge base.")

        for vuln in confirmed_vulnerabilities:
            suggestion = self.mitigation_agent.suggest_patch(vuln)
            print(f"\n--- Mitigation Suggestion for line {vuln['line']} ---")
            print(f"Code: {vuln['code']}")
            print(f"Suggestion: {suggestion}")
            print("-----------------------------------------")
