import ast
from .code_analyzer import CodeAnalyzer
from .static_analysis_agent import StaticAnalysisAgent
from .knowledge_base_agent import KnowledgeBaseAgent
class VulnerabilityDetectorAgent:
    """
    An agent that detects security flaws in source code.
    Combines static analysis rules with a fine-tuned deep learning model for verification.
    """
    def __init__(self, config_path='config.yaml'):
        self.static_analyzer = StaticAnalysisAgent()
        # Initialize CodeAnalyzer in classifier mode to use the fine-tuned model
        self.dl_classifier = CodeAnalyzer(config_path=config_path, is_classifier=True)
        self.knowledge_base = KnowledgeBaseAgent() # Initialize KnowledgeBaseAgent
        print("VulnerabilityDetectorAgent initialized with Static Analyzer, DL Classifier, and Knowledge Base.")

    def find_vulnerabilities(self, file_path):
        """
        Scans a Python file for potential vulnerabilities using static analysis,
        then uses the deep learning model to verify potential flaws.
        """
        print(f"\n--- Scanning file: {file_path} for vulnerabilities ---")
        potential_flaws_static = self.static_analyzer.find_potential_flaws(file_path)

        confirmed_vulnerabilities = []

        if not potential_flaws_static:
            print("No potential flaws detected by static analysis.")
            return []

        print(f"Static analysis found {len(potential_flaws_static)} potential flaws. Verifying with DL model...")

        for flaw_candidate in potential_flaws_static:
            print(f"  Verifying static flaw: {flaw_candidate['type']} at line {flaw_candidate['line']}")
                        # Use the deep learning model to get a prediction for this snippet
            analysis_result = self.dl_classifier.analyze_snippet(flaw_candidate["code"])

            # Only add to confirmed vulnerabilities if the DL model also predicts it's vulnerable (label 1)
                        if analysis_result["prediction"] == 1:
                vulnerability = flaw_candidate.copy() # Make a copy to add DL details
                vulnerability["dl_prediction"] = analysis_result["prediction"]
                vulnerability["confidence_vulnerable"] = analysis_result["confidence_scores"][1] # Confidence for 'Vulnerable'
                confirmed_vulnerabilities.append(vulnerability)
                self.knowledge_base.add_vulnerability(vulnerability) # Add to KB
                print(f"  --> CONFIRMED VULNERABLE by DL model (Confidence: {vulnerability['confidence_vulnerable']:.2f})")
                        else:
                print(f"  --> NOT CONFIRMED by DL model (Confidence: {analysis_result['confidence_scores'][0]:.2f} for Not Vulnerable)")
        return confirmed_vulnerabilities

