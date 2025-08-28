import ast
# Corrected Imports: Use absolute imports for modules within the 'src' directory
from src.code_analyzer import CodeAnalyzer
from src.static_analysis_agent import StaticAnalysisAgent
from src.knowledge_base_agent import KnowledgeBaseAgent

class VulnerabilityDetectorAgent:
    """
    An agent that detects security flaws in source code.
    Combines static analysis rules with a fine-tuned deep learning model for verification,
    and consults a knowledge base for mitigation suggestions.
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
        # This method is designed for scanning actual files.
        # The new method `detect_and_suggest_mitigation` below is for API input.
        print(f"\n--- Scanning file: {file_path} for vulnerabilities ---")
        potential_flaws_static = self.static_analyzer.find_potential_flaws(file_path)
        
        confirmed_vulnerabilities = []

        if not potential_flaws_static:
            print("No potential flaws detected by static analysis.")
            return []

        print(f"Static analysis found {len(potential_flaws_static)} potential flaws. Verifying with DL model...")

        for flaw_candidate in potential_flaws_static:
            print(f"  Verifying static flaw: {flaw_candidate['type']} at line {flaw_candidate['line']}")
            
            analysis_result = self.dl_classifier.analyze_snippet(flaw_candidate["code"])
            
            # --- CORRECTED INDENTATION START ---
            if analysis_result["prediction"] == 1:
                vulnerability = flaw_candidate.copy()
                vulnerability["dl_prediction"] = analysis_result["prediction"]
                vulnerability["confidence_vulnerable"] = analysis_result["confidence_scores"][1]
                confirmed_vulnerabilities.append(vulnerability)
                self.knowledge_base.add_vulnerability(vulnerability)
                print(f"  --> CONFIRMED VULNERABLE by DL model (Confidence: {vulnerability['confidence_vulnerable']:.2f})")
            else:
                print(f"  --> NOT CONFIRMED by DL model (Confidence: {analysis_result['confidence_scores'][0]:.2f} for Not Vulnerable)")
            # --- CORRECTED INDENTATION END ---
        
        return confirmed_vulnerabilities

    def detect_and_suggest_mitigation(self, code_snippet: str, file_path_hint: str = "<api_input_code>") -> dict:
        """
        Unified method to detect vulnerabilities in a given code snippet (string)
        and provide mitigation suggestions from the knowledge base.
        This method is designed to be called by the FastAPI endpoint.
        """
        print(f"\n--- Detecting vulnerabilities and suggesting mitigations for: {file_path_hint} ---")

        # 1. Run Static Analysis on the provided code string
        potential_flaws = self.static_analyzer.find_potential_flaws_in_string(code_snippet)

        results = {
            "overall_vulnerable": False,
            "detected_vulnerabilities": []
        }

        if not potential_flaws:
            print("No potential flaws detected by static analysis.")
            return results

        print(f"Static analysis found {len(potential_flaws)} potential flaws. Verifying with DL model and fetching mitigations...")

        for flaw_candidate in potential_flaws:
            # 2. Use Deep Learning Classifier to verify
            dl_analysis_result = self.dl_classifier.analyze_snippet(flaw_candidate["code"])

            if dl_analysis_result["prediction"] == 1: # Confirmed vulnerable by DL model
                results["overall_vulnerable"] = True

                vulnerability_details = flaw_candidate.copy()
                vulnerability_details["dl_prediction"] = dl_analysis_result["prediction"]
                vulnerability_details["confidence_vulnerable"] = dl_analysis_result["confidence_scores"][1]

                # 3. Get Mitigation Suggestion from Knowledge Base
                mitigation_suggestion = self.knowledge_base.get_mitigation_suggestion(flaw_candidate["type"])
                vulnerability_details["mitigation_suggestion"] = mitigation_suggestion

                # 4. (Future) Integrate RL agent here to get a concrete patch
                vulnerability_details["suggested_patch_code"] = "RL agent integration not yet complete for this type."

                results["detected_vulnerabilities"].append(vulnerability_details)
                self.knowledge_base.add_vulnerability(vulnerability_details) # Store confirmed findings

                print(f"  CONFIRMED: {flaw_candidate['type']} at line {flaw_candidate['line']} (DL Confidence: {vulnerability_details['confidence_vulnerable']:.2f})")
            else:
                print(f"  SKIPPED: {flaw_candidate['type']} at line {flaw_candidate['line']} (Not confirmed by DL, Confidence: {dl_analysis_result['confidence_scores'][0]:.2f})")

        return results

