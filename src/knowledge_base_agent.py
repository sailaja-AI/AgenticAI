import json
import os

class KnowledgeBaseAgent:
    """
    The central orchestrator for command-line scans and a repository for findings.
    It coordinates other agents, stores detected vulnerabilities, and manages
    known mitigation strategies.
    """
    def __init__(self, findings_file="findings.json"):
        self.findings_file = findings_file
        self.vulnerabilities = self._load_findings()
        self.mitigation_strategies = self._load_mitigation_strategies() # New: Load known mitigation strategies
        print(f"KnowledgeBaseAgent initialized. Loaded {len(self.vulnerabilities)} existing findings.")
        print(f"Loaded {len(self.mitigation_strategies)} mitigation strategies.")

    def _load_findings(self):
        """Loads vulnerabilities from a JSON file."""
        if os.path.exists(self.findings_file):
            with open(self.findings_file, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: {self.findings_file} is corrupted. Starting with empty findings.")
                    return []
        return []

    def _save_findings(self):
        """Saves current vulnerabilities to a JSON file."""
        with open(self.findings_file, 'w', encoding='utf-8') as f:
            json.dump(self.vulnerabilities, f, indent=4)
        print(f"Findings saved to {self.findings_file}")

    def _load_mitigation_strategies(self, strategies_file="mitigation_strategies.json"):
        """Loads predefined mitigation strategies from a JSON file."""
        # This would be a static file you define, mapping vulnerability types to general patch advice
        if os.path.exists(strategies_file):
            with open(strategies_file, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: {strategies_file} is corrupted. Starting with empty strategies.")
                    return {}
        return {} # Returns an empty dict if file not found

    def add_vulnerability(self, vulnerability_details):
        """Adds a new vulnerability finding to the knowledge base."""
        # Ensure it's not a duplicate (e.g., same type, file, line, and code)
        if vulnerability_details not in self.vulnerabilities:
            self.vulnerabilities.append(vulnerability_details)
            self._save_findings()
            print(f"Added new vulnerability: {vulnerability_details['type']} at {vulnerability_details['file']}:{vulnerability_details['line']}")
        else:
            print(f"Duplicate vulnerability found, not adding: {vulnerability_details['type']} at {vulnerability_details['file']}:{vulnerability_details['line']}")


    def get_vulnerabilities(self, file_path=None, vulnerability_type=None):
        """Retrieves stored vulnerabilities, optionally filtered by file or type."""
        filtered_vulnerabilities = self.vulnerabilities
        if file_path:
            filtered_vulnerabilities = [v for v in filtered_vulnerabilities if v.get('file') == file_path]
        if vulnerability_type:
            filtered_vulnerabilities = [v for v in filtered_vulnerabilities if v.get('type') == vulnerability_type]
        return filtered_vulnerabilities

    def get_mitigation_suggestion(self, vulnerability_type):
        """
        Provides a general mitigation suggestion for a given vulnerability type.
        This would be used as a hint for the RL agent or for user reports.
        """
        return self.mitigation_strategies.get(vulnerability_type, "No specific mitigation strategy known for this type.")

    def update_vulnerability_status(self, vulnerability_id, new_status="patched", patch_info=None):
        """Updates the status of a vulnerability (e.g., after patching)."""
        # You'd need a unique ID for each vulnerability, perhaps a hash of its details.
        # For simplicity, this is a placeholder.
        print(f"Updating status for vulnerability ID {vulnerability_id} to {new_status}")
        # Implement logic to find and update the vulnerability
        self._save_findings() # Save changes
    
    # You might add methods for:
    # - `add_mitigation_strategy(type, description, example_patch)`
    # - `report_findings()` to generate a summary report