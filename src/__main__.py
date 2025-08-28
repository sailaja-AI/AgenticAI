import os
import argparse
# Corrected Imports: Use absolute imports for modules within the 'src' directory
from src.agent import VulnerabilityDetectorAgent

def main():
    parser = argparse.ArgumentParser(description="Agentic AI for Code Vulnerability Detection.")
    parser.add_argument("file_or_code_snippet", type=str,
                        help="Path to the code file to scan, or a direct code snippet enclosed in quotes.")
    parser.add_argument("--is_file", action="store_true",
                        help="Treat the input as a file path. Otherwise, it's treated as a code snippet string.")
    
    args = parser.parse_args()

    # Initialize the main agent
    detector_agent = VulnerabilityDetectorAgent()

    if args.is_file:
        file_to_scan = args.file_or_code_snippet
        if not os.path.exists(file_to_scan):
            print(f"Error: File not found at {file_to_scan}")
            return
        
        # Read code from file for processing by the agent
        with open(file_to_scan, 'r', encoding='utf-8') as f:
            code_content = f.read()
        
        print(f"\n--- Initiating scan for file: {file_to_scan} ---")
        results = detector_agent.detect_and_suggest_mitigation(code_content, file_path_hint=file_to_scan)
    else:
        code_snippet = args.file_or_code_snippet
        if not code_snippet.strip():
            print("Error: Code snippet cannot be empty.")
            return

        print(f"\n--- Initiating scan for code snippet ---")
        results = detector_agent.detect_and_suggest_mitigation(code_snippet)
    
    # Print formatted results
    print("\n--- Scan Results ---")
    if results["overall_vulnerable"]:
        print("Vulnerabilities Detected!")
        for i, vuln in enumerate(results["detected_vulnerabilities"]):
            print(f"\n[{i+1}] Type: {vuln['type']}")
            print(f"    File: {vuln['file']}")
            print(f"    Line: {vuln['line']}")
            print(f"    Code: '''{vuln['code']}'''")
            print(f"    DL Confidence (Vulnerable): {vuln['confidence_vulnerable']:.2f}")
            print(f"    Mitigation: {vuln['mitigation_suggestion']}")
            if vuln.get("suggested_patch_code"):
                print(f"    Suggested Patch:\n'''\n{vuln['suggested_patch_code']}\n'''")
    else:
        print("No confirmed vulnerabilities found.")
    print("--- Scan Complete ---")

if __name__ == "__main__":
    main()