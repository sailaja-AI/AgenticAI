import sys
from .knowledge_base_agent import KnowledgeBaseAgent
def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src <file_to_scan>")
        sys.exit(1)

    file_to_scan = sys.argv[1]

    # Initialize the main orchestrator agent
    kb_agent = KnowledgeBaseAgent()

    # Start the workflow
    kb_agent.scan_and_mitigate(file_to_scan)
if __name__ == "__main__":
    main()