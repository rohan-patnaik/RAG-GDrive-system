# scripts/ingest_documents.py
"""
Example script to trigger document ingestion using the CLI.
This is useful for scheduled tasks or simple batch ingestion.
"""
import subprocess
import sys
from pathlib import Path

# Assuming the script is run from the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python" # Adjust if your venv is elsewhere or named differently
RAG_SYSTEM_CLI = "rag-system" # If installed via setup.py and in PATH from venv
# Alternatively, directly call the main.py script:
# RAG_SYSTEM_CLI_MAIN = str(PROJECT_ROOT / "backend" / "rag_system" / "main.py")


def run_ingestion(documents_path: str, recursive: bool = True, patterns: str = "*.txt"):
    """
    Runs the ingestion CLI command.

    Args:
        documents_path: Path to the documents to ingest.
        recursive: Whether to ingest recursively.
        patterns: Comma-separated glob patterns for file matching.
    """
    print(f"Starting ingestion for path: {documents_path}")
    print(f"Recursive: {recursive}, Patterns: {patterns}")

    command = [
        str(VENV_PYTHON), # Use python from venv to ensure correct environment
        "-m", "rag_system.main", # Run the main module that exposes the Click app
        "ingest",
        documents_path,
    ]
    if recursive:
        command.append("--recursive")
    if patterns:
        command.extend(["--patterns", patterns]) # Click handles comma-separated string for multiple patterns if option is defined that way

    # If rag-system is installed and in PATH (e.g. after pip install -e .)
    # command = [
    #     RAG_SYSTEM_CLI,
    #     "ingest",
    #     documents_path,
    # ]
    # if recursive:
    #     command.append("--recursive")
    # if patterns:
    #     command.extend(["--patterns", patterns])


    try:
        # Set PYTHONPATH if running main.py directly and not as an installed package
        env = {"PYTHONPATH": str(PROJECT_ROOT / "backend")}
        # For running 'rag-system' as installed command, env might not be needed if venv is active.

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print("Ingestion process completed successfully.")
            print("Output:\n", stdout)
        else:
            print(f"Ingestion process failed with code {process.returncode}.")
            print("Error Output:\n", stderr)
            if stdout:
                print("Standard Output (if any):\n", stdout)

    except FileNotFoundError:
        print(f"Error: Could not find Python executable at {VENV_PYTHON} or CLI '{RAG_SYSTEM_CLI}'.")
        print("Please ensure the virtual environment is set up and activated, or the package is installed.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # --- Configuration for the script ---
    # Path to the documents you want to ingest.
    # This path should be accessible from where this script is run.
    # If using Docker, this path might need to be relative to a mounted volume.
    DOCS_TO_INGEST = str(PROJECT_ROOT / "data" / "sample_documents")

    # Ingestion options
    INGEST_RECURSIVELY = True
    FILE_PATTERNS = "*.txt,*.md" # Example: ingest text and markdown files

    # Check if the documents path exists
    if not Path(DOCS_TO_INGEST).exists():
        print(f"Error: Document path '{DOCS_TO_INGEST}' does not exist.")
        print("Please provide a valid path to your documents.")
        sys.exit(1)

    run_ingestion(DOCS_TO_INGEST, recursive=INGEST_RECURSIVELY, patterns=FILE_PATTERNS)

    # Example for another directory:
    # custom_docs_path = "/path/to/your/other_documents"
    # if Path(custom_docs_path).exists():
    #     run_ingestion(custom_docs_path, recursive=False, patterns="*.pdf") # Assuming PDF support added
    # else:
    #     print(f"Skipping ingestion for '{custom_docs_path}' as it does not exist.")
