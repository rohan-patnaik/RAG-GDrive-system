# backend/rag_system/utils/helpers.py
import uuid
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_uuid(prefix: str = "id") -> str:
    """
    Generates a unique identifier string with an optional prefix.

    Args:
        prefix: An optional prefix for the UUID.

    Returns:
        A string representing the unique identifier.
    """
    return f"{prefix}_{uuid.uuid4().hex}"


def get_project_root() -> Path:
    """
    Returns the project root directory.
    Assumes this file is within backend/rag_system/utils/
    Adjust if the file structure changes.
    """
    # Resolve the path of the current file -> .../utils/helpers.py
    # Then go up three levels to reach the project root (RAG-GDrive-system/)
    return Path(__file__).resolve().parent.parent.parent.parent


# Example of another helper function
def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncates a string to a maximum length, adding a suffix if truncated.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


if __name__ == "__main__":
    # Test generate_uuid
    uuid1 = generate_uuid()
    uuid2 = generate_uuid("doc")
    logger.info(f"Generated UUID 1: {uuid1}")
    logger.info(f"Generated UUID 2: {uuid2}")

    # Test get_project_root
    project_root = get_project_root()
    logger.info(f"Project root directory: {project_root}")
    # Verify by checking if a known file/folder exists at the root
    readme_path = project_root / "README.md"
    logger.info(f"README.md exists at root: {readme_path.exists()}")

    # Test truncate_text
    long_text = "This is a very long string that needs to be truncated for display purposes."
    short_text = "Short text."
    logger.info(f"Truncated long text: '{truncate_text(long_text, 30)}'")
    logger.info(f"Truncated short text: '{truncate_text(short_text, 30)}'")
    logger.info(f"Truncated exact length: '{truncate_text('12345', 5)}'")
    logger.info(f"Truncated with custom suffix: '{truncate_text(long_text, 20, ' (...)')}'")
