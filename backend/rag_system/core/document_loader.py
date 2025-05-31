# backend/rag_system/core/document_loader.py
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from rag_system.models.schemas import Document, DocumentMetadata

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads documents from various sources.
    Initially supports loading text files from a directory.
    """

    def __init__(self) -> None:
        pass

    def load_from_directory(
        self,
        source_directory: Path | str,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[Document]:
        """
        Loads documents from a specified directory.

        Args:
            source_directory: The path to the directory containing documents.
            file_patterns: A list of glob patterns to match files (e.g., ["*.txt", "*.md"]).
                           Defaults to ["*.txt"] if None.
            recursive: Whether to search for files in subdirectories.

        Returns:
            A list of Document objects.

        Raises:
            FileNotFoundError: If the source_directory does not exist or is not a directory.
        """
        if file_patterns is None:
            file_patterns = ["*.txt"]

        source_path = Path(source_directory)
        if not source_path.exists() or not source_path.is_dir():
            msg = f"Source directory not found or is not a directory: {source_directory}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        documents: List[Document] = []
        logger.info(
            f"Starting document load from directory: {source_directory} "
            f"with patterns: {file_patterns}, recursive: {recursive}"
        )

        for pattern in file_patterns:
            glob_method = source_path.rglob if recursive else source_path.glob
            for file_path in glob_method(pattern):
                if file_path.is_file():
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        metadata = DocumentMetadata(
                            source_id=str(file_path.resolve()),
                            filename=file_path.name,
                            path=str(file_path.parent.resolve()),
                            # Add more metadata as needed, e.g., file_size, creation_date
                        )
                        doc = Document(
                            id=str(file_path.stem), # Use filename without extension as ID, or generate UUID
                            content=content,
                            metadata=metadata,
                        )
                        documents.append(doc)
                        logger.debug(f"Successfully loaded document: {file_path.name}")
                    except Exception as e:
                        logger.error(
                            f"Failed to load or read document {file_path.name}: {e}",
                            exc_info=True,
                        )
        logger.info(f"Loaded {len(documents)} documents from {source_directory}.")
        return documents

    # Placeholder for future PDF loading
    def load_pdf(self, file_path: Path | str) -> Optional[Document]:
        """
        Loads a single PDF document.
        (This is a placeholder and needs a PDF parsing library like PyPDF2 or pdfminer.six)
        """
        logger.warning(
            "PDF loading is not yet implemented. "
            f"Skipping PDF: {file_path}"
        )
        # Example (requires a PDF library):
        # try:
        #     from PyPDF2 import PdfReader # or another library
        #     reader = PdfReader(file_path)
        #     text = ""
        #     for page in reader.pages:
        #         text += page.extract_text()
        #     metadata = DocumentMetadata(source_id=str(Path(file_path).resolve()), filename=Path(file_path).name)
        #     return Document(id=Path(file_path).stem, content=text, metadata=metadata)
        # except ImportError:
        #     logger.error("PyPDF2 library not found. Please install it to load PDFs.")
        #     return None
        # except Exception as e:
        #     logger.error(f"Failed to load PDF {file_path}: {e}")
        #     return None
        return None


# Example Usage:
if __name__ == "__main__":
    from rag_system.config.logging_config import setup_logging
    setup_logging("DEBUG")

    loader = DocumentLoader()
    sample_dir = Path(__file__).parent.parent.parent.parent / "data" / "sample_documents"
    # Create dummy files for testing
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "doc1.txt").write_text("This is the first sample document about AI.")
    (sample_dir / "doc2.txt").write_text("The second document discusses Python programming.")
    (sample_dir / "subdir").mkdir(exist_ok=True)
    (sample_dir / "subdir" / "doc3.txt").write_text("A document in a subdirectory.")


    try:
        docs = loader.load_from_directory(sample_dir, file_patterns=["*.txt"], recursive=True)
        if docs:
            for doc_item in docs:
                logger.info(f"Loaded Document ID: {doc_item.id}, Source: {doc_item.metadata.source_id}")
                logger.info(f"Content Preview: {doc_item.content[:100]}...")
        else:
            logger.info("No documents loaded.")

    except FileNotFoundError as e:
        logger.error(e)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

