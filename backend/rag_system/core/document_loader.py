# backend/rag_system/core/document_loader.py
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import mimetypes # For guessing content type

from rag_system.models.schemas import Document, DocumentMetadata
from rag_system.utils.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    Loads documents from various sources.
    Currently supports loading from a local directory and from in-memory bytes.
    """

    def __init__(self) -> None:
        """Initializes the DocumentLoader."""
        # No specific initialization needed for now
        pass

    def _create_document_from_content(
        self,
        content: str,
        filename: str,
        source_id_prefix: str,
        file_path: Optional[str] = None,
        file_size: Optional[int] = None,
        content_type: Optional[str] = None,
    ) -> Document:
        """Helper to create a Document object."""
        doc_id = f"{source_id_prefix}_{Path(filename).stem.replace(' ', '_')}"
        metadata = DocumentMetadata(
            source_id=doc_id,
            filename=filename,
            path=file_path,
            file_size=file_size,
            content_type=content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream",
            # created_at and modified_at could be set here if available
        )
        return Document(id=doc_id, content=content, metadata=metadata)

    def load_document_from_bytes(
        self, content_bytes: bytes, filename: str, source_id_prefix: str = "upload"
    ) -> Document:
        """
        Loads a single document from its byte content.

        Args:
            content_bytes: The byte content of the file.
            filename: The original name of the file.
            source_id_prefix: A prefix for the document's source_id.

        Returns:
            A Document object.

        Raises:
            DocumentProcessingError: If the content cannot be decoded.
        """
        logger.info(f"Loading document from bytes: {filename}")
        try:
            # Attempt to decode as UTF-8, common for text files
            content_str = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            logger.error(f"Could not decode file {filename} as UTF-8. Attempting latin-1.")
            try:
                content_str = content_bytes.decode("latin-1") # Fallback for other encodings
            except UnicodeDecodeError as e:
                msg = f"Failed to decode content for file {filename} with UTF-8 and latin-1: {e}"
                logger.error(msg)
                raise DocumentProcessingError(msg)

        if not content_str.strip():
            msg = f"File {filename} has no text content after decoding."
            logger.warning(msg)
            # Depending on requirements, you might raise an error or return a Document with empty content
            # For now, let's proceed but it might result in no chunks.

        return self._create_document_from_content(
            content=content_str,
            filename=filename,
            source_id_prefix=source_id_prefix,
            file_size=len(content_bytes),
            # content_type will be guessed by _create_document_from_content
        )


    def load_from_directory(
        self,
        source_directory: str,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[Document]:
        """
        Loads documents from a specified directory matching given patterns.

        Args:
            source_directory: The path to the directory.
            file_patterns: A list of glob patterns (e.g., ["*.txt", "*.md"]).
                           Defaults to ["*.txt"] if None.
            recursive: Whether to search subdirectories.

        Returns:
            A list of Document objects.

        Raises:
            FileNotFoundError: If the source directory does not exist.
            DocumentProcessingError: If no files are found or other processing issues occur.
        """
        if file_patterns is None:
            file_patterns = ["*.txt"] # Default to .txt if no patterns provided

        source_path = Path(source_directory)
        if not source_path.exists() or not source_path.is_dir():
            msg = f"Source directory not found or is not a directory: {source_directory}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(
            f"Loading documents from: {source_directory}, patterns: {file_patterns}, recursive: {recursive}"
        )
        documents: List[Document] = []
        files_processed_count = 0

        glob_method = source_path.rglob if recursive else source_path.glob

        for pattern in file_patterns:
            for file_path_obj in glob_method(pattern):
                if file_path_obj.is_file():
                    files_processed_count += 1
                    logger.debug(f"Processing file: {file_path_obj}")
                    try:
                        with open(file_path_obj, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()

                        if not content.strip():
                            logger.warning(f"File {file_path_obj.name} is empty or contains only whitespace. Skipping.")
                            continue

                        doc = self._create_document_from_content(
                            content=content,
                            filename=file_path_obj.name,
                            source_id_prefix=str(source_path.name), # Use dir name as prefix
                            file_path=str(file_path_obj.resolve()),
                            file_size=file_path_obj.stat().st_size,
                            content_type=mimetypes.guess_type(file_path_obj.name)[0] or "application/octet-stream"
                        )
                        documents.append(doc)
                    except Exception as e:
                        logger.error(f"Failed to process file {file_path_obj}: {e}", exc_info=True)
                        # Optionally, collect these errors to report them

        if not documents and files_processed_count > 0:
            logger.warning(
                f"Processed {files_processed_count} files, but no valid documents were loaded (e.g., all empty or failed to decode)."
            )
        elif not documents:
            logger.warning(f"No files found matching patterns {file_patterns} in {source_directory}")
            # raise DocumentProcessingError(f"No files found for patterns {file_patterns} in {source_directory}")

        logger.info(f"Successfully loaded {len(documents)} documents from directory.")
        return documents

    # Placeholder for future PDF/other type loading
    def _load_specific_file_type(self, file_path: Path, file_content: bytes) -> Optional[str]:
        """Loads content from specific file types like PDF, DOCX."""
        file_extension = file_path.suffix.lower()
        if file_extension == ".pdf":
            # Placeholder for PDF parsing logic
            # from PyPDF2 import PdfReader
            # reader = PdfReader(io.BytesIO(file_content))
            # text = ""
            # for page in reader.pages:
            #    text += page.extract_text() or ""
            # return text
            logger.warning(f"PDF processing for {file_path.name} is not yet fully implemented.")
            return file_content.decode("utf-8", errors="ignore") # Basic attempt
        # Add other file types (docx, etc.) here
        return None

# Example Usage (for testing this file directly)
if __name__ == "__main__":
    loader = DocumentLoader()
    sample_dir_path = Path(__file__).resolve().parent.parent.parent / "data" / "sample_documents"

    print(f"\n--- Testing load_from_directory on {sample_dir_path} ---")
    if sample_dir_path.exists():
        try:
            docs_from_dir = loader.load_from_directory(str(sample_dir_path), file_patterns=["*.txt"], recursive=True)
            for doc_item in docs_from_dir:
                print(f"Loaded from dir: {doc_item.metadata.filename}, ID: {doc_item.id}, Size: {len(doc_item.content)}")
        except Exception as e:
            print(f"Error loading from directory: {e}")
    else:
        print(f"Sample directory {sample_dir_path} not found for testing.")

    print("\n--- Testing load_document_from_bytes ---")
    try:
        test_filename = "test_upload.txt"
        test_content_bytes = "This is content from an uploaded file.\nIt has multiple lines.".encode("utf-8")
        doc_from_bytes = loader.load_document_from_bytes(test_content_bytes, test_filename)
        print(f"Loaded from bytes: {doc_from_bytes.metadata.filename}, ID: {doc_from_bytes.id}, Size: {len(doc_from_bytes.content)}")
        print(f"Content type: {doc_from_bytes.metadata.content_type}")

        test_filename_empty = "empty_upload.txt"
        test_content_empty_bytes = "   ".encode("utf-8")
        doc_empty_from_bytes = loader.load_document_from_bytes(test_content_empty_bytes, test_filename_empty)
        print(f"Loaded empty from bytes: {doc_empty_from_bytes.metadata.filename}, Content: '{doc_empty_from_bytes.content}'")

    except Exception as e:
        print(f"Error loading from bytes: {e}")
