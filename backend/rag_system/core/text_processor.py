# backend/rag_system/core/text_processor.py
import logging
import re
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_system.models.schemas import Document, DocumentChunk, DocumentMetadata

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Processes text content, including cleaning and chunking.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        """
        Initializes the TextProcessor.

        Args:
            chunk_size: The target size of each text chunk (in characters).
            chunk_overlap: The number of characters to overlap between chunks.
            separators: Optional list of custom separators for splitting text.
                        If None, default separators of RecursiveCharacterTextSplitter are used.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""] # Default LangChain separators

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False, # Treat separators literally
            separators=self.separators,
        )
        logger.info(
            f"TextProcessor initialized with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}."
        )

    def clean_text(self, text: str) -> str:
        """
        Performs basic text cleaning.

        Args:
            text: The input string.

        Returns:
            The cleaned string.
        """
        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing whitespace
        text = text.strip()
        # Add more cleaning rules as needed (e.g., remove special characters, normalize unicode)
        return text

    def split_document(self, document: Document) -> List[DocumentChunk]:
        """
        Splits a Document into multiple DocumentChunks.

        Args:
            document: The Document object to split.

        Returns:
            A list of DocumentChunk objects.
        """
        if not document.content:
            logger.warning(f"Document {document.id} has no content. Skipping splitting.")
            return []

        cleaned_content = self.clean_text(document.content)
        if not cleaned_content:
            logger.warning(f"Document {document.id} has no content after cleaning. Skipping splitting.")
            return []

        # LangChain's text_splitter expects simple text or LangChain Document objects.
        # We'll pass the text and then construct our DocumentChunk objects.
        split_texts = self.text_splitter.split_text(cleaned_content)

        chunks: List[DocumentChunk] = []
        for i, text_chunk in enumerate(split_texts):
            chunk_id = f"{document.id}_chunk_{i}"
            # Create a copy of the document's metadata for the chunk
            # and add chunk-specific information.
            chunk_metadata = document.metadata.model_copy(deep=True)
            chunk_metadata.chunk_number = i
            chunk_metadata.total_chunks = len(split_texts)
            # You might want to add other metadata like character start/end positions

            chunk = DocumentChunk(
                id=chunk_id,
                document_id=document.id,
                content=text_chunk,
                metadata=chunk_metadata,
                embedding=None, # Embedding will be added later
            )
            chunks.append(chunk)

        logger.debug(
            f"Split document {document.id} into {len(chunks)} chunks."
        )
        return chunks

    def split_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """
        Splits a list of Documents into DocumentChunks.

        Args:
            documents: A list of Document objects.

        Returns:
            A list of all DocumentChunk objects from all documents.
        """
        all_chunks: List[DocumentChunk] = []
        for doc in documents:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)
        logger.info(
            f"Processed {len(documents)} documents, resulting in {len(all_chunks)} chunks."
        )
        return all_chunks


# Example Usage:
if __name__ == "__main__":
    from rag_system.config.logging_config import setup_logging
    setup_logging("DEBUG")

    processor = TextProcessor(chunk_size=100, chunk_overlap=20)

    sample_text = (
        "This is a sample text. It has multiple sentences.\n\n"
        "This is a new paragraph. It talks about AI and machine learning. "
        "Python is a great language for these tasks.    Many libraries exist."
    )
    cleaned = processor.clean_text(sample_text)
    logger.info(f"Cleaned text: '{cleaned}'")

    doc_meta = DocumentMetadata(source_id="sample_doc_1", filename="sample.txt")
    sample_doc = Document(id="doc1", content=sample_text, metadata=doc_meta)

    chunks = processor.split_document(sample_doc)
    for i, chunk_item in enumerate(chunks):
        logger.info(f"Chunk {i} ID: {chunk_item.id}")
        logger.info(f"Chunk {i} Content: '{chunk_item.content}'")
        logger.info(f"Chunk {i} Metadata: {chunk_item.metadata.model_dump_json(indent=2)}")

    docs_list = [
        Document(id="doc1", content="First document content. Short.", metadata=doc_meta),
        Document(id="doc2", content=sample_text, metadata=doc_meta),
    ]
    all_doc_chunks = processor.split_documents(docs_list)
    logger.info(f"Total chunks from list of documents: {len(all_doc_chunks)}")

