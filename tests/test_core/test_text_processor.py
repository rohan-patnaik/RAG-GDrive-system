# tests/test_core/test_text_processor.py
import pytest
from rag_system.core.text_processor import TextProcessor
from rag_system.models.schemas import Document, DocumentMetadata, DocumentChunk


@pytest.fixture
def default_text_processor() -> TextProcessor:
    return TextProcessor(chunk_size=100, chunk_overlap=20)


def test_clean_text(default_text_processor: TextProcessor):
    """Test basic text cleaning."""
    text = "  This is a    test string with   extra spaces. \nAnd new lines.  "
    expected = "This is a test string with extra spaces. \nAnd new lines."
    assert default_text_processor.clean_text(text) == expected

    text_empty = "   "
    expected_empty = ""
    assert default_text_processor.clean_text(text_empty) == expected_empty


def test_split_document_simple(default_text_processor: TextProcessor):
    """Test splitting a simple document."""
    doc_meta = DocumentMetadata(source_id="test_doc", filename="test.txt")
    doc = Document(
        id="doc1",
        content="This is the first sentence. This is the second sentence. This is the third sentence, which is a bit longer.",
        metadata=doc_meta,
    )
    # With chunk_size=100, overlap=20, this should split.
    # "This is the first sentence. This is the second sentence. This is the third sentence, which is a bit " (99 chars)
    # "sentence, which is a bit longer." (30 chars, but overlap makes the second chunk start earlier)

    chunks = default_text_processor.split_document(doc)
    assert len(chunks) > 0 # Exact number depends on splitter logic with small text

    for i, chunk in enumerate(chunks):
        assert isinstance(chunk, DocumentChunk)
        assert chunk.id == f"doc1_chunk_{i}"
        assert chunk.document_id == "doc1"
        assert len(chunk.content) > 0
        assert len(chunk.content) <= default_text_processor.chunk_size # Or slightly more due to how splitters work
        assert chunk.metadata.source_id == "test_doc"
        assert chunk.metadata.chunk_number == i
        assert chunk.metadata.total_chunks == len(chunks)

    if len(chunks) > 1:
        # Check overlap if possible (hard to assert precisely without knowing exact split points)
        # For RecursiveCharacterTextSplitter, it tries to split on separators.
        # Example: chunk1 ends with "sentence. This is the second sentence."
        # chunk2 starts with "second sentence. This is the third"
        pass # Precise overlap check is complex for this splitter


def test_split_document_with_separators():
    """Test splitting with custom separators and behavior."""
    processor = TextProcessor(
        chunk_size=50,
        chunk_overlap=10,
        separators=["\n\n", "\n", ". ", " "],
    )
    text = "Paragraph one.\n\nParagraph two starts here. It has a few sentences.\nThird paragraph."
    doc_meta = DocumentMetadata(source_id="sep_doc", filename="sep.txt")
    doc = Document(id="doc_sep", content=text, metadata=doc_meta)

    chunks = processor.split_document(doc)
    assert len(chunks) >= 3 # Expecting splits at \n\n and potentially within paragraphs

    # Verify content is split logically
    # print([c.content for c in chunks]) # For debugging
    assert "Paragraph one." in chunks[0].content
    if len(chunks) > 1:
      assert "Paragraph two starts here." in chunks[1].content or "Paragraph one." in chunks[1].content # due to overlap
    if len(chunks) > 2:
      assert "Third paragraph." in chunks[-1].content or "few sentences." in chunks[-1].content


def test_split_document_empty_content(default_text_processor: TextProcessor):
    """Test splitting a document with empty or whitespace-only content."""
    doc_meta = DocumentMetadata(source_id="empty_doc", filename="empty.txt")
    doc_empty = Document(id="doc_empty", content="", metadata=doc_meta)
    doc_whitespace = Document(id="doc_ws", content="   \n   ", metadata=doc_meta)

    chunks_empty = default_text_processor.split_document(doc_empty)
    assert len(chunks_empty) == 0

    chunks_whitespace = default_text_processor.split_document(doc_whitespace)
    assert len(chunks_whitespace) == 0


def test_split_document_content_shorter_than_chunk_size(
    default_text_processor: TextProcessor,
):
    """Test splitting document with content shorter than chunk_size."""
    short_content = "This is a short document."
    doc_meta = DocumentMetadata(source_id="short_doc", filename="short.txt")
    doc = Document(id="doc_short", content=short_content, metadata=doc_meta)

    chunks = default_text_processor.split_document(doc)
    assert len(chunks) == 1
    assert chunks[0].content == short_content
    assert chunks[0].metadata.chunk_number == 0
    assert chunks[0].metadata.total_chunks == 1


def test_split_documents_list(default_text_processor: TextProcessor):
    """Test splitting a list of documents."""
    doc_meta1 = DocumentMetadata(source_id="multi_doc1", filename="multi1.txt")
    doc1 = Document(
        id="multi1",
        content="Document one, first part. Document one, second part which is longer to ensure splitting.",
        metadata=doc_meta1,
    )
    doc_meta2 = DocumentMetadata(source_id="multi_doc2", filename="multi2.txt")
    doc2 = Document(
        id="multi2",
        content="Document two, also with enough content to be split into multiple chunks by the processor.",
        metadata=doc_meta2,
    )
    doc_meta3 = DocumentMetadata(source_id="multi_doc3", filename="multi3.txt")
    doc3 = Document(id="multi3", content="Short one.", metadata=doc_meta3)


    documents = [doc1, doc2, doc3]
    all_chunks = default_text_processor.split_documents(documents)

    assert len(all_chunks) > 3 # Expecting doc1 and doc2 to split, doc3 to be one chunk
    doc1_chunks = [c for c in all_chunks if c.document_id == "multi1"]
    doc2_chunks = [c for c in all_chunks if c.document_id == "multi2"]
    doc3_chunks = [c for c in all_chunks if c.document_id == "multi3"]

    assert len(doc1_chunks) > 0 # Should be > 1 if content is long enough
    assert len(doc2_chunks) > 0 # Should be > 1
    assert len(doc3_chunks) == 1
    assert doc3_chunks[0].content == "Short one."

    # Check metadata propagation and chunk numbering within each document's chunks
    for i, chunk in enumerate(doc1_chunks):
        assert chunk.metadata.chunk_number == i
        assert chunk.metadata.total_chunks == len(doc1_chunks)
        assert chunk.metadata.filename == "multi1.txt"

    for i, chunk in enumerate(doc2_chunks):
        assert chunk.metadata.chunk_number == i
        assert chunk.metadata.total_chunks == len(doc2_chunks)
        assert chunk.metadata.filename == "multi2.txt"


def test_text_processor_custom_config():
    """Test TextProcessor with custom chunk_size and chunk_overlap."""
    processor = TextProcessor(chunk_size=20, chunk_overlap=5)
    assert processor.chunk_size == 20
    assert processor.chunk_overlap == 5

    doc_meta = DocumentMetadata(source_id="custom_doc", filename="custom.txt")
    doc = Document(
        id="doc_custom",
        content="A small sentence. Another small sentence. Yet another one.", # Approx 60 chars
        metadata=doc_meta,
    )
    chunks = processor.split_document(doc)
    # Expecting multiple chunks due to small chunk_size
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.content) <= 20 # Or slightly more
