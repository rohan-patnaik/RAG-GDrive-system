# tests/test_core/test_vector_store.py
import pytest
import shutil
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch

import chromadb # type: ignore
from chromadb.api.models.Collection import Collection as ChromaCollection # type: ignore
from chromadb.utils import embedding_functions # type: ignore

from rag_system.core.vector_store import VectorStoreService
from rag_system.core.embeddings import EmbeddingService # For mocking
from rag_system.models.schemas import DocumentChunk, DocumentMetadata, RetrievedChunk
from rag_system.utils.exceptions import VectorStoreError
from rag_system.config.settings import AppSettings


# Fixture for test-specific settings, ensuring a unique path for ChromaDB
@pytest.fixture
def vector_store_test_settings(test_settings: AppSettings, tmp_path: Path) -> AppSettings:
    """Overrides vector store path to use a temporary directory for this test module."""
    settings_copy = test_settings.model_copy(deep=True)
    # Each test function using this might get a new tmp_path,
    # so ChromaDB instances will be isolated if this fixture is function-scoped.
    # If module-scoped, tmp_path is per-module.
    # For true isolation per test, this fixture should be function-scoped.
    settings_copy.VECTOR_STORE_PATH = str(tmp_path / "test_chroma_data")
    settings_copy.CHROMA_COLLECTION_NAME = "pytest_module_collection" # Consistent name for the module
    # Ensure the directory exists
    Path(settings_copy.VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
    return settings_copy


@pytest.fixture
def mock_embedding_service_for_vs() -> MagicMock:
    """Mocks EmbeddingService specifically for VectorStore tests."""
    mock_emb_service = MagicMock(spec=EmbeddingService)
    mock_emb_service.model_name = "sentence-transformers/all-MiniLM-L6-v2" # Must match what Chroma expects or is configured with
    mock_emb_service.device = "cpu"
    mock_emb_service.encode_query.return_value = [0.1, 0.2, 0.3] # Dummy query embedding
    mock_emb_service.get_embedding_dimension.return_value = 384 # Example dimension
    return mock_emb_service


@pytest.fixture
def vector_store_service(
    vector_store_test_settings: AppSettings,
    mock_embedding_service_for_vs: MagicMock
) -> VectorStoreService:
    """
    Provides a VectorStoreService instance initialized with a real ChromaDB
    in a temporary directory. Cleans up after tests.
    """
    # Ensure the path is clean before starting
    store_path = Path(vector_store_test_settings.VECTOR_STORE_PATH)
    if store_path.exists():
        shutil.rmtree(store_path)
    store_path.mkdir(parents=True, exist_ok=True)

    # Use the mock embedding service that provides model_name for Chroma's ef
    service = VectorStoreService(
        settings=vector_store_test_settings,
        embedding_service=mock_embedding_service_for_vs
    )
    # Ensure collection is clean before each test using this fixture
    try:
        service.clear_collection()
    except Exception as e:
        # This might happen if the collection didn't exist yet, which is fine.
        print(f"Note: Error during pre-test collection clearing (might be first run): {e}")


    yield service

    # Teardown: Clean up the ChromaDB directory after tests in this module/function
    # This is important if the tmp_path is function-scoped.
    # If tmp_path is module-scoped, this cleanup might run once per module.
    # The session-scoped cleanup in conftest handles the global test path.
    if store_path.exists():
        shutil.rmtree(store_path)


@pytest.fixture
def sample_chunks_for_vs() -> List[DocumentChunk]:
    """Provides sample DocumentChunk objects for vector store tests."""
    chunks = []
    for i in range(3):
        meta = DocumentMetadata(
            source_id=f"doc{i+1}",
            filename=f"file{i+1}.txt",
            custom_fields={"category": "test", "index": i}
        )
        # Embeddings are not pre-populated as ChromaDB will generate them
        chunks.append(
            DocumentChunk(
                id=f"chunk{i+1}",
                document_id=f"doc{i+1}",
                content=f"This is content for chunk {i+1}. It talks about topic {chr(65+i)}.",
                metadata=meta,
            )
        )
    return chunks


def test_vector_store_initialization(vector_store_service: VectorStoreService, vector_store_test_settings: AppSettings):
    """Test that VectorStoreService initializes correctly and creates/gets a collection."""
    assert vector_store_service.client is not None
    assert vector_store_service.collection is not None
    assert vector_store_service.collection.name == vector_store_test_settings.CHROMA_COLLECTION_NAME
    assert vector_store_service.collection.count() == 0 # Should be empty due to clear_collection in fixture


def test_add_chunks_and_get_stats(vector_store_service: VectorStoreService, sample_chunks_for_vs: List[DocumentChunk]):
    """Test adding chunks and verifying collection stats."""
    vector_store_service.add_chunks(sample_chunks_for_vs)
    stats = vector_store_service.get_collection_stats()
    assert stats["item_count"] == len(sample_chunks_for_vs)

    # Verify one chunk's data (optional, more of an integration test for Chroma)
    retrieved = vector_store_service.collection.get(ids=["chunk1"], include=["metadatas", "documents"])
    assert retrieved["ids"] == ["chunk1"]
    assert retrieved["documents"][0] == sample_chunks_for_vs[0].content
    assert retrieved["metadatas"][0]["source_id"] == sample_chunks_for_vs[0].metadata.source_id


def test_add_chunks_empty_list(vector_store_service: VectorStoreService):
    """Test adding an empty list of chunks."""
    vector_store_service.add_chunks([])
    stats = vector_store_service.get_collection_stats()
    assert stats["item_count"] == 0


def test_add_chunks_with_empty_content(vector_store_service: VectorStoreService):
    """Test adding chunks where some have empty content."""
    meta = DocumentMetadata(source_id="doc_empty_content", filename="empty.txt")
    chunks = [
        DocumentChunk(id="c_empty", document_id="d1", content="", metadata=meta),
        DocumentChunk(id="c_valid", document_id="d1", content="Valid content.", metadata=meta),
    ]
    vector_store_service.add_chunks(chunks)
    stats = vector_store_service.get_collection_stats()
    assert stats["item_count"] == 1 # Only the valid chunk should be added
    retrieved = vector_store_service.collection.get(ids=["c_valid"])
    assert retrieved["ids"] == ["c_valid"]


def test_search_similar_chunks(
    vector_store_service: VectorStoreService,
    sample_chunks_for_vs: List[DocumentChunk],
    mock_embedding_service_for_vs: MagicMock # Used by VectorStoreService for query embedding
):
    """Test searching for similar chunks."""
    vector_store_service.add_chunks(sample_chunks_for_vs)
    assert vector_store_service.collection.count() == 3

    # The mock_embedding_service_for_vs.encode_query will be called by RAGService/user,
    # then the result passed to vector_store_service.search_similar.
    # Here, we simulate that by providing a query embedding directly.
    # ChromaDB itself uses its internal EF for search if query_embeddings are passed.
    dummy_query_embedding = [0.1, 0.2, 0.3] # Example query embedding

    retrieved_chunks = vector_store_service.search_similar(
        query_embedding=dummy_query_embedding, top_k=2
    )
    assert len(retrieved_chunks) <= 2 # Can be less if fewer than top_k items in store or match
    assert len(retrieved_chunks) > 0 # Expect some results for a generic query if items exist

    for chunk in retrieved_chunks:
        assert isinstance(chunk, RetrievedChunk)
        assert chunk.id in [c.id for c in sample_chunks_for_vs]
        assert chunk.score is not None
        assert 0.0 <= chunk.score <= 1.0 # Assuming cosine similarity (1 - distance)

    # Test search on empty collection
    vector_store_service.clear_collection()
    empty_results = vector_store_service.search_similar(dummy_query_embedding, top_k=2)
    assert len(empty_results) == 0


def test_search_with_metadata_filter(
    vector_store_service: VectorStoreService,
    sample_chunks_for_vs: List[DocumentChunk],
):
    vector_store_service.add_chunks(sample_chunks_for_vs)
    dummy_query_embedding = [0.1, 0.2, 0.3]

    # Filter for category: "test", index: 0 (should match chunk1)
    # ChromaDB's where filter syntax: {"metadata_field": "value"} or {"$and": [...]} / {"$or": [...]}
    # Our custom_fields are nested, so Chroma needs them flattened or specific handling.
    # The current VectorStoreService flattens metadata for storage.
    # So, custom_fields_category and custom_fields_index would be the keys.
    # Let's adjust sample_chunks_for_vs metadata to be flat for easier testing here.

    # Re-create chunks with flatter metadata for this test
    flat_meta_chunks = []
    for i in range(3):
        meta = DocumentMetadata(
            source_id=f"doc{i+1}",
            filename=f"file{i+1}.txt",
            # Flatten custom fields for easier Chroma query in test
            custom_fields_category="test_cat", # All same category
            custom_fields_index_num=i # Different index (Chroma needs compatible types)
        )
        flat_meta_chunks.append(
            DocumentChunk(
                id=f"flat_chunk{i+1}",
                document_id=f"doc{i+1}",
                content=f"Content for flat chunk {i+1}.",
                metadata=meta,
            )
        )
    vector_store_service.clear_collection()
    vector_store_service.add_chunks(flat_meta_chunks)

    # Filter for custom_fields_index_num == 1 (should match flat_chunk2)
    filtered_results = vector_store_service.search_similar(
        query_embedding=dummy_query_embedding,
        top_k=3,
        filter_metadata={"custom_fields_index_num": 1}
    )
    assert len(filtered_results) == 1
    assert filtered_results[0].id == "flat_chunk2"
    assert filtered_results[0].metadata["custom_fields_index_num"] == 1


def test_delete_chunks_by_ids(vector_store_service: VectorStoreService, sample_chunks_for_vs: List[DocumentChunk]):
    """Test deleting chunks by their IDs."""
    vector_store_service.add_chunks(sample_chunks_for_vs)
    initial_count = vector_store_service.get_collection_stats()["item_count"]
    assert initial_count == len(sample_chunks_for_vs)

    ids_to_delete = [sample_chunks_for_vs[0].id, sample_chunks_for_vs[1].id]
    vector_store_service.delete_chunks(chunk_ids=ids_to_delete)

    stats_after_delete = vector_store_service.get_collection_stats()
    assert stats_after_delete["item_count"] == initial_count - len(ids_to_delete)

    # Verify they are gone
    retrieved = vector_store_service.collection.get(ids=ids_to_delete)
    assert len(retrieved["ids"]) == 0


def test_delete_chunks_by_metadata_filter(
    vector_store_service: VectorStoreService,
    sample_chunks_for_vs: List[DocumentChunk] # These have nested custom_fields
):
    """Test deleting chunks using a metadata filter."""
    # Add chunks with varied metadata for filtering
    meta1 = DocumentMetadata(source_id="filter_doc1", filename="f1.txt", custom_fields={"type": "report", "year": 2023})
    chunk1 = DocumentChunk(id="filter_c1", document_id="d1", content="Report content 2023", metadata=meta1)
    meta2 = DocumentMetadata(source_id="filter_doc2", filename="f2.txt", custom_fields={"type": "article", "year": 2023})
    chunk2 = DocumentChunk(id="filter_c2", document_id="d2", content="Article content 2023", metadata=meta2)
    meta3 = DocumentMetadata(source_id="filter_doc3", filename="f3.txt", custom_fields={"type": "report", "year": 2024})
    chunk3 = DocumentChunk(id="filter_c3", document_id="d3", content="Report content 2024", metadata=meta3)

    vector_store_service.add_chunks([chunk1, chunk2, chunk3])
    assert vector_store_service.get_collection_stats()["item_count"] == 3

    # Delete reports from 2023. Metadata keys for custom fields are flattened like "custom_fields_type"
    vector_store_service.delete_chunks(filter_metadata={"custom_fields_type": "report", "custom_fields_year": 2023})
    assert vector_store_service.get_collection_stats()["item_count"] == 2 # chunk2 and chunk3 should remain

    # Verify chunk1 is deleted, chunk2 and chunk3 remain
    remaining_ids = vector_store_service.collection.get()["ids"]
    assert "filter_c1" not in remaining_ids
    assert "filter_c2" in remaining_ids
    assert "filter_c3" in remaining_ids


def test_clear_collection(vector_store_service: VectorStoreService, sample_chunks_for_vs: List[DocumentChunk]):
    """Test clearing all items from the collection."""
    vector_store_service.add_chunks(sample_chunks_for_vs)
    assert vector_store_service.get_collection_stats()["item_count"] > 0

    vector_store_service.clear_collection()
    stats = vector_store_service.get_collection_stats()
    assert stats["item_count"] == 0
    # Verify the collection still exists but is empty
    assert vector_store_service.collection is not None
    assert vector_store_service.collection.name == vector_store_service.collection_name


def test_vector_store_error_handling(vector_store_test_settings: AppSettings, mock_embedding_service_for_vs: MagicMock):
    """Test error handling for common VectorStore operations."""
    # Simulate an error during client initialization
    with patch("chromadb.PersistentClient", side_effect=Exception("Chroma init failed")):
        with pytest.raises(VectorStoreError, match="ChromaDB client/collection initialization failed"):
            VectorStoreService(settings=vector_store_test_settings, embedding_service=mock_embedding_service_for_vs)

    # Test errors in other methods (requires a valid service instance first)
    service = VectorStoreService(settings=vector_store_test_settings, embedding_service=mock_embedding_service_for_vs)
    service.clear_collection() # Start clean

    # Mock collection methods to raise errors
    with patch.object(service.collection, "add", side_effect=Exception("Add failed")):
        with pytest.raises(VectorStoreError, match="Failed to add chunks"):
            service.add_chunks([DocumentChunk(id="err_c1", document_id="d1", content="test", metadata=DocumentMetadata(source_id="s1"))])

    with patch.object(service.collection, "query", side_effect=Exception("Query failed")):
        with pytest.raises(VectorStoreError, match="Failed to search for similar chunks"):
            service.search_similar(query_embedding=[0.1, 0.2], top_k=1)

    with patch.object(service.collection, "delete", side_effect=Exception("Delete failed")):
        with pytest.raises(VectorStoreError, match="Failed to delete chunks"):
            service.delete_chunks(chunk_ids=["c1"])

    with patch.object(service.collection, "count", side_effect=Exception("Count failed")):
        with pytest.raises(VectorStoreError, match="Failed to retrieve collection statistics"):
            service.get_collection_stats()

    # Test clearing a non-existent collection (should be handled by get_or_create_collection or specific error)
    # This is harder to test without deeper ChromaDB mocking.
    # The current clear_collection deletes and recreates, so it should handle it.
