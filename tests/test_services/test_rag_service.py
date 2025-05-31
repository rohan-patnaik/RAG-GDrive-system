# tests/test_services/test_rag_service.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, cast

from rag_system.services.rag_service import RAGService
from rag_system.models.schemas import (
    IngestionRequest,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
    Document,
    DocumentChunk,
    RetrievedChunk,
    LLMProvider,
    StatusEnum,
    ComponentStatus,
    SystemStatusResponse,
)
from rag_system.core.document_loader import DocumentLoader
from rag_system.core.text_processor import TextProcessor
from rag_system.utils.exceptions import (
    DocumentProcessingError,
    VectorStoreError,
    EmbeddingError,
    LLMError,
    RAGServiceError,
)
from rag_system.config.settings import AppSettings


@pytest.fixture
def mock_document_loader() -> MagicMock:
    mock = MagicMock(spec=DocumentLoader)
    mock.load_from_directory.return_value = [
        Document(id="doc1", content="Content doc1", metadata=MagicMock()),
        Document(id="doc2", content="Content doc2", metadata=MagicMock()),
    ]
    return mock

@pytest.fixture
def mock_text_processor() -> MagicMock:
    mock = MagicMock(spec=TextProcessor)
    # Simulate splitting each doc into 2 chunks
    def mock_split_documents(docs: List[Document]) -> List[DocumentChunk]:
        chunks = []
        for i, doc in enumerate(docs):
            for j in range(2): # 2 chunks per doc
                chunk_meta = MagicMock()
                chunk_meta.source_id = f"source_{doc.id}"
                chunks.append(
                    DocumentChunk(
                        id=f"{doc.id}_chunk{j}",
                        document_id=doc.id,
                        content=f"Chunk {j} of {doc.id}",
                        metadata=chunk_meta,
                        embedding=None # Embeddings added later or by vector store
                    )
                )
        return chunks
    mock.split_documents.side_effect = mock_split_documents
    return mock


# mock_embedding_service and mock_vector_store_service are from conftest.py
# mock_llm_service is from conftest.py

@pytest.fixture
def rag_service(
    test_settings: AppSettings,
    mock_document_loader: MagicMock,
    mock_text_processor: MagicMock,
    mock_embedding_service: MagicMock, # from conftest
    mock_vector_store_service: MagicMock, # from conftest
    mock_llm_service: AsyncMock, # from conftest
) -> RAGService:
    return RAGService(
        settings=test_settings,
        document_loader=mock_document_loader,
        text_processor=mock_text_processor,
        embedding_service=mock_embedding_service,
        vector_store_service=mock_vector_store_service,
        llm_service=mock_llm_service,
    )


@pytest.mark.asyncio
async def test_ingest_documents_success(
    rag_service: RAGService,
    mock_document_loader: MagicMock,
    mock_text_processor: MagicMock,
    mock_vector_store_service: MagicMock,
):
    """Test successful document ingestion."""
    ingest_request = IngestionRequest(source_directory="fake/path")
    response = await rag_service.ingest_documents(ingest_request)

    mock_document_loader.load_from_directory.assert_called_once()
    mock_text_processor.split_documents.assert_called_once()
    # mock_embedding_service.encode_texts was called if vector store doesn't embed
    # In our setup, ChromaDB embeds, so EmbeddingService.encode_texts is not directly called by RAGService.ingest
    mock_vector_store_service.add_chunks.assert_called_once()

    assert isinstance(response, IngestionResponse)
    assert response.documents_processed == 2 # From mock_document_loader
    assert response.chunks_added == 4       # 2 docs * 2 chunks/doc from mock_text_processor
    assert not response.errors
    assert "Ingestion completed" in response.message


@pytest.mark.asyncio
async def test_ingest_documents_no_docs_found(
    rag_service: RAGService, mock_document_loader: MagicMock
):
    """Test ingestion when no documents are found."""
    mock_document_loader.load_from_directory.return_value = []
    ingest_request = IngestionRequest(source_directory="empty/path")
    response = await rag_service.ingest_documents(ingest_request)

    assert response.documents_processed == 0
    assert response.chunks_added == 0
    assert "No documents found" in response.message
    assert len(response.errors) == 1


@pytest.mark.asyncio
async def test_ingest_documents_no_chunks_generated(
    rag_service: RAGService, mock_text_processor: MagicMock
):
    """Test ingestion when no chunks are generated from documents."""
    mock_text_processor.split_documents.return_value = []
    ingest_request = IngestionRequest(source_directory="path/with/docs")
    response = await rag_service.ingest_documents(ingest_request)

    assert response.chunks_added == 0
    # The message might vary, check for a relevant part
    assert "No chunks were generated" in response.errors[0] or "No document chunks to add" in response.message


@pytest.mark.asyncio
async def test_ingest_document_loader_error(
    rag_service: RAGService, mock_document_loader: MagicMock
):
    """Test ingestion failure if DocumentLoader raises an error."""
    mock_document_loader.load_from_directory.side_effect = DocumentProcessingError(
        "Failed to load"
    )
    ingest_request = IngestionRequest(source_directory="bad/path")
    with pytest.raises(DocumentProcessingError, match="Failed to load"):
        await rag_service.ingest_documents(ingest_request)


@pytest.mark.asyncio
async def test_ingest_text_processor_error(
    rag_service: RAGService, mock_text_processor: MagicMock
):
    """Test ingestion failure if TextProcessor raises an error."""
    mock_text_processor.split_documents.side_effect = DocumentProcessingError(
        "Failed to split"
    )
    ingest_request = IngestionRequest(source_directory="good/path")
    with pytest.raises(DocumentProcessingError, match="Failed to split"):
        await rag_service.ingest_documents(ingest_request)


@pytest.mark.asyncio
async def test_ingest_vector_store_error(
    rag_service: RAGService, mock_vector_store_service: MagicMock
):
    """Test ingestion failure if VectorStoreService raises an error during add_chunks."""
    mock_vector_store_service.add_chunks.side_effect = VectorStoreError(
        "Failed to add to Chroma"
    )
    ingest_request = IngestionRequest(source_directory="good/path")
    with pytest.raises(VectorStoreError, match="Failed to add to Chroma"):
        await rag_service.ingest_documents(ingest_request)


@pytest.mark.asyncio
async def test_query_success(
    rag_service: RAGService,
    mock_embedding_service: MagicMock,
    mock_vector_store_service: MagicMock,
    mock_llm_service: AsyncMock,
    test_settings: AppSettings,
):
    """Test successful query processing."""
    query_text = "What is AI?"
    query_request = QueryRequest(query_text=query_text)

    # Mock return values
    mock_embedding_service.encode_query.return_value = [0.1, 0.2, 0.3]
    retrieved_chunks_data = [
        RetrievedChunk(id="c1", content="AI is cool.", metadata={}, score=0.9),
        RetrievedChunk(id="c2", content="LLMs are powerful.", metadata={}, score=0.8),
    ]
    mock_vector_store_service.search_similar.return_value = retrieved_chunks_data
    mock_llm_service.generate_response.return_value = ("AI is a field of CS.", "mock-gpt")

    response = await rag_service.query(query_request)

    mock_embedding_service.encode_query.assert_called_once_with(query_text)
    mock_vector_store_service.search_similar.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=test_settings.TOP_K_RESULTS, # Default top_k from settings
    )
    mock_llm_service.generate_response.assert_called_once()
    # Check args of generate_response call
    args, kwargs = mock_llm_service.generate_response.call_args
    assert kwargs['query'] == query_text
    assert kwargs['context_chunks'] == retrieved_chunks_data
    assert kwargs['provider'] == LLMProvider(test_settings.DEFAULT_LLM_PROVIDER)


    assert isinstance(response, QueryResponse)
    assert response.query_text == query_text
    assert response.llm_answer == "AI is a field of CS."
    assert response.llm_model_used == "mock-gpt"
    assert response.retrieved_chunks == retrieved_chunks_data


@pytest.mark.asyncio
async def test_query_with_params_override(
    rag_service: RAGService,
    mock_vector_store_service: MagicMock,
    mock_llm_service: AsyncMock,
    test_settings: AppSettings,
):
    """Test query with overridden top_k, provider, and similarity_threshold."""
    query_request = QueryRequest(
        query_text="Test query",
        llm_provider=LLMProvider.ANTHROPIC,
        top_k=5,
        similarity_threshold=0.8
    )
    # Mock search_similar to return chunks that would be filtered by threshold
    mock_vector_store_service.search_similar.return_value = [
        RetrievedChunk(id="c1", content="c1", metadata={}, score=0.9), # Keep
        RetrievedChunk(id="c2", content="c2", metadata={}, score=0.75), # Filter out
        RetrievedChunk(id="c3", content="c3", metadata={}, score=0.85), # Keep
    ]

    await rag_service.query(query_request)

    mock_vector_store_service.search_similar.assert_called_once_with(
        query_embedding=mock_embedding_service.encode_query.return_value, # from conftest mock
        top_k=5, # Overridden top_k
    )
    # Check context_chunks passed to LLMService after threshold filtering
    args, kwargs = mock_llm_service.generate_response.call_args
    assert len(kwargs['context_chunks']) == 2
    assert kwargs['context_chunks'][0].id == "c1"
    assert kwargs['context_chunks'][1].id == "c3"
    assert kwargs['provider'] == LLMProvider.ANTHROPIC


@pytest.mark.asyncio
async def test_query_embedding_error(
    rag_service: RAGService, mock_embedding_service: MagicMock
):
    """Test query failure if embedding service fails."""
    mock_embedding_service.encode_query.side_effect = EmbeddingError("Embedding failed")
    query_request = QueryRequest(query_text="test")
    with pytest.raises(EmbeddingError, match="Embedding failed"):
        await rag_service.query(query_request)


@pytest.mark.asyncio
async def test_query_vector_store_error(
    rag_service: RAGService, mock_vector_store_service: MagicMock
):
    """Test query failure if vector store search fails."""
    mock_vector_store_service.search_similar.side_effect = VectorStoreError("Search failed")
    query_request = QueryRequest(query_text="test")
    with pytest.raises(VectorStoreError, match="Search failed"):
        await rag_service.query(query_request)


@pytest.mark.asyncio
async def test_query_llm_error(
    rag_service: RAGService, mock_llm_service: AsyncMock
):
    """Test query failure if LLM service fails."""
    mock_llm_service.generate_response.side_effect = LLMError("LLM generation failed")
    query_request = QueryRequest(query_text="test")
    with pytest.raises(LLMError, match="LLM generation failed"):
        await rag_service.query(query_request)


@pytest.mark.asyncio
async def test_get_system_status(
    rag_service: RAGService,
    mock_embedding_service: MagicMock,
    mock_vector_store_service: MagicMock,
    mock_llm_service: AsyncMock,
    test_settings: AppSettings,
):
    """Test the get_system_status method."""
    # Setup mock return values for component health checks
    mock_embedding_service.get_model.return_value = True # Simulate model loaded
    mock_embedding_service.get_embedding_dimension.return_value = 384
    mock_embedding_service.model_name = "mock-emb-model"

    mock_vector_store_service.get_collection_stats.return_value = {
        "item_count": 10, "collection_name": "test_coll"
    }
    mock_vector_store_service.client = True # Simulate client initialized

    # LLMService.check_provider_health is already mocked in conftest to return OK status
    # for any provider passed to it.

    status_response = await rag_service.get_system_status()

    assert isinstance(status_response, SystemStatusResponse)
    assert status_response.system_status == StatusEnum.OK
    assert status_response.app_name == test_settings.APP_NAME
    assert len(status_response.components) >= 3 # Embedding, VectorStore, at least one LLM

    comp_names = [comp.name for comp in status_response.components]
    assert "EmbeddingService" in comp_names
    assert "VectorStoreService" in comp_names
    # Check for default LLM provider status (OpenAI, Anthropic, Gemini based on test_settings keys)
    if test_settings.OPENAI_API_KEY: assert "Openai LLM" in comp_names
    if test_settings.ANTHROPIC_API_KEY: assert "Anthropic LLM" in comp_names
    if test_settings.GOOGLE_API_KEY: assert "Gemini LLM" in comp_names


    for comp in status_response.components:
        assert comp.status == StatusEnum.OK # All should be OK based on mocks

@pytest.mark.asyncio
async def test_get_system_status_component_failure(
    rag_service: RAGService,
    mock_vector_store_service: MagicMock,
):
    """Test get_system_status when a component reports an error."""
    mock_vector_store_service.get_collection_stats.side_effect = VectorStoreError("VS down")

    status_response = await rag_service.get_system_status()

    assert status_response.system_status == StatusEnum.ERROR # or DEGRADED depending on severity
    vs_status = next(c for c in status_response.components if c.name == "VectorStoreService")
    assert vs_status.status == StatusEnum.ERROR
    assert "VS down" in vs_status.message
