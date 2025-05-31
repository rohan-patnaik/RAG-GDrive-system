# tests/conftest.py
import asyncio
import os
import shutil
from pathlib import Path
from typing import AsyncGenerator, Generator, List, cast, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
# from pydantic_settings import SettingsConfigDict # Not directly used here after initial setup

from rag_system.config.settings import AppSettings, get_settings
from rag_system.api.app import create_app # Import the factory function
from rag_system.models.schemas import (
    Document, DocumentMetadata, DocumentChunk, LLMProvider,
    RetrievedChunk, StatusEnum, ComponentStatus, SystemStatusResponse
)
from rag_system.core.embeddings import EmbeddingService
from rag_system.core.vector_store import VectorStoreService
from rag_system.services.llm_service import LLMService
from rag_system.services.rag_service import RAGService


# Override settings for testing
# Store original settings to restore later if necessary, though pytest typically isolates tests.
# _original_settings_env_file = AppSettings.model_config.get('env_file') # Handled by pydantic-settings itself

@pytest.fixture(scope="session", autouse=True)
def set_test_environment_variables():
    """Set environment variables for testing before AppSettings is first loaded."""
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["LOG_LEVEL"] = "DEBUG" # Or "WARNING" to reduce noise
    os.environ["VECTOR_STORE_PATH"] = "./data/test_vector_store_pytest"
    os.environ["CHROMA_COLLECTION_NAME"] = "test_rag_pytest_collection"
    os.environ["LOG_FILE_PATH"] = "logs/pytest_rag_system.log" # Test specific log
    # Mock API keys - tests should mock external calls, but settings might expect them
    os.environ["OPENAI_API_KEY"] = "test_openai_key_pytest"
    os.environ["ANTHROPIC_API_KEY"] = "test_anthropic_key_pytest"
    os.environ["GOOGLE_API_KEY"] = "test_google_key_pytest"
    # Indicate testing mode for app lifespan
    os.environ["RAG_SYSTEM_TESTING_MODE"] = "true"
    # Ensure test vector store path exists and is clean for session
    test_store_path = Path(os.environ["VECTOR_STORE_PATH"])
    if test_store_path.exists():
        shutil.rmtree(test_store_path)
    test_store_path.mkdir(parents=True, exist_ok=True)

    # Ensure test log path exists
    test_log_path_dir = Path(os.environ["LOG_FILE_PATH"]).parent
    test_log_path_dir.mkdir(parents=True, exist_ok=True)

    yield # Allow tests to run

    # Clean up test vector store after session (optional, good practice)
    # if test_store_path.exists():
    #     shutil.rmtree(test_store_path)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for each test session."""
    # policy = asyncio.get_event_loop_policy()
    # loop = policy.new_event_loop()
    # yield loop
    # loop.close()
    # Simpler way for pytest-asyncio default behavior:
    loop = asyncio.SelectorEventLoop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> AppSettings:
    """
    Provides a fixture for application settings, ensuring it uses test-specific .env if any,
    or relies on environment variables set by `set_test_environment_variables`.
    """
    # get_settings() will be influenced by the environment variables set above.
    # If you had a .env.test, pydantic-settings would pick it up if configured.
    settings = get_settings()
    # Ensure critical test paths are what we expect
    assert settings.VECTOR_STORE_PATH == "./data/test_vector_store_pytest"
    assert settings.ENVIRONMENT == "testing"
    return settings


@pytest.fixture(scope="function") # Use "function" scope for FastAPI app if state needs to be clean per test
async def test_app(test_settings: AppSettings) -> AsyncGenerator[FastAPI, None]:
    """
    Creates a FastAPI application instance for testing, using test settings.
    The lifespan events for model loading are bypassed due to RAG_SYSTEM_TESTING_MODE.
    """
    assert os.getenv("RAG_SYSTEM_TESTING_MODE") == "true", "TESTING_MODE not set for app fixture"

    app = create_app(app_settings=test_settings)

    # Mock services that would normally be initialized in lifespan for API tests
    # These mocks can be overridden in specific tests if needed.
    # For API tests, we often want to test the route logic and how it calls the service,
    # so the service itself is mocked. For service tests, underlying components are mocked.
    app.state.embedding_service = AsyncMock(spec=EmbeddingService)
    app.state.vector_store_service = AsyncMock(spec=VectorStoreService)
    app.state.llm_service = AsyncMock(spec=LLMService)

    # Mock RAGService for API tests
    mock_rag_service = AsyncMock(spec=RAGService)
    # Setup default return values for common RAGService methods used in API routes
    mock_rag_service.get_system_status.return_value = SystemStatusResponse(
        system_status=StatusEnum.OK,
        components=[ComponentStatus(name="MockedService", status=StatusEnum.OK)]
    )
    app.state.rag_service = mock_rag_service
    app.state.settings = test_settings # Ensure settings are also on app.state

    yield app


@pytest.fixture(scope="function")
async def async_client(test_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """
    Provides an asynchronous HTTP client for making requests to the test FastAPI app.
    """
    async with AsyncClient(app=test_app, base_url="http://testserver") as client:
        yield client


@pytest.fixture(scope="session")
def temp_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Creates a temporary directory for test data (e.g., vector stores, sample docs)."""
    # This is good for things that need a unique temp path per test run if VECTOR_STORE_PATH wasn't session-fixed
    return tmp_path_factory.mktemp("rag_test_data_func_scope")


@pytest.fixture(scope="function")
def mock_embedding_service() -> MagicMock:
    """Provides a mock EmbeddingService."""
    mock = MagicMock(spec=EmbeddingService)
    mock.encode_texts = MagicMock(return_value=[[0.1, 0.2, 0.3]] * 3) # Example: 3 texts
    mock.encode_query = MagicMock(return_value=[0.1, 0.2, 0.3])
    mock.get_embedding_dimension = MagicMock(return_value=3) # Example dimension
    mock.get_model = MagicMock(return_value=MagicMock()) # Mock the underlying model object
    return mock


@pytest.fixture(scope="function")
def mock_vector_store_service(test_settings: AppSettings) -> MagicMock:
    """
    Provides a mock VectorStoreService.
    Initializes a real ChromaDB instance in a test-specific, temporary directory
    if VECTOR_STORE_PATH is set to a temp path for integration-like unit tests of VectorStore.
    Otherwise, provides a pure MagicMock.
    For this fixture, we'll provide a MagicMock. Specific tests for VectorStoreService
    will initialize a real Chroma instance.
    """
    mock = MagicMock(spec=VectorStoreService)
    mock.add_chunks = MagicMock()
    mock.search_similar = MagicMock(return_value=[
        RetrievedChunk(id="c1", content="content1", metadata={}, score=0.9)
    ])
    mock.delete_chunks = MagicMock()
    mock.get_collection_stats = MagicMock(return_value={"item_count": 0, "collection_name": test_settings.CHROMA_COLLECTION_NAME})
    mock.clear_collection = MagicMock()
    return mock


@pytest.fixture(scope="function")
def mock_llm_service() -> AsyncMock:
    """Provides a mock LLMService."""
    mock = AsyncMock(spec=LLMService)
    mock.generate_response = AsyncMock(return_value=("Mocked LLM answer", "mock-model-v1"))
    # Mock health check for providers
    mock.check_provider_health = AsyncMock(
        side_effect=lambda provider: ComponentStatus(
            name=f"{provider.value} LLM", status=StatusEnum.OK, message="Mocked OK"
        )
    )
    return mock


@pytest.fixture(scope="function")
def sample_documents() -> List[Document]:
    """Provides a list of sample Document objects for testing."""
    return [
        Document(
            id="doc1",
            content="This is the first document about AI and LLMs.",
            metadata=DocumentMetadata(source_id="file1.txt", filename="file1.txt"),
        ),
        Document(
            id="doc2",
            content="The second document discusses Python programming and FastAPI.",
            metadata=DocumentMetadata(source_id="file2.txt", filename="file2.txt"),
        ),
    ]

@pytest.fixture(scope="function")
def sample_document_chunks(sample_documents: List[Document]) -> List[DocumentChunk]:
    """Provides a list of sample DocumentChunk objects."""
    chunks = []
    for i, doc in enumerate(sample_documents):
        # Simple split for testing: one chunk per document
        chunk_id = f"{doc.id}_chunk_0"
        chunk_metadata = doc.metadata.model_copy()
        chunk_metadata.chunk_number = 0
        chunk_metadata.total_chunks = 1
        chunks.append(
            DocumentChunk(
                id=chunk_id,
                document_id=doc.id,
                content=doc.content, # Full content as one chunk for simplicity here
                metadata=chunk_metadata,
                embedding=[0.1 * (i+1), 0.2 * (i+1), 0.3 * (i+1)] # Dummy embedding
            )
        )
    return chunks


@pytest.fixture(scope="function")
def temp_dir_with_docs(tmp_path: Path) -> Path:
    """Creates a temporary directory and populates it with sample text files."""
    doc_dir = tmp_path / "sample_docs"
    doc_dir.mkdir()
    (doc_dir / "doc1.txt").write_text("Content of document 1 about apples.")
    (doc_dir / "doc2.txt").write_text("Content of document 2 about bananas.")
    sub_dir = doc_dir / "subdir"
    sub_dir.mkdir()
    (sub_dir / "doc3.txt").write_text("Content of document 3 in subdirectory about oranges.")
    return doc_dir

# Fixture to clean up the test vector store directory per function if needed,
# but session scope with initial cleanup is usually better for ChromaDB.
@pytest.fixture(scope="function", autouse=False) # Not autouse by default
def clean_test_vector_store(test_settings: AppSettings):
    """Cleans the test vector store directory before a test runs."""
    store_path = Path(test_settings.VECTOR_STORE_PATH)
    if store_path.exists():
        # This is aggressive if multiple tests run in parallel and share the session-scoped path.
        # For true isolation, each test needing a pristine ChromaDB should use a unique path.
        # The session-scoped cleanup in set_test_environment_variables handles initial cleanup.
        # This fixture is more for ensuring a clean state *before* a specific test *function*.
        # For Chroma, it might be better to delete and recreate the collection.
        try:
            # If using a persistent client, deleting the directory might be problematic
            # if the client is still active. Better to use client.delete_collection.
            # For now, we assume tests manage their client instances carefully.
            # shutil.rmtree(store_path) # Potentially problematic
            pass # Rely on collection-level cleanup within tests if needed
        except Exception as e:
            print(f"Warning: Could not fully clean test vector store {store_path}: {e}")
    store_path.mkdir(parents=True, exist_ok=True) # Ensure it exists
    yield
    # No cleanup after function here, session cleanup handles final removal.
