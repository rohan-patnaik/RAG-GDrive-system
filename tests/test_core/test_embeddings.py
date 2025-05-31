# tests/test_core/test_embeddings.py
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from rag_system.core.embeddings import EmbeddingService
from rag_system.utils.exceptions import EmbeddingError
from rag_system.config.settings import AppSettings


@pytest.fixture
def embedding_settings(test_settings: AppSettings) -> AppSettings:
    # Use test_settings by default, can override specific fields if needed
    # e.g., test_settings.EMBEDDING_MODEL_NAME = "mock-model"
    return test_settings


# Patch SentenceTransformer for all tests in this module
@patch("sentence_transformers.SentenceTransformer")
def test_embedding_service_init_success(
    MockSentenceTransformer: MagicMock, embedding_settings: AppSettings
):
    """Test successful initialization of EmbeddingService."""
    mock_model_instance = MagicMock()
    mock_model_instance.get_sentence_embedding_dimension.return_value = 384 # Example dim
    MockSentenceTransformer.return_value = mock_model_instance

    service = EmbeddingService(settings=embedding_settings)

    MockSentenceTransformer.assert_called_once_with(
        embedding_settings.EMBEDDING_MODEL_NAME,
        device=embedding_settings.EMBEDDING_MODEL_DEVICE,
    )
    assert service.model is not None
    assert service.model_name == embedding_settings.EMBEDDING_MODEL_NAME
    assert service.get_embedding_dimension() == 384


@patch("sentence_transformers.SentenceTransformer")
def test_embedding_service_init_failure(
    MockSentenceTransformer: MagicMock, embedding_settings: AppSettings
):
    """Test EmbeddingService initialization failure if model loading fails."""
    MockSentenceTransformer.side_effect = Exception("Model loading failed")

    with pytest.raises(EmbeddingError) as exc_info:
        EmbeddingService(settings=embedding_settings)
    assert "Could not load embedding model" in str(exc_info.value)
    assert "Model loading failed" in str(exc_info.value.__cause__)


@patch("sentence_transformers.SentenceTransformer")
def test_get_model_not_loaded(
    MockSentenceTransformer: MagicMock, embedding_settings: AppSettings
):
    """Test get_model raises error if model is not loaded."""
    # Simulate model failing to load by not setting up MockSentenceTransformer's return_value
    # or by manually setting service.model to None after a failed init (harder to test directly)
    MockSentenceTransformer.side_effect = Exception("Failed to load")
    service = None
    try:
        service = EmbeddingService(settings=embedding_settings)
    except EmbeddingError:
        # Expected, now create a dummy service instance to test get_model
        service = EmbeddingService.__new__(EmbeddingService) # Create instance without calling __init__
        service.model = None # Ensure model is None

    if service: # service might not be created if EmbeddingError is raised from __new__
        with pytest.raises(EmbeddingError, match="Embedding model is not loaded"):
            service.get_model()


@patch("sentence_transformers.SentenceTransformer")
def test_encode_texts_success(
    MockSentenceTransformer: MagicMock, embedding_settings: AppSettings
):
    """Test successful encoding of a list of texts."""
    mock_model_instance = MagicMock()
    # SentenceTransformer.encode returns a list of ndarray or ndarray of ndarrays
    mock_embeddings_np = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
    mock_model_instance.encode.return_value = mock_embeddings_np
    MockSentenceTransformer.return_value = mock_model_instance

    service = EmbeddingService(settings=embedding_settings)
    texts = ["hello world", "another text"]
    embeddings = service.encode_texts(texts)

    mock_model_instance.encode.assert_called_once_with(texts, convert_to_tensor=False)
    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]


def test_encode_texts_empty_list(embedding_settings: AppSettings):
    """Test encoding an empty list of texts returns an empty list."""
    # No need to mock SentenceTransformer if _load_model is not called or encode_texts handles empty early
    with patch.object(EmbeddingService, "_load_model", return_value=None): # Prevent actual model load
        service = EmbeddingService(settings=embedding_settings)
        service.model = MagicMock() # Mock the model attribute directly
        embeddings = service.encode_texts([])
        assert embeddings == []
        service.model.encode.assert_not_called()


@patch("sentence_transformers.SentenceTransformer")
def test_encode_texts_failure(
    MockSentenceTransformer: MagicMock, embedding_settings: AppSettings
):
    """Test encode_texts failure if model's encode method fails."""
    mock_model_instance = MagicMock()
    mock_model_instance.encode.side_effect = Exception("Encoding failed")
    MockSentenceTransformer.return_value = mock_model_instance

    service = EmbeddingService(settings=embedding_settings)
    with pytest.raises(EmbeddingError) as exc_info:
        service.encode_texts(["some text"])
    assert "Failed to generate embeddings for texts" in str(exc_info.value)
    assert "Encoding failed" in str(exc_info.value.__cause__)


@patch("sentence_transformers.SentenceTransformer")
def test_encode_query_success(
    MockSentenceTransformer: MagicMock, embedding_settings: AppSettings
):
    """Test successful encoding of a single query."""
    mock_model_instance = MagicMock()
    mock_embedding_np = np.array([0.5, 0.6, 0.7])
    mock_model_instance.encode.return_value = mock_embedding_np
    MockSentenceTransformer.return_value = mock_model_instance

    service = EmbeddingService(settings=embedding_settings)
    query = "test query"
    embedding = service.encode_query(query)

    mock_model_instance.encode.assert_called_once_with(query, convert_to_tensor=False)
    assert embedding == [0.5, 0.6, 0.7]


def test_encode_query_empty_string(embedding_settings: AppSettings):
    """Test encoding an empty query string returns an empty list."""
    with patch.object(EmbeddingService, "_load_model", return_value=None):
        service = EmbeddingService(settings=embedding_settings)
        service.model = MagicMock()
        embedding = service.encode_query("")
        assert embedding == []
        service.model.encode.assert_not_called()


@patch("sentence_transformers.SentenceTransformer")
def test_encode_query_failure(
    MockSentenceTransformer: MagicMock, embedding_settings: AppSettings
):
    """Test encode_query failure."""
    mock_model_instance = MagicMock()
    mock_model_instance.encode.side_effect = Exception("Query encoding failed")
    MockSentenceTransformer.return_value = mock_model_instance

    service = EmbeddingService(settings=embedding_settings)
    with pytest.raises(EmbeddingError) as exc_info:
        service.encode_query("a query")
    assert "Failed to generate embedding for the query" in str(exc_info.value)
    assert "Query encoding failed" in str(exc_info.value.__cause__)


@patch("sentence_transformers.SentenceTransformer")
def test_get_embedding_dimension(
    MockSentenceTransformer: MagicMock, embedding_settings: AppSettings
):
    """Test getting the embedding dimension."""
    mock_model_instance = MagicMock()
    mock_model_instance.get_sentence_embedding_dimension.return_value = 768
    MockSentenceTransformer.return_value = mock_model_instance

    service = EmbeddingService(settings=embedding_settings)
    dimension = service.get_embedding_dimension()
    assert dimension == 768


@patch("sentence_transformers.SentenceTransformer")
def test_get_embedding_dimension_failure(
    MockSentenceTransformer: MagicMock, embedding_settings: AppSettings
):
    """Test failure in getting embedding dimension."""
    mock_model_instance = MagicMock()
    mock_model_instance.get_sentence_embedding_dimension.side_effect = Exception("Dim error")
    MockSentenceTransformer.return_value = mock_model_instance

    service = EmbeddingService(settings=embedding_settings)
    dimension = service.get_embedding_dimension() # Should log error and return None
    assert dimension is None
