# tests/test_api/test_documents_endpoint.py
import pytest
from httpx import AsyncClient
from fastapi import status, FastAPI
from unittest.mock import AsyncMock, cast

from rag_system.models.schemas import IngestionRequest, IngestionResponse
from rag_system.utils.exceptions import DocumentProcessingError, VectorStoreError


@pytest.mark.asyncio
async def test_ingest_documents_success(async_client: AsyncClient, test_app: FastAPI):
    """Test successful document ingestion via /documents/ingest endpoint."""
    ingest_payload = {
        "source_directory": "data/test_docs",
        "file_patterns": ["*.txt"],
        "recursive": True,
    }
    expected_response_data = {
        "message": "Ingestion successful",
        "documents_processed": 5,
        "chunks_added": 50,
        "errors": [],
    }

    mock_rag_service = cast(AsyncMock, test_app.state.rag_service)
    mock_rag_service.ingest_documents.return_value = IngestionResponse(**expected_response_data)

    response = await async_client.post("/documents/ingest", json=ingest_payload)

    assert response.status_code == status.HTTP_201_CREATED
    response_data = response.json()
    assert response_data["message"] == expected_response_data["message"]
    assert response_data["documents_processed"] == expected_response_data["documents_processed"]
    assert response_data["chunks_added"] == expected_response_data["chunks_added"]
    mock_rag_service.ingest_documents.assert_called_once()
    # Check if the IngestionRequest passed to the service matches the payload
    call_args = mock_rag_service.ingest_documents.call_args[0][0] # Get the IngestionRequest argument
    assert isinstance(call_args, IngestionRequest)
    assert call_args.source_directory == ingest_payload["source_directory"]


@pytest.mark.asyncio
async def test_ingest_documents_partial_success_with_errors(async_client: AsyncClient, test_app: FastAPI):
    """Test ingestion with some errors reported by RAGService."""
    ingest_payload = {"source_directory": "data/mixed_docs"}
    expected_response_data = {
        "message": "Ingestion completed with some errors.",
        "documents_processed": 10,
        "chunks_added": 80,
        "errors": ["Failed to process file1.txt", "Could not read file2.log"],
    }
    mock_rag_service = cast(AsyncMock, test_app.state.rag_service)
    mock_rag_service.ingest_documents.return_value = IngestionResponse(**expected_response_data)

    response = await async_client.post("/documents/ingest", json=ingest_payload)

    assert response.status_code == status.HTTP_201_CREATED # Still 201 if some success
    response_data = response.json()
    assert response_data["errors"] == expected_response_data["errors"]
    assert response_data["chunks_added"] == expected_response_data["chunks_added"]


@pytest.mark.asyncio
async def test_ingest_documents_processing_error(async_client: AsyncClient, test_app: FastAPI):
    """Test ingestion failure due to DocumentProcessingError from RAGService."""
    ingest_payload = {"source_directory": "data/bad_docs"}
    mock_rag_service = cast(AsyncMock, test_app.state.rag_service)
    mock_rag_service.ingest_documents.side_effect = DocumentProcessingError(
        message="Source directory not found", detail="/path/does/not/exist", status_code=404
    )

    response = await async_client.post("/documents/ingest", json=ingest_payload)

    assert response.status_code == status.HTTP_404_NOT_FOUND
    response_data = response.json()
    assert response_data["error"] == "DocumentProcessingError"
    assert "Source directory not found" in response_data["message"]
    assert "/path/does/not/exist" in response_data["detail"]


@pytest.mark.asyncio
async def test_ingest_documents_vector_store_error(async_client: AsyncClient, test_app: FastAPI):
    """Test ingestion failure due to VectorStoreError from RAGService."""
    ingest_payload = {"source_directory": "data/docs_to_fail_vs"}
    mock_rag_service = cast(AsyncMock, test_app.state.rag_service)
    mock_rag_service.ingest_documents.side_effect = VectorStoreError(
        message="Failed to connect to vector database", status_code=503
    )

    response = await async_client.post("/documents/ingest", json=ingest_payload)

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    response_data = response.json()
    assert response_data["error"] == "VectorStoreError"
    assert "Failed to connect to vector database" in response_data["message"]


@pytest.mark.asyncio
async def test_ingest_documents_file_not_found_error(async_client: AsyncClient, test_app: FastAPI):
    """Test ingestion failure due to FileNotFoundError (e.g., from DocumentLoader)."""
    ingest_payload = {"source_directory": "non_existent_path"}
    mock_rag_service = cast(AsyncMock, test_app.state.rag_service)
    # RAGService catches FileNotFoundError and re-raises as DocumentProcessingError,
    # or the route handler catches it directly. Let's assume RAGService handles it.
    mock_rag_service.ingest_documents.side_effect = FileNotFoundError(
        "Directory 'non_existent_path' does not exist."
    )
    # The route handler for /ingest catches FileNotFoundError and returns 404

    response = await async_client.post("/documents/ingest", json=ingest_payload)

    assert response.status_code == status.HTTP_404_NOT_FOUND
    response_data = response.json()
    assert "Directory 'non_existent_path' does not exist." in response_data["detail"]


@pytest.mark.asyncio
async def test_ingest_documents_uses_defaults_from_settings(async_client: AsyncClient, test_app: FastAPI):
    """Test that endpoint uses default file_patterns and recursive from settings if not provided."""
    ingest_payload = {"source_directory": "data/default_check"} # No patterns or recursive flag

    expected_response_data = {"message": "Default check done", "documents_processed": 1, "chunks_added": 1, "errors": []}
    mock_rag_service = cast(AsyncMock, test_app.state.rag_service)
    mock_rag_service.ingest_documents.return_value = IngestionResponse(**expected_response_data)

    app_settings = test_app.state.settings

    await async_client.post("/documents/ingest", json=ingest_payload)

    mock_rag_service.ingest_documents.assert_called_once()
    call_args: IngestionRequest = mock_rag_service.ingest_documents.call_args[0][0]
    assert call_args.source_directory == ingest_payload["source_directory"]
    assert call_args.file_patterns == app_settings.DEFAULT_FILE_PATTERNS
    assert call_args.recursive == app_settings.DEFAULT_RECURSIVE_INGESTION


@pytest.mark.asyncio
async def test_ingest_documents_overrides_defaults(async_client: AsyncClient, test_app: FastAPI):
    """Test that endpoint uses provided file_patterns and recursive, overriding settings."""
    ingest_payload = {
        "source_directory": "data/override_check",
        "file_patterns": ["*.md"],
        "recursive": False
    }

    expected_response_data = {"message": "Override check done", "documents_processed": 1, "chunks_added": 1, "errors": []}
    mock_rag_service = cast(AsyncMock, test_app.state.rag_service)
    mock_rag_service.ingest_documents.return_value = IngestionResponse(**expected_response_data)

    await async_client.post("/documents/ingest", json=ingest_payload)

    mock_rag_service.ingest_documents.assert_called_once()
    call_args: IngestionRequest = mock_rag_service.ingest_documents.call_args[0][0]
    assert call_args.source_directory == ingest_payload["source_directory"]
    assert call_args.file_patterns == ingest_payload["file_patterns"]
    assert call_args.recursive == ingest_payload["recursive"]
