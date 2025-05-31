# tests/test_api/test_query_endpoint.py
import pytest
from httpx import AsyncClient
from fastapi import status, FastAPI
from unittest.mock import AsyncMock, cast

from rag_system.models.schemas import (
    QueryRequest, QueryResponse, LLMProvider, RetrievedChunk,
    SystemStatusResponse, StatusEnum, ComponentStatus
)
from rag_system.utils.exceptions import QueryProcessingError, LLMError, VectorStoreError
from rag_system.config.settings import AppSettings


@pytest.mark.asyncio
async def test_query_rag_system_success(async_client: AsyncClient, test_app: FastAPI):
    """Test successful query via /query/ endpoint."""
    query_payload = {"query_text": "What is the capital of France?"}
    expected_response_data = {
        "query_text": query_payload["query_text"],
        "llm_answer": "The capital of France is Paris.",
        "llm_provider_used": LLMProvider.GEMINI.value, # Default from test_settings
        "llm_model_used": "gemini-test-model",
        "retrieved_chunks": [
            {"id": "chunk1", "content": "Paris is the capital.", "metadata": {}, "score": 0.95}
        ],
    }

    mock_rag_service = cast(AsyncMock, test_app.state.rag_service)
    # Ensure retrieved_chunks are instances of RetrievedChunk for the mock
    expected_response_data_obj = expected_response_data.copy()
    expected_response_data_obj["retrieved_chunks"] = [
        RetrievedChunk(**chunk_data) for chunk_data in expected_response_data["retrieved_chunks"]
    ]
    mock_rag_service.query.return_value = QueryResponse(**expected_response_data_obj)

    response = await async_client.post("/query/", json=query_payload)

    assert response.status_code == status.HTTP_200_OK
    response_data = response.json()
    assert response_data["llm_answer"] == expected_response_data["llm_answer"]
    assert response_data["llm_provider_used"] == expected_response_data["llm_provider_used"]
    assert len(response_data["retrieved_chunks"]) == 1

    mock_rag_service.query.assert_called_once()
    call_args: QueryRequest = mock_rag_service.query.call_args[0][0]
    assert call_args.query_text == query_payload["query_text"]
    # Check defaults from settings are applied if not in payload
    app_settings: AppSettings = test_app.state.settings
    assert call_args.llm_provider == LLMProvider(app_settings.DEFAULT_LLM_PROVIDER)
    assert call_args.top_k == app_settings.TOP_K_RESULTS


@pytest.mark.asyncio
async def test_query_rag_system_with_overrides(async_client: AsyncClient, test_app: FastAPI):
    """Test /query/ endpoint with provider, top_k, and model overrides."""
    query_payload = {
        "query_text": "Tell me about Claude.",
        "llm_provider": "anthropic",
        "llm_model_name": "claude-3-opus-20240229",
        "top_k": 5,
        "similarity_threshold": 0.75
    }
    expected_response_data = {
        "query_text": query_payload["query_text"],
        "llm_answer": "Claude is an AI assistant by Anthropic.",
        "llm_provider_used": LLMProvider.ANTHROPIC.value,
        "llm_model_used": query_payload["llm_model_name"], # Expect override to be used
        "retrieved_chunks": [],
    }
    mock_rag_service = cast(AsyncMock, test_app.state.rag_service)
    mock_rag_service.query.return_value = QueryResponse(**expected_response_data)

    response = await async_client.post("/query/", json=query_payload)

    assert response.status_code == status.HTTP_200_OK
    response_data = response.json()
    assert response_data["llm_provider_used"] == "anthropic"
    assert response_data["llm_model_used"] == query_payload["llm_model_name"]

    mock_rag_service.query.assert_called_once()
    call_args: QueryRequest = mock_rag_service.query.call_args[0][0]
    assert call_args.llm_provider == LLMProvider.ANTHROPIC
    assert call_args.llm_model_name == query_payload["llm_model_name"]
    assert call_args.top_k == query_payload["top_k"]
    assert call_args.similarity_threshold == query_payload["similarity_threshold"]


@pytest.mark.asyncio
async def test_query_rag_system_processing_error(async_client: AsyncClient, test_app: FastAPI):
    """Test /query/ endpoint when RAGService.query raises QueryProcessingError."""
    query_payload = {"query_text": "This query will fail."}
    mock_rag_service = cast(AsyncMock, test_app.state.rag_service)
    mock_rag_service.query.side_effect = QueryProcessingError(
        message="Failed to process query", detail="Internal component failure", status_code=500
    )

    response = await async_client.post("/query/", json=query_payload)

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    response_data = response.json()
    assert response_data["error"] == "QueryProcessingError"
    assert "Failed to process query" in response_data["message"]


@pytest.mark.asyncio
async def test_query_rag_system_llm_error(async_client: AsyncClient, test_app: FastAPI):
    """Test /query/ endpoint when RAGService.query raises LLMError."""
    query_payload = {"query_text": "LLM error query."}
    mock_rag_service = cast(AsyncMock, test_app.state.rag_service)
    mock_rag_service.query.side_effect = LLMError(
        message="LLM unavailable", provider="openai", status_code=503
    )

    response = await async_client.post("/query/", json=query_payload)

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    response_data = response.json()
    assert response_data["error"] == "LLMError"
    assert "LLM unavailable" in response_data["message"]


@pytest.mark.asyncio
async def test_query_rag_system_validation_error(async_client: AsyncClient):
    """Test /query/ endpoint with invalid payload (e.g., missing query_text)."""
    invalid_payload = {"llm_provider": "openai"} # Missing query_text
    response = await async_client.post("/query/", json=invalid_payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_get_query_system_status_ok(async_client: AsyncClient, test_app: FastAPI):
    """Test the /query/status endpoint when system is healthy."""
    expected_status_data = {
        "system_status": "OK",
        "app_name": test_app.state.settings.APP_NAME,
        "environment": test_app.state.settings.ENVIRONMENT,
        "version": "0.1.0", # This comes from app_version in health.py, adjust if dynamic
        "components": [
            {"name": "Mocked LLM", "status": "OK", "message": "Healthy"}
        ]
    }
    mock_rag_service = cast(AsyncMock, test_app.state.rag_service)
    # RAGService.get_system_status is used by /query/status route
    mock_rag_service.get_system_status.return_value = SystemStatusResponse(
        system_status=StatusEnum.OK,
        app_name=test_app.state.settings.APP_NAME,
        environment=test_app.state.settings.ENVIRONMENT,
        version="0.1.0",
        components=[ComponentStatus(name="Mocked LLM", status=StatusEnum.OK, message="Healthy")]
    )


    response = await async_client.get("/query/status")
    assert response.status_code == status.HTTP_200_OK
    response_data = response.json()

    assert response_data["system_status"] == expected_status_data["system_status"]
    assert response_data["app_name"] == expected_status_data["app_name"]
    assert len(response_data["components"]) == 1
    assert response_data["components"][0]["name"] == "Mocked LLM"


@pytest.mark.asyncio
async def test_get_query_system_status_error(async_client: AsyncClient, test_app: FastAPI):
    """Test /query/status when RAGService.get_system_status indicates an error."""
    mock_rag_service = cast(AsyncMock, test_app.state.rag_service)
    mock_rag_service.get_system_status.return_value = SystemStatusResponse(
        system_status=StatusEnum.ERROR,
        components=[ComponentStatus(name="VectorStore", status=StatusEnum.ERROR, message="Down")]
    )

    response = await async_client.get("/query/status")
    assert response.status_code == status.HTTP_200_OK # Endpoint is up, body shows error
    response_data = response.json()
    assert response_data["system_status"] == "ERROR"
    assert response_data["components"][0]["name"] == "VectorStore"
    assert response_data["components"][0]["status"] == "ERROR"
