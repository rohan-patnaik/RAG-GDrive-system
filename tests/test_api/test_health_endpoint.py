# tests/test_api/test_health_endpoint.py
import pytest
from httpx import AsyncClient
from fastapi import status, FastAPI

from rag_system.models.schemas import HealthResponse, StatusEnum, ComponentStatus, SystemStatusResponse
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_health_check_all_ok(async_client: AsyncClient, test_app: FastAPI):
    """Test the /health endpoint when all components are OK."""
    # Mock the RAGService's get_system_status to return an all-OK response
    mock_status_response = SystemStatusResponse(
        system_status=StatusEnum.OK,
        app_name="TestRAG",
        environment="testing",
        version="0.1.0-test",
        components=[
            ComponentStatus(name="EmbeddingService", status=StatusEnum.OK, message="Loaded"),
            ComponentStatus(name="VectorStoreService", status=StatusEnum.OK, message="Connected"),
            ComponentStatus(name="Gemini LLM", status=StatusEnum.OK, message="Healthy"),
        ]
    )
    # Get the mock RAGService from the app state (configured in conftest.py for test_app)
    mock_rag_service = cast(AsyncMock, test_app.state.rag_service)
    mock_rag_service.get_system_status.return_value = mock_status_response

    response = await async_client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    health_data = HealthResponse(**response.json()) # Validate response schema

    assert health_data.system_status == StatusEnum.OK
    assert health_data.app_name == test_app.state.settings.APP_NAME # From test_settings via app.state
    assert len(health_data.components) == 3
    for comp in health_data.components:
        assert comp.status == StatusEnum.OK


@pytest.mark.asyncio
async def test_health_check_component_error(async_client: AsyncClient, test_app: FastAPI):
    """Test the /health endpoint when a component reports an error."""
    mock_status_response = SystemStatusResponse(
        system_status=StatusEnum.ERROR, # Overall status reflects component error
        components=[
            ComponentStatus(name="EmbeddingService", status=StatusEnum.OK, message="Loaded"),
            ComponentStatus(name="VectorStoreService", status=StatusEnum.ERROR, message="Connection failed"),
        ]
    )
    mock_rag_service = cast(AsyncMock, test_app.state.rag_service)
    mock_rag_service.get_system_status.return_value = mock_status_response

    response = await async_client.get("/health")

    assert response.status_code == status.HTTP_200_OK # Endpoint itself is OK, body indicates error
    health_data = HealthResponse(**response.json())

    assert health_data.system_status == StatusEnum.ERROR
    vs_status = next(c for c in health_data.components if c.name == "VectorStoreService")
    assert vs_status.status == StatusEnum.ERROR
    assert "Connection failed" in vs_status.message


@pytest.mark.asyncio
async def test_health_check_rag_service_unavailable(async_client: AsyncClient, test_app: FastAPI):
    """Test /health when RAGService itself is unavailable (e.g., failed to init in lifespan)."""
    # Simulate RAGService not being available on app.state
    # In conftest, test_app fixture already mocks rag_service.
    # To test this scenario, we'd need a different app fixture or modify app.state.
    # For this test, let's assume get_system_status raises an exception.
    mock_rag_service = cast(AsyncMock, test_app.state.rag_service)
    mock_rag_service.get_system_status.side_effect = Exception("RAGService completely down")

    response = await async_client.get("/health")
    assert response.status_code == status.HTTP_200_OK # Endpoint is up, but reports internal error

    health_data = HealthResponse(**response.json())
    assert health_data.system_status == StatusEnum.ERROR
    # Check if a component status reflects this failure
    assert any(
        "RAGService Internal Status" in comp.name and comp.status == StatusEnum.ERROR
        for comp in health_data.components
    )


@pytest.mark.asyncio
async def test_health_check_startup_error(async_client: AsyncClient, test_app: FastAPI):
    """Test /health when a startup error is registered on app.state."""
    # Simulate a startup error being set on app.state
    # This would typically happen in the lifespan manager if a critical service fails to load.
    original_startup_error = getattr(test_app.state, 'startup_error', None)
    test_app.state.startup_error = "Critical service X failed during startup."
    # We don't need to mock rag_service.get_system_status because the health check
    # should prioritize showing the startup_error.

    response = await async_client.get("/health")
    assert response.status_code == status.HTTP_200_OK

    health_data = HealthResponse(**response.json())
    assert health_data.system_status == StatusEnum.ERROR
    app_startup_status = next(c for c in health_data.components if c.name == "Application Startup")
    assert app_startup_status.status == StatusEnum.ERROR
    assert "Critical service X failed" in app_startup_status.message

    # Clean up the modified app.state
    if original_startup_error is not None:
        test_app.state.startup_error = original_startup_error
    else:
        delattr(test_app.state, 'startup_error')
