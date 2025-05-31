import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from pydantic import SecretStr

from rag_system.services.llm_service import LLMService
from rag_system.models.schemas import LLMProvider, ComponentStatus, StatusEnum
from rag_system.config.settings import AppSettings
from rag_system.utils.exceptions import ConfigurationError, LLMError

# Fixtures for AppSettings
@pytest.fixture
def mock_settings_all_configured():
    """AppSettings with all major LLM providers configured."""
    return AppSettings(
        OPENAI_API_KEY=SecretStr("fake_openai_key"), DEFAULT_OPENAI_MODEL="gpt-3.5-turbo",
        ANTHROPIC_API_KEY=SecretStr("fake_anthropic_key"), DEFAULT_ANTHROPIC_MODEL="claude-2",
        GOOGLE_API_KEY=SecretStr("fake_google_key"), DEFAULT_GEMINI_MODEL="gemini-pro",
        # Minimal other settings that might be required by AppSettings or LLMService init
        RAG_APP_NAME="TestAppHealthCheck",
        LOG_LEVEL="DEBUG",
        DB_TYPE="memory",
        VECTOR_DB_TYPE="chroma",
        DEFAULT_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
        CHROMA_PERSIST_DIRECTORY="/tmp/chroma_test_health",
        CHROMA_COLLECTION_NAME="test_collection_health"
    )

@pytest.fixture
def mock_settings_openai_only():
    """AppSettings with only OpenAI configured."""
    return AppSettings(
        OPENAI_API_KEY=SecretStr("fake_openai_key"), DEFAULT_OPENAI_MODEL="gpt-3.5-turbo",
        ANTHROPIC_API_KEY=None, DEFAULT_ANTHROPIC_MODEL="claude-2", # Explicitly None
        GOOGLE_API_KEY=SecretStr(""), DEFAULT_GEMINI_MODEL="gemini-pro",       # Explicitly empty SecretStr
        RAG_APP_NAME="TestAppHealthCheckOpenAI",
        LOG_LEVEL="DEBUG",
        DB_TYPE="memory",
        VECTOR_DB_TYPE="chroma",
        DEFAULT_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
        CHROMA_PERSIST_DIRECTORY="/tmp/chroma_test_health_openai",
        CHROMA_COLLECTION_NAME="test_collection_health_openai"
    )

@pytest.mark.asyncio
class TestLLMServiceHealthChecks:

    async def test_check_provider_health_healthy(self, mock_settings_all_configured: AppSettings):
        llm_service = LLMService(settings=mock_settings_all_configured)

        mock_client = MagicMock()
        # Make check_health an AsyncMock if it's not already by virtue of being part of a MagicMock
        # that is later used in an async context. Explicitly making it AsyncMock is safer.
        mock_client.check_health = AsyncMock(return_value=(True, "Provider is Healthy", {"details": "all_ok"}))

        # Patch _get_client for the LLMService instance
        with patch.object(llm_service, '_get_client', return_value=mock_client) as mock_get_client_method:
            for provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, LLMProvider.GEMINI]:
                status_report = await llm_service.check_provider_health(provider)

                mock_get_client_method.assert_called_with(provider)
                mock_client.check_health.assert_called_once() # Assert it was called

                assert status_report.status == StatusEnum.OK
                assert status_report.message == "Provider is Healthy"
                # The details are merged, so check for inclusion
                assert status_report.details.get("configured") is True
                assert status_report.details.get("connection") == "success"
                assert status_report.details.get("details") == "all_ok"

                mock_client.check_health.reset_mock() # Reset for the next provider in the loop
                mock_get_client_method.reset_mock() # Reset for the next provider

    async def test_check_provider_health_unhealthy(self, mock_settings_all_configured: AppSettings):
        llm_service = LLMService(settings=mock_settings_all_configured)

        mock_client = MagicMock()
        mock_client.check_health = AsyncMock(return_value=(False, "Provider is Unhealthy", {"details": "connection_failed"}))

        with patch.object(llm_service, '_get_client', return_value=mock_client) as mock_get_client_method:
            status_report = await llm_service.check_provider_health(LLMProvider.OPENAI)

            mock_get_client_method.assert_called_with(LLMProvider.OPENAI)
            mock_client.check_health.assert_called_once()

            assert status_report.status == StatusEnum.ERROR
            assert status_report.message == "Provider is Unhealthy"
            assert status_report.details.get("configured") is True
            assert status_report.details.get("connection") == "failed"
            assert status_report.details.get("details") == "connection_failed"

    async def test_check_provider_health_key_not_configured(self, mock_settings_openai_only: AppSettings):
        llm_service = LLMService(settings=mock_settings_openai_only)

        # Test Anthropic (API key is None)
        status_anthropic = await llm_service.check_provider_health(LLMProvider.ANTHROPIC)
        assert status_anthropic.status == StatusEnum.DEGRADED
        assert "API key not configured for anthropic (checked ANTHROPIC_API_KEY)" in status_anthropic.message
        assert status_anthropic.details == {"configured": False, "reason": "missing_api_key"}

        # Test Gemini (API key is "")
        status_gemini = await llm_service.check_provider_health(LLMProvider.GEMINI)
        assert status_gemini.status == StatusEnum.DEGRADED
        assert "API key not configured for gemini (checked GOOGLE_API_KEY)" in status_gemini.message
        assert status_gemini.details == {"configured": False, "reason": "missing_api_key"}

    async def test_check_provider_health_client_check_health_raises_llm_error(self, mock_settings_all_configured: AppSettings):
        llm_service = LLMService(settings=mock_settings_all_configured)

        mock_client = MagicMock()
        # Simulate LLMError from client's check_health
        llm_error_instance = LLMError(message="Client-side LLM Error", detail="Connection Timeout")
        mock_client.check_health = AsyncMock(side_effect=llm_error_instance)

        with patch.object(llm_service, '_get_client', return_value=mock_client) as mock_get_client_method:
            status_report = await llm_service.check_provider_health(LLMProvider.OPENAI)

            mock_get_client_method.assert_called_with(LLMProvider.OPENAI)
            mock_client.check_health.assert_called_once()

            assert status_report.status == StatusEnum.ERROR
            assert f"Health check failed: {str(llm_error_instance)}" in status_report.message
            assert status_report.details.get("configured") is True
            assert str(llm_error_instance) in status_report.details.get("error", "")

    async def test_check_provider_health_client_check_health_raises_general_exception(self, mock_settings_all_configured: AppSettings):
        llm_service = LLMService(settings=mock_settings_all_configured)

        mock_client = MagicMock()
        general_exception_instance = Exception("Some other error")
        mock_client.check_health = AsyncMock(side_effect=general_exception_instance)

        with patch.object(llm_service, '_get_client', return_value=mock_client) as mock_get_client_method:
            status_report = await llm_service.check_provider_health(LLMProvider.OPENAI)

            mock_get_client_method.assert_called_with(LLMProvider.OPENAI)
            mock_client.check_health.assert_called_once()

            assert status_report.status == StatusEnum.ERROR
            assert f"Health check failed: {str(general_exception_instance)}" in status_report.message
            assert status_report.details.get("configured") is True
            assert str(general_exception_instance) in status_report.details.get("error", "")


    async def test_check_provider_health_get_client_raises_config_error(self, mock_settings_all_configured: AppSettings):
        llm_service = LLMService(settings=mock_settings_all_configured)

        config_error_instance = ConfigurationError("Client init failed for test")
        # Patch _get_client to raise ConfigurationError
        with patch.object(llm_service, '_get_client', side_effect=config_error_instance) as mock_get_client_method:
            status_report = await llm_service.check_provider_health(LLMProvider.GEMINI)

            mock_get_client_method.assert_called_with(LLMProvider.GEMINI)

            assert status_report.status == StatusEnum.NOT_CONFIGURED
            assert f"Provider gemini not configured: {str(config_error_instance)}" in status_report.message
            assert status_report.details == {"configured": False, "reason": str(config_error_instance)}

    async def test_check_provider_health_local_provider_placeholder(self, mock_settings_all_configured: AppSettings):
        llm_service = LLMService(settings=mock_settings_all_configured)

        status_report = await llm_service.check_provider_health(LLMProvider.LOCAL)

        assert status_report.status == StatusEnum.OK
        assert "Local LLM health check not fully implemented yet." in status_report.message
        assert status_report.details == {"configured": True, "info": "placeholder_status"}

    async def test_check_provider_health_openai_key_missing_direct_check(self, mock_settings_openai_only: AppSettings):
        """Test case where a key is missing, relying on the direct check in check_provider_health."""
        # Modify settings to remove OpenAI key after init (less common, but tests the check)
        llm_service = LLMService(settings=mock_settings_openai_only)

        # Test Anthropic (key is None in fixture)
        status_anthropic = await llm_service.check_provider_health(LLMProvider.ANTHROPIC)
        assert status_anthropic.status == StatusEnum.DEGRADED
        assert "API key not configured for anthropic" in status_anthropic.message.lower()
        assert status_anthropic.details.get("reason") == "missing_api_key"

        # Test Gemini (key is "" in fixture)
        status_gemini = await llm_service.check_provider_health(LLMProvider.GEMINI)
        assert status_gemini.status == StatusEnum.DEGRADED
        assert "API key not configured for gemini" in status_gemini.message.lower()
        assert status_gemini.details.get("reason") == "missing_api_key"

        # For OpenAI, key IS configured in this fixture, so it should try to get client
        # We need to mock _get_client for this part for OpenAI to avoid actual client calls
        mock_client_openai = MagicMock()
        mock_client_openai.check_health = AsyncMock(return_value=(True, "OpenAI Healthy", {}))
        with patch.object(llm_service, '_get_client', return_value=mock_client_openai) as mock_get_client_openai:
            status_openai = await llm_service.check_provider_health(LLMProvider.OPENAI)
            assert status_openai.status == StatusEnum.OK
            mock_get_client_openai.assert_called_with(LLMProvider.OPENAI)
            mock_client_openai.check_health.assert_called_once()
            assert "OpenAI Healthy" in status_openai.message
