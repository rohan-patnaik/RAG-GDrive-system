# tests/test_services/test_llm_service.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rag_system.services.llm_service import LLMService, MAX_CONTEXT_CHAR_LENGTH
from rag_system.models.schemas import LLMProvider, RetrievedChunk, ComponentStatus, StatusEnum
from rag_system.utils.exceptions import LLMError, ConfigurationError
from rag_system.config.settings import AppSettings
from rag_system.integrations.openai_client import OpenAIClient
from rag_system.integrations.anthropic_client import AnthropicClient
from rag_system.integrations.gemini_client import GeminiClient


@pytest.fixture
def llm_test_settings(test_settings: AppSettings) -> AppSettings:
    # Ensure API keys are set for testing client initialization paths
    settings_copy = test_settings.model_copy(deep=True)
    settings_copy.OPENAI_API_KEY = "fake_openai_key"
    settings_copy.ANTHROPIC_API_KEY = "fake_anthropic_key"
    settings_copy.GOOGLE_API_KEY = "fake_google_key"
    return settings_copy


@pytest.fixture
def mock_openai_client() -> AsyncMock:
    client = AsyncMock(spec=OpenAIClient)
    client.generate_response = AsyncMock(return_value=("OpenAI answer", "gpt-test"))
    client.check_health = AsyncMock(return_value=(True, "OpenAI OK", {}))
    return client

@pytest.fixture
def mock_anthropic_client() -> AsyncMock:
    client = AsyncMock(spec=AnthropicClient)
    client.generate_response = AsyncMock(return_value=("Anthropic answer", "claude-test"))
    client.check_health = AsyncMock(return_value=(True, "Anthropic OK", {}))
    return client

@pytest.fixture
def mock_gemini_client() -> AsyncMock:
    client = AsyncMock(spec=GeminiClient)
    client.generate_response = AsyncMock(return_value=("Gemini answer", "gemini-test"))
    client.check_health = AsyncMock(return_value=(True, "Gemini OK", {}))
    return client


@pytest.fixture
def llm_service_with_mocks(
    llm_test_settings: AppSettings,
    mock_openai_client: AsyncMock,
    mock_anthropic_client: AsyncMock,
    mock_gemini_client: AsyncMock,
) -> LLMService:
    with patch("rag_system.services.llm_service.OpenAIClient", return_value=mock_openai_client), \
         patch("rag_system.services.llm_service.AnthropicClient", return_value=mock_anthropic_client), \
         patch("rag_system.services.llm_service.GeminiClient", return_value=mock_gemini_client):
        service = LLMService(settings=llm_test_settings)
        return service


def test_llm_service_initialization(llm_test_settings: AppSettings):
    """Test LLMService initializes clients based on API key presence."""
    with patch("rag_system.services.llm_service.OpenAIClient") as MockOAI, \
         patch("rag_system.services.llm_service.AnthropicClient") as MockAnthropic, \
         patch("rag_system.services.llm_service.GeminiClient") as MockGemini:

        # Case 1: All keys present
        settings_all_keys = llm_test_settings.model_copy(deep=True)
        LLMService(settings=settings_all_keys)
        MockOAI.assert_called_once()
        MockAnthropic.assert_called_once()
        MockGemini.assert_called_once()

        MockOAI.reset_mock()
        MockAnthropic.reset_mock()
        MockGemini.reset_mock()

        # Case 2: Only OpenAI key
        settings_openai_only = llm_test_settings.model_copy(deep=True)
        settings_openai_only.ANTHROPIC_API_KEY = None
        settings_openai_only.GOOGLE_API_KEY = None
        LLMService(settings=settings_openai_only)
        MockOAI.assert_called_once()
        MockAnthropic.assert_not_called()
        MockGemini.assert_not_called()


def test_get_client_not_configured(llm_test_settings: AppSettings):
    """Test _get_client raises ConfigurationError if provider not configured."""
    settings_no_openai = llm_test_settings.model_copy(deep=True)
    settings_no_openai.OPENAI_API_KEY = None # Disable OpenAI

    with patch("rag_system.services.llm_service.AnthropicClient"), \
         patch("rag_system.services.llm_service.GeminiClient"): # Mock others to prevent their init issues
        service = LLMService(settings=settings_no_openai)
        with pytest.raises(ConfigurationError, match="Openai LLM provider is not configured"):
            service._get_client(LLMProvider.OPENAI)


@pytest.mark.asyncio
async def test_generate_response_openai(
    llm_service_with_mocks: LLMService, mock_openai_client: AsyncMock
):
    """Test generating response with OpenAI provider."""
    query = "Test query"
    chunks = [RetrievedChunk(id="c1", content="ctx1", metadata={}, score=0.9)]
    answer, model_used = await llm_service_with_mocks.generate_response(
        query, chunks, LLMProvider.OPENAI
    )
    assert answer == "OpenAI answer"
    assert model_used == "gpt-test"
    mock_openai_client.generate_response.assert_called_once()
    # We can inspect the prompt passed to the client if needed
    prompt_arg = mock_openai_client.generate_response.call_args[1]['prompt']
    assert query in prompt_arg
    assert "ctx1" in prompt_arg


@pytest.mark.asyncio
async def test_generate_response_anthropic(
    llm_service_with_mocks: LLMService, mock_anthropic_client: AsyncMock
):
    """Test generating response with Anthropic provider."""
    query = "Test query"
    chunks = [] # Test with no chunks
    answer, model_used = await llm_service_with_mocks.generate_response(
        query, chunks, LLMProvider.ANTHROPIC, model_name_override="claude-custom"
    )
    assert answer == "Anthropic answer"
    assert model_used == "claude-test" # Mock returns this, not override, for simplicity of mock
    mock_anthropic_client.generate_response.assert_called_once()
    # Check that model_name_override was passed to the client
    assert mock_anthropic_client.generate_response.call_args[1]['model_name_override'] == "claude-custom"
    prompt_arg = mock_anthropic_client.generate_response.call_args[1]['prompt']
    assert "no specific context was found" in prompt_arg.lower() # Check no-context prompt part


@pytest.mark.asyncio
async def test_generate_response_gemini(
    llm_service_with_mocks: LLMService, mock_gemini_client: AsyncMock
):
    """Test generating response with Gemini provider."""
    query = "Test query"
    chunks = [RetrievedChunk(id="c1", content="ctx_gemini", metadata={}, score=0.9)]
    answer, model_used = await llm_service_with_mocks.generate_response(
        query, chunks, LLMProvider.GEMINI
    )
    assert answer == "Gemini answer"
    assert model_used == "gemini-test"
    mock_gemini_client.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_generate_response_client_error(llm_service_with_mocks: LLMService, mock_openai_client: AsyncMock):
    """Test generate_response when the client raises an LLMError."""
    mock_openai_client.generate_response.side_effect = LLMError("OpenAI client failed")
    with pytest.raises(LLMError, match="OpenAI client failed"):
        await llm_service_with_mocks.generate_response("q", [], LLMProvider.OPENAI)


def test_construct_prompt_with_context(llm_service_with_mocks: LLMService):
    """Test prompt construction with context."""
    query = "What is X?"
    chunks = [
        RetrievedChunk(id="id1", content="Context about X.", metadata={}, score=0.9),
        RetrievedChunk(id="id2", content="More context on X.", metadata={}, score=0.8),
    ]
    prompt = llm_service_with_mocks._construct_prompt(query, chunks, LLMProvider.OPENAI)
    assert query in prompt
    assert "Context about X" in prompt
    assert "More context on X" in prompt
    assert "Source 1 (ID: id1, Score: 0.9000)" in prompt


def test_construct_prompt_no_context(llm_service_with_mocks: LLMService):
    """Test prompt construction with no context."""
    query = "What is Y?"
    prompt = llm_service_with_mocks._construct_prompt(query, [], LLMProvider.GEMINI)
    assert query in prompt
    assert "no specific context was found" in prompt.lower()


def test_construct_prompt_context_truncation(llm_service_with_mocks: LLMService):
    """Test prompt construction with context truncation."""
    query = "Long query"
    long_content = "A" * (MAX_CONTEXT_CHAR_LENGTH // 2 + 100) # Ensure one chunk exceeds half
    chunks = [
        RetrievedChunk(id="c1", content=long_content, metadata={}, score=0.9),
        RetrievedChunk(id="c2", content="Short context.", metadata={}, score=0.8), # This one should be truncated
    ]
    prompt = llm_service_with_mocks._construct_prompt(query, chunks, LLMProvider.OPENAI)
    assert long_content in prompt
    assert "Short context." not in prompt # Assuming MAX_CONTEXT_CHAR_LENGTH is set appropriately for this test
    # This test is sensitive to the exact value of MAX_CONTEXT_CHAR_LENGTH and chunk formatting.


@pytest.mark.asyncio
async def test_check_provider_health_success(llm_service_with_mocks: LLMService, mock_openai_client: AsyncMock):
    """Test successful health check for a provider."""
    status = await llm_service_with_mocks.check_provider_health(LLMProvider.OPENAI)
    assert status.status == StatusEnum.OK
    assert status.name == "Openai LLM"
    assert status.message == "OpenAI OK"
    mock_openai_client.check_health.assert_called_once()


@pytest.mark.asyncio
async def test_check_provider_health_failure(llm_service_with_mocks: LLMService, mock_anthropic_client: AsyncMock):
    """Test health check failure for a provider."""
    mock_anthropic_client.check_health.return_value = (False, "Anthropic down", {"code": 500})
    status = await llm_service_with_mocks.check_provider_health(LLMProvider.ANTHROPIC)
    assert status.status == StatusEnum.ERROR
    assert status.name == "Anthropic LLM"
    assert status.message == "Anthropic down"
    assert status.details == {"code": 500}


@pytest.mark.asyncio
async def test_check_provider_health_not_configured(llm_test_settings: AppSettings):
    """Test health check for a provider that is not configured."""
    settings_no_gemini = llm_test_settings.model_copy(deep=True)
    settings_no_gemini.GOOGLE_API_KEY = None # Disable Gemini

    with patch("rag_system.services.llm_service.OpenAIClient"), \
         patch("rag_system.services.llm_service.AnthropicClient"):
        service = LLMService(settings=settings_no_gemini)
        status = await service.check_provider_health(LLMProvider.GEMINI)
        assert status.status == StatusEnum.ERROR
        assert "Not configured" in status.message
