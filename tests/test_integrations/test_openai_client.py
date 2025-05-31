# tests/test_integrations/test_openai_client.py
import pytest
from unittest.mock import AsyncMock, patch
import openai # For exception types

from rag_system.integrations.openai_client import OpenAIClient
from rag_system.utils.exceptions import LLMError, ConfigurationError


@pytest.fixture
def openai_api_key() -> str:
    return "test_sk_openai_key"


@pytest.fixture
@patch("rag_system.integrations.openai_client.AsyncOpenAI")
def openai_client(MockAsyncOpenAI: AsyncMock, openai_api_key: str) -> OpenAIClient:
    mock_async_openai_instance = AsyncMock()
    MockAsyncOpenAI.return_value = mock_async_openai_instance
    client = OpenAIClient(api_key=openai_api_key, default_model="gpt-3.5-turbo")
    client.async_client = mock_async_openai_instance # Ensure the instance uses the mock
    return client


def test_openai_client_init_success(openai_api_key: str):
    """Test successful initialization of OpenAIClient."""
    with patch("rag_system.integrations.openai_client.AsyncOpenAI") as MockAsyncOpenAI:
        mock_instance = AsyncMock()
        MockAsyncOpenAI.return_value = mock_instance
        client = OpenAIClient(api_key=openai_api_key, default_model="gpt-4")
        MockAsyncOpenAI.assert_called_once_with(api_key=openai_api_key, timeout=60.0)
        assert client.default_model == "gpt-4"
        assert client.async_client == mock_instance


def test_openai_client_init_no_api_key():
    """Test OpenAIClient initialization fails if API key is not provided."""
    with pytest.raises(ConfigurationError, match="OpenAI API key not provided"):
        OpenAIClient(api_key=None)


@patch("rag_system.integrations.openai_client.AsyncOpenAI")
def test_openai_client_init_sdk_error(MockAsyncOpenAI: AsyncMock):
    """Test OpenAIClient init fails if AsyncOpenAI constructor raises an error."""
    MockAsyncOpenAI.side_effect = Exception("SDK init failed")
    with pytest.raises(ConfigurationError, match="AsyncOpenAI client initialization failed: SDK init failed"):
        OpenAIClient(api_key="some_key")


@pytest.mark.asyncio
async def test_generate_response_success(openai_client: OpenAIClient):
    """Test successful response generation."""
    mock_completion_response = AsyncMock()
    mock_completion_response.choices = [AsyncMock()]
    mock_completion_response.choices[0].message = AsyncMock()
    mock_completion_response.choices[0].message.content = "Generated OpenAI response."
    mock_completion_response.model = "gpt-3.5-turbo-0125" # Actual model string from API

    openai_client.async_client.chat.completions.create.return_value = mock_completion_response

    prompt = "Test prompt for OpenAI"
    answer, model_used = await openai_client.generate_response(prompt)

    assert answer == "Generated OpenAI response."
    assert model_used == "gpt-3.5-turbo-0125"
    openai_client.async_client.chat.completions.create.assert_called_once_with(
        model=openai_client.default_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )


@pytest.mark.asyncio
async def test_generate_response_with_model_override(openai_client: OpenAIClient):
    """Test response generation with a model override."""
    custom_model = "gpt-4-turbo"
    mock_completion_response = AsyncMock() # As above
    mock_completion_response.choices = [AsyncMock(message=AsyncMock(content="GPT-4 response"))]
    mock_completion_response.model = custom_model
    openai_client.async_client.chat.completions.create.return_value = mock_completion_response

    answer, model_used = await openai_client.generate_response("Prompt", model_name_override=custom_model)
    assert answer == "GPT-4 response"
    assert model_used == custom_model
    openai_client.async_client.chat.completions.create.assert_called_once_with(
        model=custom_model, # Check overridden model
        messages=[{"role": "user", "content": "Prompt"}],
        temperature=0.7,
    )


@pytest.mark.asyncio
async def test_generate_response_api_error_handling(openai_client: OpenAIClient):
    """Test handling of various OpenAI API errors."""
    error_map = {
        openai.APIConnectionError(request=MagicMock()): "OpenAI API connection error",
        openai.RateLimitError(message="Rate limit.", response=MagicMock(), body=None): "OpenAI API rate limit exceeded",
        openai.AuthenticationError(message="Auth error.", response=MagicMock(), body=None): "OpenAI API authentication error",
        openai.APIStatusError(message="Bad request.", request=MagicMock(), response=MagicMock(status_code=400)): "OpenAI API error (400)",
    }

    for api_error, expected_msg_part in error_map.items():
        openai_client.async_client.chat.completions.create.side_effect = api_error
        with pytest.raises(LLMError) as exc_info:
            await openai_client.generate_response("prompt")
        assert expected_msg_part in str(exc_info.value)
        assert exc_info.value.provider == "OpenAI"
        if isinstance(api_error, (openai.APIConnectionError, openai.RateLimitError)) or \
           (isinstance(api_error, openai.APIStatusError) and api_error.status_code >= 500):
            assert exc_info.value.is_retryable is True
        elif isinstance(api_error, openai.AuthenticationError):
             assert exc_info.value.is_retryable is False


@pytest.mark.asyncio
async def test_generate_response_no_choices(openai_client: OpenAIClient):
    """Test response generation when API returns no choices."""
    mock_response = AsyncMock()
    mock_response.choices = []
    mock_response.model_dump_json.return_value = "{}" # For logging
    openai_client.async_client.chat.completions.create.return_value = mock_response

    with pytest.raises(LLMError, match="OpenAI API returned no choices or message"):
        await openai_client.generate_response("prompt")


@pytest.mark.asyncio
async def test_generate_response_empty_content(openai_client: OpenAIClient):
    """Test response generation when API returns empty message content."""
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock(message=AsyncMock(content=None))]
    mock_response.model = "gpt-test"
    openai_client.async_client.chat.completions.create.return_value = mock_response

    with pytest.raises(LLMError, match="OpenAI API returned empty message content"):
        await openai_client.generate_response("prompt")


@pytest.mark.asyncio
async def test_check_health_success(openai_client: OpenAIClient):
    """Test successful health check."""
    mock_models_list = AsyncMock()
    mock_models_list.data = [MagicMock(id="gpt-3.5-turbo"), MagicMock(id="gpt-4")]
    openai_client.async_client.models.list.return_value = mock_models_list

    is_healthy, msg, details = await openai_client.check_health()

    assert is_healthy is True
    assert "Successfully connected" in msg
    assert details["models_count"] == 2
    assert details["default_model_family_available"] is True # Assuming default is gpt-3.5


@pytest.mark.asyncio
async def test_check_health_no_models_returned(openai_client: OpenAIClient):
    """Test health check when API returns no models."""
    mock_models_list = AsyncMock()
    mock_models_list.data = []
    openai_client.async_client.models.list.return_value = mock_models_list

    is_healthy, msg, _ = await openai_client.check_health()
    assert is_healthy is False
    assert "No models returned from API" in msg


@pytest.mark.asyncio
async def test_check_health_api_error(openai_client: OpenAIClient):
    """Test health check when models.list raises an API error."""
    openai_client.async_client.models.list.side_effect = openai.AuthenticationError(
        message="Invalid API key", response=MagicMock(), body=None
    )
    is_healthy, msg, details = await openai_client.check_health()
    assert is_healthy is False
    assert "Authentication failed" in msg
    assert details["error_type"] == "AuthenticationError"
