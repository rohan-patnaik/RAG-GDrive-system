# tests/test_integrations/test_anthropic_client.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import anthropic # For exception types

from rag_system.integrations.anthropic_client import AnthropicClient
from rag_system.utils.exceptions import LLMError, ConfigurationError


@pytest.fixture
def anthropic_api_key() -> str:
    return "test_sk_anthropic_key"


@pytest.fixture
@patch("rag_system.integrations.anthropic_client.AsyncAnthropic")
def anthropic_client(MockAsyncAnthropic: AsyncMock, anthropic_api_key: str) -> AnthropicClient:
    mock_async_anthropic_instance = AsyncMock()
    MockAsyncAnthropic.return_value = mock_async_anthropic_instance
    client = AnthropicClient(api_key=anthropic_api_key, default_model="claude-3-haiku-20240307")
    client.async_client = mock_async_anthropic_instance
    return client


def test_anthropic_client_init_success(anthropic_api_key: str):
    with patch("rag_system.integrations.anthropic_client.AsyncAnthropic") as MockAsyncAnthropic:
        mock_instance = AsyncMock()
        MockAsyncAnthropic.return_value = mock_instance
        client = AnthropicClient(api_key=anthropic_api_key, default_model="claude-2.1")
        MockAsyncAnthropic.assert_called_once_with(api_key=anthropic_api_key, timeout=90.0)
        assert client.default_model == "claude-2.1"
        assert client.async_client == mock_instance


def test_anthropic_client_init_no_api_key():
    with pytest.raises(ConfigurationError, match="Anthropic API key not provided"):
        AnthropicClient(api_key=None)


@patch("rag_system.integrations.anthropic_client.AsyncAnthropic")
def test_anthropic_client_init_sdk_error(MockAsyncAnthropic: AsyncMock):
    MockAsyncAnthropic.side_effect = Exception("SDK init failed")
    with pytest.raises(ConfigurationError, match="AsyncAnthropic client initialization failed: SDK init failed"):
        AnthropicClient(api_key="some_key")


@pytest.mark.asyncio
async def test_generate_response_success(anthropic_client: AnthropicClient):
    mock_response = AsyncMock()
    mock_response.content = [MagicMock(type="text", text="Generated Anthropic response.")]
    mock_response.model = "claude-3-haiku-test" # Actual model string from API
    anthropic_client.async_client.messages.create.return_value = mock_response

    prompt = "Test prompt for Anthropic"
    answer, model_used = await anthropic_client.generate_response(prompt)

    assert answer == "Generated Anthropic response."
    assert model_used == "claude-3-haiku-test"
    anthropic_client.async_client.messages.create.assert_called_once_with(
        model=anthropic_client.default_model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )


@pytest.mark.asyncio
async def test_generate_response_with_model_override(anthropic_client: AnthropicClient):
    custom_model = "claude-3-opus-20240229"
    mock_response = AsyncMock(content=[MagicMock(type="text", text="Opus response")], model=custom_model)
    anthropic_client.async_client.messages.create.return_value = mock_response

    answer, model_used = await anthropic_client.generate_response("Prompt", model_name_override=custom_model)
    assert answer == "Opus response"
    assert model_used == custom_model
    anthropic_client.async_client.messages.create.assert_called_once_with(
        model=custom_model,
        max_tokens=2048,
        messages=[{"role": "user", "content": "Prompt"}],
        temperature=0.7,
    )


@pytest.mark.asyncio
async def test_generate_response_api_error_handling(anthropic_client: AnthropicClient):
    # Mock the response object for APIStatusError
    mock_http_response = MagicMock()
    mock_http_response.text = "Error details from API"

    error_map = {
        anthropic.APIConnectionError(request=MagicMock()): "Anthropic API connection error",
        anthropic.RateLimitError(message="Rate limit.", response=mock_http_response, body=None): "Anthropic API rate limit exceeded",
        anthropic.AuthenticationError(message="Auth error.", response=mock_http_response, body=None): "Anthropic API authentication error",
        anthropic.APIStatusError(message="Bad request.", request=MagicMock(), response=mock_http_response, status_code=400): "Anthropic API error (400)",
    }

    for api_error, expected_msg_part in error_map.items():
        anthropic_client.async_client.messages.create.side_effect = api_error
        with pytest.raises(LLMError) as exc_info:
            await anthropic_client.generate_response("prompt")
        assert expected_msg_part in str(exc_info.value)
        assert exc_info.value.provider == "Anthropic"
        if isinstance(api_error, (anthropic.APIConnectionError, anthropic.RateLimitError)) or \
           (isinstance(api_error, anthropic.APIStatusError) and api_error.status_code >= 500):
            assert exc_info.value.is_retryable is True
        elif isinstance(api_error, anthropic.AuthenticationError):
            assert exc_info.value.is_retryable is False


@pytest.mark.asyncio
async def test_generate_response_no_content(anthropic_client: AnthropicClient):
    mock_response = AsyncMock(content=None) # No content block
    mock_response.model_dump_json.return_value = "{}"
    anthropic_client.async_client.messages.create.return_value = mock_response
    with pytest.raises(LLMError, match="Anthropic API returned no content or unexpected format"):
        await anthropic_client.generate_response("prompt")


@pytest.mark.asyncio
async def test_generate_response_non_text_block(anthropic_client: AnthropicClient):
    mock_response = AsyncMock(content=[MagicMock(type="image", source=MagicMock())])
    anthropic_client.async_client.messages.create.return_value = mock_response
    with pytest.raises(LLMError, match="Anthropic API returned non-text block: image"):
        await anthropic_client.generate_response("prompt")


@pytest.mark.asyncio
async def test_check_health_success(anthropic_client: AnthropicClient):
    # The health check itself calls messages.create, so we mock that
    anthropic_client.async_client.messages.create.return_value = AsyncMock() # Minimal successful response

    is_healthy, msg, details = await anthropic_client.check_health()
    assert is_healthy is True
    assert "Successfully connected" in msg
    assert details["default_model_tested"] == anthropic_client.default_model
    anthropic_client.async_client.messages.create.assert_called_once_with(
        model=anthropic_client.default_model,
        max_tokens=1,
        messages=[{"role": "user", "content": "Health check ping."}],
        timeout=10.0
    )


@pytest.mark.asyncio
async def test_check_health_api_error(anthropic_client: AnthropicClient):
    anthropic_client.async_client.messages.create.side_effect = anthropic.AuthenticationError(
        message="Invalid API key", response=MagicMock(), body=None
    )
    is_healthy, msg, details = await anthropic_client.check_health()
    assert is_healthy is False
    assert "Authentication failed" in msg
    assert details["error_type"] == "AuthenticationError"
