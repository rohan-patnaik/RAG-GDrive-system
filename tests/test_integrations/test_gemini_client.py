# tests/test_integrations/test_gemini_client.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import google.generativeai as genai # For exception types and model classes
from google.api_core.exceptions import PermissionDenied, ResourceExhausted, GoogleAPIError # type: ignore

from rag_system.integrations.gemini_client import GeminiClient
from rag_system.utils.exceptions import LLMError, ConfigurationError


@pytest.fixture
def google_api_key() -> str:
    return "test_google_gemini_api_key"


@pytest.fixture
def mock_generative_model_instance() -> AsyncMock:
    """Mocks an instance of genai.GenerativeModel."""
    mock_model = AsyncMock(spec=genai.GenerativeModel)
    # Mock the generate_content_async method
    mock_response = AsyncMock()
    mock_response.text = "Generated Gemini response."
    # Simulate candidates and finish reason for successful case
    mock_candidate = MagicMock()
    mock_candidate.finish_reason = 1 # "STOP"
    mock_candidate.safety_ratings = []
    mock_response.candidates = [mock_candidate]
    mock_response.prompt_feedback = MagicMock(block_reason=None)
    mock_model.generate_content_async.return_value = mock_response
    return mock_model


@pytest.fixture
@patch("rag_system.integrations.gemini_client.genai.configure")
@patch("rag_system.integrations.gemini_client.genai.GenerativeModel")
def gemini_client(
    MockGenerativeModel: MagicMock, # This is the class mock
    mock_genai_configure: MagicMock,
    google_api_key: str,
    mock_generative_model_instance: AsyncMock # This is the instance mock
) -> GeminiClient:
    MockGenerativeModel.return_value = mock_generative_model_instance # Class mock returns instance mock
    client = GeminiClient(api_key=google_api_key, default_model="gemini-2.5-flash-preview-05-20")
    # The client's self.model should be the mock_generative_model_instance
    return client


def test_gemini_client_init_success(google_api_key: str):
    with patch("rag_system.integrations.gemini_client.genai.configure") as mock_configure, \
         patch("rag_system.integrations.gemini_client.genai.GenerativeModel") as MockGenerativeModel:
        mock_model_instance = AsyncMock()
        MockGenerativeModel.return_value = mock_model_instance

        client = GeminiClient(api_key=google_api_key, default_model="gemini-pro")

        mock_configure.assert_called_once_with(api_key=google_api_key)
        MockGenerativeModel.assert_called_once_with("gemini-pro")
        assert client.default_model_name == "gemini-pro"
        assert client.model == mock_model_instance


def test_gemini_client_init_no_api_key():
    with pytest.raises(ConfigurationError, match="Google API key (for Gemini) not provided"):
        GeminiClient(api_key=None)


@patch("rag_system.integrations.gemini_client.genai.configure")
def test_gemini_client_init_configure_error(mock_genai_configure: MagicMock):
    mock_genai_configure.side_effect = Exception("genai.configure failed")
    with pytest.raises(ConfigurationError, match="Google Gemini client configuration failed: genai.configure failed"):
        GeminiClient(api_key="some_key")


@patch("rag_system.integrations.gemini_client.genai.configure")
@patch("rag_system.integrations.gemini_client.genai.GenerativeModel")
def test_gemini_client_init_model_init_error(MockGenerativeModel: MagicMock, mock_genai_configure: MagicMock):
    MockGenerativeModel.side_effect = Exception("GenerativeModel init failed")
    with pytest.raises(ConfigurationError, match="Google Gemini client configuration failed: GenerativeModel init failed"):
        GeminiClient(api_key="some_key")


@pytest.mark.asyncio
async def test_generate_response_success(gemini_client: GeminiClient, mock_generative_model_instance: AsyncMock):
    prompt = "Test prompt for Gemini"
    answer, model_used = await gemini_client.generate_response(prompt)

    assert answer == "Generated Gemini response."
    assert model_used == gemini_client.default_model_name # Default model was used
    mock_generative_model_instance.generate_content_async.assert_called_once_with(
        contents=prompt
    )


@pytest.mark.asyncio
@patch("rag_system.integrations.gemini_client.genai.GenerativeModel") # To mock creation of new model instance
async def test_generate_response_with_model_override(
    MockGenerativeModelOverride: MagicMock,
    gemini_client: GeminiClient, # Original client with its default model mock
    google_api_key: str # Needed if GeminiClient re-initializes model
):
    custom_model_name = "gemini-1.5-pro-latest"
    # Mock the new instance that would be created for the override
    mock_override_model_instance = AsyncMock(spec=genai.GenerativeModel)
    mock_override_response = AsyncMock(text="Pro model response.")
    mock_override_candidate = MagicMock(finish_reason=1, safety_ratings=[])
    mock_override_response.candidates = [mock_override_candidate]
    mock_override_response.prompt_feedback = MagicMock(block_reason=None)
    mock_override_model_instance.generate_content_async.return_value = mock_override_response
    MockGenerativeModelOverride.return_value = mock_override_model_instance

    # Ensure the original client's model mock is not used for this call
    # gemini_client.model is already a mock from the fixture

    answer, model_used = await gemini_client.generate_response("Prompt", model_name_override=custom_model_name)

    assert answer == "Pro model response."
    assert model_used == custom_model_name
    MockGenerativeModelOverride.assert_called_once_with(custom_model_name) # Check new model was initialized
    mock_override_model_instance.generate_content_async.assert_called_once_with(contents="Prompt")
    # Ensure original model mock was not called
    assert gemini_client.model.generate_content_async.call_count == 0


@pytest.mark.asyncio
async def test_generate_response_api_error_handling(gemini_client: GeminiClient, mock_generative_model_instance: AsyncMock):
    error_map = {
        PermissionDenied("Permission denied error", errors=()): "Gemini API error (None): Permission denied error", # Code might be None for some GoogleAPIError
        ResourceExhausted("Quota exceeded", errors=()): "Gemini API error (None): Quota exceeded",
        GoogleAPIError("Generic Google API error", errors=()): "Gemini API error (None): Generic Google API error",
    }
    # For GoogleAPIError, the code attribute might not always be present or might be None.
    # The string representation of the exception usually includes the status code if available.

    for api_error_instance, expected_msg_part in error_map.items():
        # Manually set code if it's usually part of the error type for more specific matching
        if isinstance(api_error_instance, PermissionDenied): setattr(api_error_instance, 'code', 403)
        if isinstance(api_error_instance, ResourceExhausted): setattr(api_error_instance, 'code', 429)

        mock_generative_model_instance.generate_content_async.side_effect = api_error_instance
        with pytest.raises(LLMError) as exc_info:
            await gemini_client.generate_response("prompt")

        # More flexible matching due to variability in GoogleAPIError string format
        assert "Gemini API error" in str(exc_info.value)
        assert exc_info.value.provider == "Gemini"
        if hasattr(api_error_instance, 'code') and (api_error_instance.code == 429 or api_error_instance.code >= 500):
            assert exc_info.value.is_retryable is True


@pytest.mark.asyncio
async def test_generate_response_no_candidates(gemini_client: GeminiClient, mock_generative_model_instance: AsyncMock):
    mock_response = AsyncMock(candidates=None, prompt_feedback=MagicMock(block_reason="SAFETY"))
    mock_generative_model_instance.generate_content_async.return_value = mock_response
    with pytest.raises(LLMError, match="Gemini API returned no candidates. Block reason: SAFETY"):
        await gemini_client.generate_response("prompt")


@pytest.mark.asyncio
async def test_generate_response_blocked_by_safety(gemini_client: GeminiClient, mock_generative_model_instance: AsyncMock):
    mock_response = AsyncMock()
    mock_candidate = MagicMock(finish_reason="SAFETY", safety_ratings=[MagicMock(category="HARM_CATEGORY_SEXUAL", probability="HIGH")])
    mock_response.candidates = [mock_candidate]
    mock_response.text = "" # No text when blocked
    mock_generative_model_instance.generate_content_async.return_value = mock_response

    with pytest.raises(LLMError, match="Blocked by API. Reason: SAFETY. Safety: HARM_CATEGORY_SEXUAL: HIGH"):
        await gemini_client.generate_response("prompt")


@pytest.mark.asyncio
async def test_check_health_success(gemini_client: GeminiClient, mock_generative_model_instance: AsyncMock):
    # Health check calls generate_content_async with a ping
    mock_health_response = AsyncMock(text="ok", candidates=[MagicMock(finish_reason=1)], prompt_feedback=None)
    mock_generative_model_instance.generate_content_async.return_value = mock_health_response

    is_healthy, msg, details = await gemini_client.check_health()

    assert is_healthy is True
    assert "Successfully connected" in msg
    assert details["default_model_tested"] == gemini_client.default_model_name
    mock_generative_model_instance.generate_content_async.assert_called_once_with(
        "Health check ping", generation_config=genai.types.GenerationConfig(max_output_tokens=1)
    )


@pytest.mark.asyncio
async def test_check_health_api_error(gemini_client: GeminiClient, mock_generative_model_instance: AsyncMock):
    api_error = PermissionDenied("Health check permission denied")
    setattr(api_error, 'code', 403) # Manually set code for test
    mock_generative_model_instance.generate_content_async.side_effect = api_error

    is_healthy, msg, details = await gemini_client.check_health()

    assert is_healthy is False
    assert "Health check failed with API error (403): Health check permission denied" in msg
    assert details["error_type"] == "PermissionDenied"
    assert details["error_code"] == 403
