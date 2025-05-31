# backend/rag_system/integrations/anthropic_client.py
import logging
from typing import Optional, Tuple, Dict, Any
import anthropic # Using the official Anthropic Python library
from anthropic import AsyncAnthropic

from rag_system.utils.exceptions import LLMError, ConfigurationError

logger = logging.getLogger(__name__)

# Default timeout for API requests in seconds
DEFAULT_TIMEOUT = 90.0 # Anthropic models can sometimes take longer

class AnthropicClient:
    """
    Client for interacting with the Anthropic API (Claude models).
    """

    def __init__(self, api_key: Optional[str] = None, default_model: str = "claude-3-haiku-20240307"):
        """
        Initializes the AnthropicClient.

        Args:
            api_key: The Anthropic API key.
            default_model: The default Claude model to use.
                           Examples: "claude-3-opus-20240229", "claude-3-sonnet-20240229",
                                     "claude-3-haiku-20240307", "claude-2.1".

        Raises:
            ConfigurationError: If the API key is not provided.
        """
        if not api_key:
            msg = "Anthropic API key not provided. Please set ANTHROPIC_API_KEY."
            logger.error(msg)
            raise ConfigurationError(msg)

        self.api_key = api_key
        self.default_model = default_model
        try:
            self.async_client = AsyncAnthropic(api_key=self.api_key, timeout=DEFAULT_TIMEOUT)
            logger.info(f"AsyncAnthropic client initialized successfully for model family: {default_model.split('-')[0]}.")
        except Exception as e:
            logger.error(f"Failed to initialize AsyncAnthropic client: {e}", exc_info=True)
            raise ConfigurationError(f"AsyncAnthropic client initialization failed: {e}") from e

    async def generate_response(
        self, prompt: str, model_name_override: Optional[str] = None
    ) -> Tuple[str, str]: # (answer, model_name_used)
        """
        Generates a response from Anthropic Claude using the messages API.

        Args:
            prompt: The user's part of the prompt. The client will wrap this.
                    It should be the full content for the "user" role message.
            model_name_override: Specific model name to use, overriding the client's default.

        Returns:
            A tuple containing the generated text content and the model name used.

        Raises:
            LLMError: If the API call fails or returns an unexpected response.
        """
        model_to_use = model_name_override or self.default_model
        logger.debug(
            f"Requesting Anthropic completion. Model: {model_to_use}. Prompt (start): '{prompt[:100]}...'"
        )

        # Anthropic's Messages API expects a list of messages.
        # The provided `prompt` is considered the user's message.
        # A system prompt can be added if desired.
        messages = [{"role": "user", "content": prompt}]
        # system_prompt = "You are a helpful RAG assistant." # Optional

        try:
            response = await self.async_client.messages.create(
                model=model_to_use,
                max_tokens=2048, # Adjust as needed, Claude models have large context windows
                # system=system_prompt, # Pass system prompt if using one
                messages=messages,
                temperature=0.7, # Adjust as needed
            )

            if response.content and isinstance(response.content, list) and len(response.content) > 0:
                # Content is a list of blocks, usually one TextBlock
                first_block = response.content[0]
                if first_block.type == "text":
                    answer_text = first_block.text
                    model_used = response.model # Actual model string used
                    logger.info(f"Anthropic response received successfully from model {model_used}.")
                    return answer_text.strip(), model_used
                else:
                    logger.warning(f"Anthropic API returned non-text block: {first_block.type}")
                    raise LLMError(f"Anthropic API returned non-text block: {first_block.type}", provider="Anthropic")
            else:
                logger.error(f"Anthropic API returned no content or unexpected format. Response: {response.model_dump_json(indent=2)}")
                raise LLMError("Anthropic API returned no content or unexpected format.", provider="Anthropic")

        except anthropic.APIConnectionError as e:
            logger.error(f"Anthropic API connection error: {e}", exc_info=True)
            raise LLMError(f"Anthropic API connection error: {e}", provider="Anthropic", is_retryable=True) from e
        except anthropic.RateLimitError as e:
            logger.error(f"Anthropic API rate limit exceeded: {e}", exc_info=True)
            raise LLMError(f"Anthropic API rate limit exceeded: {e}", provider="Anthropic", is_retryable=True) from e
        except anthropic.AuthenticationError as e:
            logger.error(f"Anthropic API authentication error: {e}", exc_info=True)
            raise LLMError(f"Anthropic API authentication error: {e}. Check your API key.", provider="Anthropic") from e
        except anthropic.APIStatusError as e: # Covers 4xx and 5xx errors not caught above
            logger.error(f"Anthropic API status error ({e.status_code}): {e.response.text if e.response else 'No response body'}", exc_info=True)
            error_message = e.message or "Unknown API error"
            raise LLMError(f"Anthropic API error ({e.status_code}): {error_message}", provider="Anthropic", is_retryable=e.status_code >=500) from e
        except Exception as e:
            logger.error(f"An unexpected error occurred with Anthropic API: {e}", exc_info=True)
            raise LLMError(f"Unexpected error with Anthropic API: {str(e)}", provider="Anthropic") from e

    async def check_health(self) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Checks the health of the Anthropic API.
        Anthropic doesn't have a dedicated "list models" or simple ping.
        A lightweight way is to try a very short, cheap completion.
        Alternatively, assume if client initializes, basic connectivity is fine.
        For a more robust check, a minimal messages.create call can be made.
        """
        logger.debug("Checking Anthropic API health...")
        try:
            # Attempt a very small, fast completion to check connectivity and auth
            # This will incur a very small cost.
            await self.async_client.messages.create(
                model=self.default_model, # Use a fast model like Haiku if possible
                max_tokens=1,
                messages=[{"role": "user", "content": "Health check ping."}],
                timeout=10.0 # Short timeout for health check
            )
            message = f"Successfully connected and received response from model {self.default_model}."
            logger.info(f"Anthropic health check successful: {message}")
            return True, message, {"default_model_tested": self.default_model}
        except anthropic.AuthenticationError:
            msg = "Authentication failed. Check API key."
            logger.error(f"Anthropic health check: {msg}")
            return False, msg, {"error_type": "AuthenticationError"}
        except Exception as e:
            error_msg = f"Health check failed: {str(e)}"
            logger.error(f"Anthropic health check error: {e}", exc_info=True)
            # Try to parse specific error types if possible for better detail
            error_type = type(e).__name__
            error_detail_dict = {"error_type": error_type, "error_detail": str(e)}
            if isinstance(e, anthropic.APIStatusError):
                error_detail_dict["status_code"] = e.status_code
            return False, error_msg, error_detail_dict


# Example Usage:
if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv
    from rag_system.config.logging_config import setup_logging
    setup_logging("DEBUG")
    load_dotenv()

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    async def main():
        if not anthropic_api_key:
            logger.error("ANTHROPIC_API_KEY not found in environment variables.")
            return

        try:
            # Using claude-3-haiku as it's fast and cost-effective for testing
            client = AnthropicClient(api_key=anthropic_api_key, default_model="claude-3-haiku-20240307")
            logger.info("AnthropicClient initialized for example.")

            # Health Check
            healthy, msg, details = await client.check_health()
            logger.info(f"Anthropic Health: {'OK' if healthy else 'Error'} - {msg} {details or ''}")

            if healthy:
                # Test generation
                # The prompt for Anthropic should be the user's turn.
                # If you have a system prompt, it's passed separately to `messages.create`.
                test_user_prompt = (
                    "You are a helpful AI. Based on the context: 'The sky is blue.', "
                    "answer the question: 'What color is the sky?'"
                )
                try:
                    answer, model_used = await client.generate_response(test_user_prompt)
                    logger.info(f"Test User Prompt: {test_user_prompt}")
                    logger.info(f"Model Used: {model_used}")
                    logger.info(f"Test Answer: {answer}")
                except LLMError as e:
                    logger.error(f"LLMError during test generation: {e.message} (Detail: {e.detail})")

        except ConfigurationError as e:
            logger.error(f"ConfigurationError: {e.message}")
        except Exception as e:
            logger.error(f"An unexpected error occurred in example: {e}", exc_info=True)

    asyncio.run(main())
