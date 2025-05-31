# backend/rag_system/integrations/openai_client.py
import logging
from typing import Optional, Tuple, List, Dict, Any
import openai # Using the official OpenAI Python library v1.x.x
from openai import AsyncOpenAI

from rag_system.utils.exceptions import LLMError, ConfigurationError

logger = logging.getLogger(__name__)

# Default timeout for API requests in seconds
DEFAULT_TIMEOUT = 60.0 # seconds


class OpenAIClient:
    """
    Client for interacting with the OpenAI API.
    Handles chat completions.
    """

    def __init__(self, api_key: Optional[str] = None, default_model: str = "gpt-3.5-turbo"):
        """
        Initializes the OpenAIClient.

        Args:
            api_key: The OpenAI API key.
            default_model: The default model to use for completions.

        Raises:
            ConfigurationError: If the API key is not provided.
        """
        if not api_key:
            msg = "OpenAI API key not provided. Please set OPENAI_API_KEY."
            logger.error(msg)
            raise ConfigurationError(msg)

        self.api_key = api_key
        self.default_model = default_model
        try:
            # Initialize the asynchronous client
            self.async_client = AsyncOpenAI(api_key=self.api_key, timeout=DEFAULT_TIMEOUT)
            logger.info(f"AsyncOpenAI client initialized successfully for model family: {default_model.split('-')[0]}.")
        except Exception as e:
            logger.error(f"Failed to initialize AsyncOpenAI client: {e}", exc_info=True)
            raise ConfigurationError(f"AsyncOpenAI client initialization failed: {e}") from e

    async def generate_response(
        self, prompt: str, model_name_override: Optional[str] = None
    ) -> Tuple[str, str]: # (answer, model_name_used)
        """
        Generates a response from OpenAI using the chat completions endpoint.

        Args:
            prompt: The combined prompt including context and query.
            model_name_override: Specific model name to use, overriding the client's default.

        Returns:
            A tuple containing the generated text content and the model name used.

        Raises:
            LLMError: If the API call fails or returns an unexpected response.
        """
        model_to_use = model_name_override or self.default_model
        logger.debug(
            f"Requesting OpenAI completion. Model: {model_to_use}. Prompt (start): '{prompt[:100]}...'"
        )

        try:
            # For chat models, the prompt is typically the content of the user message.
            # A more complex setup might involve a list of messages (system, user, assistant).
            # Here, we assume the input `prompt` is a fully formed user message content.
            chat_completion = await self.async_client.chat.completions.create(
                model=model_to_use,
                messages=[
                    # {"role": "system", "content": "You are a helpful RAG assistant."}, # Optional system message
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7, # Adjust as needed
                # max_tokens=1000, # Adjust as needed, or let API decide
            )

            if chat_completion.choices and chat_completion.choices[0].message:
                content = chat_completion.choices[0].message.content
                model_used = chat_completion.model # The actual model string used by API
                if content:
                    logger.info(f"OpenAI response received successfully from model {model_used}.")
                    return content.strip(), model_used
                else:
                    logger.warning(f"OpenAI API returned an empty message content for model {model_used}.")
                    raise LLMError("OpenAI API returned empty message content.", provider="OpenAI")
            else:
                logger.error(f"OpenAI API returned no choices or message. Response: {chat_completion.model_dump_json(indent=2)}")
                raise LLMError("OpenAI API returned no choices or message.", provider="OpenAI")

        except openai.APIConnectionError as e:
            logger.error(f"OpenAI API connection error: {e}", exc_info=True)
            raise LLMError(f"OpenAI API connection error: {e}", provider="OpenAI", is_retryable=True) from e
        except openai.RateLimitError as e:
            logger.error(f"OpenAI API rate limit exceeded: {e}", exc_info=True)
            raise LLMError(f"OpenAI API rate limit exceeded: {e}", provider="OpenAI", is_retryable=True) from e
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI API authentication error: {e}", exc_info=True)
            raise LLMError(f"OpenAI API authentication error: {e}. Check your API key.", provider="OpenAI") from e
        except openai.APIStatusError as e: # Covers 4xx and 5xx errors not caught above
            logger.error(f"OpenAI API status error ({e.status_code}): {e.response}", exc_info=True)
            raise LLMError(f"OpenAI API error ({e.status_code}): {e.message}", provider="OpenAI", is_retryable=e.status_code >=500) from e
        except Exception as e:
            logger.error(f"An unexpected error occurred with OpenAI API: {e}", exc_info=True)
            raise LLMError(f"Unexpected error with OpenAI API: {str(e)}", provider="OpenAI") from e

    async def check_health(self) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Checks the health of the OpenAI API by trying to list available models.
        Returns: (is_healthy, message, details_dict)
        """
        logger.debug("Checking OpenAI API health...")
        try:
            models = await self.async_client.models.list(timeout=10.0) # Short timeout for health check
            if models and models.data:
                # Check if the default model or a common one is available
                available_model_ids = [m.id for m in models.data]
                default_model_family = self.default_model.split('-')[0] # e.g., "gpt-3.5" or "gpt-4"
                found_family = any(default_model_family in m_id for m_id in available_model_ids)

                message = f"Successfully connected. Found {len(available_model_ids)} models. "
                message += f"Default model family '{default_model_family}' {'found' if found_family else 'not explicitly found in list'}."

                details = {
                    "models_count": len(available_model_ids),
                    "default_model_family_available": found_family,
                    # "sample_models": available_model_ids[:3] # Optional: list a few
                }
                logger.info(f"OpenAI health check successful: {message}")
                return True, message, details
            else:
                logger.warning("OpenAI health check: No models returned from API.")
                return False, "No models returned from API.", None
        except openai.AuthenticationError:
            msg = "Authentication failed. Check API key."
            logger.error(f"OpenAI health check: {msg}")
            return False, msg, {"error_type": "AuthenticationError"}
        except Exception as e:
            error_msg = f"Health check failed: {str(e)}"
            logger.error(f"OpenAI health check error: {e}", exc_info=True)
            return False, error_msg, {"error_type": type(e).__name__, "error_detail": str(e)}


# Example Usage:
if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv
    from rag_system.config.logging_config import setup_logging
    setup_logging("DEBUG")
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")

    async def main():
        if not openai_api_key:
            logger.error("OPENAI_API_KEY not found in environment variables.")
            return

        try:
            client = OpenAIClient(api_key=openai_api_key, default_model="gpt-3.5-turbo") # or "gpt-4-turbo-preview"
            logger.info("OpenAIClient initialized for example.")

            # Health Check
            healthy, msg, details = await client.check_health()
            logger.info(f"OpenAI Health: {'OK' if healthy else 'Error'} - {msg} {details or ''}")

            if healthy:
                # Test generation
                test_prompt = "What is the weather like in Paris today? (answer concisely)"
                try:
                    answer, model_used = await client.generate_response(test_prompt)
                    logger.info(f"Test Query: {test_prompt}")
                    logger.info(f"Model Used: {model_used}")
                    logger.info(f"Test Answer: {answer}")
                except LLMError as e:
                    logger.error(f"LLMError during test generation: {e.message} (Detail: {e.detail})")

        except ConfigurationError as e:
            logger.error(f"ConfigurationError: {e.message}")
        except Exception as e:
            logger.error(f"An unexpected error occurred in example: {e}", exc_info=True)

    asyncio.run(main())
