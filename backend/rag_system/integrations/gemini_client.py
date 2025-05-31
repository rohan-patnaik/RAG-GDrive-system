# backend/rag_system/integrations/gemini_client.py
import logging
from typing import Optional, Tuple, Dict, Any
import google.generativeai as genai
from google.generativeai.types import GenerationConfig # For typing if needed
from google.api_core.exceptions import GoogleAPIError # General Google API error

from rag_system.utils.exceptions import LLMError, ConfigurationError

logger = logging.getLogger(__name__)

# Default timeout for API requests (not directly configurable in google-genai client, managed by underlying http client)
# We can implement our own timeout logic if needed, e.g., with asyncio.wait_for.

class GeminiClient:
    """
    Client for interacting with the Google Gemini API.
    """

    def __init__(self, api_key: Optional[str] = None, default_model: str = "gemini-2.5-flash-preview-05-20"):
        """
        Initializes the GeminiClient.

        Args:
            api_key: The Google API key for Gemini.
            default_model: The default Gemini model to use (e.g., "gemini-pro", "gemini-1.0-pro", "gemini-2.5-flash-preview-05-20", "gemini-1.5-pro-latest").

        Raises:
            ConfigurationError: If the API key is not provided or configuration fails.
        """
        if not api_key:
            msg = "Google API key (for Gemini) not provided. Please set GOOGLE_API_KEY."
            logger.error(msg)
            raise ConfigurationError(msg)

        self.api_key = api_key
        self.default_model_name = default_model # Store the string name
        self.model = None # This will be an instance of GenerativeModel

        try:
            genai.configure(api_key=self.api_key)
            # Initialize the model instance here or per request.
            # Initializing here allows reuse if the model name doesn't change often.
            self.model = genai.GenerativeModel(self.default_model_name)
            logger.info(f"Google Gemini client configured successfully for model: {self.default_model_name}.")
        except Exception as e:
            logger.error(f"Failed to configure Google Gemini client: {e}", exc_info=True)
            raise ConfigurationError(f"Google Gemini client configuration failed: {e}") from e

    async def generate_response(
        self, prompt: str, model_name_override: Optional[str] = None
    ) -> Tuple[str, str]: # (answer, model_name_used)
        """
        Generates a response from Google Gemini.

        Args:
            prompt: The prompt string for the model.
            model_name_override: Specific model name to use, overriding the client's default.
                                 If provided, a new GenerativeModel instance will be used for this call.

        Returns:
            A tuple containing the generated text content and the model name used.

        Raises:
            LLMError: If the API call fails or returns an unexpected response.
        """
        model_to_use_name = model_name_override or self.default_model_name
        logger.debug(
            f"Requesting Gemini completion. Model: {model_to_use_name}. Prompt (start): '{prompt[:100]}...'"
        )

        current_model_instance = self.model
        if model_name_override and model_name_override != self.default_model_name:
            try:
                logger.info(f"Using overridden model for this request: {model_name_override}")
                current_model_instance = genai.GenerativeModel(model_name_override)
            except Exception as e:
                logger.error(f"Failed to initialize overridden Gemini model '{model_name_override}': {e}", exc_info=True)
                raise LLMError(f"Failed to use overridden Gemini model '{model_name_override}'. Error: {e}", provider="Gemini") from e
        
        if not current_model_instance: # Should have been initialized in __init__ or above block
             logger.error(f"Gemini model instance not available for model: {model_to_use_name}")
             raise LLMError(f"Gemini model instance not available for model: {model_to_use_name}", provider="Gemini")


        try:
            # Use generate_content_async for asynchronous operation
            # The library handles retries for certain errors internally.
            response = await current_model_instance.generate_content_async(
                contents=prompt,
                # generation_config=GenerationConfig(temperature=0.7) # Optional config
            )

            if response.candidates:
                # Check for finish_reason if needed (e.g., SAFETY, RECITATION)
                if response.candidates[0].finish_reason not in [1, "STOP", "MAX_TOKENS"]: # 1 is "STOP" for older versions
                    finish_reason_val = response.candidates[0].finish_reason
                    # Try to get safety ratings if available
                    safety_ratings_str = ""
                    if response.candidates[0].safety_ratings:
                        safety_ratings_str = ", ".join([f"{sr.category.name}: {sr.probability.name}" for sr in response.candidates[0].safety_ratings])

                    logger.warning(
                        f"Gemini generation finished with reason: {finish_reason_val}. "
                        f"Safety ratings: [{safety_ratings_str or 'N/A'}]"
                    )
                    # If blocked due to safety, text might be missing or empty.
                    if not response.text: # Check if text is empty
                        block_message = f"Blocked by API. Reason: {finish_reason_val}."
                        if safety_ratings_str:
                            block_message += f" Safety: {safety_ratings_str}"
                        raise LLMError(block_message, provider="Gemini", is_retryable=False)


                answer_text = response.text # response.text directly gives the combined text
                logger.info(f"Gemini response received successfully from model {model_to_use_name}.")
                return answer_text.strip(), model_to_use_name # Model name used is what we passed
            else:
                # This case might indicate an issue, or if prompt_feedback has block reason
                block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                block_message = f"Gemini API returned no candidates. Block reason: {block_reason}."
                logger.error(f"{block_message} Response: {response}")
                raise LLMError(block_message, provider="Gemini")

        except GoogleAPIError as e: # Catch specific Google API errors
            logger.error(f"Google Gemini API error: {e}", exc_info=True)
            # e.g. google.api_core.exceptions.PermissionDenied: 403 Your API key is invalid.
            # e.g. google.api_core.exceptions.ResourceExhausted: 429 Quota exceeded.
            is_retryable = e.code == 429 or e.code >= 500 # Rate limit or server error
            raise LLMError(f"Gemini API error ({e.code}): {e.message}", provider="Gemini", is_retryable=is_retryable) from e
        except Exception as e: # Catch other unexpected errors
            logger.error(f"An unexpected error occurred with Gemini API: {e}", exc_info=True)
            raise LLMError(f"Unexpected error with Gemini API: {str(e)}", provider="Gemini") from e

    async def check_health(self) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Checks the health of the Google Gemini API by trying to list available models
        or get information about the default model.
        """
        logger.debug("Checking Google Gemini API health...")
        try:
            # Option 1: List models (can be slow or paginated)
            # models = [m async for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            # if models:
            #     model_names = [m.name for m in models]
            #     message = f"Successfully connected. Found {len(model_names)} usable models."
            #     details = {"models_count": len(model_names), "sample_models": model_names[:3]}
            #     logger.info(f"Gemini health check successful: {message}")
            #     return True, message, details
            # else:
            #     return False, "No usable models found.", None

            # Option 2: Get the default model (faster if it works)
            # This implicitly checks if the API key is valid and the service is reachable.
            if not self.model: # Should be initialized in constructor
                 return False, "Gemini model client not initialized.", {"error_type": "InitializationError"}

            # A lightweight way to check is to try to get model info or a tiny generation
            # For now, we'll assume if self.model exists and was configured, basic connectivity is okay.
            # A more robust check would be a tiny generate_content_async call.
            await self.model.generate_content_async("Health check ping", generation_config=genai.types.GenerationConfig(max_output_tokens=1))

            message = f"Successfully connected and received test response from model {self.default_model_name}."
            logger.info(f"Gemini health check successful: {message}")
            return True, message, {"default_model_tested": self.default_model_name}

        except GoogleAPIError as e:
            error_msg = f"Health check failed with API error ({e.code}): {e.message}"
            logger.error(f"Gemini health check error: {e}", exc_info=True)
            return False, error_msg, {"error_type": type(e).__name__, "error_code": e.code, "error_detail": e.message}
        except Exception as e:
            error_msg = f"Health check failed: {str(e)}"
            logger.error(f"Gemini health check error: {e}", exc_info=True)
            return False, error_msg, {"error_type": type(e).__name__, "error_detail": str(e)}


# Example Usage:
if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv
    from rag_system.config.logging_config import setup_logging
    setup_logging("DEBUG")
    load_dotenv()

    google_api_key = os.getenv("GOOGLE_API_KEY")

    async def main():
        if not google_api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables.")
            return

        try:
            # Using gemini-2.5-flash-preview-05-20 as it's fast and cost-effective for testing
            client = GeminiClient(api_key=google_api_key, default_model="gemini-2.5-flash-preview-05-20")
            logger.info("GeminiClient initialized for example.")

            # Health Check
            healthy, msg, details = await client.check_health()
            logger.info(f"Gemini Health: {'OK' if healthy else 'Error'} - {msg} {details or ''}")

            if healthy:
                # Test generation
                test_prompt = (
                    "You are a helpful AI. Based on the context: 'The grass is green.', "
                    "answer the question: 'What color is the grass?'"
                )
                try:
                    answer, model_used = await client.generate_response(test_prompt)
                    logger.info(f"Test Prompt: {test_prompt}")
                    logger.info(f"Model Used: {model_used}")
                    logger.info(f"Test Answer: {answer}")

                    # Test with model override
                    # answer_pro, model_used_pro = await client.generate_response(
                    #     test_prompt, model_name_override="gemini-1.5-pro-latest" # or "gemini-pro"
                    # )
                    # logger.info(f"Test Prompt (Pro): {test_prompt}")
                    # logger.info(f"Model Used (Pro): {model_used_pro}")
                    # logger.info(f"Test Answer (Pro): {answer_pro}")

                except LLMError as e:
                    logger.error(f"LLMError during test generation: {e.message} (Detail: {e.detail})")

        except ConfigurationError as e:
            logger.error(f"ConfigurationError: {e.message}")
        except Exception as e:
            logger.error(f"An unexpected error occurred in example: {e}", exc_info=True)

    asyncio.run(main())
