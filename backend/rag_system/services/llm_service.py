# backend/rag_system/services/llm_service.py
import asyncio
import logging
from typing import List, Optional, Tuple, cast

from rag_system.config.settings import AppSettings, get_settings
from rag_system.models.schemas import LLMProvider, RetrievedChunk, ComponentStatus, StatusEnum
from rag_system.integrations.openai_client import OpenAIClient
from rag_system.integrations.anthropic_client import AnthropicClient
from rag_system.integrations.gemini_client import GeminiClient
from rag_system.utils.exceptions import LLMError, ConfigurationError

logger = logging.getLogger(__name__)

# Maximum context length to send to LLM (in characters, approximate)
# This is a safeguard, actual token limits are model-specific and handled by clients.
MAX_CONTEXT_CHAR_LENGTH = 15000 # Adjust based on typical model context windows (e.g., 8k, 32k, 128k tokens)


class LLMService:
    """
    Manages interactions with different Large Language Model (LLM) providers.
    It selects the appropriate client based on the specified provider and
    constructs prompts for generation.
    """

    def __init__(self, settings: Optional[AppSettings] = None):
        """
        Initializes the LLMService.

        Args:
            settings: Application settings. If None, loads default settings.
        """
        self.settings = settings or get_settings()
        self._clients: dict[LLMProvider, OpenAIClient | AnthropicClient | GeminiClient | None] = {
            LLMProvider.OPENAI: None,
            LLMProvider.ANTHROPIC: None,
            LLMProvider.GEMINI: None,
            LLMProvider.LOCAL: None, # Placeholder for future local LLM
        }
        self._initialize_clients()
        logger.info("LLMService initialized.")

    def _initialize_clients(self) -> None:
        """Initializes clients for configured LLM providers."""
        if self.settings.OPENAI_API_KEY:
            try:
                self._clients[LLMProvider.OPENAI] = OpenAIClient(
                    api_key=self.settings.OPENAI_API_KEY.get_secret_value() if self.settings.OPENAI_API_KEY else None,
                    default_model=self.settings.DEFAULT_OPENAI_MODEL,
                )
                logger.info("OpenAI client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
                # Non-fatal, service can still run if other providers are configured
        else:
            logger.warning("OpenAI API key not configured. OpenAI client will not be available.")

        if self.settings.ANTHROPIC_API_KEY:
            try:
                self._clients[LLMProvider.ANTHROPIC] = AnthropicClient(
                    api_key=self.settings.ANTHROPIC_API_KEY.get_secret_value() if self.settings.ANTHROPIC_API_KEY else None,
                    default_model=self.settings.DEFAULT_ANTHROPIC_MODEL,
                )
                logger.info("Anthropic client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}", exc_info=True)
        else:
            logger.warning("Anthropic API key not configured. Anthropic client will not be available.")

        if self.settings.GOOGLE_API_KEY:
            try:
                self._clients[LLMProvider.GEMINI] = GeminiClient(
                    api_key=self.settings.GOOGLE_API_KEY.get_secret_value() if self.settings.GOOGLE_API_KEY else None,
                    default_model=self.settings.DEFAULT_GEMINI_MODEL,
                )
                logger.info("Google Gemini client initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Google Gemini client: {e}", exc_info=True)
        else:
            logger.warning("Google API key (for Gemini) not configured. Gemini client will not be available.")

        # Initialize local LLM client if configured (future)
        # if self.settings.DEFAULT_LOCAL_MODEL_ENDPOINT:
        #     self._clients[LLMProvider.LOCAL] = LocalLLMClient(endpoint=self.settings.DEFAULT_LOCAL_MODEL_ENDPOINT)
        #     logger.info("Local LLM client initialized.")

    def _get_client(
        self, provider: LLMProvider
    ) -> OpenAIClient | AnthropicClient | GeminiClient: # Add LocalLLMClient in future
        """
        Retrieves the initialized client for the given provider.

        Raises:
            ConfigurationError: If the provider is not configured or client not initialized.
        """
        client = self._clients.get(provider)
        if client is None:
            api_key_name = ""
            if provider == LLMProvider.OPENAI: api_key_name = "OPENAI_API_KEY"
            elif provider == LLMProvider.ANTHROPIC: api_key_name = "ANTHROPIC_API_KEY"
            elif provider == LLMProvider.GEMINI: api_key_name = "GOOGLE_API_KEY"

            error_message = f"{provider.value.capitalize()} LLM provider is not configured or its client failed to initialize."
            if api_key_name and not getattr(self.settings, api_key_name, None):
                 error_message += f" Please ensure '{api_key_name}' is set in your environment."
            logger.error(error_message)
            raise ConfigurationError(error_message)
        return client


    def _construct_prompt(
        self, query: str, context_chunks: List[RetrievedChunk], provider: LLMProvider
    ) -> str:
        """
        Constructs a prompt for the LLM using the query and retrieved context chunks.
        Tailors the prompt format based on the LLM provider if necessary.

        Args:
            query: The user's original query.
            context_chunks: A list of relevant document chunks.
            provider: The LLM provider being used (to potentially customize prompt structure).

        Returns:
            The constructed prompt string.
        """
        if not context_chunks:
            # If no context, just pass the query, perhaps with a note.
            # Some models might perform better if explicitly told there's no context.
            prompt = f"Question: {query}\n\nAnswer the question based on your general knowledge as no specific context was found."
            logger.info("No context chunks provided for prompt construction. Using query directly.")
            return prompt

        context_str = ""
        current_length = 0
        for i, chunk in enumerate(context_chunks):
            chunk_text = f"Source {i+1} (ID: {chunk.id}, Score: {chunk.score:.4f}):\n{chunk.content}\n\n"
            if current_length + len(chunk_text) > MAX_CONTEXT_CHAR_LENGTH:
                logger.warning(
                    f"Context length limit ({MAX_CONTEXT_CHAR_LENGTH} chars) reached. "
                    f"Truncating context to {i} chunks."
                )
                break
            context_str += chunk_text
            current_length += len(chunk_text)

        # Basic prompt template - can be customized per provider if needed
        # For Anthropic, ensure "Human:" and "Assistant:" turns are clear if building a conversational prompt.
        # For Gemini, it's generally fine with direct instruction.
        # For OpenAI, standard instruction-following prompts work well.

        if provider == LLMProvider.ANTHROPIC:
            # Anthropic Claude models respond well to Human/Assistant turn structure.
            # However, for a RAG task, a direct instruction within the "Human" turn is also fine.
            prompt = (
                "You are a helpful AI assistant. Based on the following context, please answer the question. "
                "If the context does not contain the answer, say that you cannot answer based on the provided information.\n\n"
                "Context:\n"
                "---------------------\n"
                f"{context_str.strip()}"
                "\n---------------------\n\n"
                f"Question: {query}\n\n"
                "Answer:" # Anthropic expects the model to start generation after "Assistant:" or a clear prompt end.
            )
        else: # Default for OpenAI, Gemini, and potentially others
            prompt = (
                "You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. "
                "If you don't know the answer from the context, just say that you don't know, don't try to make up an answer. "
                "Be concise and informative.\n\n"
                "Context:\n"
                "---------------------\n"
                f"{context_str.strip()}"
                "\n---------------------\n\n"
                f"Question: {query}\n\n"
                "Helpful Answer:"
            )
        logger.debug(f"Constructed prompt for provider {provider.value}:\n{prompt[:500]}...") # Log beginning of prompt
        return prompt

    async def generate_response(
        self,
        query: str,
        context_chunks: List[RetrievedChunk],
        provider: LLMProvider,
        model_name_override: Optional[str] = None,
    ) -> Tuple[str, str]: # Returns (answer, model_name_used)
        """
        Generates a response from the specified LLM provider.

        Args:
            query: The user's query.
            context_chunks: Relevant context chunks.
            provider: The LLMProvider enum member.
            model_name_override: Optional specific model name to use, overriding the client's default.

        Returns:
            A tuple containing the generated answer string and the actual model name used.

        Raises:
            LLMError: If generation fails or provider is not configured.
            ConfigurationError: If the selected provider is not available.
        """
        logger.info(
            f"Generating response using {provider.value} "
            f"(model override: {model_name_override or 'None'}). Query: '{query[:50]}...'"
        )
        client = self._get_client(provider) # Raises ConfigurationError if not available

        prompt = self._construct_prompt(query, context_chunks, provider)

        try:
            # The generate_response method in each client should handle model selection
            # (default or override) and return (answer, model_name_used)
            answer, model_used = await client.generate_response(
                prompt=prompt,
                model_name_override=model_name_override
            )
            logger.info(f"Response generated successfully by {provider.value} using model {model_used}.")
            return answer, model_used
        except LLMError: # Re-raise LLMErrors from clients
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while generating response from {provider.value}: {e}",
                exc_info=True,
            )
            raise LLMError(
                f"Failed to generate response from {provider.value}. Detail: {str(e)}"
            ) from e

    async def check_provider_health(self, provider: LLMProvider) -> ComponentStatus:
        """Check the health of a specific LLM provider."""
        logger.info(f"Checking health for provider: {provider.value}")
        
        status = ComponentStatus(name=f"{provider.value.capitalize()} LLM", status=StatusEnum.UNKNOWN)
        api_key = None
        api_key_name = ""

        try:
            # Check if API key is configured
            if provider == LLMProvider.OPENAI:
                api_key = self.settings.OPENAI_API_KEY
                api_key_name = "OPENAI_API_KEY"
            elif provider == LLMProvider.ANTHROPIC:
                api_key = self.settings.ANTHROPIC_API_KEY
                api_key_name = "ANTHROPIC_API_KEY"
            elif provider == LLMProvider.GEMINI:
                api_key = self.settings.GOOGLE_API_KEY
                api_key_name = "GOOGLE_API_KEY"
            # Add local provider check here in the future if it uses API keys
            # elif provider == LLMProvider.LOCAL:
            #     # For local, health might mean the endpoint is reachable
            #     # This part needs to be defined based on how local LLM health is determined
            #     pass

            if provider != LLMProvider.LOCAL and \
               (api_key is None or len(api_key.get_secret_value().strip()) == 0):
                status.status = StatusEnum.DEGRADED
                status.message = f"API key not configured for {provider.value} (checked {api_key_name})"
                status.details = {"configured": False, "reason": "missing_api_key"}
                logger.warning(status.message)
                return status
            
            # Try to get the client and check health
            # For local provider, this might involve a different health check logic
            if provider == LLMProvider.LOCAL:
                # TODO: Implement health check for local LLM
                # For now, assume local is healthy if configured, or specific check needed
                # depending on local LLM client implementation
                status.status = StatusEnum.OK # Or UNKNOWN if no check implemented
                status.message = "Local LLM health check not fully implemented yet."
                status.details = {"configured": True, "info": "placeholder_status"}
                logger.info(f"Local LLM ({provider.value}) health check placeholder.")
                return status

            client = self._get_client(provider) # This will raise ConfigurationError if not properly set up
            is_healthy, message, details = await client.check_health()

            if is_healthy:
                status.status = StatusEnum.OK
                status.message = message
                status.details = details or {"configured": True, "connection": "success"}
            else:
                status.status = StatusEnum.ERROR
                status.message = message
                status.details = details or {"configured": True, "connection": "failed"}

        except ConfigurationError as ce:
            logger.warning(f"Configuration error during health check for {provider.value}: {str(ce)}")
            status.status = StatusEnum.NOT_CONFIGURED # Or DEGRADED
            status.message = f"Provider {provider.value} not configured: {str(ce)}"
            status.details = {"configured": False, "reason": str(ce)}
        except Exception as e:
            logger.error(f"Health check failed for {provider.value}: {str(e)}", exc_info=True)
            status.status = StatusEnum.ERROR
            status.message = f"Health check failed: {str(e)}"
            # Ensure details includes that it was configured if we passed the API key check stage
            status.details = status.details or {} # Initialize if None
            status.details.update({"error": str(e)})
            if api_key_name: # If we determined an api_key_name, it means we expected it to be configured
                 status.details["configured"] = True
    
        return status

# Example Usage (typically not run directly like this in production)
if __name__ == "__main__":
    from rag_system.config.logging_config import setup_logging
    setup_logging("DEBUG")

    # This example assumes .env is set up with API keys
    try:
        llm_service = LLMService()

        sample_query = "What is the capital of France?"
        sample_chunks = [
            RetrievedChunk(id="doc1_chunk0", content="France is a country in Europe.", metadata={"source": "doc1"}, score=0.9),
            RetrievedChunk(id="doc2_chunk0", content="Paris is a famous city known for the Eiffel Tower.", metadata={"source": "doc2"}, score=0.85),
        ]

        async def test_provider(provider_enum: LLMProvider):
            console_prefix = f"[{provider_enum.value.upper()}]"
            print(f"\n{console_prefix} Testing provider...")
            health_status = await llm_service.check_provider_health(provider_enum)
            print(f"{console_prefix} Health: {health_status.status.value} - {health_status.message} {health_status.details or ''}")

            if health_status.status == StatusEnum.OK:
                try:
                    answer, model_used = await llm_service.generate_response(
                        query=sample_query,
                        context_chunks=sample_chunks,
                        provider=provider_enum,
                    )
                    print(f"{console_prefix} Query: {sample_query}")
                    print(f"{console_prefix} Model Used: {model_used}")
                    print(f"{console_prefix} Answer: {answer}")
                except (LLMError, ConfigurationError) as e:
                    print(f"{console_prefix} Error generating response: {e.message} - {e.detail}")
                except Exception as e:
                    print(f"{console_prefix} Unexpected error: {e}")
            else:
                print(f"{console_prefix} Skipping generation due to health check failure.")

        async def main():
            # Test OpenAI
            if llm_service.settings.OPENAI_API_KEY:
                await test_provider(LLMProvider.OPENAI)
            else:
                print("[OPENAI] Skipping test: OPENAI_API_KEY not set.")

            # Test Anthropic
            if llm_service.settings.ANTHROPIC_API_KEY:
                await test_provider(LLMProvider.ANTHROPIC)
            else:
                print("[ANTHROPIC] Skipping test: ANTHROPIC_API_KEY not set.")

            # Test Gemini
            if llm_service.settings.GOOGLE_API_KEY:
                await test_provider(LLMProvider.GEMINI)
            else:
                print("[GEMINI] Skipping test: GOOGLE_API_KEY not set.")

        asyncio.run(main())

    except ConfigurationError as e:
        print(f"Configuration Error during LLMService init: {e.message}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logger.error("Main example error", exc_info=True)

