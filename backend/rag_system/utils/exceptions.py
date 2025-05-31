# backend/rag_system/utils/exceptions.py
from typing import Optional, Any

class BaseRAGError(Exception):
    """Base class for custom exceptions in the RAG system."""
    status_code: int = 500
    error_type: str = "RAGSystemError"
    message: str = "An unexpected error occurred in the RAG system."
    detail: Optional[Any] = None

    def __init__(
        self,
        message: Optional[str] = None,
        detail: Optional[Any] = None,
        status_code: Optional[int] = None,
        error_type: Optional[str] = None,
    ):
        super().__init__(message or self.message)
        if message is not None:
            self.message = message
        if detail is not None:
            self.detail = detail
        if status_code is not None:
            self.status_code = status_code
        if error_type is not None:
            self.error_type = error_type

    def __str__(self) -> str:
        return f"{self.error_type} (Status {self.status_code}): {self.message}{f' - Detail: {self.detail}' if self.detail else ''}"


class ConfigurationError(BaseRAGError):
    """Exception for configuration-related errors."""
    status_code = 500  # Internal server error, as config issues are usually server-side
    error_type = "ConfigurationError"
    message = "A configuration error occurred."


class DocumentProcessingError(BaseRAGError):
    """Exception for errors during document loading or processing."""
    status_code = 400  # Bad request if input docs are problematic, or 500 if internal
    error_type = "DocumentProcessingError"
    message = "An error occurred while processing documents."


class EmbeddingError(BaseRAGError):
    """Exception for errors related to generating embeddings."""
    status_code = 500
    error_type = "EmbeddingError"
    message = "An error occurred while generating embeddings."


class VectorStoreError(BaseRAGError):
    """Exception for errors related to the vector store."""
    status_code = 500
    error_type = "VectorStoreError"
    message = "An error occurred with the vector store."


class LLMError(BaseRAGError):
    """Exception for errors related to LLM interactions."""
    status_code = 502  # Bad Gateway, as we're calling an external service
    error_type = "LLMError"
    message = "An error occurred while interacting with the LLM."
    provider: Optional[str] = None
    is_retryable: bool = False

    def __init__(
        self,
        message: Optional[str] = None,
        detail: Optional[Any] = None,
        status_code: Optional[int] = None,
        error_type: Optional[str] = None,
        provider: Optional[str] = None,
        is_retryable: bool = False,
    ):
        super().__init__(message, detail, status_code, error_type)
        if provider is not None:
            self.provider = provider
        self.is_retryable = is_retryable

    def __str__(self) -> str:
        provider_info = f" (Provider: {self.provider})" if self.provider else ""
        retry_info = " (Retryable)" if self.is_retryable else ""
        return f"{self.error_type}{provider_info}{retry_info} (Status {self.status_code}): {self.message}{f' - Detail: {self.detail}' if self.detail else ''}"


class QueryProcessingError(BaseRAGError):
    """Exception for errors during the query processing pipeline."""
    status_code = 500
    error_type = "QueryProcessingError"
    message = "An error occurred while processing the query."


class RAGServiceError(BaseRAGError):
    """Generic exception for errors within the RAGService orchestration logic."""
    status_code = 500
    error_type = "RAGServiceError"
    message = "An error occurred within the RAG service."


class IntegrationError(BaseRAGError):
    """Exception for errors related to third-party integrations (e.g., Google Drive)."""
    status_code = 503  # Service Unavailable, if an external integration fails
    error_type = "IntegrationError"
    message = "An error occurred with an external integration."


# Example of how to use:
# raise ConfigurationError("API key for OpenAI is missing.", detail={"missing_key": "OPENAI_API_KEY"})
# raise LLMError("OpenAI API request timed out.", provider="OpenAI", is_retryable=True)
