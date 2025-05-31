# netlify/functions/query.py
import json
import os
import sys
import logging
import asyncio
from typing import Dict, Any, Optional

# --- Path Setup ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except Exception as e:
    print(f"Error setting up sys.path in query function: {e}", file=sys.stderr)


# --- Imports from backend ---
try:
    from backend.rag_system.config.settings import AppSettings, get_settings, clear_settings_cache
    from backend.rag_system.services.rag_service import RAGService
    from backend.rag_system.services.embedding_service import EmbeddingService
    from backend.rag_system.services.vector_store import VectorStoreService
    from backend.rag_system.services.llm_service import LLMService
    from backend.rag_system.core.document_loader import DocumentLoader
    from backend.rag_system.core.text_processor import TextProcessor
    from backend.rag_system.models.schemas import QueryRequest, QueryResponse
    from backend.rag_system.utils.exceptions import BaseRAGError, QueryProcessingError, ConfigurationError
    from pydantic import ValidationError
except ImportError as e:
    print(f"ImportError in query.py: {e}. Check sys.path and backend module availability.", file=sys.stderr)
    class BaseRAGError(Exception): pass # type: ignore
    class QueryRequest: # type: ignore
        def __init__(self, **kwargs): raise NotImplementedError(f"QueryRequest not loaded: {e}")
    class QueryResponse: # type: ignore
        def __init__(self, **kwargs): raise NotImplementedError(f"QueryResponse not loaded: {e}")
        def model_dump(self, **kwargs): return {"error": "QueryResponse not loaded", "detail": str(e)}


# --- Logging ---
logger = logging.getLogger("netlify.functions.query")
logging.basicConfig(stream=sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO").upper())


# --- Global Instances for Caching ---
_settings_cache: Optional[AppSettings] = None
_rag_service_cache: Optional[RAGService] = None
_initialization_error: Optional[str] = None

def _get_cached_settings() -> AppSettings:
    global _settings_cache
    if _settings_cache is None:
        logger.info("Initializing AppSettings for Netlify query function...")
        _settings_cache = get_settings()
        logger.info(f"AppSettings loaded in query: AppName='{_settings_cache.APP_NAME}', VectorProvider='{_settings_cache.VECTOR_STORE_PROVIDER}'")
    return _settings_cache

def _initialize_rag_service_if_needed() -> RAGService:
    global _rag_service_cache, _initialization_error
    if _rag_service_cache is not None:
        return _rag_service_cache
    if _initialization_error is not None:
        raise ConfigurationError(f"RAGService previously failed to initialize: {_initialization_error}")

    logger.info("Attempting to initialize RAGService for Netlify query function...")
    try:
        current_settings = _get_cached_settings()
        doc_loader = DocumentLoader()
        text_processor = TextProcessor(
            chunk_size=current_settings.CHUNK_SIZE,
            chunk_overlap=current_settings.CHUNK_OVERLAP
        )
        embedding_service = EmbeddingService(settings=current_settings)
        vector_store_service = VectorStoreService(settings=current_settings, embedding_service=embedding_service)
        llm_service = LLMService(settings=current_settings)

        _rag_service_cache = RAGService(
            settings=current_settings,
            document_loader=doc_loader,
            text_processor=text_processor,
            embedding_service=embedding_service,
            vector_store_service=vector_store_service,
            llm_service=llm_service
        )
        logger.info("RAGService initialized successfully in query function.")
        _initialization_error = None
        return _rag_service_cache
    except Exception as e:
        _initialization_error = str(e)
        logger.error(f"CRITICAL: Failed to initialize RAGService in query function: {e}", exc_info=True)
        raise ConfigurationError(f"Failed to initialize RAGService: {e}")


# --- Netlify Function Handler ---
def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    response_headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Allow-Methods": "POST, OPTIONS", # Query is POST
    }

    http_method = event.get("httpMethod", "POST").upper()

    if http_method == "OPTIONS":
        return {"statusCode": 204, "headers": response_headers, "body": ""}
    if http_method != "POST":
        return {"statusCode": 405, "headers": response_headers, "body": json.dumps({"error": "Method Not Allowed"})}

    logger.info(f"Query function invoked. Method: {http_method}")

    try:
        rag_service = _initialize_rag_service_if_needed()

        event_body_str = event.get("body", "{}")
        if not event_body_str:
            logger.warning("Received empty request body for query.")
            return {"statusCode": 400, "headers": response_headers, "body": json.dumps({"error": "Bad Request", "message": "Request body is empty."})}
        
        try:
            request_data_dict = json.loads(event_body_str)
            logger.info(f"Received query request data: {request_data_dict}")
            query_request = QueryRequest(**request_data_dict)
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from request body.", exc_info=True)
            return {"statusCode": 400, "headers": response_headers, "body": json.dumps({"error": "Bad Request", "message": "Invalid JSON format."})}
        except ValidationError as ve:
            logger.error(f"Validation error for QueryRequest: {ve.errors()}", exc_info=True)
            return {"statusCode": 422, "headers": response_headers, "body": json.dumps({"error": "Unprocessable Entity", "detail": ve.errors()})}


        query_response_model: QueryResponse = asyncio.run(rag_service.query(query_request))
        response_body_dict = query_response_model.model_dump(mode="json")

        logger.info(f"Query processed successfully for: '{query_request.query_text[:50]}...'")
        return {
            "statusCode": 200,
            "headers": response_headers,
            "body": json.dumps(response_body_dict),
        }

    except BaseRAGError as e:
        logger.error(f"RAG Error during query processing: {type(e).__name__} - {e.message}", exc_info=True)
        return {
            "statusCode": getattr(e, 'status_code', 500),
            "headers": response_headers,
            "body": json.dumps({"error": e.error_type, "message": e.message, "detail": str(e.detail)}),
        }
    except Exception as e:
        logger.error(f"Unexpected critical error during query processing: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": response_headers,
            "body": json.dumps({"error": "Internal Server Error", "message": "An unexpected error occurred."}),
        }

# --- Local Testing (Optional) ---
if __name__ == "__main__":
    print("--- Running query.py locally for testing ---")
    # Mock event and context
    mock_query_body = {
        "query_text": "What is the capital of France?",
        "llm_provider": "gemini" # Ensure your .env has GOOGLE_API_KEY
    }
    mock_event_post = {
        "httpMethod": "POST",
        "body": json.dumps(mock_query_body)
    }
    mock_context = None
    
    print("\n--- Testing POST request ---")
    # Ensure your .env is configured, especially PINECONE keys and LLM keys
    # Also, ensure you have ingested some data into your Pinecone index.
    # If testing ingestion first, comment out this query test.
    response_post = handler(mock_event_post, mock_context)
    print(f"Status Code: {response_post['statusCode']}")
    try:
        print("Body:", json.dumps(json.loads(response_post.get("body", "{}")), indent=2))
    except json.JSONDecodeError:
        print("Body (raw):", response_post.get("body"))
    print("--- End of local test ---")
