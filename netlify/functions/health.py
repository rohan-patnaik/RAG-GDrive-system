# netlify/functions/health.py
import json
import os
import sys
import logging
import asyncio # Required for asyncio.run
from typing import Dict, Any, Optional

# --- Path Setup ---
# Add project root to sys.path to allow imports from 'backend'
# This assumes 'netlify/functions/' is one level down from the project root.
# Adjust if your directory structure is different.
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # print(f"Health Function: Project root added to sys.path: {project_root}", file=sys.stderr)
    # print(f"Health Function: sys.path: {sys.path}", file=sys.stderr)
except Exception as e:
    print(f"Error setting up sys.path in health function: {e}", file=sys.stderr)
    # Fallback or re-raise if critical
    pass


# --- Imports from backend ---
# These imports must happen AFTER sys.path is configured.
try:
    from backend.rag_system.config.settings import AppSettings, get_settings, clear_settings_cache
    from backend.rag_system.config.logging_config import setup_logging # For full logging setup
    from backend.rag_system.services.rag_service import RAGService
    from backend.rag_system.services.embedding_service import EmbeddingService
    from backend.rag_system.services.vector_store import VectorStoreService
    from backend.rag_system.services.llm_service import LLMService
    from backend.rag_system.core.document_loader import DocumentLoader # RAGService needs it
    from backend.rag_system.core.text_processor import TextProcessor   # RAGService needs it
    from backend.rag_system.models.schemas import SystemStatusResponse, ComponentStatus, StatusEnum
    from backend.rag_system.utils.exceptions import BaseRAGError
except ImportError as e:
    print(f"ImportError in health.py: {e}. Check sys.path and backend module availability.", file=sys.stderr)
    # Define dummy classes or raise to make it clear initialization failed
    class BaseRAGError(Exception): pass # type: ignore
    class StatusEnum: ERROR = "ERROR" # type: ignore
    class SystemStatusResponse: # type: ignore
        def __init__(self, **kwargs): self.kwargs = kwargs
        def model_dump(self, **kwargs): return {"system_status": "ERROR", "message": f"ImportError: {e}"}
    # This ensures the function can still return a JSON error if imports fail.


# --- Logging ---
# Configure basic logging. For more advanced, call setup_logging from your config.
# Netlify captures stdout/stderr, so print statements or basic logging work.
logger = logging.getLogger("netlify.functions.health")
logging.basicConfig(stream=sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO").upper())


# --- Global Instances for Caching (leveraging Netlify's function instance reuse) ---
_settings_cache: Optional[AppSettings] = None
_rag_service_cache: Optional[RAGService] = None
_initialization_error: Optional[str] = None


def _get_cached_settings() -> AppSettings:
    global _settings_cache
    if _settings_cache is None:
        logger.info("Initializing AppSettings for Netlify health function...")
        # clear_settings_cache() # Call this if you need to force-reload from env vars on each invocation
        _settings_cache = get_settings()
        logger.info(f"AppSettings loaded in health: AppName='{_settings_cache.APP_NAME}', VectorProvider='{_settings_cache.VECTOR_STORE_PROVIDER}'")
    return _settings_cache

def _initialize_rag_service_if_needed() -> RAGService:
    global _rag_service_cache, _initialization_error
    if _rag_service_cache is not None:
        return _rag_service_cache
    if _initialization_error is not None: # If previous attempt failed
        raise ConfigurationError(f"RAGService previously failed to initialize: {_initialization_error}")

    logger.info("Attempting to initialize RAGService for Netlify health function...")
    try:
        current_settings = _get_cached_settings()
        # Explicitly create dependencies for RAGService
        # DocumentLoader and TextProcessor are needed by RAGService constructor
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
        logger.info("RAGService initialized successfully in health function.")
        _initialization_error = None # Clear any previous error
        return _rag_service_cache
    except Exception as e:
        _initialization_error = str(e)
        logger.error(f"CRITICAL: Failed to initialize RAGService in health function: {e}", exc_info=True)
        # Raise a specific error to be caught by the handler
        raise ConfigurationError(f"Failed to initialize RAGService: {e}")


# --- Netlify Function Handler ---
def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    response_headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
    }

    http_method = event.get("httpMethod", "GET").upper()

    if http_method == "OPTIONS":
        return {"statusCode": 204, "headers": response_headers, "body": ""}
    if http_method != "GET":
        return {"statusCode": 405, "headers": response_headers, "body": json.dumps({"error": "Method Not Allowed"})}

    logger.info(f"Health function invoked. Method: {http_method}")

    try:
        # Ensure RAGService is initialized (or attempt to)
        rag_service = _initialize_rag_service_if_needed()

        # Run the async get_system_status method
        # Netlify's Python runtime supports top-level await or asyncio.run
        status_report_model: SystemStatusResponse = asyncio.run(rag_service.get_system_status())
        response_body_dict = status_report_model.model_dump(mode="json")
        
        # Determine overall HTTP status based on system status (optional)
        # http_status_code = 200 if status_report_model.system_status == StatusEnum.OK else 503
        http_status_code = 200 # Always return 200, body indicates detailed status

        logger.info(f"Health check successful. Overall status: {status_report_model.system_status}")
        return {
            "statusCode": http_status_code,
            "headers": response_headers,
            "body": json.dumps(response_body_dict),
        }

    except BaseRAGError as e: # Catch custom RAG errors (includes ConfigurationError)
        logger.error(f"RAG Error during health check: {type(e).__name__} - {e.message}", exc_info=True)
        # Try to get settings for app_name, otherwise use default
        app_name_fallback = "RAG System"
        env_fallback = "unknown"
        try:
            s = _get_cached_settings()
            app_name_fallback = s.APP_NAME
            env_fallback = s.ENVIRONMENT
        except Exception:
            pass

        error_status = SystemStatusResponse(
            system_status=StatusEnum.ERROR,
            app_name=app_name_fallback,
            environment=env_fallback,
            components=[
                ComponentStatus(name="SystemInitialization", status=StatusEnum.ERROR, message=f"{type(e).__name__}: {e.message}", details=e.detail)
            ]
        )
        return {
            "statusCode": getattr(e, 'status_code', 500), # Use error's status code or default to 500
            "headers": response_headers,
            "body": json.dumps(error_status.model_dump(mode="json")),
        }
    except Exception as e:
        logger.error(f"Unexpected critical error during health check: {e}", exc_info=True)
        # Fallback error response if something went very wrong
        critical_error_body = {
            "system_status": "ERROR",
            "message": "A critical unexpected error occurred in the health check.",
            "detail": str(e),
        }
        return {
            "statusCode": 500,
            "headers": response_headers,
            "body": json.dumps(critical_error_body),
        }

# --- Local Testing (Optional) ---
# To test this function locally:
# 1. Ensure your .env file is in the project root (two levels up from this file).
# 2. Run `python netlify/functions/health.py` from the project root.
if __name__ == "__main__":
    print("--- Running health.py locally for testing ---")
    # Mock event and context
    mock_event_get = {"httpMethod": "GET"}
    mock_context = None
    
    print("\n--- Testing GET request ---")
    response_get = handler(mock_event_get, mock_context)
    print(f"Status Code: {response_get['statusCode']}")
    try:
        print("Body:", json.dumps(json.loads(response_get.get("body", "{}")), indent=2))
    except json.JSONDecodeError:
        print("Body (raw):", response_get.get("body"))

    print("\n--- Testing OPTIONS request ---")
    mock_event_options = {"httpMethod": "OPTIONS"}
    response_options = handler(mock_event_options, mock_context)
    print(f"Status Code: {response_options['statusCode']}")
    print("Body:", response_options.get("body"))
    print("--- End of local test ---")

