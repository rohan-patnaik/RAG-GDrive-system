# netlify/functions/ingest.py
import json
import os
import sys
import logging
import asyncio
import cgi # For parsing multipart/form-data
import io # For BytesIO
import base64 # If files are base64 encoded in the event
from typing import Dict, Any, Optional, List, Tuple

# --- Path Setup ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except Exception as e:
    print(f"Error setting up sys.path in ingest function: {e}", file=sys.stderr)

# --- Imports from backend ---
try:
    from backend.rag_system.config.settings import AppSettings, get_settings, clear_settings_cache
    from backend.rag_system.services.rag_service import RAGService
    from backend.rag_system.services.embedding_service import EmbeddingService
    from backend.rag_system.services.vector_store import VectorStoreService
    from backend.rag_system.services.llm_service import LLMService
    from backend.rag_system.core.document_loader import DocumentLoader
    from backend.rag_system.core.text_processor import TextProcessor
    from backend.rag_system.models.schemas import IngestionResponse # Assuming this is the correct response model
    from backend.rag_system.utils.exceptions import BaseRAGError, DocumentProcessingError, ConfigurationError
except ImportError as e:
    print(f"ImportError in ingest.py: {e}. Check sys.path and backend module availability.", file=sys.stderr)
    class BaseRAGError(Exception): pass # type: ignore
    class IngestionResponse: # type: ignore
        def __init__(self, **kwargs): self.kwargs = kwargs
        def model_dump(self, **kwargs): return {"error": "IngestionResponse not loaded", "detail": str(e)}


# --- Logging ---
logger = logging.getLogger("netlify.functions.ingest")
logging.basicConfig(stream=sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO").upper())

# --- Global Instances for Caching ---
_settings_cache: Optional[AppSettings] = None
_rag_service_cache: Optional[RAGService] = None
_initialization_error: Optional[str] = None


def _get_cached_settings() -> AppSettings:
    global _settings_cache
    if _settings_cache is None:
        logger.info("Initializing AppSettings for Netlify ingest function...")
        _settings_cache = get_settings()
        logger.info(f"AppSettings loaded in ingest: AppName='{_settings_cache.APP_NAME}', VectorProvider='{_settings_cache.VECTOR_STORE_PROVIDER}'")
    return _settings_cache

def _initialize_rag_service_if_needed() -> RAGService:
    global _rag_service_cache, _initialization_error
    if _rag_service_cache is not None:
        return _rag_service_cache
    if _initialization_error is not None:
        raise ConfigurationError(f"RAGService previously failed to initialize: {_initialization_error}")

    logger.info("Attempting to initialize RAGService for Netlify ingest function...")
    try:
        current_settings = _get_cached_settings()
        doc_loader = DocumentLoader()
        text_processor = TextProcessor(
            chunk_size=current_settings.CHUNK_SIZE,
            chunk_overlap=current_settings.CHUNK_OVERLAP
        )
        embedding_service = EmbeddingService(settings=current_settings)
        vector_store_service = VectorStoreService(settings=current_settings, embedding_service=embedding_service)
        llm_service = LLMService(settings=current_settings) # Not directly used by ingest, but part of RAGService

        _rag_service_cache = RAGService(
            settings=current_settings,
            document_loader=doc_loader,
            text_processor=text_processor,
            embedding_service=embedding_service,
            vector_store_service=vector_store_service,
            llm_service=llm_service
        )
        logger.info("RAGService initialized successfully in ingest function.")
        _initialization_error = None
        return _rag_service_cache
    except Exception as e:
        _initialization_error = str(e)
        logger.error(f"CRITICAL: Failed to initialize RAGService in ingest function: {e}", exc_info=True)
        raise ConfigurationError(f"Failed to initialize RAGService: {e}")

def _parse_multipart_form_data(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parses multipart/form-data from the Netlify event.
    Assumes 'cgi.FieldStorage' can handle the body.
    """
    uploaded_files_data: List[Dict[str, Any]] = []
    
    content_type_header = event.get('headers', {}).get('content-type', 
                                                    event.get('headers', {}).get('Content-Type', ''))
    
    if 'multipart/form-data' not in content_type_header:
        logger.warning(f"Content-Type is not multipart/form-data: {content_type_header}")
        # Could try to handle as single file if body is just raw content, but less robust
        return []

    body_str = event.get('body', '')
    is_base64_encoded = event.get('isBase64Encoded', False)

    if is_base64_encoded:
        try:
            body_bytes = base64.b64decode(body_str)
        except Exception as e:
            logger.error(f"Failed to base64 decode body: {e}")
            return [] # Or raise error
    else:
        # For multipart/form-data, the body might be a string that needs to be encoded
        # to bytes using an encoding that preserves the binary data (like latin-1 or utf-8 with error handling)
        # cgi.FieldStorage expects a file-like object of bytes.
        try:
            body_bytes = body_str.encode('latin-1') # Common for HTTP bodies if not specified
        except Exception as e:
            logger.warning(f"Could not encode body string as latin-1 for cgi parser, trying utf-8: {e}")
            try:
                body_bytes = body_str.encode('utf-8') # try utf-8
            except Exception as e2:
                logger.error(f"Failed to encode body string as bytes for cgi parser: {e2}")
                return []


    # Create a file-like object for cgi.FieldStorage
    fp = io.BytesIO(body_bytes)

    # Prepare environment for cgi.FieldStorage
    # It needs CONTENT_TYPE and CONTENT_LENGTH (if available)
    environ = {
        'REQUEST_METHOD': 'POST',
        'CONTENT_TYPE': content_type_header,
        'CONTENT_LENGTH': str(len(body_bytes)) # Calculate length of the byte body
    }

    try:
        form = cgi.FieldStorage(fp=fp, environ=environ, keep_blank_values=True)
        
        for field_name in form.keys():
            field_item = form[field_name]
            
            if isinstance(field_item, list): # Multiple files with the same field name
                for item in field_item:
                    if item.filename and item.file:
                        file_content_bytes = item.file.read()
                        uploaded_files_data.append({
                            "filename": item.filename,
                            "content_bytes": file_content_bytes
                        })
                        logger.info(f"Parsed uploaded file (list item): {item.filename}, size: {len(file_content_bytes)} bytes")
            elif field_item.filename and field_item.file: # Single file
                file_content_bytes = field_item.file.read()
                uploaded_files_data.append({
                    "filename": field_item.filename,
                    "content_bytes": file_content_bytes
                })
                logger.info(f"Parsed uploaded file (single item): {field_item.filename}, size: {len(file_content_bytes)} bytes")
            # else:
                # logger.debug(f"Skipping form field '{field_name}' as it's not a file upload.")

    except Exception as e:
        logger.error(f"Error parsing multipart/form-data with cgi: {e}", exc_info=True)
        # This might happen if the body is not well-formed multipart

    if not uploaded_files_data:
        logger.warning("No files successfully parsed from multipart/form-data.")

    return uploaded_files_data


# --- Netlify Function Handler ---
def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    response_headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type, Authorization", # Allow Content-Type for multipart
        "Access-Control-Allow-Methods": "POST, OPTIONS", # Ingest is POST
    }

    http_method = event.get("httpMethod", "POST").upper()

    if http_method == "OPTIONS":
        return {"statusCode": 204, "headers": response_headers, "body": ""}
    if http_method != "POST":
        return {"statusCode": 405, "headers": response_headers, "body": json.dumps({"error": "Method Not Allowed"})}

    logger.info(f"Ingest function invoked. Method: {http_method}, Headers: {event.get('headers')}")

    try:
        rag_service = _initialize_rag_service_if_needed()
        
        # Parse uploaded files from the event
        # This part is highly dependent on how Netlify (and API Gateway, if any) passes multipart/form-data
        uploaded_files_data = _parse_multipart_form_data(event)

        if not uploaded_files_data:
            logger.warning("No files found in the request for ingestion.")
            return {
                "statusCode": 400,
                "headers": response_headers,
                "body": json.dumps({"error": "Bad Request", "message": "No files provided for ingestion or failed to parse."})
            }
        
        logger.info(f"Attempting to ingest {len(uploaded_files_data)} parsed files.")
        ingestion_response_model: IngestionResponse = asyncio.run(
            rag_service.ingest_uploaded_documents(uploaded_files_data)
        )
        response_body_dict = ingestion_response_model.model_dump(mode="json")

        logger.info(f"Ingestion process completed. Message: {ingestion_response_model.message}")
        return {
            "statusCode": 201, # 201 Created is often used for successful ingestion
            "headers": response_headers,
            "body": json.dumps(response_body_dict),
        }

    except BaseRAGError as e:
        logger.error(f"RAG Error during ingestion: {type(e).__name__} - {e.message}", exc_info=True)
        return {
            "statusCode": getattr(e, 'status_code', 500),
            "headers": response_headers,
            "body": json.dumps({"error": e.error_type, "message": e.message, "detail": str(e.detail)}),
        }
    except Exception as e:
        logger.error(f"Unexpected critical error during ingestion: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": response_headers,
            "body": json.dumps({"error": "Internal Server Error", "message": "An unexpected error occurred during ingestion."}),
        }

# --- Local Testing (Optional - More Complex for Multipart) ---
if __name__ == "__main__":
    print("--- Running ingest.py locally for testing (Multipart is tricky here) ---")
    # To test this locally, you'd need to construct a mock 'event' dictionary
    # that accurately simulates a Netlify multipart/form-data request.
    # This is non-trivial. It's often easier to test with `netlify dev` or by deploying.

    # Example of a very basic mock (actual Netlify event is more complex):
    # file_content = "This is a test file for local ingest."
    # boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    # body_parts = [
    #     f"--{boundary}",
    #     'Content-Disposition: form-data; name="files"; filename="test.txt"',
    #     'Content-Type: text/plain',
    #     '',
    #     file_content,
    #     f"--{boundary}--",
    #     ''
    # ]
    # mock_body = "\r\n".join(body_parts)
    
    # mock_event_post_multipart = {
    #     "httpMethod": "POST",
    #     "headers": {
    #         "content-type": f"multipart/form-data; boundary={boundary}",
    #         "Content-Type": f"multipart/form-data; boundary={boundary}" # Case variations
    #     },
    #     "body": mock_body, # For cgi, this needs to be bytes
    #     "isBase64Encoded": False
    # }
    # mock_context = None
    
    # print("\n--- Testing POST request (mocked multipart) ---")
    # # Ensure .env is configured
    # # response_post = handler(mock_event_post_multipart, mock_context)
    # # print(f"Status Code: {response_post['statusCode']}")
    # # try:
    # #     print("Body:", json.dumps(json.loads(response_post.get("body", "{}")), indent=2))
    # # except json.JSONDecodeError:
    # #     print("Body (raw):", response_post.get("body"))
    print("Local testing of multipart ingest is complex. Use `netlify dev` or deploy.")
    print("--- End of local test ---")
