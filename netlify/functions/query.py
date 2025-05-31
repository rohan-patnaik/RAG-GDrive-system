import json
import logging
import os
import sys
import asyncio
from typing import Dict, Any

# Adjust sys.path to include the project root (where 'backend' directory is located)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure basic logging for the function
# In a real Netlify environment, you might rely more on Netlify's own logging,
# but this helps for local testing and provides a basic setup.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global instance of RAGService to leverage Netlify's function instance reuse
# and reduce cold start times for subsequent invocations.
# This means RAGService (and its components like EmbeddingService) will be initialized
# once per function instance, not on every request.
rag_service_instance = None

def init_rag_service():
    """Initializes the RAGService if it hasn't been already."""
    global rag_service_instance
    if rag_service_instance is None:
        logger.info("Initializing RAGService instance...")
        try:
            from backend.rag_system.services.rag_service import RAGService
            # RAGService will load AppSettings and its components internally
            rag_service_instance = RAGService()
            logger.info("RAGService initialized successfully.")
        except ImportError as e:
            logger.error(f"Failed to import RAGService: {str(e)}." \
                         f" Current sys.path: {sys.path}", exc_info=True)
            # This error will be caught by the handler if init fails during a request
            raise
        except Exception as e:
            logger.error(f"Error during RAGService initialization: {str(e)}", exc_info=True)
            raise
    return rag_service_instance


async def process_query(event_body: str) -> Dict[str, Any]:
    """
    Helper async function to process the query using RAGService.
    Separated to manage asyncio event loop if needed.
    """
    from backend.rag_system.models.schemas import QueryRequest, QueryResponse
    from backend.rag_system.utils.exceptions import RAGBaseError

    try:
        rag_service = init_rag_service() # Get or initialize the service
        if rag_service is None: # Should not happen if init_rag_service raises on failure
             return {
                "statusCode": 500,
                "body": json.dumps({"error": "ServiceInitializationError", "message": "RAGService could not be initialized."})
            }

        try:
            request_data = json.loads(event_body)
            query_request = QueryRequest(**request_data)
            logger.info(f"Parsed QueryRequest: {query_request.query_text[:50]}...")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {str(e)}", exc_info=True)
            return {"statusCode": 400, "body": json.dumps({"error": "InvalidJSON", "message": str(e)})}
        except Exception as e: # Catches Pydantic validation errors too
            logger.error(f"Error parsing QueryRequest: {str(e)}", exc_info=True)
            return {"statusCode": 400, "body": json.dumps({"error": "InvalidRequest", "message": str(e)})}

        try:
            query_response: QueryResponse = await rag_service.query(query_request)
            logger.info(f"Successfully processed query. LLM answer: {query_response.llm_answer[:50]}...")
            return {
                "statusCode": 200,
                "body": query_response.model_dump_json() # Use Pydantic's method for proper serialization
            }
        except RAGBaseError as e: # Catch custom errors from the RAG system
            logger.error(f"RAGService query error: {e.message}", exc_info=True)
            return {
                "statusCode": e.status_code if hasattr(e, 'status_code') else 500,
                "body": json.dumps({"error": e.error_type if hasattr(e, 'error_type') else type(e).__name__, "message": e.message, "detail": e.detail})
            }
        except Exception as e: # Catch any other unexpected error
            logger.error(f"Unexpected error during query processing: {str(e)}", exc_info=True)
            return {
                "statusCode": 500,
                "body": json.dumps({"error": "InternalServerError", "message": "An unexpected error occurred."})
            }

    except ImportError as e: # Catch import errors during lazy loading of schemas etc.
        logger.error(f"ImportError during query processing: {str(e)}." \
                     f" Current sys.path: {sys.path}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "ImportFailure", "message": "Failed to import backend modules for query processing."})
        }
    except Exception as e: # Catch errors from init_rag_service() or other unexpected setup issues
        logger.error(f"Critical error in query function (e.g., service init): {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "ServiceUnavailable", "message": f"The service is currently unavailable: {str(e)}"})
        }


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Netlify function handler for RAG queries.
    """
    logger.info("Query function invoked.")

    response_headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",  # Adjust for production
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST, OPTIONS" # Allow POST and OPTIONS
    }

    # Handle preflight OPTIONS requests for CORS
    if event.get('httpMethod', '').upper() == 'OPTIONS':
        logger.info("Handling OPTIONS preflight request for query endpoint.")
        return {
            "statusCode": 204, # No Content
            "headers": response_headers,
            "body": ""
        }

    if event.get('httpMethod', '').upper() != 'POST':
        logger.warning(f"Query endpoint called with non-POST method: {event.get('httpMethod')}")
        return {
            "statusCode": 405, # Method Not Allowed
            "headers": response_headers,
            "body": json.dumps({"error": "MethodNotAllowed", "message": "Only POST requests are supported."})
        }

    event_body = event.get("body", "{}")
    if not event_body: # Check for empty body
        logger.warning("Received empty request body.")
        return {
            "statusCode": 400,
            "headers": response_headers,
            "body": json.dumps({"error": "InvalidRequest", "message": "Request body cannot be empty."})
        }

    # Netlify's Python runtime might handle top-level async handlers,
    # but using asyncio.run() is a safe way to ensure the async code runs correctly.
    # If the runtime already manages an event loop, asyncio.run() might not be ideal.
    # However, for a simple `def handler`, it's common.
    # A more advanced setup might use an ASGI adapter if Netlify supports it directly for plain functions.
    try:
        # For Python 3.7+, asyncio.run can be used.
        # If running in an environment that already has an event loop (like some serverless runtimes),
        # this might need adjustment (e.g., just `await process_query(...)` if the handler itself can be async).
        # Let's assume `asyncio.run` is safe here.
        response = asyncio.run(process_query(event_body))
    except RuntimeError as e:
        # This can happen if asyncio.run() is called when an event loop is already running.
        # In such cases, we might need to schedule the task differently.
        # For now, log and return an error.
        logger.error(f"RuntimeError with asyncio: {str(e)}. This might indicate an event loop issue.", exc_info=True)
        response = {
            "statusCode": 500,
            "body": json.dumps({"error": "AsyncError", "message": "Error managing asynchronous operations."})
        }

    # Add common headers to the response
    response["headers"] = response_headers
    return response


# Example for local testing (not run by Netlify)
if __name__ == "__main__":
    logger.info("Running query function locally for testing...")

    # Mock event and context
    mock_query_request_body = {
        "query_text": "What is the capital of France?",
        "llm_provider": "gemini", # Ensure your .env has GOOGLE_API_KEY for this to attempt
        "top_k": 1
    }
    mock_event = {
        "httpMethod": "POST",
        "body": json.dumps(mock_query_request_body)
    }
    mock_context = None

    # Ensure environment variables (PINECONE_API_KEY, etc.) are set in your local .env file
    # for RAGService initialization to succeed.

    # Test POST
    response = handler(mock_event, mock_context)
    print("\n--- Test POST Response ---")
    try:
        # Try to pretty-print if body is JSON, otherwise print as is
        body_content = json.loads(response.get("body", "{}"))
        print(json.dumps(body_content, indent=2))
    except json.JSONDecodeError:
        print(response.get("body"))
    print(f"Status Code: {response.get('statusCode')}")


    # Test OPTIONS
    mock_event_options = {"httpMethod": "OPTIONS"}
    response_options = handler(mock_event_options, mock_context)
    print("\n--- Test OPTIONS Response ---")
    print(json.dumps(response_options, indent=2))


    # Test GET (should be rejected)
    mock_event_get = {"httpMethod": "GET"}
    response_get = handler(mock_event_get, mock_context)
    print("\n--- Test GET Response (expect 405) ---")
    try:
        body_content = json.loads(response_get.get("body", "{}"))
        print(json.dumps(body_content, indent=2))
    except json.JSONDecodeError:
        print(response_get.get("body"))
    print(f"Status Code: {response_get.get('statusCode')}")

    # Test empty body
    mock_event_empty_body = {"httpMethod": "POST", "body": ""}
    response_empty_body = handler(mock_event_empty_body, mock_context)
    print("\n--- Test Empty Body POST Response (expect 400) ---")
    try:
        body_content = json.loads(response_empty_body.get("body", "{}"))
        print(json.dumps(body_content, indent=2))
    except json.JSONDecodeError:
        print(response_empty_body.get("body"))
    print(f"Status Code: {response_empty_body.get('statusCode')}")


    # Test invalid JSON
    mock_event_invalid_json = {"httpMethod": "POST", "body": "{'query_text': 'test'"} # Invalid JSON
    response_invalid_json = handler(mock_event_invalid_json, mock_context)
    print("\n--- Test Invalid JSON POST Response (expect 400) ---")
    try:
        body_content = json.loads(response_invalid_json.get("body", "{}"))
        print(json.dumps(body_content, indent=2))
    except json.JSONDecodeError:
        print(response_invalid_json.get("body"))
    print(f"Status Code: {response_invalid_json.get('statusCode')}")

    logger.info("Local query function test finished.")
```
