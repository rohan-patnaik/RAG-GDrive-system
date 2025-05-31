import json
import logging
import os
import sys
import asyncio
import cgi
import base64
import tempfile
import shutil
from typing import Dict, Any, List
from io import BytesIO

# Adjust sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global RAGService instance
rag_service_instance = None

def init_rag_service():
    global rag_service_instance
    if rag_service_instance is None:
        logger.info("Initializing RAGService for ingest function...")
        try:
            from backend.rag_system.services.rag_service import RAGService
            rag_service_instance = RAGService()
            logger.info("RAGService initialized successfully for ingest.")
        except Exception as e:
            logger.error(f"Error during RAGService initialization for ingest: {str(e)}", exc_info=True)
            raise
    return rag_service_instance

async def process_ingestion(event: Dict[str, Any]) -> Dict[str, Any]:
    from backend.rag_system.models.schemas import IngestionRequest, IngestionResponse
    from backend.rag_system.utils.exceptions import RAGBaseError

    temp_request_dir = None # To store path to temporary directory for this request

    try:
        rag_service = init_rag_service()
        if not rag_service:
             return {"statusCode": 500, "body": json.dumps({"error": "ServiceInitializationError", "message": "RAGService could not be initialized for ingestion."})}

        # Parse multipart/form-data
        content_type_header = event.get('headers', {}).get('content-type', event.get('headers', {}).get('Content-Type', ''))
        if not content_type_header or 'multipart/form-data' not in content_type_header.lower():
            logger.warning(f"Invalid content type: {content_type_header}")
            return {"statusCode": 400, "body": json.dumps({"error": "InvalidContentType", "message": "Request must be multipart/form-data."})}

        # Decode body if base64 encoded
        request_body_str = event.get('body', '')
        if event.get('isBase64Encoded', False):
            logger.info("Request body is base64 encoded. Decoding...")
            try:
                request_body_bytes = base64.b64decode(request_body_str)
            except Exception as e:
                logger.error(f"Base64 decoding failed: {str(e)}", exc_info=True)
                return {"statusCode": 400, "body": json.dumps({"error": "InvalidRequestBody", "message": "Failed to decode base64 body."})}
        else:
            # For cgi.FieldStorage, it might expect bytes, but sometimes works with string if headers are right.
            # Let's try to encode to latin-1 as cgi module sometimes has issues with UTF-8 strings directly.
            try:
                request_body_bytes = request_body_str.encode('latin-1') # Common encoding for HTTP bodies if not specified
            except Exception as e: # Fallback if it's already bytes somehow or other issue
                 logger.warning(f"Could not encode body string to latin-1, using raw string if it's an error: {e}")
                 if isinstance(request_body_str, str): # if it's a string, needs encoding
                    request_body_bytes = request_body_str.encode('utf-8') # try utf-8
                 else: # if it's already bytes, use as is (should not happen often with Netlify event['body'])
                    request_body_bytes = request_body_str


        # Prepare fp and headers for cgi.FieldStorage
        fp = BytesIO(request_body_bytes)
        environ = {'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': content_type_header, 'CONTENT_LENGTH': str(len(request_body_bytes))}

        try:
            form = cgi.FieldStorage(fp=fp, environ=environ, keep_blank_values=True)
        except Exception as e:
            logger.error(f"Failed to parse multipart form data: {str(e)}", exc_info=True)
            return {"statusCode": 400, "body": json.dumps({"error": "FormParsingError", "message": f"Could not parse form data: {str(e)}"})}

        if not form.list:
             logger.warning("Received empty form or FieldStorage could not parse fields.")
             return {"statusCode": 400, "body": json.dumps({"error": "EmptyForm", "message": "No files or form fields found in the request."})}

        # Create a unique temporary directory for this request's files
        # Netlify functions can write to /tmp
        temp_request_dir = tempfile.mkdtemp(prefix="rag_ingest_")
        logger.info(f"Created temporary directory for uploaded files: {temp_request_dir}")

        file_patterns = ["*.*"] # Process all files saved in the temp dir
        saved_files_count = 0

        for field_name in form.keys():
            field_item = form[field_name]
            if isinstance(field_item, list): # Multiple files with same name
                for item in field_item:
                    if item.filename:
                        file_content = item.file.read()
                        file_path = os.path.join(temp_request_dir, item.filename)
                        with open(file_path, 'wb') as f:
                            f.write(file_content)
                        logger.info(f"Saved uploaded file to temporary path: {file_path}")
                        saved_files_count += 1
            elif field_item.filename: # Single file
                file_content = field_item.file.read()
                file_path = os.path.join(temp_request_dir, field_item.filename)
                with open(file_path, 'wb') as f:
                    f.write(file_content)
                logger.info(f"Saved uploaded file to temporary path: {file_path}")
                saved_files_count +=1
            else:
                # Handle other form fields if necessary, e.g., parameters
                # For now, we only care about files.
                logger.info(f"Skipping non-file form field: {field_name}")

        if saved_files_count == 0:
            logger.warning("No files were uploaded or saved from the form.")
            # Clean up temp dir if created
            if temp_request_dir:
                shutil.rmtree(temp_request_dir)
                logger.info(f"Cleaned up empty temporary directory: {temp_request_dir}")
            return {"statusCode": 400, "body": json.dumps({"error": "NoFilesUploaded", "message": "No files found in the upload request."})}

        # Call RAGService to ingest from the temporary directory
        ingestion_request = IngestionRequest(
            source_directory=temp_request_dir,
            file_patterns=file_patterns, # Process all files we just saved
            recursive=False # Files are directly in temp_request_dir
        )

        logger.info(f"Calling RAGService.ingest_documents for directory: {temp_request_dir}")
        ingestion_response: IngestionResponse = await rag_service.ingest_documents(ingestion_request)

        return {
            "statusCode": 200,
            "body": ingestion_response.model_dump_json()
        }

    except RAGBaseError as e:
        logger.error(f"RAGService ingest error: {e.message}", exc_info=True)
        return {"statusCode": e.status_code if hasattr(e, 'status_code') else 500, "body": json.dumps({"error": e.error_type if hasattr(e, 'error_type') else type(e).__name__, "message": e.message, "detail": e.detail})}
    except Exception as e:
        logger.error(f"Unexpected error during ingestion processing: {str(e)}", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": "InternalServerError", "message": f"An unexpected error occurred: {str(e)}"})}
    finally:
        # Clean up the temporary directory
        if temp_request_dir and os.path.exists(temp_request_dir):
            try:
                shutil.rmtree(temp_request_dir)
                logger.info(f"Successfully cleaned up temporary directory: {temp_request_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory {temp_request_dir}: {e}", exc_info=True)


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    logger.info("Ingest function invoked.")

    response_headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*", # Adjust for production
        "Access-Control-Allow-Headers": "Content-Type, Authorization", # Allow Content-Type and Auth
        "Access-Control-Allow-Methods": "POST, OPTIONS"
    }

    if event.get('httpMethod', '').upper() == 'OPTIONS':
        logger.info("Handling OPTIONS preflight request for ingest endpoint.")
        return {"statusCode": 204, "headers": response_headers, "body": ""}

    if event.get('httpMethod', '').upper() != 'POST':
        logger.warning(f"Ingest endpoint called with non-POST method: {event.get('httpMethod')}")
        return {"statusCode": 405, "headers": response_headers, "body": json.dumps({"error": "MethodNotAllowed", "message": "Only POST requests are supported."})}

    try:
        response = asyncio.run(process_ingestion(event))
    except RuntimeError as e:
        logger.error(f"RuntimeError with asyncio for ingest: {str(e)}.", exc_info=True)
        response = {"statusCode": 500, "body": json.dumps({"error": "AsyncError", "message": "Error managing asynchronous operations for ingest."})}

    # Ensure headers are part of the final response
    if "headers" not in response:
        response["headers"] = {}
    response["headers"].update(response_headers)

    return response

# Example for local testing (requires more setup for multipart form data)
if __name__ == "__main__":
    logger.info("Running ingest function locally for testing (requires manual multipart body setup)...")
    # To test this locally, you'd need to construct a mock 'event' dictionary
    # that accurately simulates a Netlify multipart/form-data request.
    # This includes base64 encoding the body and setting correct Content-Type headers.
    # This is non-trivial to do directly in a simple script.
    # Consider using a tool like Postman or `curl` to send a multipart/form-data
    # request to a locally running Netlify Dev server (`netlify dev`).

    # A very simplified mock event (likely won't work fully with cgi.FieldStorage without a proper body):
    mock_event_empty_form = {
        "httpMethod": "POST",
        "headers": {"content-type": "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW"},
        "body": "", # A real body would be complex and base64 encoded
        "isBase64Encoded": False
    }

    print("\n--- Test Empty Form POST (will likely fail parsing or show no files) ---")
    # response = handler(mock_event_empty_form, None)
    # print(json.dumps(response, indent=2))
    print("Local testing of multipart/form-data is complex. Use 'netlify dev' and a tool like Postman.")

    logger.info("Local ingest function test finished.")
```
