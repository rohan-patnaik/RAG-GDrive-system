import json
import logging
import os
import sys
from typing import Dict, Any

# Adjust sys.path to include the project root (where 'backend' directory is located)
# This allows importing modules from 'backend.rag_system.*'
# The path needs to be relative to this file's location within the Netlify deployment structure.
# Assuming 'netlify/functions/health.py' and 'backend/' are siblings under a common root.
# Netlify typically places the function file inside a structure like /var/task/netlify/functions/health.py
# and the rest of the repo might be at /var/task/
# So, adding /var/task (which is effectively two levels up from 'netlify/functions') should work.
# Or, more robustly, find the project root dynamically.

# For local testing, this might need adjustment. For Netlify, this structure is common.
# Adding project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure basic logging for the function
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Netlify function handler for health check.
    Attempts to load application settings as a basic health indicator.
    """
    logger.info("Health check function invoked.")

    response_headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",  # Allow requests from any origin
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "GET, OPTIONS" # Allow GET and OPTIONS
    }

    # Handle preflight OPTIONS requests for CORS
    if event.get('httpMethod', '').upper() == 'OPTIONS':
        logger.info("Handling OPTIONS preflight request.")
        return {
            "statusCode": 204, # No Content
            "headers": response_headers,
            "body": ""
        }

    if event.get('httpMethod', '').upper() != 'GET':
        logger.warning(f"Health check called with non-GET method: {event.get('httpMethod')}")
        return {
            "statusCode": 405, # Method Not Allowed
            "headers": response_headers,
            "body": json.dumps({"error": "Method Not Allowed", "message": "Only GET requests are supported."})
        }

    try:
        # Dynamically import AppSettings after adjusting sys.path
        from backend.rag_system.config.settings import AppSettings, get_settings

        # Attempt to load settings
        settings = get_settings() # This will use cached settings if already loaded

        settings_loaded_successfully = settings is not None

        if settings_loaded_successfully:
            logger.info("Application settings loaded successfully.")
            response_body = {
                "status": "OK",
                "message": "Service healthy, settings loaded.",
                "settings_loaded": True,
                "app_name": settings.APP_NAME,
                "environment": settings.ENVIRONMENT,
                "vector_store_provider": settings.VECTOR_STORE_PROVIDER,
                # Avoid logging sensitive keys here
            }
            return {
                "statusCode": 200,
                "headers": response_headers,
                "body": json.dumps(response_body),
            }
        else:
            # This case should ideally not be reached if get_settings() raises an exception on failure
            logger.error("Settings object was None after calling get_settings().")
            return {
                "statusCode": 500,
                "headers": response_headers,
                "body": json.dumps({
                    "status": "ERROR",
                    "message": "Service unhealthy, settings object is None.",
                    "settings_loaded": False,
                }),
            }

    except ImportError as e:
        logger.error(f"ImportError during health check: {str(e)}." \
                     f" Current sys.path: {sys.path}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": response_headers,
            "body": json.dumps({
                "status": "ERROR",
                "message": "Failed to import backend modules.",
                "settings_loaded": False,
                "error_details": str(e)
            }),
        }
    except Exception as e:
        logger.error(f"Error during health check: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": response_headers,
            "body": json.dumps({
                "status": "ERROR",
                "message": "Service unhealthy due to an internal error.",
                "settings_loaded": False,
                "error_details": str(e)
            }),
        }

# Example for local testing (not run by Netlify)
if __name__ == "__main__":
    # To test this locally, you might need to simulate the Netlify environment
    # or adjust paths. For example, ensure 'backend' is in PYTHONPATH.
    # Example: PYTHONPATH=$PYTHONPATH:$(pwd) python netlify/functions/health.py
    logger.info("Running health check locally for testing...")

    # Mock event and context
    mock_event = {"httpMethod": "GET"}
    mock_context = None # Netlify context object, not strictly needed for this simple test

    response = handler(mock_event, mock_context)
    print(json.dumps(response, indent=2))

    # Test OPTIONS
    mock_event_options = {"httpMethod": "OPTIONS"}
    response_options = handler(mock_event_options, mock_context)
    print(json.dumps(response_options, indent=2))

    # Test POST
    mock_event_post = {"httpMethod": "POST"}
    response_post = handler(mock_event_post, mock_context)
    print(json.dumps(response_post, indent=2))

    # To test the ImportError, you might need to temporarily break the sys.path logic
    # or rename the 'backend' directory.
    # For instance, temporarily remove project_root from sys.path for a test:
    # original_sys_path = list(sys.path)
    # if project_root in sys.path:
    #     sys.path.remove(project_root)
    # response_import_error = handler(mock_event, mock_context)
    # print("Import Error Test Response:", json.dumps(response_import_error, indent=2))
    # sys.path = original_sys_path # Restore

    # To test general exception, you could mock get_settings to raise an error:
    # from unittest.mock import patch
    # with patch('backend.rag_system.config.settings.get_settings', side_effect=Exception("Mocked loading error")):
    #    response_general_error = handler(mock_event, mock_context)
    #    print("General Error Test Response:", json.dumps(response_general_error, indent=2))

    logger.info("Local health check test finished.")
```
