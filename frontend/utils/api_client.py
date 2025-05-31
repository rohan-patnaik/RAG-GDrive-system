import httpx
import asyncio
from typing import Dict, Any, Optional

BACKEND_API_URL = "http://localhost:8000"

async def make_api_request(method: str, endpoint: str, json_data=None, params=None) -> Optional[Dict[str, Any]]:
    """Make an API request to the backend."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            url = f"{BACKEND_API_URL}{endpoint}"
            
            if method.upper() == "GET":
                response = await client.get(url, params=params)
            elif method.upper() == "POST":
                response = await client.post(url, json=json_data, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
    except httpx.ConnectError:
        raise ConnectionError(f"Could not connect to backend at {BACKEND_API_URL}. Make sure the API server is running.")
    except httpx.HTTPStatusError as e:
        raise Exception(f"API request failed with status {e.response.status_code}: {e.response.text}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")

async def query_backend(query_data: dict) -> Optional[Dict[str, Any]]:
    """Send a query to the backend RAG system."""
    return await make_api_request("POST", "/query/", json_data=query_data)

async def get_system_status() -> Optional[Dict[str, Any]]:
    """Get system status from the backend.""" 
    return await make_api_request("GET", "/health")

async def ingest_documents_on_backend(ingestion_data: dict) -> Optional[Dict[str, Any]]:
    """Trigger document ingestion on the backend."""
    return await make_api_request("POST", "/documents/ingest", json_data=ingestion_data)

# Note: For file upload functionality, you would need:
# async def upload_and_ingest_files(files: List[UploadedFile]) -> Optional[Dict[str, Any]]:
#     """
#     Uploads files to a backend endpoint for ingestion.
#     This requires a new backend endpoint that accepts multipart/form-data.
#     """
#     # Implementation would depend on backend endpoint design
#     pass
