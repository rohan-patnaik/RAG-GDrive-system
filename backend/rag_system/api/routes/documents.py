# backend/rag_system/api/routes/documents.py
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Body, Request, status

from rag_system.services.rag_service import RAGService
from rag_system.models.schemas import IngestionRequest, IngestionResponse
from rag_system.utils.exceptions import DocumentProcessingError, VectorStoreError
from rag_system.config.settings import AppSettings

logger = logging.getLogger(__name__)
router = APIRouter()


# Dependency to get RAGService instance from application state
def get_rag_service(request: Request) -> RAGService:
    rag_service = request.app.state.rag_service
    if not rag_service:
        logger.error("RAGService not initialized or found in application state.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is not available at the moment. Please try again later."
        )
    return rag_service

def get_app_settings(request: Request) -> AppSettings:
    app_settings = request.app.state.settings
    if not app_settings:
        logger.error("AppSettings not initialized or found in application state.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Application settings are not available."
        )
    return app_settings


@router.post(
    "/ingest",
    response_model=IngestionResponse,
    summary="Ingest documents into the RAG system",
    description=(
        "Loads documents from the specified source directory, processes them "
        "into chunks, generates embeddings, and stores them in the vector store."
    ),
    status_code=status.HTTP_201_CREATED,
)
async def ingest_documents_endpoint(
    request_data: Annotated[IngestionRequest, Body(
        examples=[
            {
                "summary": "Basic Ingestion",
                "description": "Ingest all .txt files from a specified directory.",
                "value": {
                    "source_directory": "data/sample_documents",
                    "file_patterns": ["*.txt"],
                    "recursive": True,
                },
            },
            {
                "summary": "Ingestion with custom patterns",
                "description": "Ingest .md and .log files.",
                "value": {
                    "source_directory": "/path/to/my/docs",
                    "file_patterns": ["*.md", "*.log"],
                    "recursive": False,
                },
            }
        ]
    )],
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
    settings: Annotated[AppSettings, Depends(get_app_settings)],
) -> IngestionResponse:
    """
    Endpoint to trigger the document ingestion pipeline.
    """
    logger.info(
        f"Received document ingestion request: {request_data.model_dump_json(indent=2)}"
    )

    # Use defaults from settings if not provided in request
    if request_data.file_patterns is None:
        request_data.file_patterns = settings.DEFAULT_FILE_PATTERNS
    if request_data.recursive is None:
        request_data.recursive = settings.DEFAULT_RECURSIVE_INGESTION

    try:
        response = await rag_service.ingest_documents(request_data)
        if response.errors:
            # Partial success or some failures
            logger.warning(f"Ingestion completed with some errors: {response.errors}")
            # Decide on status code: 201 if some success, or 207 Multi-Status
            # For simplicity, stick to 201 if any chunks were added.
            if response.chunks_added == 0 and response.documents_processed == 0:
                 raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Ingestion failed. Errors: {'; '.join(response.errors)}"
                )

        logger.info(
            f"Ingestion successful: {response.documents_processed} documents processed, "
            f"{response.chunks_added} chunks added."
        )
        return response
    except DocumentProcessingError as e:
        logger.error(f"Document processing error during ingestion: {e.message}", exc_info=True)
        raise HTTPException(status_code=e.status_code, detail=f"{e.message}: {e.detail}")
    except VectorStoreError as e:
        logger.error(f"Vector store error during ingestion: {e.message}", exc_info=True)
        raise HTTPException(status_code=e.status_code, detail=f"{e.message}: {e.detail}")
    except FileNotFoundError as e: # From DocumentLoader
        logger.error(f"File not found error during ingestion: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during document ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during document ingestion: {str(e)}",
        )

# Example of another document-related endpoint (e.g., list ingested sources)
# @router.get("/sources", summary="List ingested document sources")
# async def list_document_sources(
#     rag_service: Annotated[RAGService, Depends(get_rag_service)]
# ):
#     # This would require RAGService to have a method to query distinct sources
#     # from the vector store's metadata.
#     try:
#         sources = await rag_service.list_ingested_sources() # Hypothetical method
#         return {"sources": sources}
#     except Exception as e:
#         logger.error(f"Error listing document sources: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Failed to list document sources.")
