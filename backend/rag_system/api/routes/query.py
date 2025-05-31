# backend/rag_system/api/routes/query.py
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Body, Request, status

from rag_system.services.rag_service import RAGService
from rag_system.models.schemas import (
    QueryRequest,
    QueryResponse,
    SystemStatusResponse,
    LLMProvider,
)
from rag_system.utils.exceptions import (
    QueryProcessingError,
    LLMError,
    VectorStoreError,
)
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
    "/",
    response_model=QueryResponse,
    summary="Query the RAG system",
    description=(
        "Submits a natural language query. The system retrieves relevant document "
        "chunks, augments the query, and generates an answer using the selected LLM."
    ),
)
async def query_rag_system_endpoint(
    request_data: Annotated[QueryRequest, Body(
        examples=[
            {
                "summary": "Basic Query",
                "description": "Query with default LLM provider and parameters.",
                "value": {"query_text": "What are the main applications of AI?"},
            },
            {
                "summary": "Query with specific LLM",
                "description": "Query using Anthropic Claude and custom top_k.",
                "value": {
                    "query_text": "Explain the concept of zero-shot learning.",
                    "llm_provider": "anthropic",
                    "top_k": 5,
                },
            },
            {
                "summary": "Query with Gemini and specific model",
                "description": "Query using Google Gemini with a specific model.",
                "value": {
                    "query_text": "Summarize the latest advancements in renewable energy.",
                    "llm_provider": "gemini",
                    "llm_model_name": "gemini-2.5-flash-preview-05-20",
                    "top_k": 3,
                },
            },
        ]
    )],
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
    settings: Annotated[AppSettings, Depends(get_app_settings)],
) -> QueryResponse:
    """
    Endpoint to ask a question to the RAG system.
    """
    logger.info(f"Received query request: {request_data.model_dump_json(indent=2)}")

    # Apply defaults from settings if not provided in request
    if request_data.llm_provider is None:
        request_data.llm_provider = LLMProvider(settings.DEFAULT_LLM_PROVIDER) # Cast to enum
    if request_data.top_k is None:
        request_data.top_k = settings.TOP_K_RESULTS
    if request_data.similarity_threshold is None:
        request_data.similarity_threshold = settings.SIMILARITY_THRESHOLD
    # Note: llm_model_name defaults are handled within LLMService based on provider

    try:
        response = await rag_service.query(request_data)
        logger.info(
            f"Query processed successfully. LLM Answer: '{response.llm_answer[:100]}...'"
        )
        return response
    except (QueryProcessingError, LLMError, VectorStoreError) as e:
        logger.error(f"Error processing query: {e.message}", exc_info=False) # exc_info=False if detail is enough
        raise HTTPException(status_code=e.status_code, detail=f"{e.message}: {e.detail}")
    except ValueError as e: # e.g. invalid LLMProvider string
        logger.error(f"Validation error in query request: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during query processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while processing your query: {str(e)}",
        )


@router.get(
    "/status",
    response_model=SystemStatusResponse,
    summary="Get RAG Query System Status",
    description="Provides the operational status of query-related components like LLMs and Vector Store.",
)
async def get_query_system_status(
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
    settings: Annotated[AppSettings, Depends(get_app_settings)],
) -> SystemStatusResponse:
    """
    Endpoint to get the status of the RAG query system components.
    This is similar to the main /health endpoint but can be focused on query pipeline readiness.
    """
    logger.info("Received request for query system status.")
    try:
        # The RAGService's get_system_status can be reused or a more specific one created.
        status_response = await rag_service.get_system_status()
        # Augment with app-specific info if needed, though HealthResponse already has some
        status_response.app_name = settings.APP_NAME
        status_response.environment = settings.ENVIRONMENT
        # status_response.version = "0.1.0" # Or get from settings/package
        logger.info(f"Query system status: {status_response.system_status}")
        return status_response
    except Exception as e:
        logger.error(f"Failed to retrieve query system status: {e}", exc_info=True)
        # Return a degraded status if the status check itself fails
        return SystemStatusResponse(
            system_status="ERROR",
            app_name=settings.APP_NAME,
            environment=settings.ENVIRONMENT,
            components=[{"name": "StatusCheck", "status": "ERROR", "message": str(e)}]
        )
