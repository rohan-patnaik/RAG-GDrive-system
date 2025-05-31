# backend/rag_system/api/routes/health.py
import logging
from typing import Annotated
import datetime

from fastapi import APIRouter, Depends, Request, status

from rag_system.models.schemas import HealthResponse, StatusEnum, ComponentStatus
from rag_system.services.rag_service import RAGService # To check its dependencies
from rag_system.config.settings import AppSettings, get_settings
from rag_system import __version__ as app_version

logger = logging.getLogger(__name__)
router = APIRouter()


# Dependency to get RAGService instance from application state
def get_rag_service_optional(request: Request) -> RAGService | None:
    # This allows the health check to function even if RAGService failed to initialize
    return getattr(request.app.state, 'rag_service', None)

def get_app_settings(request: Request) -> AppSettings:
    app_settings = getattr(request.app.state, 'settings', None)
    if not app_settings:
        # Fallback if not in state (e.g., during early startup failure)
        logger.warning("AppSettings not found in request.app.state, loading directly for health check.")
        return get_settings()
    return app_settings


@router.get(
    "", # Route path is just "/health" due to router prefix
    response_model=HealthResponse,
    summary="System Health Check",
    description="Provides the operational status of the RAG system and its core components.",
)
async def health_check(
    request: Request, # Access startup error if any
    rag_service: Annotated[RAGService | None, Depends(get_rag_service_optional)],
    settings: Annotated[AppSettings, Depends(get_app_settings)],
) -> HealthResponse:
    """
    Performs a health check of the RAG system.
    Checks connectivity to essential services like the vector store and configured LLMs.
    """
    logger.debug("Health check endpoint called.")
    overall_status = StatusEnum.OK
    components: List[ComponentStatus] = []

    # Check for startup errors
    startup_error = getattr(request.app.state, 'startup_error', None)
    if startup_error:
        logger.error(f"Health check: Startup error detected: {startup_error}")
        overall_status = StatusEnum.ERROR
        components.append(ComponentStatus(
            name="Application Startup",
            status=StatusEnum.ERROR,
            message=f"Application failed to start critical services: {startup_error}"
        ))
        # If startup failed critically, other checks might not be meaningful or possible
        return HealthResponse(
            system_status=overall_status,
            app_name=settings.APP_NAME,
            environment=settings.ENVIRONMENT,
            version=app_version,
            components=components,
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )


    if rag_service:
        try:
            status_report = await rag_service.get_system_status()
            components.extend(status_report.components)
            if status_report.system_status != StatusEnum.OK:
                overall_status = StatusEnum.DEGRADED # Or ERROR if any component is ERROR
                for comp in status_report.components:
                    if comp.status == StatusEnum.ERROR:
                        overall_status = StatusEnum.ERROR
                        break
        except Exception as e:
            logger.error(f"Error getting system status from RAGService: {e}", exc_info=True)
            overall_status = StatusEnum.ERROR
            components.append(ComponentStatus(
                name="RAGService Internal Status",
                status=StatusEnum.ERROR,
                message=f"Failed to retrieve detailed status: {str(e)}"
            ))
    else:
        # This case should ideally be caught by startup_error if RAGService is critical
        logger.warning("Health check: RAGService not available.")
        overall_status = StatusEnum.ERROR
        components.append(ComponentStatus(
            name="RAGService Initialization",
            status=StatusEnum.ERROR,
            message="RAGService is not initialized or available."
        ))

    # You can add more specific checks here if not covered by RAGService.get_system_status()
    # For example, a simple database ping if RAGService doesn't cover it deeply enough.

    # Determine final overall status based on components
    if overall_status == StatusEnum.OK: # Re-evaluate if not already ERROR
        for comp in components:
            if comp.status == StatusEnum.ERROR:
                overall_status = StatusEnum.ERROR
                break
            if comp.status == StatusEnum.DEGRADED:
                overall_status = StatusEnum.DEGRADED # Keep degraded if no errors

    response = HealthResponse(
        system_status=overall_status,
        app_name=settings.APP_NAME,
        environment=settings.ENVIRONMENT,
        version=app_version,
        components=components,
        timestamp=datetime.datetime.now(datetime.timezone.utc)
    )

    logger.info(f"Health check completed. Overall status: {response.system_status}")
    if response.system_status != StatusEnum.OK:
        # Return 503 if not fully healthy, as per common practice for load balancers
        # However, FastAPI will use 200 OK by default unless an HTTPException is raised.
        # For now, we return 200 with the status in the body.
        # If you need a non-200 status for unhealthy, you'd raise HTTPException here.
        pass

    return response
