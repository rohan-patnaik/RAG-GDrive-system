# backend/rag_system/api/app.py
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import structlog

from rag_system.config.settings import get_settings, AppSettings
from rag_system.config.logging_config import setup_logging
from rag_system.api.routes import health, documents, query
from rag_system.api.middleware import LoggingMiddleware
from rag_system.services.rag_service import RAGService
from rag_system.core.embeddings import EmbeddingService
from rag_system.core.vector_store import VectorStoreService
from rag_system.services.llm_service import LLMService
from rag_system.utils.exceptions import BaseRAGError
from rag_system import __version__

# Initialize settings first
settings = get_settings()

# Setup logging with correct parameter names
logger = setup_logging(
    log_level=settings.LOG_LEVEL,
    log_file=settings.LOG_FILE_PATH,
    enable_json=settings.is_production,
    enable_colors=settings.is_development
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan event handler for startup and shutdown.
    Initializes core services on startup and cleans up on shutdown.
    """
    logger.info("Starting RAG System API", version=__version__, environment=settings.ENVIRONMENT)
    
    # Check if we're in testing mode
    if settings.RAG_SYSTEM_TESTING_MODE:
        logger.info("Running in testing mode - skipping service initialization")
        app.state.settings = settings
        app.state.startup_error = None
        yield
        return

    startup_error = None
    
    try:
        # Initialize core services
        logger.info("Initializing core services...")
        
        # 1. Initialize Embedding Service
        logger.info("Loading embedding model...")
        embedding_service = EmbeddingService(settings=settings)
        app.state.embedding_service = embedding_service
        logger.info("Embedding service initialized", model=settings.EMBEDDING_MODEL_NAME)
        
        # 2. Initialize Vector Store Service
        logger.info("Connecting to vector store...")
        vector_store_service = VectorStoreService(
            settings=settings, 
            embedding_service=embedding_service
        )
        app.state.vector_store_service = vector_store_service
        logger.info("Vector store service initialized", path=settings.VECTOR_STORE_PATH)
        
        # 3. Initialize LLM Service
        logger.info("Initializing LLM clients...")
        llm_service = LLMService(settings=settings)
        app.state.llm_service = llm_service
        logger.info("LLM service initialized", providers=settings.configured_llm_providers)
        
        # 4. Initialize RAG Service (orchestrator)
        logger.info("Setting up RAG orchestrator...")
        rag_service = RAGService(
            settings=settings,
            embedding_service=embedding_service,
            vector_store_service=vector_store_service,
            llm_service=llm_service,
        )
        app.state.rag_service = rag_service
        logger.info("RAG service initialized successfully")
        
        # Store settings
        app.state.settings = settings
        app.state.startup_error = None
        
        logger.info("All services initialized successfully", services_count=4)
        
    except Exception as e:
        error_msg = f"Failed to initialize services: {str(e)}"
        startup_error = error_msg
        logger.error("Service initialization failed", error=error_msg, exc_info=True)
        
        # Store error state for health checks
        app.state.startup_error = startup_error
        app.state.settings = settings
        
        # Create mock services to prevent AttributeError
        app.state.embedding_service = None
        app.state.vector_store_service = None
        app.state.llm_service = None
        app.state.rag_service = None
    
    # Yield control to the application
    yield
    
    # Cleanup (shutdown)
    logger.info("Shutting down RAG System API...")
    
    # Perform any necessary cleanup
    if hasattr(app.state, 'vector_store_service') and app.state.vector_store_service:
        try:
            # Close any connections if needed
            logger.info("Cleaning up vector store connections...")
            # Add cleanup code here if needed
        except Exception as e:
            logger.error("Error during vector store cleanup", error=str(e))
    
    logger.info("RAG System API shutdown complete")


def create_exception_handler(app: FastAPI) -> None:
    """Create custom exception handlers for the application."""
    
    @app.exception_handler(BaseRAGError)
    async def rag_exception_handler(request: Request, exc: BaseRAGError) -> JSONResponse:
        """Handle custom RAG system exceptions."""
        logger.error(
            "RAG system error",
            error_type=exc.error_type,
            message=exc.message,
            detail=exc.detail,
            path=request.url.path,
            method=request.method
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_type,
                "message": exc.message,
                "detail": exc.detail,
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions."""
        logger.error(
            "Unexpected error", 
            error=str(exc),
            path=request.url.path,
            method=request.method,
            exc_info=True
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "detail": str(exc) if settings.is_development else "Please contact support"
            }
        )


def create_app(app_settings: AppSettings = None) -> FastAPI:
    """
    FastAPI application factory.
    
    Args:
        app_settings: Optional settings override for testing
        
    Returns:
        Configured FastAPI application
    """
    # Use provided settings or global settings
    app_settings = app_settings or settings
    
    # Create FastAPI app
    app = FastAPI(
        title="RAG GDrive System API",
        description="Production-ready Retrieval-Augmented Generation system with multi-LLM support",
        version=__version__,
        docs_url="/docs" if app_settings.is_development else None,
        redoc_url="/redoc" if app_settings.is_development else None,
        lifespan=lifespan,
    )
    
    # Add middleware
    
    # 1. Trusted Host Middleware (security)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=app_settings.ALLOWED_HOSTS if app_settings.is_production else ["*"]
    )
    
    # 2. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if app_settings.is_development else app_settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # 3. Custom Logging Middleware
    app.add_middleware(LoggingMiddleware)
    
    # Add exception handlers
    create_exception_handler(app)
    
    # Include routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(documents.router, prefix="/documents", tags=["Documents"])
    app.include_router(query.router, prefix="/query", tags=["Query"])
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with basic API information."""
        return {
            "name": "RAG GDrive System API",
            "version": __version__,
            "environment": app_settings.ENVIRONMENT,
            "docs_url": "/docs" if app_settings.is_development else None,
            "health_url": "/health"
        }
    
    logger.info("FastAPI application created", environment=app_settings.ENVIRONMENT)
    return app


# Create the app instance
app = create_app()


# Dependency functions for route handlers
def get_rag_service(request: Request) -> RAGService:
    """Dependency to get RAGService from application state."""
    rag_service = getattr(request.app.state, 'rag_service', None)
    if not rag_service:
        startup_error = getattr(request.app.state, 'startup_error', None)
        if startup_error:
            raise BaseRAGError(
                message="RAG service is not available due to startup error",
                detail=startup_error,
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        else:
            raise BaseRAGError(
                message="RAG service is not initialized",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            )
    return rag_service


def get_app_settings(request: Request) -> AppSettings:
    """Dependency to get application settings."""
    return getattr(request.app.state, 'settings', settings)


# Make dependencies available for import
__all__ = ["app", "create_app", "get_rag_service", "get_app_settings"]
