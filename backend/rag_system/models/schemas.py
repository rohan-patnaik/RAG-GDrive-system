# backend/rag_system/models/schemas.py
import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, ConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    LOCAL = "local"  # For future local model support


class StatusEnum(str, Enum):
    """System component status levels."""
    OK = "OK"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


# =============================================================================
# DOCUMENT MODELS
# =============================================================================

class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    model_config = ConfigDict(extra="allow")
    
    source_id: str = Field(..., description="Unique identifier for the document source")
    filename: Optional[str] = Field(None, description="Original filename")
    path: Optional[str] = Field(None, description="File path or URL")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    created_at: Optional[datetime.datetime] = Field(None, description="Creation timestamp")
    modified_at: Optional[datetime.datetime] = Field(None, description="Last modification timestamp")
    content_type: Optional[str] = Field(None, description="MIME type or content type")
    
    # Chunk-specific metadata (when used for DocumentChunk)
    chunk_number: Optional[int] = Field(None, ge=0, description="Chunk index within document")
    total_chunks: Optional[int] = Field(None, ge=1, description="Total chunks in document")
    
    # Custom fields (flattened for vector store compatibility)
    custom_fields: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional custom metadata")

    @field_validator('path')
    @classmethod
    def validate_path(cls, v: Optional[str]) -> Optional[str]:
        if v and not v.startswith(('http://', 'https://', 'ftp://')):
            # It's a file path, validate it exists or is reasonable
            try:
                path_obj = Path(v)
                return str(path_obj.resolve()) if path_obj.exists() else v
            except (OSError, ValueError):
                # Invalid path, but don't fail - just return as-is
                return v
        return v


class Document(BaseModel):
    """A document with content and metadata."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "doc_123",
                "content": "This is the document content...",
                "metadata": {
                    "source_id": "file_abc.txt",
                    "filename": "abc.txt",
                    "path": "/documents/abc.txt"
                }
            }
        }
    )
    
    id: str = Field(..., min_length=1, description="Unique document identifier")
    content: str = Field(..., min_length=1, description="Document text content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")


class DocumentChunk(BaseModel):
    """A chunk of a document with embeddings."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "doc_123_chunk_0",
                "document_id": "doc_123",
                "content": "This is a chunk of the document...",
                "metadata": {
                    "source_id": "file_abc.txt",
                    "filename": "abc.txt", 
                    "chunk_number": 0,
                    "total_chunks": 5
                },
                "embedding": [0.1, 0.2, 0.3]
            }
        }
    )
    
    id: str = Field(..., min_length=1, description="Unique chunk identifier")
    document_id: str = Field(..., min_length=1, description="Parent document ID")
    content: str = Field(..., min_length=1, description="Chunk text content")
    metadata: DocumentMetadata = Field(..., description="Chunk metadata (inherits from document)")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding for the chunk")

    @field_validator('embedding')
    @classmethod
    def validate_embedding(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None:
            if len(v) == 0:
                raise ValueError("Embedding cannot be empty if provided")
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("Embedding must contain only numeric values")
        return v


class RetrievedChunk(BaseModel):
    """Schema for retrieved document chunks with similarity scores."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    id: str = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0.0 to 1.0)")
    
    @field_validator('score')
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Ensure score is in valid range."""
        if v < 0.0:
            # Log warning but clamp the value instead of failing
            import logging
            logging.getLogger(__name__).warning(f"Similarity score {v} is negative, clamping to 0.0")
            return 0.0
        elif v > 1.0:
            import logging
            logging.getLogger(__name__).warning(f"Similarity score {v} is > 1.0, clamping to 1.0")
            return 1.0
        return v

    
    id: str = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")


# =============================================================================
# API REQUEST/RESPONSE MODELS
# =============================================================================

class IngestionRequest(BaseModel):
    """Request model for document ingestion."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source_directory": "data/documents",
                "file_patterns": ["*.txt", "*.md"],
                "recursive": True
            }
        }
    )
    
    source_directory: str = Field(..., min_length=1, description="Path to directory containing documents")
    file_patterns: Optional[List[str]] = Field(None, description="File patterns to match (e.g., ['*.txt', '*.pdf'])")
    recursive: Optional[bool] = Field(None, description="Search subdirectories recursively")

    @field_validator('source_directory')
    @classmethod
    def validate_source_directory(cls, v: str) -> str:
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Source directory does not exist: {v}")
        if not path.is_dir():
            raise ValueError(f"Source path is not a directory: {v}")
        return str(path.resolve())


class IngestionResponse(BaseModel):
    """Response model for document ingestion."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Ingestion completed successfully",
                "documents_processed": 25,
                "chunks_added": 150,
                "errors": []
            }
        }
    )
    
    message: str = Field(..., description="Human-readable status message")
    documents_processed: int = Field(..., ge=0, description="Number of documents processed")
    chunks_added: int = Field(..., ge=0, description="Number of chunks added to vector store")
    errors: List[str] = Field(default_factory=list, description="List of error messages")


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query_text": "What is machine learning?",
                "llm_provider": "gemini",
                "llm_model_name": "gemini-1.5-pro-latest",
                "top_k": 3,
                "similarity_threshold": 0.7
            }
        }
    )
    
    query_text: str = Field(..., min_length=1, max_length=2000, description="User's natural language query")
    llm_provider: Optional[LLMProvider] = Field(None, description="LLM provider to use")
    llm_model_name: Optional[str] = Field(None, description="Specific model name (overrides provider default)")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity threshold")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query_text": "What is machine learning?",
                "llm_answer": "Machine learning is a subset of artificial intelligence...",
                "llm_provider_used": "gemini",
                "llm_model_used": "gemini-1.5-flash-latest",
                "retrieved_chunks": [
                    {
                        "id": "chunk_1",
                        "content": "Machine learning involves...",
                        "metadata": {"source_id": "ml_guide.txt"},
                        "score": 0.92
                    }
                ]
            }
        }
    )
    
    query_text: str = Field(..., description="Original user query")
    llm_answer: str = Field(..., description="Generated answer from LLM")
    llm_provider_used: LLMProvider = Field(..., description="LLM provider that generated the answer")
    llm_model_used: str = Field(..., description="Specific model used for generation")
    retrieved_chunks: List[RetrievedChunk] = Field(..., description="Chunks retrieved for context")


# =============================================================================
# SYSTEM STATUS MODELS
# =============================================================================

class ComponentStatus(BaseModel):
    """Status information for a system component."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "VectorStore",
                "status": "OK",
                "message": "Connected to ChromaDB",
                "details": {"collection_size": 1500}
            }
        }
    )
    
    name: str = Field(..., description="Component name")
    status: StatusEnum = Field(..., description="Component status")
    message: Optional[str] = Field(None, description="Status message or error description")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional status details")


class SystemStatusResponse(BaseModel):
    """System-wide status response."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "system_status": "OK",
                "timestamp": "2025-01-07T10:30:00Z",
                "app_name": "RAG GDrive System",
                "environment": "production",
                "version": "0.1.0",
                "components": [
                    {
                        "name": "EmbeddingService",
                        "status": "OK",
                        "message": "Model loaded successfully"
                    }
                ]
            }
        }
    )
    
    system_status: StatusEnum = Field(..., description="Overall system status")
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc), description="Status check timestamp")
    app_name: Optional[str] = Field(None, description="Application name")
    environment: Optional[str] = Field(None, description="Environment (dev, staging, prod)")
    version: Optional[str] = Field(None, description="Application version")
    components: List[ComponentStatus] = Field(default_factory=list, description="Individual component statuses")


# Alias for backward compatibility and API docs
HealthResponse = SystemStatusResponse


# =============================================================================
# ERROR RESPONSE MODELS
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response model."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "ValidationError",
                "message": "Invalid input provided",
                "detail": "Field 'query_text' is required",
                "timestamp": "2025-01-07T10:30:00Z"
            }
        }
    )
    
    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc), description="Error timestamp")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_document_metadata(
    source_id: str,
    filename: Optional[str] = None,
    **kwargs: Any
) -> DocumentMetadata:
    """Helper function to create DocumentMetadata with common fields."""
    return DocumentMetadata(
        source_id=source_id,
        filename=filename,
        created_at=datetime.datetime.now(datetime.timezone.utc),
        **kwargs
    )


def create_chunk_from_document(
    document: Document,
    chunk_content: str,
    chunk_number: int,
    total_chunks: int,
    chunk_id_suffix: Optional[str] = None
) -> DocumentChunk:
    """Helper function to create a DocumentChunk from a Document."""
    chunk_id = f"{document.id}_chunk_{chunk_number}"
    if chunk_id_suffix:
        chunk_id += f"_{chunk_id_suffix}"
    
    # Copy document metadata and add chunk-specific fields
    chunk_metadata = DocumentMetadata(
        **document.metadata.model_dump(),
        chunk_number=chunk_number,
        total_chunks=total_chunks
    )
    
    return DocumentChunk(
        id=chunk_id,
        document_id=document.id,
        content=chunk_content,
        metadata=chunk_metadata
    )
