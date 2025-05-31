# backend/rag_system/config/settings.py
import os
import json
from pathlib import Path
from typing import List, Optional, Any, Dict, Union
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr, field_validator, computed_field
import structlog

logger = structlog.get_logger(__name__)


def parse_list_from_env(value: Union[str, List[str]]) -> List[str]:
    """Parse a list from environment variable."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        # Try to parse as JSON first
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: split by comma and clean up
        return [item.strip().strip('"\'') for item in value.split(',') if item.strip()]
    
    return []


class AppSettings(BaseSettings):
    """
    Production-grade application settings with comprehensive validation.
    
    Loads configuration from environment variables and .env files.
    Includes security, performance, and operational settings.
    """
    
    model_config = SettingsConfigDict(
        env_file=os.getenv("RAG_CONFIG_FILE", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        validate_assignment=True,
    )

    # =============================================================================
    # APPLICATION SETTINGS
    # =============================================================================
    APP_NAME: str = Field(default="RAG GDrive System", description="Application name")
    ENVIRONMENT: str = Field(default="development", description="Environment: development, staging, production")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FILE_PATH: str = Field(default="logs/rag_system.log", description="Log file path")

    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, ge=1024, le=65535, description="API port")
    API_RELOAD: bool = Field(default=False, description="Enable API auto-reload (development only)")

    # =============================================================================
    # SECURITY SETTINGS
    # =============================================================================
    SECRET_KEY: SecretStr = Field(default="change-me-in-production", description="Secret key for security")
    ALLOWED_HOSTS: List[str] = Field(default=["localhost", "127.0.0.1"], description="Allowed hosts")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, description="Enable rate limiting")
    RATE_LIMIT_REQUESTS: int = Field(default=100, ge=1, description="Requests per window")
    RATE_LIMIT_WINDOW: int = Field(default=3600, ge=60, description="Rate limit window in seconds")

    # =============================================================================
    # LLM API KEYS
    # =============================================================================
    OPENAI_API_KEY: Optional[SecretStr] = Field(default=None, description="OpenAI API key")
    ANTHROPIC_API_KEY: Optional[SecretStr] = Field(default=None, description="Anthropic API key")
    GOOGLE_API_KEY: Optional[SecretStr] = Field(default=None, description="Google API key")

    # LLM Configuration
    DEFAULT_LLM_PROVIDER: str = Field(default="gemini", description="Default LLM provider")
    DEFAULT_OPENAI_MODEL: str = Field(default="gpt-3.5-turbo", description="Default OpenAI model")
    DEFAULT_ANTHROPIC_MODEL: str = Field(default="claude-3-haiku-20240307", description="Default Anthropic model")
    DEFAULT_GEMINI_MODEL: str = Field(default="gemini-1.5-flash-latest", description="Default Gemini model")

    # =============================================================================
    # VECTOR STORE CONFIGURATION
    # =============================================================================
    VECTOR_STORE_PROVIDER: str = Field(default="chromadb", description="Vector store provider: 'chromadb' or 'pinecone'")
    VECTOR_STORE_PATH: str = Field(default="./data/vector_store", description="Vector store data path (for ChromaDB)")
    CHROMA_COLLECTION_NAME: str = Field(default="rag_documents_v1", description="ChromaDB collection name")

    # Pinecone Configuration (optional, only if VECTOR_STORE_PROVIDER is 'pinecone')
    PINECONE_API_KEY: Optional[SecretStr] = Field(default=None, description="Pinecone API key")
    PINECONE_ENVIRONMENT: Optional[str] = Field(default=None, description="Pinecone environment (e.g., 'us-west1-gcp')")
    PINECONE_INDEX_NAME: Optional[str] = Field(default=None, description="Pinecone index name")

    # Embedding Configuration
    EMBEDDING_MODEL_NAME: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", 
        description="Sentence transformer model name"
    )
    EMBEDDING_MODEL_DEVICE: str = Field(default="cpu", description="Device for embedding model")

    # =============================================================================
    # RAG PARAMETERS
    # =============================================================================
    CHUNK_SIZE: int = Field(default=1000, ge=100, le=8000, description="Text chunk size")
    CHUNK_OVERLAP: int = Field(default=200, ge=0, le=1000, description="Text chunk overlap")
    TOP_K_RESULTS: int = Field(default=3, ge=1, le=20, description="Number of retrieved chunks")
    SIMILARITY_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")

    # Document Processing (with custom parsing)
    DEFAULT_FILE_PATTERNS: List[str] = Field(default=["*.txt", "*.md"], description="Default file patterns")
    DEFAULT_RECURSIVE_INGESTION: bool = Field(default=True, description="Default recursive ingestion")

    # =============================================================================
    # PERFORMANCE SETTINGS
    # =============================================================================
    DB_POOL_SIZE: int = Field(default=10, ge=1, le=100, description="Database connection pool size")
    DB_MAX_OVERFLOW: int = Field(default=20, ge=0, le=100, description="Database max overflow connections")
    DB_POOL_TIMEOUT: int = Field(default=30, ge=5, le=300, description="Database pool timeout")

    HTTP_TIMEOUT: int = Field(default=60, ge=5, le=300, description="HTTP client timeout")
    HTTP_MAX_CONNECTIONS: int = Field(default=100, ge=10, le=1000, description="HTTP max connections")
    HTTP_MAX_KEEPALIVE: int = Field(default=20, ge=5, le=100, description="HTTP max keepalive connections")

    # =============================================================================
    # MONITORING & OBSERVABILITY
    # =============================================================================
    PROMETHEUS_ENABLED: bool = Field(default=False, description="Enable Prometheus metrics")
    PROMETHEUS_PORT: int = Field(default=9090, ge=1024, le=65535, description="Prometheus metrics port")
    HEALTH_CHECK_TIMEOUT: int = Field(default=30, ge=5, le=120, description="Health check timeout")

    # =============================================================================
    # TESTING CONFIGURATION
    # =============================================================================
    RAG_SYSTEM_TESTING_MODE: bool = Field(default=False, description="Enable testing mode")

    # =============================================================================
    # FIELD VALIDATORS (Pydantic v2 style)
    # =============================================================================
    
    @field_validator('ENVIRONMENT')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        # Clean up the value first (remove quotes and comments)
        v = v.strip().strip('"\'').split('#')[0].strip()
        valid_envs = {'development', 'staging', 'production'}
        if v.lower() not in valid_envs:
            raise ValueError(f"ENVIRONMENT must be one of: {valid_envs}")
        return v.lower()

    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        # Clean up the value first
        v = v.strip().strip('"\'').split('#')[0].strip()
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of: {valid_levels}")
        return v.upper()

    @field_validator('DEFAULT_LLM_PROVIDER')
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        # Clean up the value first
        v = v.strip().strip('"\'').split('#')[0].strip()
        valid_providers = {'openai', 'anthropic', 'gemini'}
        if v.lower() not in valid_providers:
            raise ValueError(f"DEFAULT_LLM_PROVIDER must be one of: {valid_providers}")
        return v.lower()

    @field_validator('VECTOR_STORE_PROVIDER')
    @classmethod
    def validate_vector_store_provider(cls, v: str) -> str:
        # Clean up the value first
        v = v.strip().strip('"\'').split('#')[0].strip()
        valid_providers = {'chromadb', 'pinecone'}
        if v.lower() not in valid_providers:
            raise ValueError(f"VECTOR_STORE_PROVIDER must be one of: {valid_providers}")
        return v.lower()

    @field_validator('EMBEDDING_MODEL_DEVICE')
    @classmethod
    def validate_embedding_device(cls, v: str) -> str:
        # Clean up the value first
        v = v.strip().strip('"\'').split('#')[0].strip()
        valid_devices = {'cpu', 'cuda', 'mps'}
        if v.lower() not in valid_devices:
            raise ValueError(f"EMBEDDING_MODEL_DEVICE must be one of: {valid_devices}")
        return v.lower()

    @field_validator('DEFAULT_FILE_PATTERNS', mode='before')
    @classmethod
    def validate_file_patterns(cls, v) -> List[str]:
        return parse_list_from_env(v)

    @field_validator('ALLOWED_HOSTS', mode='before')
    @classmethod  
    def validate_allowed_hosts(cls, v) -> List[str]:
        return parse_list_from_env(v)

    @field_validator('VECTOR_STORE_PATH')
    @classmethod
    def validate_vector_store_path(cls, v: str) -> str:
        path = Path(v).resolve()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValueError(f"Cannot create vector store directory {path.parent}: {e}")
        return str(path)

    @field_validator('LOG_FILE_PATH')
    @classmethod
    def validate_log_file_path(cls, v: str) -> str:
        path = Path(v).resolve()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValueError(f"Cannot create log directory {path.parent}: {e}")
        return str(path)

    @field_validator('CHUNK_OVERLAP')
    @classmethod
    def validate_chunk_overlap(cls, v: int) -> int:
        if v < 0:
            raise ValueError("CHUNK_OVERLAP must be non-negative")
        return v

    @field_validator('SECRET_KEY')
    @classmethod
    def validate_secret_key(cls, v: SecretStr) -> SecretStr:
        secret = v.get_secret_value()
        if len(secret) < 8:  # Minimum reasonable length
            raise ValueError("SECRET_KEY must be at least 8 characters")
        return v

    # =============================================================================
    # COMPUTED PROPERTIES
    # =============================================================================
    
    @computed_field
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == 'development'

    @computed_field
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == 'production'

    @computed_field
    @property
    def configured_llm_providers(self) -> List[str]:
        providers = []
        if self.OPENAI_API_KEY:
            providers.append('openai')
        if self.ANTHROPIC_API_KEY:
            providers.append('anthropic')
        if self.GOOGLE_API_KEY:
            providers.append('gemini')
        return providers

    def get_api_key(self, provider: str) -> Optional[str]:
        """Safely get API key for a provider."""
        key_map = {
            'openai': self.OPENAI_API_KEY,
            'anthropic': self.ANTHROPIC_API_KEY,
            'gemini': self.GOOGLE_API_KEY,
        }
        key = key_map.get(provider.lower())
        return key.get_secret_value() if key else None

    def model_dump_safe(self) -> Dict[str, Any]:
        """Return model data with secrets masked."""
        data = self.model_dump()
        for key in data:
            if any(sensitive in key.lower() for sensitive in ['key', 'secret', 'password']):
                if data[key]:
                    data[key] = "***MASKED***"
        return data


@lru_cache()
def get_settings() -> AppSettings:
    """Get cached application settings."""
    try:
        settings = AppSettings()
        logger.info(
            "Settings loaded successfully",
            environment=settings.ENVIRONMENT,
            app_name=settings.APP_NAME,
            configured_providers=settings.configured_llm_providers,
            vector_store_provider=settings.VECTOR_STORE_PROVIDER,
        )
        if settings.VECTOR_STORE_PROVIDER == 'pinecone':
            if settings.PINECONE_INDEX_NAME:
                logger.info("Pinecone index name", index_name=settings.PINECONE_INDEX_NAME)
            else:
                logger.warning("Pinecone is selected as vector store, but PINECONE_INDEX_NAME is not set.")
        return settings
    except Exception as e:
        logger.error("Failed to load settings", error=str(e))
        raise


def clear_settings_cache() -> None:
    """Clear the settings cache. Useful for testing."""
    get_settings.cache_clear()
