import os
from typing import Optional

class Config:
    """Enhanced configuration for serverless functions"""
    
    # Pinecone Configuration
    PINECONE_API_KEY: Optional[str] = os.getenv('PINECONE_API_KEY')
    PINECONE_INDEX_NAME: Optional[str] = os.getenv('PINECONE_INDEX_NAME', 'rag-documents')
    PINECONE_ENVIRONMENT: Optional[str] = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
    
    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY: Optional[str] = os.getenv('ANTHROPIC_API_KEY')
    GOOGLE_API_KEY: Optional[str] = os.getenv('GOOGLE_API_KEY')
    DEFAULT_LLM_PROVIDER: str = os.getenv('DEFAULT_LLM_PROVIDER', 'openai')
    
    # RAG Parameters
    TOP_K_RESULTS: int = int(os.getenv('TOP_K_RESULTS', '3'))
    SIMILARITY_THRESHOLD: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
    
    # Authentication
    SECRET_KEY: Optional[str] = os.getenv('SECRET_KEY')
    API_KEY: Optional[str] = os.getenv('API_KEY')  # Single API key fallback
    API_KEYS: Optional[str] = os.getenv('API_KEYS')  # JSON string of multiple keys
    
    # Rate Limiting
    ENABLE_RATE_LIMITING: bool = os.getenv('ENABLE_RATE_LIMITING', 'true').lower() == 'true'
    DEFAULT_RATE_LIMIT_REQUESTS: int = int(os.getenv('DEFAULT_RATE_LIMIT_REQUESTS', '50'))
    DEFAULT_RATE_LIMIT_WINDOW: int = int(os.getenv('DEFAULT_RATE_LIMIT_WINDOW', '3600'))
    
    # Caching
    ENABLE_CACHING: bool = os.getenv('ENABLE_CACHING', 'true').lower() == 'true'
    CACHE_TTL: int = int(os.getenv('CACHE_TTL', '3600'))
    ENABLE_SEMANTIC_CACHE: bool = os.getenv('ENABLE_SEMANTIC_CACHE', 'false').lower() == 'true'
    SEMANTIC_CACHE_THRESHOLD: float = float(os.getenv('SEMANTIC_CACHE_THRESHOLD', '0.9'))
    
    # File Processing
    MAX_FILE_SIZE: int = int(os.getenv('MAX_FILE_SIZE', '10485760'))  # 10MB
    ALLOWED_FILE_TYPES: list = os.getenv('ALLOWED_FILE_TYPES', 'txt,pdf,docx').split(',')
    CHUNK_SIZE: int = int(os.getenv('CHUNK_SIZE', '1000'))
    CHUNK_OVERLAP: int = int(os.getenv('CHUNK_OVERLAP', '200'))

config = Config()