import os
from typing import Optional, List # Added List

class Config:
    # ... (other LLM and Pinecone keys as before) ...
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY: Optional[str] = os.getenv('ANTHROPIC_API_KEY')
    GOOGLE_API_KEY: Optional[str] = os.getenv('GOOGLE_API_KEY') # From your .env
    DEFAULT_LLM_PROVIDER: str = os.getenv('DEFAULT_LLM_PROVIDER', 'google') # Changed default

    PINECONE_API_KEY: Optional[str] = os.getenv('PINECONE_API_KEY') # From your .env
    PINECONE_INDEX_NAME: Optional[str] = os.getenv('PINECONE_INDEX_NAME', 'rag-jules') # From your .env
    PINECONE_ENVIRONMENT: Optional[str] = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1') # From your .env

    # This is the "Backend Access Key" the frontend will send.
    # The backend checks against the value set in Netlify's environment variables.
    # Let's use 'API_KEY' as the env var name, matching your .env example for a single key.
    BACKEND_ACCESS_KEY: Optional[str] = os.getenv('API_KEY') # This is what auth.py will check against

    # You can also keep the API_KEYS for multi-client setup if needed,
    # but for simplicity, BACKEND_ACCESS_KEY is clearer for a single shared key.
    API_KEYS_JSON: Optional[str] = os.getenv('API_KEYS_JSON') # For multiple client keys in JSON format

    SECRET_KEY: Optional[str] = os.getenv('SECRET_KEY', 'default-fallback-secret-for-dev') # Used by auth.py if generating keys

    # ... (other RAG, Rate Limiting, Caching params) ...
    TOP_K_RESULTS: int = int(os.getenv('TOP_K_RESULTS', '3'))
    SIMILARITY_THRESHOLD: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
    CHUNK_SIZE: int = int(os.getenv('CHUNK_SIZE', '1000'))
    CHUNK_OVERLAP: int = int(os.getenv('CHUNK_OVERLAP', '200'))

    ENABLE_RATE_LIMITING: bool = os.getenv('ENABLE_RATE_LIMITING', 'true').lower() == 'true'
    DEFAULT_RATE_LIMIT_REQUESTS: int = int(os.getenv('DEFAULT_RATE_LIMIT_REQUESTS', '50'))
    DEFAULT_RATE_LIMIT_WINDOW: int = int(os.getenv('DEFAULT_RATE_LIMIT_WINDOW', '3600'))

    ENABLE_CACHING: bool = os.getenv('ENABLE_CACHING', 'true').lower() == 'true'
    CACHE_TTL: int = int(os.getenv('CACHE_TTL', '3600'))
    ENABLE_SEMANTIC_CACHE: bool = os.getenv('ENABLE_SEMANTIC_CACHE', 'false').lower() == 'true'
    SEMANTIC_CACHE_THRESHOLD: float = float(os.getenv('SEMANTIC_CACHE_THRESHOLD', '0.9'))

    MAX_FILE_SIZE: int = int(os.getenv('MAX_FILE_SIZE', '10485760'))  # 10MB
    ALLOWED_FILE_TYPES: List[str] = os.getenv('ALLOWED_FILE_TYPES', 'txt,pdf,docx').split(',')


config = Config()