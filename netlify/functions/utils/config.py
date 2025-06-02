import os
import hmac
import hashlib
from typing import Optional

class Config:
    """Enhanced configuration with proper API key management"""
    
    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY: Optional[str] = os.getenv('ANTHROPIC_API_KEY')
    GOOGLE_API_KEY: Optional[str] = os.getenv('GOOGLE_API_KEY')
    DEFAULT_LLM_PROVIDER: str = os.getenv('DEFAULT_LLM_PROVIDER', 'gemini')
    
    # Vector Store
    PINECONE_API_KEY: Optional[str] = os.getenv('PINECONE_API_KEY')
    PINECONE_ENVIRONMENT: str = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
    PINECONE_INDEX_NAME: str = os.getenv('PINECONE_INDEX_NAME', 'rag-jules')
    
    # Security
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'fallback-secret-change-me')
    SYSTEM_API_KEY: Optional[str] = os.getenv('SYSTEM_API_KEY')
    
    # RAG Parameters
    CHUNK_SIZE: int = int(os.getenv('CHUNK_SIZE', '1000'))
    CHUNK_OVERLAP: int = int(os.getenv('CHUNK_OVERLAP', '200'))
    TOP_K_RESULTS: int = int(os.getenv('TOP_K_RESULTS', '3'))
    SIMILARITY_THRESHOLD: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.4'))
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
    RATE_LIMIT_REQUESTS: int = int(os.getenv('RATE_LIMIT_REQUESTS', '100'))
    RATE_LIMIT_WINDOW: int = int(os.getenv('RATE_LIMIT_WINDOW', '3600'))
    
    @classmethod
    def generate_system_api_key(cls, client_id: str = "frontend") -> str:
        """Generate system API key from SECRET_KEY"""
        return hmac.new(
            cls.SECRET_KEY.encode(),
            client_id.encode(),
            hashlib.sha256
        ).hexdigest()
    
    @classmethod
    def verify_system_api_key(cls, provided_key: str, client_id: str = "frontend") -> bool:
        """Verify system API key"""
        if cls.SYSTEM_API_KEY:
            return provided_key == cls.SYSTEM_API_KEY
        
        expected_key = cls.generate_system_api_key(client_id)
        return provided_key == expected_key

config = Config()