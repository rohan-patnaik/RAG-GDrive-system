import openai
from typing import List
from .config import config

class EmbeddingService:
    """Embedding service using OpenAI"""
    
    def __init__(self):
        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key required for embeddings")
        
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    
    def encode_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return [data.embedding for data in response.data]