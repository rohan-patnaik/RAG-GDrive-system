import openai
import numpy as np
from typing import List
from .config import config

class EmbeddingService:
    """Lightweight embedding service using OpenAI's API"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    
    def encode_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
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