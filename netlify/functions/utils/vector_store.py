import pinecone
from typing import List, Dict, Any, Optional
from .config import config

class VectorStore:
    """Pinecone-based vector store for serverless deployment"""
    
    def __init__(self):
        pinecone.init(
            api_key=config.PINECONE_API_KEY,
            environment=config.PINECONE_ENVIRONMENT
        )
        self.index = pinecone.Index(config.PINECONE_INDEX_NAME)
    
    def upsert_chunks(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to vector store"""
        vectors = []
        for chunk in chunks:
            vectors.append({
                'id': chunk['id'],
                'values': chunk['embedding'],
                'metadata': {
                    'content': chunk['content'],
                    'source': chunk.get('source', ''),
                    'filename': chunk.get('filename', '')
                }
            })
        
        self.index.upsert(vectors=vectors)
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        chunks = []
        for match in results['matches']:
            chunks.append({
                'id': match['id'],
                'content': match['metadata']['content'],
                'score': match['score'],
                'metadata': {
                    'source': match['metadata'].get('source', ''),
                    'filename': match['metadata'].get('filename', '')
                }
            })
        
        return chunks