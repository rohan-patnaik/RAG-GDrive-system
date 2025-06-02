import pinecone
from typing import List, Dict, Any
from .config import config

class VectorStore:
    """Pinecone vector store implementation"""
    
    def __init__(self):
        if not config.PINECONE_API_KEY:
            raise ValueError("Pinecone API key not configured")
        
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
                    'filename': chunk.get('filename', ''),
                    'chunk_index': chunk.get('chunk_index', 0)
                }
            })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
    
    def search(self, query_embedding: List[float], top_k: int = None) -> List[Dict[str, Any]]:
        """Search for similar chunks"""
        top_k = top_k or config.TOP_K_RESULTS
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        chunks = []
        for match in results['matches']:
            if match['score'] >= config.SIMILARITY_THRESHOLD:
                chunks.append({
                    'id': match['id'],
                    'content': match['metadata']['content'],
                    'score': match['score'],
                    'metadata': {
                        'source': match['metadata'].get('source', ''),
                        'filename': match['metadata'].get('filename', ''),
                        'chunk_index': match['metadata'].get('chunk_index', 0)
                    }
                })
        
        return chunks