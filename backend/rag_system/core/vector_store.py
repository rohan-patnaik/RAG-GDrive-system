import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from pinecone import Pinecone, Vector
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

from rag_system.config.settings import AppSettings, get_settings
from rag_system.core.embeddings import EmbeddingService
from rag_system.models.schemas import DocumentChunk, RetrievedChunk, DocumentMetadata
from rag_system.utils.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

class VectorStoreService:
    """Vector store service that supports both ChromaDB and Pinecone"""
    
    def __init__(self, settings: Optional[AppSettings] = None, embedding_service: Optional[EmbeddingService] = None):
        self.settings = settings or get_settings()
        self.embedding_service = embedding_service
        
        # Store provider for routing
        self.provider = self.settings.VECTOR_STORE_PROVIDER.lower()
        
        # Initialize the appropriate vector store
        if self.provider == "pinecone":
            self._init_pinecone()
        elif self.provider == "chromadb":
            self._init_chromadb()
        else:
            raise VectorStoreError(f"Unsupported vector store provider: {self.settings.VECTOR_STORE_PROVIDER}")
        
        logger.info(f"VectorStoreService initialized with provider: {self.provider}")
    
    def _init_pinecone(self) -> None:
        """Initialize Pinecone vector store"""
        if not PINECONE_AVAILABLE:
            raise VectorStoreError("Pinecone client not available. Install with: pip install pinecone-client")
        
        # Validate required settings
        if not self.settings.PINECONE_API_KEY:
            raise VectorStoreError("PINECONE_API_KEY is required when using Pinecone provider")
        if not self.settings.PINECONE_ENVIRONMENT:
            raise VectorStoreError("PINECONE_ENVIRONMENT is required when using Pinecone provider")
        if not self.settings.PINECONE_INDEX_NAME:
            raise VectorStoreError("PINECONE_INDEX_NAME is required when using Pinecone provider")
        
        try:
            # Initialize Pinecone client
            self.pinecone_client = Pinecone(
                api_key=self.settings.PINECONE_API_KEY.get_secret_value(),
                environment=self.settings.PINECONE_ENVIRONMENT
            )
            
            # Get the index
            self.index_name = self.settings.PINECONE_INDEX_NAME
            if self.index_name not in self.pinecone_client.list_indexes().names():
                raise VectorStoreError(f"Pinecone index '{self.index_name}' does not exist.")
            
            self.index = self.pinecone_client.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Pinecone: {str(e)}")
    
    def _init_chromadb(self) -> None:
        """Initialize ChromaDB vector store"""
        if not CHROMADB_AVAILABLE:
            raise VectorStoreError("ChromaDB not available. Install with: pip install chromadb")
        
        try:
            store_path = Path(self.settings.VECTOR_STORE_PATH)
            store_path.mkdir(parents=True, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(store_path),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            self.collection_name = self.settings.CHROMA_COLLECTION_NAME
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Connected to ChromaDB collection: {self.collection_name}")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB: {str(e)}")
    
    def _serialize_metadata(self, metadata: DocumentMetadata) -> Dict[str, Any]:
        """Serialize metadata for storage in vector database"""
        meta_dict = metadata.model_dump(exclude_none=True)
        
        if self.provider == "pinecone":
            # Pinecone-specific serialization
            serialized_meta: Dict[str, Any] = {}
            
            # Handle custom fields
            if "custom_fields" in meta_dict and meta_dict["custom_fields"]:
                for ck, cv in meta_dict["custom_fields"].items():
                    if isinstance(cv, (str, int, float, bool)):
                        serialized_meta[f"custom_{ck}"] = cv
                    elif isinstance(cv, list):
                        serialized_meta[f"custom_{ck}"] = [str(item) for item in cv]
                    else:
                        serialized_meta[f"custom_{ck}"] = str(cv)
            
            # Handle other fields
            for k, v in meta_dict.items():
                if k != "custom_fields":
                    if isinstance(v, (str, int, float, bool)):
                        serialized_meta[k] = v
                    elif isinstance(v, list):
                        serialized_meta[k] = [str(item) for item in v]
                    else:
                        serialized_meta[k] = str(v)
            
            return serialized_meta
        else:
            # ChromaDB can handle more complex metadata
            return meta_dict
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector store"""
        if not chunks:
            logger.warning("No chunks provided for addition")
            return
        
        if self.provider == "pinecone":
            self._add_chunks_pinecone(chunks)
        elif self.provider == "chromadb":
            self._add_chunks_chromadb(chunks)
    
    def _add_chunks_pinecone(self, chunks: List[DocumentChunk]) -> None:
        """Add chunks to Pinecone"""
        vectors_to_upsert: List[Vector] = []
        
        for chunk in chunks:
            if not chunk.embedding:
                logger.warning(f"Chunk {chunk.id} has no embedding, skipping")
                continue
            
            metadata_payload = self._serialize_metadata(chunk.metadata)
            # Add chunk content to metadata for retrieval
            metadata_payload["text_content"] = chunk.content
            
            vector = Vector(
                id=chunk.id,
                values=chunk.embedding,
                metadata=metadata_payload
            )
            vectors_to_upsert.append(vector)
        
        if vectors_to_upsert:
            try:
                upsert_response = self.index.upsert(vectors=vectors_to_upsert)
                logger.info(f"Added {len(vectors_to_upsert)} chunks to Pinecone. Upserted: {upsert_response.upserted_count}")
            except Exception as e:
                raise VectorStoreError(f"Failed to add chunks to Pinecone: {str(e)}")
    
    def _add_chunks_chromadb(self, chunks: List[DocumentChunk]) -> None:
        """Add chunks to ChromaDB"""
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        for chunk in chunks:
            if not chunk.content.strip():
                logger.warning(f"Chunk {chunk.id} has empty content, skipping")
                continue
            
            ids.append(chunk.id)
            documents.append(chunk.content)
            metadatas.append(self._serialize_metadata(chunk.metadata))
            
            # If embedding is provided, use it; otherwise ChromaDB will generate it
            if chunk.embedding:
                embeddings.append(chunk.embedding)
        
        if ids:
            try:
                if embeddings and len(embeddings) == len(ids):
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                        embeddings=embeddings
                    )
                else:
                    # Let ChromaDB generate embeddings
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                    )
                
                logger.info(f"Added {len(ids)} chunks to ChromaDB collection")
            except Exception as e:
                raise VectorStoreError(f"Failed to add chunks to ChromaDB: {str(e)}")
    
    def search_similar(self, query_embedding: List[float], top_k: int = 3, 
                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        """Search for similar chunks"""
        if self.provider == "pinecone":
            return self._search_pinecone(query_embedding, top_k, filter_metadata)
        elif self.provider == "chromadb":
            return self._search_chromadb(query_embedding, top_k, filter_metadata)
        
        return []
    
    def _search_pinecone(self, query_embedding: List[float], top_k: int, 
                        filter_metadata: Optional[Dict[str, Any]]) -> List[RetrievedChunk]:
        """Search Pinecone for similar chunks"""
        try:
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_metadata,
                include_metadata=True,
                include_values=False
            )
            
            retrieved_chunks: List[RetrievedChunk] = []
            
            for match in query_response.matches:
                metadata = match.metadata if match.metadata else {}
                content = metadata.pop("text_content", "")
                
                retrieved_chunk = RetrievedChunk(
                    id=match.id,
                    content=content,
                    metadata=metadata,
                    score=match.score if match.score is not None else 0.0
                )
                retrieved_chunks.append(retrieved_chunk)
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks from Pinecone")
            return retrieved_chunks
            
        except Exception as e:
            raise VectorStoreError(f"Failed to search Pinecone: {str(e)}")
    
    def _search_chromadb(self, query_embedding: List[float], top_k: int, 
                        filter_metadata: Optional[Dict[str, Any]]) -> List[RetrievedChunk]:
        """Search ChromaDB for similar chunks"""
        try:
            query_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"]
            }
            
            if filter_metadata:
                query_kwargs["where"] = filter_metadata
            
            results = self.collection.query(**query_kwargs)
            
            retrieved_chunks: List[RetrievedChunk] = []
            
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    chunk_id = results["ids"][0][i]
                    content = results["documents"][0][i] if results["documents"] else ""
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else 1.0
                    
                    # Convert distance to similarity score (ChromaDB returns distances)
                    score = max(0.0, 1.0 - distance)
                    
                    retrieved_chunk = RetrievedChunk(
                        id=chunk_id,
                        content=content,
                        metadata=metadata,
                        score=score
                    )
                    retrieved_chunks.append(retrieved_chunk)
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks from ChromaDB")
            return retrieved_chunks
            
        except Exception as e:
            raise VectorStoreError(f"Failed to search ChromaDB: {str(e)}")
    
    def delete_chunks(self, chunk_ids: Optional[List[str]] = None, 
                     filter_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Delete chunks from the vector store"""
        if self.provider == "pinecone":
            self._delete_chunks_pinecone(chunk_ids, filter_metadata)
        elif self.provider == "chromadb":
            self._delete_chunks_chromadb(chunk_ids, filter_metadata)
    
    def _delete_chunks_pinecone(self, chunk_ids: Optional[List[str]], 
                               filter_metadata: Optional[Dict[str, Any]]) -> None:
        """Delete chunks from Pinecone"""
        try:
            delete_kwargs = {}
            
            if chunk_ids:
                delete_kwargs["ids"] = chunk_ids
            if filter_metadata:
                delete_kwargs["filter"] = filter_metadata
            
            if not delete_kwargs:
                raise VectorStoreError("Either chunk_ids or filter_metadata must be provided for deletion")
            
            self.index.delete(**delete_kwargs)
            logger.info(f"Deleted chunks from Pinecone with params: {delete_kwargs}")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to delete chunks from Pinecone: {str(e)}")
    
    def _delete_chunks_chromadb(self, chunk_ids: Optional[List[str]], 
                               filter_metadata: Optional[Dict[str, Any]]) -> None:
        """Delete chunks from ChromaDB"""
        try:
            delete_kwargs = {}
            
            if chunk_ids:
                delete_kwargs["ids"] = chunk_ids
            if filter_metadata:
                delete_kwargs["where"] = filter_metadata
            
            if not delete_kwargs:
                raise VectorStoreError("Either chunk_ids or filter_metadata must be provided for deletion")
            
            self.collection.delete(**delete_kwargs)
            logger.info(f"Deleted chunks from ChromaDB with params: {delete_kwargs}")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to delete chunks from ChromaDB: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        if self.provider == "pinecone":
            return self._get_pinecone_stats()
        elif self.provider == "chromadb":
            return self._get_chromadb_stats()
        
        return {}
    
    def _get_pinecone_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "provider": "pinecone",
                "index_name": self.index_name,
                "item_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness
            }
        except Exception as e:
            logger.error(f"Failed to get Pinecone stats: {e}")
            return {"provider": "pinecone", "error": str(e)}
    
    def _get_chromadb_stats(self) -> Dict[str, Any]:
        """Get ChromaDB collection statistics"""
        try:
            count = self.collection.count()
            return {
                "provider": "chromadb",
                "collection_name": self.collection_name,
                "item_count": count
            }
        except Exception as e:
            logger.error(f"Failed to get ChromaDB stats: {e}")
            return {"provider": "chromadb", "error": str(e)}
    
    def clear_collection(self, namespace: Optional[str] = None) -> None:
        """Clear all items from the collection"""
        if self.provider == "pinecone":
            self._clear_pinecone(namespace)
        elif self.provider == "chromadb":
            self._clear_chromadb()
    
    def _clear_pinecone(self, namespace: Optional[str] = None) -> None:
        """Clear Pinecone index"""
        try:
            action = f"namespace '{namespace}'" if namespace else "entire index"
            logger.warning(f"Clearing {action} in Pinecone index '{self.index_name}'")
            
            delete_kwargs = {"delete_all": True}
            if namespace:
                delete_kwargs["namespace"] = namespace
            
            self.index.delete(**delete_kwargs)
            logger.info(f"Cleared {action}")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to clear Pinecone index: {str(e)}")
    
    def _clear_chromadb(self) -> None:
        """Clear ChromaDB collection"""
        try:
            logger.warning(f"Clearing ChromaDB collection '{self.collection_name}'")
            
            # Delete the collection and recreate it
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("ChromaDB collection cleared and recreated")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to clear ChromaDB collection: {str(e)}")


# Example usage and testing (if running this file directly)
if __name__ == "__main__":
    # Example usage - you would typically not run this directly
    pass
