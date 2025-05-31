# backend/rag_system/core/vector_store.py
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
        if not self.settings.PINECONE_API_KEY or not self.settings.PINECONE_API_KEY.get_secret_value():
            raise VectorStoreError("PINECONE_API_KEY is required when using Pinecone provider")
        if not self.settings.PINECONE_ENVIRONMENT:
            raise VectorStoreError("PINECONE_ENVIRONMENT is required when using Pinecone provider")
        if not self.settings.PINECONE_INDEX_NAME:
            raise VectorStoreError("PINECONE_INDEX_NAME is required when using Pinecone provider")

        try:
            # Initialize Pinecone client
            self.pinecone_client = Pinecone(
                api_key=self.settings.PINECONE_API_KEY.get_secret_value(),
                # environment=self.settings.PINECONE_ENVIRONMENT # Environment is often part of spec now
            )

            # Get the index
            self.index_name = self.settings.PINECONE_INDEX_NAME
            # Check if index exists
            existing_indexes = []
            index_list_response = self.pinecone_client.list_indexes()
            if index_list_response and hasattr(index_list_response, 'indexes') and index_list_response.indexes:
                existing_indexes = [idx_model.name for idx_model in index_list_response.indexes if hasattr(idx_model, 'name')]

            if self.index_name not in existing_indexes:
                # Consider logging a warning or raising an error if auto-creation is not desired
                logger.warning(f"Pinecone index '{self.index_name}' does not exist. Please create it manually or ensure your setup script runs.")
                # For now, we'll raise an error as the manage_pinecone_index.py script should handle creation
                raise VectorStoreError(f"Pinecone index '{self.index_name}' does not exist. Create it first.")


            self.index = self.pinecone_client.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            # Optionally, log index stats on connection
            # stats = self.index.describe_index_stats()
            # logger.info(f"Pinecone index '{self.index_name}' stats: {stats}")


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
                metadata={"hnsw:space": "cosine"} # Ensure cosine metric for Chroma
            )

            logger.info(f"Connected to ChromaDB collection: {self.collection_name}")

        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB: {str(e)}")

    def _serialize_metadata(self, metadata: DocumentMetadata) -> Dict[str, Any]:
        """Serialize metadata for storage in vector database"""
        meta_dict = metadata.model_dump(exclude_none=True, mode='json') # Use mode='json' for better serialization

        if self.provider == "pinecone":
            serialized_meta: Dict[str, Any] = {}
            # Pinecone metadata values must be string, number, boolean, or list of strings.
            # No nested objects directly.

            # Handle custom fields
            if "custom_fields" in meta_dict and meta_dict["custom_fields"]:
                for ck, cv in meta_dict["custom_fields"].items():
                    if isinstance(cv, (str, int, float, bool)):
                        serialized_meta[f"custom_{ck}"] = cv
                    elif isinstance(cv, list) and all(isinstance(item, str) for item in cv):
                        serialized_meta[f"custom_{ck}"] = cv
                    else:
                        # Convert other types to string or skip
                        serialized_meta[f"custom_{ck}"] = str(cv)
            meta_dict.pop("custom_fields", None) # Remove original custom_fields

            # Handle other fields
            for k, v in meta_dict.items():
                if isinstance(v, (str, int, float, bool)):
                    serialized_meta[k] = v
                elif isinstance(v, list) and all(isinstance(item, str) for item in v):
                    serialized_meta[k] = v
                else:
                    # Convert datetimes and other complex types to string
                    serialized_meta[k] = str(v)
            return serialized_meta
        else:
            # ChromaDB can handle more complex metadata, but ensure it's JSON serializable
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
            # Add chunk content to metadata for retrieval, as Pinecone doesn't store it separately
            metadata_payload["text_content"] = chunk.content

            vector = Vector(
                id=chunk.id,
                values=chunk.embedding,
                metadata=metadata_payload
            )
            vectors_to_upsert.append(vector)

        if vectors_to_upsert:
            try:
                # Upsert in batches if necessary (Pinecone has limits, e.g., 100 vectors or 2MB per request)
                batch_size = 100
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i:i + batch_size]
                    upsert_response = self.index.upsert(vectors=batch)
                    logger.info(f"Upserted batch of {len(batch)} chunks to Pinecone. Upserted count from response: {upsert_response.upserted_count}")
                logger.info(f"Total {len(vectors_to_upsert)} chunks prepared for Pinecone.")
            except Exception as e:
                logger.error(f"Failed to add chunks to Pinecone: {str(e)}", exc_info=True)
                raise VectorStoreError(f"Failed to add chunks to Pinecone: {str(e)}")

    def _add_chunks_chromadb(self, chunks: List[DocumentChunk]) -> None:
        """Add chunks to ChromaDB"""
        ids = []
        documents = []
        metadatas = []
        embeddings_list = [] # Renamed to avoid conflict

        for chunk in chunks:
            if not chunk.content.strip():
                logger.warning(f"Chunk {chunk.id} has empty content, skipping")
                continue

            ids.append(chunk.id)
            documents.append(chunk.content)
            metadatas.append(self._serialize_metadata(chunk.metadata))

            if chunk.embedding:
                embeddings_list.append(chunk.embedding)

        if ids:
            try:
                if embeddings_list and len(embeddings_list) == len(ids):
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                        embeddings=embeddings_list
                    )
                else:
                    # Let ChromaDB generate embeddings if not provided for all
                    logger.info("Adding chunks to ChromaDB, allowing Chroma to generate embeddings if not all provided.")
                    self.collection.add(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                        # embeddings=None # Explicitly None if some are missing
                    )

                logger.info(f"Added {len(ids)} chunks to ChromaDB collection")
            except Exception as e:
                logger.error(f"Failed to add chunks to ChromaDB: {str(e)}", exc_info=True)
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
        """Search Pinecone for similar chunks with enhanced logging"""
        try:
            logger.info(f"Searching Pinecone: top_k={top_k}, query_embedding_dim={len(query_embedding)}, filter={filter_metadata}")
            # Ensure query_embedding is not empty and has the correct dimension
            if not query_embedding:
                logger.error("Query embedding is empty. Cannot search Pinecone.")
                return []
            
            # It's good practice to log a snippet of the query embedding for debugging, but be mindful of length
            # logger.debug(f"Query embedding (first 5 dims): {query_embedding[:5]}")

            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_metadata, # Pinecone filter object
                include_metadata=True,
                include_values=False # Set to True if you need to inspect vectors, False for production
            )
            logger.info(f"Pinecone raw query_response: {query_response}")

            retrieved_chunks: List[RetrievedChunk] = []

            if query_response and query_response.matches:
                logger.info(f"Found {len(query_response.matches)} matches in Pinecone response.")
                for match in query_response.matches:
                    logger.info(f"Processing Pinecone match: id={match.id}, score={match.score:.4f}")
                    # logger.debug(f"Match metadata: {match.metadata}") # Verbose

                    metadata = match.metadata if match.metadata else {}
                    content = metadata.pop("text_content", "") # Retrieve and remove from metadata
                    if not content:
                        logger.warning(f"Match {match.id} has no 'text_content' in metadata.")

                    retrieved_chunk = RetrievedChunk(
                        id=match.id,
                        content=content,
                        metadata=metadata, # Remaining metadata
                        score=match.score if match.score is not None else 0.0
                    )
                    retrieved_chunks.append(retrieved_chunk)
            else:
                logger.warning("No matches found in Pinecone response or query_response.matches is empty.")

            logger.info(f"Retrieved {len(retrieved_chunks)} chunks from Pinecone after processing.")
            return retrieved_chunks

        except Exception as e:
            logger.error(f"Failed to search Pinecone: {str(e)}", exc_info=True)
            # Optionally, re-raise as VectorStoreError or return empty list
            # raise VectorStoreError(f"Failed to search Pinecone: {str(e)}")
            return [] # Return empty list on error to prevent breaking the flow

    def _search_chromadb(self, query_embedding: List[float], top_k: int,
                        filter_metadata: Optional[Dict[str, Any]]) -> List[RetrievedChunk]:
        """Search ChromaDB for similar chunks"""
        try:
            logger.info(f"Searching ChromaDB: top_k={top_k}, query_embedding_dim={len(query_embedding)}, filter={filter_metadata}")
            if not query_embedding:
                logger.error("Query embedding is empty. Cannot search ChromaDB.")
                return []

            query_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"]
            }

            if filter_metadata:
                # ChromaDB's 'where' filter syntax: {"metadata_field": "value"}
                # Or for more complex: {"$and": [{"field1": "val1"}, {"field2": "val2"}]}
                query_kwargs["where"] = filter_metadata
                logger.info(f"Using ChromaDB where filter: {filter_metadata}")


            results = self.collection.query(**query_kwargs)
            logger.info(f"ChromaDB raw query results: {results}")


            retrieved_chunks: List[RetrievedChunk] = []

            if results and results.get("ids") and results["ids"][0]:
                logger.info(f"Found {len(results['ids'][0])} matches in ChromaDB response.")
                for i in range(len(results["ids"][0])):
                    chunk_id = results["ids"][0][i]
                    content = results["documents"][0][i] if results.get("documents") and results["documents"][0] else ""
                    metadata = results["metadatas"][0][i] if results.get("metadatas") and results["metadatas"][0] else {}
                    distance = results["distances"][0][i] if results.get("distances") and results["distances"][0] else 1.0

                    # Convert distance to similarity score (ChromaDB returns distances, 0 is most similar)
                    # Cosine distance is 1 - cosine_similarity. So, similarity = 1 - distance.
                    score = max(0.0, 1.0 - distance)
                    logger.info(f"Processing ChromaDB match: id={chunk_id}, distance={distance:.4f}, score={score:.4f}")


                    retrieved_chunk = RetrievedChunk(
                        id=chunk_id,
                        content=content,
                        metadata=metadata,
                        score=score
                    )
                    retrieved_chunks.append(retrieved_chunk)
            else:
                logger.warning("No matches found in ChromaDB response or results['ids'][0] is empty.")

            logger.info(f"Retrieved {len(retrieved_chunks)} chunks from ChromaDB after processing.")
            return retrieved_chunks

        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {str(e)}", exc_info=True)
            # raise VectorStoreError(f"Failed to search ChromaDB: {str(e)}")
            return []

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
            if filter_metadata: # Pinecone uses 'filter' for delete by metadata
                delete_kwargs["filter"] = filter_metadata

            if not delete_kwargs:
                logger.warning("Delete called for Pinecone without chunk_ids or filter_metadata.")
                # raise VectorStoreError("Either chunk_ids or filter_metadata must be provided for Pinecone deletion")
                return # Or simply return if this is not considered an error

            self.index.delete(**delete_kwargs)
            logger.info(f"Deleted chunks from Pinecone with params: {delete_kwargs}")

        except Exception as e:
            logger.error(f"Failed to delete chunks from Pinecone: {str(e)}", exc_info=True)
            raise VectorStoreError(f"Failed to delete chunks from Pinecone: {str(e)}")

    def _delete_chunks_chromadb(self, chunk_ids: Optional[List[str]],
                               filter_metadata: Optional[Dict[str, Any]]) -> None:
        """Delete chunks from ChromaDB"""
        try:
            delete_kwargs = {}

            if chunk_ids:
                delete_kwargs["ids"] = chunk_ids
            if filter_metadata: # ChromaDB uses 'where' for delete by metadata
                delete_kwargs["where"] = filter_metadata

            if not delete_kwargs:
                logger.warning("Delete called for ChromaDB without chunk_ids or where filter.")
                # raise VectorStoreError("Either chunk_ids or where filter must be provided for ChromaDB deletion")
                return

            self.collection.delete(**delete_kwargs)
            logger.info(f"Deleted chunks from ChromaDB with params: {delete_kwargs}")

        except Exception as e:
            logger.error(f"Failed to delete chunks from ChromaDB: {str(e)}", exc_info=True)
            raise VectorStoreError(f"Failed to delete chunks from ChromaDB: {str(e)}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        if self.provider == "pinecone":
            return self._get_pinecone_stats()
        elif self.provider == "chromadb":
            return self._get_chromadb_stats()

        return {"provider": self.provider, "status": "unknown_provider"}

    def _get_pinecone_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        try:
            stats = self.index.describe_index_stats()
            # The structure of stats can vary slightly based on client version and index type
            return {
                "provider": "pinecone",
                "index_name": self.index_name,
                "item_count": stats.get("total_vector_count", stats.get("total_record_count", 0)),
                "dimension": stats.get("dimension", 0),
                "index_fullness": stats.get("index_fullness", 0.0),
                "namespaces": stats.get("namespaces", {})
            }
        except Exception as e:
            logger.error(f"Failed to get Pinecone stats: {e}", exc_info=True)
            return {"provider": "pinecone", "index_name": self.index_name, "error": str(e)}

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
            logger.error(f"Failed to get ChromaDB stats: {e}", exc_info=True)
            return {"provider": "chromadb", "collection_name": self.collection_name, "error": str(e)}

    def clear_collection(self, namespace: Optional[str] = None) -> None:
        """Clear all items from the collection/index or a specific namespace for Pinecone."""
        if self.provider == "pinecone":
            self._clear_pinecone(namespace)
        elif self.provider == "chromadb":
            self._clear_chromadb() # ChromaDB doesn't have namespaces in the same way for this operation

    def _clear_pinecone(self, namespace: Optional[str] = None) -> None:
        """Clear Pinecone index or a specific namespace."""
        try:
            action = f"namespace '{namespace}'" if namespace else "entire index"
            logger.warning(f"Clearing {action} in Pinecone index '{self.index_name}'")

            delete_kwargs = {"delete_all": True}
            if namespace:
                delete_kwargs["namespace"] = namespace # Pinecone SDK uses 'namespace' for this

            self.index.delete(**delete_kwargs)
            logger.info(f"Pinecone: Cleared {action} in index '{self.index_name}'.")

        except Exception as e:
            logger.error(f"Failed to clear Pinecone {action}: {str(e)}", exc_info=True)
            raise VectorStoreError(f"Failed to clear Pinecone {action}: {str(e)}")

    def _clear_chromadb(self) -> None:
        """Clear ChromaDB collection by deleting and recreating it."""
        try:
            logger.warning(f"Clearing ChromaDB collection '{self.collection_name}' by deleting and recreating.")

            # Store current collection metadata if needed, though get_or_create_collection handles it
            # current_metadata = self.collection.metadata
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"} # Re-apply metadata
                # embedding_function=... # If you were using a specific one for Chroma
            )

            logger.info(f"ChromaDB collection '{self.collection_name}' cleared and recreated.")

        except Exception as e:
            logger.error(f"Failed to clear ChromaDB collection: {str(e)}", exc_info=True)
            raise VectorStoreError(f"Failed to clear ChromaDB collection: {str(e)}")

# Example usage and testing (if running this file directly)
if __name__ == "__main__":
    # This section is for direct testing of this file, not part of the main app flow.
    # You would need to set up a .env file or configure settings manually.
    print("VectorStoreService direct execution example (requires .env setup)")

    # Example: Initialize and test Pinecone
    # test_settings = get_settings()
    # if test_settings.VECTOR_STORE_PROVIDER.lower() == "pinecone":
    #     print("\n--- Testing Pinecone ---")
    #     try:
    #         # Embedding service needed if you pre-compute embeddings
    #         # from rag_system.core.embeddings import EmbeddingService
    #         # embedding_service_instance = EmbeddingService(settings=test_settings)

    #         vs_service = VectorStoreService(settings=test_settings) #, embedding_service=embedding_service_instance)
    #         print(f"Pinecone Stats: {vs_service.get_collection_stats()}")

    #         # Dummy chunk for testing (ensure embedding dimension matches index)
    #         # dummy_embedding = [0.1] * 384 # Replace with actual embedding
    #         # dummy_metadata = DocumentMetadata(source_id="test_doc", filename="test.txt", custom_fields={"category": "test"})
    #         # dummy_chunk = DocumentChunk(id="test_chunk_1", document_id="test_doc", content="This is a test chunk for Pinecone.", metadata=dummy_metadata, embedding=dummy_embedding)
    #         # vs_service.add_chunks([dummy_chunk])
    #         # print(f"Pinecone Stats after add: {vs_service.get_collection_stats()}")

    #         # search_results = vs_service.search_similar(query_embedding=dummy_embedding, top_k=1)
    #         # print(f"Search results: {search_results}")

    #     except Exception as e:
    #         print(f"Error during Pinecone test: {e}")
    # else:
    #     print("Skipping Pinecone test as provider is not 'pinecone'.")
    pass
