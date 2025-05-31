# backend/rag_system/core/vector_store.py
import logging
import json # For serializing metadata if necessary
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path # For example usage if kept

from pinecone import Pinecone, Index # Keep these if they are still valid
from pinecone.exceptions import PineconeApiException

from pinecone.core.client.models import Vector

from rag_system.models.schemas import DocumentChunk, DocumentMetadata, RetrievedChunk
# DocumentChunk will need an `embedding` field or a new `DocumentChunkWithEmbedding` model
from rag_system.utils.exceptions import VectorStoreError, EmbeddingError # EmbeddingError for consistency
from rag_system.config.settings import AppSettings, get_settings
from rag_system.core.embeddings import EmbeddingService # Kept for embedding_dimension

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Service for interacting with a Pinecone vector store.
    Manages adding, searching, and deleting document chunks.
    Embeddings are assumed to be pre-computed and provided with the chunks.
    """

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        embedding_service: Optional[EmbeddingService] = None, # Primarily for dimension info
    ):
        """
        Initializes the VectorStoreService for Pinecone.

        Args:
            settings: Application settings. If None, loads default settings.
            embedding_service: An instance of EmbeddingService. Used to get embedding dimension.

        Raises:
            VectorStoreError: If configuration is missing or Pinecone initialization fails.
        """
        if settings is None:
            settings = get_settings()

        if settings.VECTOR_STORE_PROVIDER != "pinecone":
            msg = "VectorStoreService is configured for Pinecone, but provider is " \
                  f"'{settings.VECTOR_STORE_PROVIDER}'."
            logger.error(msg)
            raise VectorStoreError(msg)

        if not all([settings.PINECONE_API_KEY, settings.PINECONE_ENVIRONMENT, settings.PINECONE_INDEX_NAME]):
            msg = "Pinecone API key, environment, or index name is missing in settings."
            logger.error(msg)
            raise VectorStoreError(msg)

        self.index_name: str = settings.PINECONE_INDEX_NAME
        self._embedding_service = embedding_service # Store for embedding dimension

        try:
            logger.info("Initializing Pinecone client...")
            pinecone_client = Pinecone(
                api_key=settings.PINECONE_API_KEY.get_secret_value(),
                environment=settings.PINECONE_ENVIRONMENT,
            )

            logger.info(f"Connecting to Pinecone index: '{self.index_name}'")
            if self.index_name not in pinecone_client.list_indexes().names:
                msg = f"Pinecone index '{self.index_name}' does not exist."
                logger.error(msg)
                # Note: Index creation is usually a separate setup step, not done on-the-fly here.
                # If dynamic creation were desired, it would require knowing dimensions, metric type, etc.
                # For example:
                # if self._embedding_service:
                #     dimension = self._embedding_service.get_embedding_dimension()
                #     pinecone_client.create_index(self.index_name, dimension=dimension, metric='cosine')
                # else:
                #    raise VectorStoreError("Embedding service needed to get dimension for index creation.")
                raise VectorStoreError(msg)

            self.index: Index = pinecone_client.Index(self.index_name)

            stats = self.index.describe_index_stats()
            logger.info(
                f"Successfully connected to Pinecone index '{self.index_name}'. "
                f"Total vectors: {stats.get('total_vector_count', 0)}"
            )

        except ApiException as e:
            logger.error(f"Pinecone API error during initialization: {e}", exc_info=True)
            raise VectorStoreError(f"Pinecone API error: {e.reason}") from e
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client or index '{self.index_name}': {e}", exc_info=True)
            raise VectorStoreError("Pinecone client/index initialization failed.") from e

    def _serialize_metadata(self, metadata: DocumentMetadata) -> Dict[str, Any]:
        """
        Serializes DocumentMetadata to a flat dictionary suitable for Pinecone.
        Pinecone metadata values can be string, number, boolean, or list of strings.
        Complex nested objects need to be flattened or serialized (e.g., to JSON strings).
        """
        meta_dict = metadata.model_dump(exclude_none=True)

        # Example: Flatten 'custom_fields' if it's a dict.
        # Pinecone supports nested metadata up to a certain depth, but direct key-value is safer.
        # For simplicity, we'll ensure all custom_fields are strings or lists of strings.
        # More complex serialization might be needed for arbitrary custom_fields.

        serialized_meta: Dict[str, Any] = {}
        for k, v in meta_dict.items():
            if isinstance(v, dict): # E.g. custom_fields
                 # Serialize dicts to JSON strings or flatten them
                 # For now, let's assume simple custom_fields or they are pre-processed
                 for ck, cv in v.items():
                     if isinstance(cv, (str, int, float, bool, list)):
                         serialized_meta[f"custom_{ck}"] = cv
                     else:
                         serialized_meta[f"custom_{ck}"] = str(cv) # Fallback to string
            elif isinstance(v, (str, int, float, bool, list)):
                serialized_meta[k] = v
            else:
                serialized_meta[k] = str(v) # Fallback for other types

        # Ensure content of lists are strings for Pinecone
        for k, v in serialized_meta.items():
            if isinstance(v, list):
                serialized_meta[k] = [str(item) for item in v]

        return serialized_meta


    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Adds document chunks with pre-computed embeddings to the Pinecone index.

        Args:
            chunks: A list of DocumentChunk objects. Each chunk MUST have its `embedding`
                    field populated with a valid vector.

        Raises:
            VectorStoreError: If adding chunks fails or embeddings are missing.
            EmbeddingError: If a chunk is missing its embedding.
        """
        if not chunks:
            logger.info("No chunks provided to add to the vector store.")
            return

        vectors_to_upsert: List[Vector] = []
        for chunk in chunks:
            if not chunk.content:
                logger.warning(f"Chunk {chunk.id} has no content, skipping.")
                continue
            if not chunk.embedding: # Crucial check
                logger.error(f"Chunk {chunk.id} is missing pre-computed embedding.")
                raise EmbeddingError(f"Chunk {chunk.id} missing embedding. Pre-compute embeddings before adding.")

            metadata_payload = self._serialize_metadata(chunk.metadata)
            # Add chunk content to metadata for potential retrieval if not storing original docs elsewhere
            metadata_payload["text_content"] = chunk.content

            vectors_to_upsert.append(
                Vector(id=chunk.id, values=chunk.embedding, metadata=metadata_payload)
            )

        if not vectors_to_upsert:
            logger.info("No valid chunks with embeddings to add after filtering.")
            return

        try:
            logger.debug(f"Upserting {len(vectors_to_upsert)} vectors to index '{self.index_name}'.")
            upsert_response = self.index.upsert(vectors=vectors_to_upsert)
            logger.info(
                f"Successfully upserted {upsert_response.upserted_count} vectors to '{self.index_name}'."
            )
        except ApiException as e:
            logger.error(f"Pinecone API error during upsert: {e}", exc_info=True)
            raise VectorStoreError(f"Pinecone API error during upsert: {e.reason}") from e
        except Exception as e:
            logger.error(f"Failed to upsert vectors to index '{self.index_name}': {e}", exc_info=True)
            raise VectorStoreError("Failed to upsert vectors to Pinecone.") from e

    def search_similar(
        self, 
        query_embedding: List[float], 
        top_k: int = 3, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """
        Search for chunks similar to the query embedding in Pinecone.

        Args:
            query_embedding: The embedding vector for the query.
            top_k: The number of similar results to retrieve.
            filter_metadata: A dictionary for metadata filtering (Pinecone syntax).
                             Example: {"genre": "drama", "year": {"$gte": 2020}}

        Returns:
            A list of RetrievedChunk objects.
        """
        if not query_embedding:
            logger.warning("Search called with an empty query embedding.")
            return []

        try:
            logger.info(
                f"Searching for {top_k} similar chunks in index '{self.index_name}'."
                f"{' With filter: ' + str(filter_metadata) if filter_metadata else ''}"
            )
            
            # Pinecone query
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_metadata,
                include_metadata=True,
                include_values=False # Usually not needed for retrieved chunks, saves bandwidth
            )
            
            retrieved_chunks: List[RetrievedChunk] = []
            if query_response.matches:
                for match in query_response.matches:
                    metadata = match.metadata if match.metadata else {}
                    # Extract original content if stored, otherwise it might be empty or from metadata
                    content = metadata.pop("text_content", "") # Retrieve and remove from metadata to avoid duplication
                    
                    retrieved_chunk = RetrievedChunk(
                        id=match.id,
                        content=content, # Content now comes from metadata
                        metadata=metadata, # Remaining metadata
                        score=match.score if match.score is not None else 0.0 # Pinecone score is similarity
                    )
                    retrieved_chunks.append(retrieved_chunk)
            
            logger.info(f"Found {len(retrieved_chunks)} similar chunks.")
            return retrieved_chunks
            
        except ApiException as e:
            logger.error(f"Pinecone API error during search: {e}", exc_info=True)
            raise VectorStoreError(f"Pinecone API error during search: {e.reason}") from e
        except Exception as e:
            logger.error(f"Failed to search in index '{self.index_name}': {e}", exc_info=True)
            raise VectorStoreError("Failed to search for similar chunks in Pinecone.") from e

    def delete_chunks(self, chunk_ids: Optional[List[str]] = None, filter_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Deletes chunks from the Pinecone index by IDs or metadata filter.

        Args:
            chunk_ids: A list of chunk IDs to delete.
            filter_metadata: A dictionary for metadata filtering (Pinecone syntax).

        Raises:
            VectorStoreError: If deletion fails.
        """
        if not chunk_ids and not filter_metadata:
            logger.warning("Attempted to delete chunks without specifying IDs or a filter.")
            return

        try:
            delete_kwargs = {}
            if chunk_ids:
                delete_kwargs["ids"] = chunk_ids
                logger.info(f"Deleting {len(chunk_ids)} chunks by ID from '{self.index_name}'.")
            if filter_metadata: # Pinecone uses 'filter' for delete
                delete_kwargs["filter"] = filter_metadata
                logger.info(f"Deleting chunks by metadata filter from '{self.index_name}': {filter_metadata}")

            if not delete_kwargs: # Should not happen if initial check passes
                return

            self.index.delete(**delete_kwargs)
            logger.info("Chunks deletion request sent to Pinecone successfully.")
        except ApiException as e:
            logger.error(f"Pinecone API error during delete: {e}", exc_info=True)
            raise VectorStoreError(f"Pinecone API error during delete: {e.reason}") from e
        except Exception as e:
            logger.error(f"Failed to delete chunks from '{self.index_name}': {e}", exc_info=True)
            raise VectorStoreError("Failed to delete chunks from Pinecone.") from e

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Retrieves statistics about the Pinecone index.

        Returns:
            A dictionary containing index statistics (e.g., total vector count, dimensions).
        """
        try:
            stats = self.index.describe_index_stats()
            # Convert Pinecone's DescribeIndexStatsResponse to a dict for consistent return type
            # Example: stats.total_vector_count, stats.dimension, stats.namespaces
            logger.debug(f"Pinecone index '{self.index_name}' stats: {stats}")
            return {
                "item_count": stats.total_vector_count or 0,
                "dimension": stats.dimension or 0,
                "namespaces": stats.namespaces or {}, # Provides counts per namespace
                "index_fullness": stats.index_fullness or 0.0, # If available
                "index_name": self.index_name,
            }
        except ApiException as e:
            logger.error(f"Pinecone API error during describe_index_stats: {e}", exc_info=True)
            raise VectorStoreError(f"Pinecone API error retrieving stats: {e.reason}") from e
        except Exception as e:
            logger.error(f"Failed to get stats for index '{self.index_name}': {e}", exc_info=True)
            raise VectorStoreError("Failed to retrieve Pinecone index statistics.") from e

    def clear_collection(self, namespace: Optional[str] = None) -> None:
        """
        Deletes all vectors from the Pinecone index or a specific namespace.
        This is a destructive operation.

        Args:
            namespace: If provided, deletes all vectors in this namespace.
                       If None, deletes all vectors in the entire index.
        """
        try:
            action = f"namespace '{namespace}'" if namespace else "entire index"
            logger.warning(
                f"Clearing all vectors from {action} in index '{self.index_name}'. "
                "This operation is destructive."
            )

            delete_kwargs = {"delete_all": True}
            if namespace:
                delete_kwargs["namespace"] = namespace

            self.index.delete(**delete_kwargs)
            logger.info(f"Successfully cleared all vectors from {action} in index '{self.index_name}'.")

        except ApiException as e:
            logger.error(f"Pinecone API error during clear_collection (delete all): {e}", exc_info=True)
            raise VectorStoreError(f"Pinecone API error clearing collection: {e.reason}") from e
        except Exception as e:
            logger.error(f"Failed to clear index '{self.index_name}': {e}", exc_info=True)
            raise VectorStoreError(f"Failed to clear Pinecone index '{self.index_name}'.") from e


# Example Usage (Simplified and adapted for Pinecone, assuming env vars are set for API key etc.):
# Note: This example would require DocumentChunk to have an `embedding` field.
# And Pinecone index must exist and be configured.
if __name__ == "__main__":
    from rag_system.config.logging_config import setup_logging
    setup_logging("DEBUG")

    # Ensure PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME are in .env or environment
    # For this example, we assume they are.
    # Also, the DocumentChunk model would need `embedding: List[float]` field.

    logger.info("Starting Pinecone VectorStoreService example...")

    # Create dummy settings, relying on .env for Pinecone specifics
    # For a real test, you'd mock AppSettings or ensure .env is correctly set up.
    class MockPineconeSettings(AppSettings):
        VECTOR_STORE_PROVIDER: str = "pinecone"
        # PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME loaded from .env
        # EMBEDDING_MODEL_NAME required by EmbeddingService
        EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
        # Ensure the dummy DocumentChunk below has an embedding of the correct dimension for this model (384)

    test_settings = None
    try:
        test_settings = MockPineconeSettings()
    except Exception as e:
        logger.warning(f"Could not load MockPineconeSettings (likely missing .env for Pinecone): {e}")
        logger.warning("Skipping Pinecone example execution. Ensure Pinecone env vars are set.")
        exit()


    if not all([test_settings.PINECONE_API_KEY, test_settings.PINECONE_ENVIRONMENT, test_settings.PINECONE_INDEX_NAME]):
        logger.warning("Pinecone credentials or index name not found in settings/env. Skipping example.")
        exit()

    embed_service = None
    vector_store = None

    try:
        embed_service = EmbeddingService(settings=test_settings)
        vector_store = VectorStoreService(settings=test_settings, embedding_service=embed_service)

        logger.info(f"Initial index stats: {vector_store.get_collection_stats()}")

        # Example: Clear a specific namespace or the whole index before running tests
        # vector_store.clear_collection() # Be careful with this!
        # logger.info(f"Index stats after clearing: {vector_store.get_collection_stats()}")

        # Create sample chunks with embeddings
        # Dimension for all-MiniLM-L6-v2 is 384
        dummy_embedding_dim = embed_service.get_embedding_dimension()
        if not dummy_embedding_dim: # Should be set by EmbeddingService
             dummy_embedding_dim = 384
             logger.warning(f"Could not get embedding dimension, defaulting to {dummy_embedding_dim}")


        sample_chunks_data = [
            ("pc_id1", "First Pinecone doc about apples.", {"source": "docPineA", "type": "fruit"}, [0.1] * dummy_embedding_dim),
            ("pc_id2", "Second Pinecone doc about bananas.", {"source": "docPineB", "type": "fruit"}, [0.2] * dummy_embedding_dim),
            ("pc_id3", "Pinecone oranges are citrus.", {"source": "docPineC", "type": "fruit"}, [0.3] * dummy_embedding_dim),
            ("pc_id4", "Pinecone document about cars.", {"source": "docPineD", "type": "vehicle"}, [0.4] * dummy_embedding_dim),
        ]

        # This part requires DocumentChunk to have an `embedding` field.
        # Let's assume it does for this example.
        chunks_to_add: List[DocumentChunk] = []
        for id_val, content_val, meta_val, emb_val in sample_chunks_data:
            doc_meta = DocumentMetadata(
                source_id=meta_val["source"],
                filename=f"{meta_val['source']}.txt",
                custom_fields=meta_val
            )
            # Assuming DocumentChunk now has an embedding field
            # Modify DocumentChunk in schemas.py: embedding: Optional[List[float]] = None
            chunk = DocumentChunk(id=id_val, document_id=meta_val["source"], content=content_val, metadata=doc_meta)
            chunk.embedding = emb_val # Assign the embedding
            chunks_to_add.append(chunk)


        if chunks_to_add:
             vector_store.add_chunks(chunks_to_add)
             logger.info(f"Index stats after adding chunks: {vector_store.get_collection_stats()}")

        # Search for similar chunks
        query_text = "Tell me about fruits with Pinecone"
        # Actual embedding generation should be robust
        query_embedding = embed_service.encode_queries([query_text])[0]


        if query_embedding:
            similar_chunks = vector_store.search_similar(query_embedding, top_k=2)
            logger.info(f"Found {len(similar_chunks)} similar chunks for query: '{query_text}'")
            for i, r_chunk in enumerate(similar_chunks):
                logger.info(f"Result {i+1}: ID={r_chunk.id}, Score={r_chunk.score:.4f}, Content='{r_chunk.content[:50]}...'")
                logger.info(f"   Metadata: {r_chunk.metadata}")

            # Example filter (syntax depends on how metadata was stored)
            # Pinecone filter syntax: {"field_name": "field_value"}
            # If custom_fields were stored as custom_type: "vehicle"
            filtered_chunks = vector_store.search_similar(
                 query_embedding, top_k=2, filter_metadata={"custom_type": "vehicle"}
            )
            logger.info(f"Found {len(filtered_chunks)} similar chunks with filter {{'custom_type': 'vehicle'}}:")
            for i, r_chunk in enumerate(filtered_chunks):
                logger.info(f"Filtered Result {i+1}: ID={r_chunk.id}, Score={r_chunk.score:.4f}, Content='{r_chunk.content[:50]}...'")
        else:
            logger.error("Could not generate query embedding for search.")

        # Delete a chunk
        if chunks_to_add:
            vector_store.delete_chunks(chunk_ids=[chunks_to_add[0].id])
            logger.info(f"Index stats after deleting '{chunks_to_add[0].id}': {vector_store.get_collection_stats()}")

            # Delete by metadata filter
            # vector_store.delete_chunks(filter_metadata={"custom_type": "fruit"})
            # logger.info(f"Index stats after deleting fruit type (via filter): {vector_store.get_collection_stats()}")


    except (VectorStoreError, EmbeddingError) as e:
        logger.error(f"Vector store or embedding service error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred in Pinecone example: {e}", exc_info=True)
    finally:
        logger.info("Pinecone VectorStoreService example finished.")
        # No local cleanup like rmtree needed for Pinecone (cloud service)
        pass
