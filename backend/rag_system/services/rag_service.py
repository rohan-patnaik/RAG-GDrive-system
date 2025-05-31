# backend/rag_system/services/rag_service.py
import logging
import asyncio
from typing import List, Optional, Tuple
import datetime

from rag_system.config.settings import AppSettings, get_settings
from rag_system.core.document_loader import DocumentLoader
from rag_system.core.text_processor import TextProcessor
from rag_system.core.embeddings import EmbeddingService
from rag_system.core.vector_store import VectorStoreService
from rag_system.services.llm_service import LLMService
from rag_system.models.schemas import (
    IngestionRequest,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
    Document,
    DocumentChunk,
    RetrievedChunk,
    LLMProvider,
    SystemStatusResponse,
    ComponentStatus,
    StatusEnum,
)
from rag_system.utils.exceptions import (
    RAGServiceError,
    DocumentProcessingError,
    VectorStoreError,
    EmbeddingError,
    LLMError,
)
from rag_system import __version__ as app_version


logger = logging.getLogger(__name__)


class RAGService:
    """
    Orchestrates the entire RAG pipeline, including document ingestion and querying.
    """

    def __init__(
        self,
        settings: Optional[AppSettings] = None,
        document_loader: Optional[DocumentLoader] = None,
        text_processor: Optional[TextProcessor] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store_service: Optional[VectorStoreService] = None,
        llm_service: Optional[LLMService] = None,
    ):
        """
        Initializes the RAGService with necessary components.
        If components are not provided, they are initialized with default settings.
        """
        self.settings = settings or get_settings()

        self.document_loader = document_loader or DocumentLoader()
        self.text_processor = text_processor or TextProcessor(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
        )
        self.embedding_service = embedding_service or EmbeddingService(
            settings=self.settings
        )
        self.vector_store_service = vector_store_service or VectorStoreService(
            settings=self.settings, embedding_service=self.embedding_service
        )
        self.llm_service = llm_service or LLMService(settings=self.settings)

        logger.info("RAGService initialized with all components.")

    async def ingest_documents(
        self, request: IngestionRequest
    ) -> IngestionResponse:
        """
        Handles the document ingestion pipeline.
        1. Loads documents from the source.
        2. Processes and chunks the documents.
        3. Generates embeddings for chunks (if not handled by vector store).
        4. Adds chunks to the vector store.

        Args:
            request: The IngestionRequest containing details for ingestion.

        Returns:
            An IngestionResponse summarizing the outcome.
        """
        logger.info(
            f"Starting document ingestion from: {request.source_directory}, "
            f"patterns: {request.file_patterns}, recursive: {request.recursive}"
        )
        errors: List[str] = []
        documents_processed_count = 0
        chunks_added_count = 0

        try:
            # 1. Load documents
            try:
                raw_documents: List[Document] = self.document_loader.load_from_directory(
                    source_directory=request.source_directory,
                    file_patterns=request.file_patterns or self.settings.DEFAULT_FILE_PATTERNS,
                    recursive=request.recursive if request.recursive is not None else self.settings.DEFAULT_RECURSIVE_INGESTION,
                )
                documents_processed_count = len(raw_documents)
                if not raw_documents:
                    msg = "No documents found or loaded from the specified source."
                    logger.warning(msg)
                    return IngestionResponse(
                        message=msg,
                        documents_processed=0,
                        chunks_added=0,
                        errors=[msg]
                    )
            except FileNotFoundError as e:
                logger.error(f"Source directory not found: {request.source_directory}", exc_info=True)
                raise DocumentProcessingError(message=f"Source directory not found: {request.source_directory}", detail=str(e)) from e
            except Exception as e:
                logger.error(f"Error loading documents: {e}", exc_info=True)
                raise DocumentProcessingError(message="Failed during document loading phase.", detail=str(e)) from e

            # 2. Process and chunk documents
            try:
                document_chunks: List[DocumentChunk] = self.text_processor.split_documents(
                    raw_documents
                )
                if not document_chunks:
                    msg = "No chunks were generated from the loaded documents."
                    logger.warning(msg)
                    # Potentially return early if no chunks, or let vector store handle empty list
                    errors.append(msg)
                    # return IngestionResponse(...)
            except Exception as e:
                logger.error(f"Error processing/chunking documents: {e}", exc_info=True)
                raise DocumentProcessingError(message="Failed during text processing and chunking.", detail=str(e)) from e

            # 3. Generate embeddings for chunks (required by Pinecone)
            if document_chunks:
                try:
                    logger.info(f"Generating embeddings for {len(document_chunks)} document chunks.")
                    chunk_contents = [chunk.content for chunk in document_chunks]
                    embeddings = self.embedding_service.encode_texts(chunk_contents)
                    if len(embeddings) != len(document_chunks):
                        # This should ideally not happen if encode_texts is robust
                        logger.error(
                            f"Mismatch in number of chunks ({len(document_chunks)}) "
                            f"and generated embeddings ({len(embeddings)})."
                        )
                        raise EmbeddingError(
                            message="Embedding generation returned a mismatched number of embeddings.",
                            detail=f"Expected {len(document_chunks)}, got {len(embeddings)}"
                        )
                    for i, chunk in enumerate(document_chunks):
                        chunk.embedding = embeddings[i]
                    logger.info(f"Successfully generated and assigned embeddings to {len(document_chunks)} chunks.")
                except EmbeddingError as e:
                    logger.error(f"Error generating embeddings: {e.message}", exc_info=True)
                    # Propagate this error to the main try-except block for IngestionResponse
                    raise DocumentProcessingError(
                        message="Failed during embedding generation phase.",
                        detail=str(e),
                        error_type="EmbeddingError" # Custom attribute for clarity
                    ) from e
                except Exception as e: # Catch any other unexpected error from embedding service
                    logger.error(f"Unexpected error during embedding generation: {e}", exc_info=True)
                    raise DocumentProcessingError(
                        message="Unexpected error during embedding generation.",
                        detail=str(e),
                        error_type="EmbeddingGenerationError"
                    ) from e
            else:
                logger.info("No document chunks to generate embeddings for.")

            # 4. Add chunks to vector store
            try:
                if document_chunks: # Only add if there are chunks
                    self.vector_store_service.add_chunks(document_chunks)
                    chunks_added_count = len(document_chunks) # Assuming all were added successfully
                    logger.info(f"Successfully added {chunks_added_count} chunks to the vector store.")
                else:
                    logger.info("No document chunks to add to the vector store.")

            except VectorStoreError as e:
                logger.error(f"Error adding chunks to vector store: {e.message}", exc_info=True)
                # This is a critical error for ingestion
                raise # Re-raise to be caught by the outer try-except
            except Exception as e: # Catch any other unexpected error from vector store
                logger.error(f"Unexpected error during vector store operation: {e}", exc_info=True)
                raise VectorStoreError(message="Unexpected error interacting with vector store during ingestion.", detail=str(e)) from e


            message = f"Ingestion completed. Processed {documents_processed_count} documents, added/updated {chunks_added_count} chunks."
            if errors:
                message += " Some issues encountered."
            logger.info(message)
            return IngestionResponse(
                message=message,
                documents_processed=documents_processed_count,
                chunks_added=chunks_added_count,
                errors=errors,
            )

        except (DocumentProcessingError, VectorStoreError, EmbeddingError) as e:
            # These are specific errors from our system
            errors.append(f"{e.error_type if hasattr(e, 'error_type') else type(e).__name__}: {e.message} - {e.detail}")
            logger.error(f"Ingestion failed due to {type(e).__name__}: {e.message}", exc_info=False) # exc_info=False if detail is enough
            # Re-raise the original custom exception for API to handle
            raise e
        except Exception as e:
            # Catch-all for other unexpected errors
            error_msg = f"An unexpected error occurred during ingestion: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            raise RAGServiceError(message="Ingestion pipeline failed unexpectedly.", detail=str(e)) from e


    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Handles the RAG query pipeline.
        1. Generates an embedding for the user's query.
        2. Retrieves relevant document chunks from the vector store.
        3. Constructs a prompt using the query and retrieved chunks.
        4. Sends the prompt to the selected LLM to generate an answer.

        Args:
            request: The QueryRequest containing the user's query and parameters.

        Returns:
            A QueryResponse with the LLM's answer and source information.
        """
        logger.info(f"Processing query: '{request.query_text[:100]}...' with provider: {request.llm_provider}")

        # Apply defaults from settings if not provided in request
        llm_provider = request.llm_provider or LLMProvider(self.settings.DEFAULT_LLM_PROVIDER)
        top_k = request.top_k or self.settings.TOP_K_RESULTS
        similarity_threshold = request.similarity_threshold # Can be None

        try:
            # 1. Generate query embedding
            try:
                query_embedding: List[float] = self.embedding_service.encode_query(
                    request.query_text
                )
                if not query_embedding: # Should not happen if encode_query raises error on failure
                    raise EmbeddingError("Generated empty query embedding.")
            except EmbeddingError as e:
                logger.error(f"Failed to embed query: {e.message}", exc_info=True)
                raise # Re-raise for API to handle

            # 2. Retrieve relevant chunks
            try:
                retrieved_chunks: List[RetrievedChunk] = self.vector_store_service.search_similar(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    # filter_metadata can be added here if needed
                )
                # Optionally filter by similarity_threshold *after* retrieval
                if similarity_threshold is not None:
                    original_count = len(retrieved_chunks)
                    retrieved_chunks = [
                        chunk for chunk in retrieved_chunks if chunk.score >= similarity_threshold
                    ]
                    logger.debug(f"Filtered {original_count - len(retrieved_chunks)} chunks by similarity threshold {similarity_threshold}.")

            except VectorStoreError as e:
                logger.error(f"Failed to retrieve chunks from vector store: {e.message}", exc_info=True)
                raise # Re-raise

            # 3. Construct prompt and 4. Generate answer using LLMService
            # LLMService will handle prompt construction internally
            try:
                llm_answer, llm_model_used = await self.llm_service.generate_response(
                    query=request.query_text,
                    context_chunks=retrieved_chunks,
                    provider=llm_provider,
                    model_name_override=request.llm_model_name # Pass optional model override
                )
            except LLMError as e:
                logger.error(f"LLM failed to generate answer: {e.message}", exc_info=True)
                raise # Re-raise
            except Exception as e: # Catch any other error from LLMService
                logger.error(f"Unexpected error during LLM response generation: {e}", exc_info=True)
                raise LLMError(message="Unexpected error from LLM service.", detail=str(e)) from e


            logger.info(f"Successfully generated LLM answer for query: '{request.query_text[:50]}...'")
            return QueryResponse(
                query_text=request.query_text,
                llm_answer=llm_answer,
                llm_provider_used=llm_provider, # The provider chosen for this query
                llm_model_used=llm_model_used, # Actual model name returned by LLMService
                retrieved_chunks=retrieved_chunks,
            )

        except (EmbeddingError, VectorStoreError, LLMError) as e:
            # These are specific errors from our system components
            logger.error(f"Query processing failed due to {type(e).__name__}: {e.message}", exc_info=False)
            raise e # Re-raise the original custom exception
        except Exception as e:
            # Catch-all for other unexpected errors during the query process
            error_msg = f"An unexpected error occurred during query processing: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RAGServiceError(message="Query pipeline failed unexpectedly.", detail=str(e)) from e

    async def get_system_status(self) -> SystemStatusResponse:
        """
        Checks the status of all core components of the RAG system.
        """
        logger.debug("RAGService: Getting system status.")
        components: List[ComponentStatus] = []
        overall_status = StatusEnum.OK # Assume OK initially

        # 1. Embedding Service Status
        emb_status = ComponentStatus(name="EmbeddingService", status=StatusEnum.UNKNOWN)
        try:
            if self.embedding_service and self.embedding_service.get_model():
                dim = self.embedding_service.get_embedding_dimension()
                emb_status.status = StatusEnum.OK
                emb_status.message = f"Model '{self.embedding_service.model_name}' loaded."
                emb_status.details = {"model_name": self.embedding_service.model_name, "dimension": dim}
            else: # Should not happen if constructor succeeded
                emb_status.status = StatusEnum.ERROR
                emb_status.message = "Embedding model not loaded."
        except Exception as e:
            logger.warning(f"EmbeddingService status check failed: {e}", exc_info=False)
            emb_status.status = StatusEnum.ERROR
            emb_status.message = f"Error: {str(e)}"
        components.append(emb_status)
        if emb_status.status != StatusEnum.OK: overall_status = StatusEnum.DEGRADED


        # 2. Vector Store Service Status
        vs_status = ComponentStatus(name="VectorStoreService", status=StatusEnum.UNKNOWN)
        try:
            if self.vector_store_service and self.vector_store_service.client:
                stats = self.vector_store_service.get_collection_stats()
                vs_status.status = StatusEnum.OK
                vs_status.message = f"Connected to collection '{stats.get('collection_name')}'."
                vs_status.details = stats
            else: # Should not happen
                vs_status.status = StatusEnum.ERROR
                vs_status.message = "Vector store client not initialized."
        except Exception as e:
            logger.warning(f"VectorStoreService status check failed: {e}", exc_info=False)
            vs_status.status = StatusEnum.ERROR
            vs_status.message = f"Error: {str(e)}"
        components.append(vs_status)
        if vs_status.status != StatusEnum.OK and overall_status == StatusEnum.OK : overall_status = StatusEnum.DEGRADED
        if vs_status.status == StatusEnum.ERROR: overall_status = StatusEnum.ERROR


        # 3. LLM Service Status (check configured providers)
        llm_service_overall_status = StatusEnum.OK
        if self.llm_service:
            # Check OpenAI
            if self.settings.OPENAI_API_KEY:
                openai_status = await self.llm_service.check_provider_health(LLMProvider.OPENAI)
                components.append(openai_status)
                if openai_status.status != StatusEnum.OK: llm_service_overall_status = StatusEnum.DEGRADED
                if openai_status.status == StatusEnum.ERROR: llm_service_overall_status = StatusEnum.ERROR

            # Check Anthropic
            if self.settings.ANTHROPIC_API_KEY:
                anthropic_status = await self.llm_service.check_provider_health(LLMProvider.ANTHROPIC)
                components.append(anthropic_status)
                if anthropic_status.status != StatusEnum.OK and llm_service_overall_status == StatusEnum.OK:
                    llm_service_overall_status = StatusEnum.DEGRADED
                if anthropic_status.status == StatusEnum.ERROR: llm_service_overall_status = StatusEnum.ERROR

            # Check Gemini
            if self.settings.GOOGLE_API_KEY:
                gemini_status = await self.llm_service.check_provider_health(LLMProvider.GEMINI)
                components.append(gemini_status)
                if gemini_status.status != StatusEnum.OK and llm_service_overall_status == StatusEnum.OK:
                    llm_service_overall_status = StatusEnum.DEGRADED
                if gemini_status.status == StatusEnum.ERROR: llm_service_overall_status = StatusEnum.ERROR
        else: # Should not happen
            llm_service_overall_status = StatusEnum.ERROR
            components.append(ComponentStatus(name="LLMService", status=StatusEnum.ERROR, message="LLMService not initialized."))

        if llm_service_overall_status != StatusEnum.OK and overall_status == StatusEnum.OK:
            overall_status = StatusEnum.DEGRADED
        if llm_service_overall_status == StatusEnum.ERROR:
            overall_status = StatusEnum.ERROR


        # Final overall status determination
        for comp in components:
            if comp.status == StatusEnum.ERROR:
                overall_status = StatusEnum.ERROR
                break
            if comp.status == StatusEnum.DEGRADED and overall_status == StatusEnum.OK:
                overall_status = StatusEnum.DEGRADED


        return SystemStatusResponse(
            system_status=overall_status,
            app_name=self.settings.APP_NAME,
            environment=self.settings.ENVIRONMENT,
            version=app_version,
            components=components,
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )

