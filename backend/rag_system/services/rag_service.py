# backend/rag_system/services/rag_service.py
import logging
import asyncio
from typing import List, Optional, Dict, Any, Tuple

from rag_system.config.settings import AppSettings, get_settings
from rag_system.core.document_loader import DocumentLoader
from rag_system.core.text_processor import TextProcessor
from rag_system.core.embeddings import EmbeddingService
from rag_system.core.vector_store import VectorStoreService
from rag_system.services.llm_service import LLMService, LLMProvider
from rag_system.models.schemas import (
    IngestionRequest,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
    Document,
    DocumentChunk,
    RetrievedChunk,
    SystemStatusResponse,
    ComponentStatus,
    StatusEnum,
    DocumentMetadata,
)
from rag_system.utils.exceptions import (
    BaseRAGError,
    DocumentProcessingError,
    VectorStoreError,
    QueryProcessingError,
    ConfigurationError,
)

logger = logging.getLogger(__name__)

class RAGService:
    """
    Orchestrates the RAG pipeline, including document ingestion,
    embedding, vector storage, retrieval, and LLM interaction.
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
        self.settings = settings or get_settings()

        # Initialize components if not provided (allows for dependency injection)
        self.document_loader = document_loader or DocumentLoader()
        self.text_processor = text_processor or TextProcessor(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
        )
        logger.info(f"RAGService using TextProcessor with chunk_size={self.settings.CHUNK_SIZE}, overlap={self.settings.CHUNK_OVERLAP}")

        self.embedding_service = embedding_service or EmbeddingService(settings=self.settings)
        self.vector_store_service = vector_store_service or VectorStoreService(
            settings=self.settings, embedding_service=self.embedding_service
        )
        self.llm_service = llm_service or LLMService(settings=self.settings)
        logger.info("RAGService initialized with all components.")

    async def ingest_documents_from_directory(self, request: IngestionRequest) -> IngestionResponse:
        """
        Ingests documents from a specified server-side directory.
        This method is suitable for CLI or backend-initiated ingestion.
        """
        logger.info(
            f"Starting ingestion from directory: {request.source_directory}, "
            f"patterns: {request.file_patterns}, recursive: {request.recursive}"
        )
        errors: List[str] = []
        documents_processed_count = 0
        chunks_added_count = 0

        try:
            # Use settings defaults if request parameters are None
            file_patterns = request.file_patterns or self.settings.DEFAULT_FILE_PATTERNS
            recursive = request.recursive if request.recursive is not None else self.settings.DEFAULT_RECURSIVE_INGESTION

            raw_documents: List[Document] = self.document_loader.load_from_directory(
                source_directory=request.source_directory,
                file_patterns=file_patterns,
                recursive=recursive,
            )
            documents_processed_count = len(raw_documents)

            if not raw_documents:
                msg = "No documents found or loaded from the specified source directory."
                logger.warning(msg)
                return IngestionResponse(
                    message=msg, documents_processed=0, chunks_added=0, errors=[]
                )

            logger.info(f"Loaded {documents_processed_count} documents from directory.")

            document_chunks: List[DocumentChunk] = self.text_processor.split_documents(raw_documents)
            logger.info(f"Generated {len(document_chunks)} chunks from {documents_processed_count} documents.")


            if not document_chunks:
                msg = "No chunks were generated from the loaded documents."
                logger.warning(msg)
                return IngestionResponse(
                    message=msg,
                    documents_processed=documents_processed_count,
                    chunks_added=0,
                    errors=errors,
                )

            # Generate embeddings for chunks
            chunk_contents = [chunk.content for chunk in document_chunks]
            logger.info(f"Generating embeddings for {len(chunk_contents)} chunk contents...")
            embeddings = self.embedding_service.encode_texts(chunk_contents)
            logger.info(f"Embeddings generated. Count: {len(embeddings)}")


            for i, chunk in enumerate(document_chunks):
                chunk.embedding = embeddings[i]

            self.vector_store_service.add_chunks(document_chunks)
            chunks_added_count = len(document_chunks)
            logger.info(f"Added {chunks_added_count} chunks to vector store.")


            message = f"Ingestion from directory completed. Processed {documents_processed_count} documents, added/updated {chunks_added_count} chunks."
            logger.info(message)
            return IngestionResponse(
                message=message,
                documents_processed=documents_processed_count,
                chunks_added=chunks_added_count,
                errors=errors,
            )

        except FileNotFoundError as e:
            logger.error(f"Ingestion failed: Directory not found - {e}", exc_info=True)
            errors.append(f"Directory not found: {str(e)}")
            raise DocumentProcessingError(f"Source directory not found: {request.source_directory}") from e
        except DocumentProcessingError as e:
            logger.error(f"Ingestion failed: Document processing error - {e}", exc_info=True)
            errors.append(f"Document processing error: {str(e)}")
            # Re-raise or return error response
            raise
        except VectorStoreError as e:
            logger.error(f"Ingestion failed: Vector store error - {e}", exc_info=True)
            errors.append(f"Vector store error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during directory ingestion: {e}", exc_info=True)
            errors.append(f"Unexpected error: {str(e)}")
            # Re-raise as a generic RAG error or a specific ingestion error
            raise BaseRAGError(f"Unexpected ingestion error: {str(e)}")


    async def ingest_uploaded_documents(
        self, uploaded_files_data: List[Dict[str, Any]]
    ) -> IngestionResponse:
        """
        Ingests documents from uploaded file data (filename and content bytes).
        Suitable for serverless functions handling file uploads.

        Args:
            uploaded_files_data: A list of dictionaries, each containing:
                                 {'filename': str, 'content_bytes': bytes}
        """
        logger.info(f"Starting ingestion for {len(uploaded_files_data)} uploaded files.")
        errors: List[str] = []
        documents_processed_count = 0
        chunks_added_count = 0
        raw_documents: List[Document] = []

        if not uploaded_files_data:
            return IngestionResponse(message="No files provided for ingestion.", documents_processed=0, chunks_added=0)

        for file_data in uploaded_files_data:
            filename = file_data.get("filename")
            content_bytes = file_data.get("content_bytes")

            if not filename or content_bytes is None: # Allow empty content_bytes if filename exists
                logger.warning(f"Skipping file data due to missing filename or content_bytes: {file_data}")
                errors.append(f"Invalid file data provided (missing name or content): {filename or 'N/A'}")
                continue
            try:
                doc = self.document_loader.load_document_from_bytes(
                    content_bytes=content_bytes,
                    filename=filename,
                    source_id_prefix="upload" # Differentiate source
                )
                raw_documents.append(doc)
                documents_processed_count += 1
            except DocumentProcessingError as e:
                logger.error(f"Failed to load document from bytes for {filename}: {e}", exc_info=True)
                errors.append(f"Error processing {filename}: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error loading {filename} from bytes: {e}", exc_info=True)
                errors.append(f"Unexpected error with {filename}: {str(e)}")


        if not raw_documents:
            msg = "No documents were successfully loaded from the uploaded files."
            logger.warning(msg)
            return IngestionResponse(
                message=msg, documents_processed=documents_processed_count, chunks_added=0, errors=errors
            )

        logger.info(f"Successfully loaded {len(raw_documents)} documents from uploads.")

        try:
            document_chunks: List[DocumentChunk] = self.text_processor.split_documents(raw_documents)
            logger.info(f"Generated {len(document_chunks)} chunks from {len(raw_documents)} uploaded documents.")

            if not document_chunks:
                msg = "No chunks were generated from the uploaded documents."
                logger.warning(msg)
                return IngestionResponse(
                    message=msg,
                    documents_processed=documents_processed_count,
                    chunks_added=0,
                    errors=errors,
                )

            chunk_contents = [chunk.content for chunk in document_chunks]
            logger.info(f"Generating embeddings for {len(chunk_contents)} chunk contents from uploads...")
            embeddings = self.embedding_service.encode_texts(chunk_contents)
            logger.info(f"Embeddings generated for uploaded content. Count: {len(embeddings)}")


            for i, chunk in enumerate(document_chunks):
                chunk.embedding = embeddings[i]

            self.vector_store_service.add_chunks(document_chunks)
            chunks_added_count = len(document_chunks)
            logger.info(f"Added {chunks_added_count} chunks from uploads to vector store.")

            message = f"Ingestion of uploaded files completed. Processed {documents_processed_count} documents, added {chunks_added_count} chunks."
            logger.info(message)
            return IngestionResponse(
                message=message,
                documents_processed=documents_processed_count,
                chunks_added=chunks_added_count,
                errors=errors,
            )
        except VectorStoreError as e:
            logger.error(f"Uploaded documents ingestion failed: Vector store error - {e}", exc_info=True)
            errors.append(f"Vector store error: {str(e)}")
            raise # Re-raise to be caught by the function handler
        except Exception as e:
            logger.error(f"An unexpected error occurred during uploaded documents ingestion: {e}", exc_info=True)
            errors.append(f"Unexpected error during chunking/embedding: {str(e)}")
            raise BaseRAGError(f"Unexpected ingestion error for uploaded files: {str(e)}")


    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Processes a user query, retrieves relevant chunks, and generates an answer using an LLM.
        """
        logger.info(
            f"Processing query: '{request.query_text[:50]}...' with provider: {request.llm_provider or self.settings.DEFAULT_LLM_PROVIDER}"
        )
        try:
            llm_provider = request.llm_provider or LLMProvider(self.settings.DEFAULT_LLM_PROVIDER)
            top_k = request.top_k or self.settings.TOP_K_RESULTS
            # Use provided threshold, or settings default, or None if neither (LLMService will handle None)
            similarity_threshold = request.similarity_threshold
            if similarity_threshold is None: # Explicitly check for None from request
                similarity_threshold = self.settings.SIMILARITY_THRESHOLD # Fallback to settings default

            logger.debug(f"Using top_k={top_k}, similarity_threshold={similarity_threshold}")


            query_embedding: List[float] = self.embedding_service.encode_query(request.query_text)

            retrieved_chunks_from_vs: List[RetrievedChunk] = self.vector_store_service.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                # filter_metadata=request.filter_metadata # If you add metadata filtering
            )
            logger.info(f"Retrieved {len(retrieved_chunks_from_vs)} chunks from vector store.")


            # Filter by similarity threshold
            final_retrieved_chunks: List[RetrievedChunk] = []
            if similarity_threshold is not None:
                original_count = len(retrieved_chunks_from_vs)
                final_retrieved_chunks = [
                    chunk for chunk in retrieved_chunks_from_vs if chunk.score >= similarity_threshold
                ]
                chunks_filtered_out = original_count - len(final_retrieved_chunks)
                logger.debug(f"Filtered {chunks_filtered_out} chunks by similarity threshold {similarity_threshold:.2f}. Kept {len(final_retrieved_chunks)}.")

            else: # No threshold, keep all retrieved
                final_retrieved_chunks = retrieved_chunks_from_vs
                logger.debug("No similarity threshold applied, keeping all retrieved chunks.")


            llm_answer, model_used = await self.llm_service.generate_response(
                query=request.query_text,
                context_chunks=final_retrieved_chunks,
                provider=llm_provider,
                model_name_override=request.llm_model_name,
            )
            logger.info(f"Successfully generated LLM answer for query: '{request.query_text[:50]}...'")


            return QueryResponse(
                query_text=request.query_text,
                llm_answer=llm_answer,
                llm_provider_used=llm_provider, # This should be the actual provider used
                llm_model_used=model_used,       # This should be the actual model used
                retrieved_chunks=final_retrieved_chunks,
            )
        except ConfigurationError as e: # Catch specific config errors if LLM provider is bad
            logger.error(f"Query failed due to configuration error: {e}", exc_info=True)
            raise
        except QueryProcessingError as e: # Catch errors from LLMService or other query stages
            logger.error(f"Query failed: Query processing error - {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during query processing: {e}", exc_info=True)
            raise QueryProcessingError(f"Unexpected query error: {str(e)}")


    async def get_system_status(self) -> SystemStatusResponse:
        """Checks the status of all major components of the RAG system."""
        logger.debug("Gathering system status...")
        components: List[ComponentStatus] = []
        overall_status = StatusEnum.OK # Assume OK initially

        # 1. Embedding Service Status
        emb_status = ComponentStatus(name="EmbeddingService", status=StatusEnum.UNKNOWN)
        try:
            dim = self.embedding_service.get_embedding_dimension()
            if dim and dim > 0:
                emb_status.status = StatusEnum.OK
                emb_status.message = f"Model '{self.embedding_service.model_name}' loaded. Dimension: {dim}."
                emb_status.details = {"model_name": self.embedding_service.model_name, "dimension": dim, "device": self.embedding_service.device}
            else:
                emb_status.status = StatusEnum.ERROR
                emb_status.message = "Embedding model not loaded or dimension is invalid."
        except Exception as e:
            logger.error(f"Error checking EmbeddingService status: {e}", exc_info=True)
            emb_status.status = StatusEnum.ERROR
            emb_status.message = str(e)
        components.append(emb_status)
        if emb_status.status != StatusEnum.OK:
            overall_status = StatusEnum.DEGRADED if overall_status == StatusEnum.OK else overall_status
            if emb_status.status == StatusEnum.ERROR: overall_status = StatusEnum.ERROR


        # 2. Vector Store Service Status
        vs_status_val = StatusEnum.UNKNOWN
        vs_message = "Not checked"
        vs_details = None
        try:
            stats = self.vector_store_service.get_collection_stats()
            if "error" in stats:
                vs_status_val = StatusEnum.ERROR
                vs_message = f"Error connecting or getting stats: {stats['error']}"
            else:
                vs_status_val = StatusEnum.OK
                vs_message = f"Connected to {self.vector_store_service.provider}. Collection: {stats.get('collection_name', stats.get('index_name', 'N/A'))}. Items: {stats.get('item_count', 'N/A')}."
                vs_details = stats
        except Exception as e:
            logger.error(f"Error checking VectorStoreService status: {e}", exc_info=True)
            vs_status_val = StatusEnum.ERROR
            vs_message = str(e)
        vs_status = ComponentStatus(name="VectorStoreService", status=vs_status_val, message=vs_message, details=vs_details)
        components.append(vs_status)
        if vs_status.status != StatusEnum.OK:
            overall_status = StatusEnum.DEGRADED if overall_status == StatusEnum.OK else overall_status
            if vs_status.status == StatusEnum.ERROR: overall_status = StatusEnum.ERROR


        # 3. LLM Service Status (check each configured provider)
        llm_service_overall_status = StatusEnum.OK
        llm_providers_to_check = []
        if self.settings.OPENAI_API_KEY and self.settings.OPENAI_API_KEY.get_secret_value() not in [None, "", "sk-your_openai_api_key_here"]:
            llm_providers_to_check.append(LLMProvider.OPENAI)
        if self.settings.ANTHROPIC_API_KEY and self.settings.ANTHROPIC_API_KEY.get_secret_value() not in [None, "", "sk-ant-your_anthropic_api_key_here"]:
            llm_providers_to_check.append(LLMProvider.ANTHROPIC)
        if self.settings.GOOGLE_API_KEY and self.settings.GOOGLE_API_KEY.get_secret_value() not in [None, "", "your_google_api_key_here"]:
            llm_providers_to_check.append(LLMProvider.GEMINI)

        if not llm_providers_to_check:
            no_llm_status = ComponentStatus(name="LLMService", status=StatusEnum.DEGRADED, message="No LLM providers appear to be configured with API keys.")
            components.append(no_llm_status)
            llm_service_overall_status = StatusEnum.DEGRADED
        else:
            for provider in llm_providers_to_check:
                try:
                    provider_health = await self.llm_service.check_provider_health(provider)
                    components.append(provider_health)
                    if provider_health.status != StatusEnum.OK:
                        llm_service_overall_status = StatusEnum.DEGRADED
                    if provider_health.status == StatusEnum.ERROR:
                        llm_service_overall_status = StatusEnum.ERROR # One error makes the whole LLM service error
                        break # No need to check further if one is in error state
                except Exception as e:
                    logger.error(f"Error checking LLM provider {provider.value} health: {e}", exc_info=True)
                    error_comp_status = ComponentStatus(name=f"{provider.value.capitalize()} LLM", status=StatusEnum.ERROR, message=str(e))
                    components.append(error_comp_status)
                    llm_service_overall_status = StatusEnum.ERROR
                    break

        if llm_service_overall_status != StatusEnum.OK:
            overall_status = StatusEnum.DEGRADED if overall_status == StatusEnum.OK and llm_service_overall_status == StatusEnum.DEGRADED else overall_status
            if llm_service_overall_status == StatusEnum.ERROR: overall_status = StatusEnum.ERROR


        # Construct final response
        status_response = SystemStatusResponse(
            system_status=overall_status,
            app_name=self.settings.APP_NAME,
            environment=self.settings.ENVIRONMENT,
            version=getattr(__import__("rag_system"), "__version__", "0.0.0"), # Get version dynamically
            components=components,
        )
        logger.debug(f"System status collected: {status_response.system_status}")
        return status_response
