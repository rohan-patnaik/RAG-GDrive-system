# RAG GDrive System Architecture

This document provides an overview of the RAG GDrive System's architecture, its main components, and how they interact.

## System Overview

The RAG GDrive System is designed to:
1.  **Ingest** documents from various sources (initially local files, planned Google Drive).
2.  **Process** these documents by cleaning text, splitting into manageable chunks.
3.  **Embed** these chunks into vector representations.
4.  **Store** the chunks and their embeddings in a vector database (ChromaDB).
5.  **Retrieve** relevant chunks based on a user's natural language query.
6.  **Generate** a synthesized answer by providing the query and retrieved context to a Large Language Model (LLM).

The system supports multiple LLM providers (OpenAI, Anthropic, Google Gemini) and offers interaction via a REST API and a Command-Line Interface (CLI). A Streamlit frontend is planned for user interaction.

## Core Components (Backend)

The backend is built using Python and FastAPI.

```mermaid
graph TD
    User_CLI[CLI User] --> CLI_Interface[CLI (Click)];
    User_API[API User/Frontend] --> REST_API[REST API (FastAPI)];

    subgraph BackendApplication [Backend Application: RAG GDrive System]
        CLI_Interface --> RAGService;
        REST_API --> RAGService[RAG Service];

        RAGService --> DocumentLoader[Document Loader];
        RAGService --> TextProcessor[Text Processor];
        RAGService --> EmbeddingService[Embedding Service];
        RAGService --> VectorStoreService[Vector Store Service (ChromaDB)];
        RAGService --> LLMService[LLM Service];

        DocumentLoader -- Documents --> TextProcessor;
        TextProcessor -- Chunks --> EmbeddingService;
        TextProcessor -- Chunks --> VectorStoreService;
        EmbeddingService -- Embeddings for Query --> VectorStoreService;
        EmbeddingService -- Embeddings for Chunks --> VectorStoreService; (*If VS doesn't embed*)

        VectorStoreService -- Stores/Retrieves Chunks & Embeddings --> ChromaDB[(ChromaDB)];
        LLMService -- Manages LLM Clients --> OpenAIClient[OpenAI Client];
        LLMService --> AnthropicClient[Anthropic Client];
        LLMService --> GeminiClient[Google Gemini Client];
        LLMService --> LocalLLMClient[Local LLM Client (Future)];

        OpenAIClient -- Interacts --> OpenAI_API[OpenAI API];
        AnthropicClient -- Interacts --> Anthropic_API[Anthropic API];
        GeminiClient -- Interacts --> Google_Gemini_API[Google Gemini API];
    end

    Config[Configuration (.env, Pydantic Settings)] --> BackendApplication;
    Logging[Logging Service] --> BackendApplication;

    style RAGService fill:#f9f,stroke:#333,stroke-width:2px
    style REST_API fill:#ccf,stroke:#333,stroke-width:2px
    style CLI_Interface fill:#ccf,stroke:#333,stroke-width:2px
```

### 1. Configuration (`config/`)
   - **`settings.py`**: Uses `pydantic-settings` to load all configurations from environment variables (`.env` file). This includes API keys, model names, paths, RAG parameters, etc.
   - **`logging_config.py`**: Sets up structured logging for the application (console and file output).

### 2. Models (`models/`)
   - **`schemas.py`**: Defines Pydantic models for data validation and serialization. This includes:
     - API request/response schemas (e.g., `QueryRequest`, `IngestionResponse`).
     - Internal data structures (e.g., `Document`, `DocumentChunk`, `RetrievedChunk`).
     - Enums (e.g., `LLMProvider`, `StatusEnum`).

### 3. Core Components (`core/`)
   - **`document_loader.py`**: Responsible for loading documents from specified sources (e.g., local directories). Converts raw files into `Document` objects.
   - **`text_processor.py`**: Cleans loaded document content and splits it into smaller `DocumentChunk`s using techniques like `RecursiveCharacterTextSplitter`.
   - **`embeddings.py` (`EmbeddingService`):** Generates vector embeddings for text (document chunks and user queries) using a sentence-transformer model (e.g., `all-MiniLM-L6-v2`).
   - **`vector_store.py` (`VectorStoreService`):** Manages interactions with the vector database (ChromaDB). Responsibilities include:
     - Adding document chunks and their embeddings.
     - Searching for chunks similar to a query embedding.
     - Deleting chunks and managing the collection.
     - ChromaDB is configured to run locally and persist data to disk. It uses its own embedding function if embeddings are not pre-calculated and passed by `EmbeddingService`.

### 4. Integrations (`integrations/`)
   - **`openai_client.py`**: Client for OpenAI's API (GPT models).
   - **`anthropic_client.py`**: Client for Anthropic's API (Claude models).
   - **`gemini_client.py`**: Client for Google's Gemini API.
   - Each client encapsulates API key management, request formatting, and response parsing for its respective LLM provider. They typically offer a `generate_response(prompt)` method and a `check_health()` method.
   - **`google_drive.py` (Placeholder):** For future integration to load documents from Google Drive.

### 5. Services (`services/`)
   - **`llm_service.py` (`LLMService`):**
     - Manages instances of the different LLM clients (`OpenAIClient`, `AnthropicClient`, `GeminiClient`).
     - Selects the appropriate client based on user request or default configuration.
     - Constructs the final prompt by combining the user query with retrieved context chunks.
     - Calls the selected LLM client to get a synthesized answer.
   - **`rag_service.py` (`RAGService`):**
     - The central orchestrator for the RAG pipeline.
     - **Ingestion:** Coordinates `DocumentLoader`, `TextProcessor`, `EmbeddingService` (if needed for pre-computation, though Chroma can embed internally), and `VectorStoreService` to ingest and store documents.
     - **Querying:**
       1. Uses `EmbeddingService` to embed the user's query.
       2. Uses `VectorStoreService` to retrieve relevant `RetrievedChunk`s.
       3. Uses `LLMService` to generate a final answer based on the query and retrieved chunks.
     - Provides a method to check the overall system status (`get_system_status`).

### 6. API (`api/`)
   - **`app.py`**: Creates the FastAPI application instance, configures middleware (CORS, error handling), and includes routers. Manages application lifespan events (startup/shutdown) for initializing services.
   - **`routes/`**: Defines API endpoints:
     - `health.py`: `/health` endpoint for system health checks.
     - `documents.py`: `/documents/ingest` for document ingestion.
     - `query.py`: `/query/` for submitting queries and `/query/status` for query system status.
   - **`middleware.py` (Optional):** For custom middleware like request logging or future authentication.

### 7. CLI (`cli/`)
   - **`commands.py`**: Defines command-line interface commands using `Click`.
     - `ingest`: Triggers document ingestion.
     - `query`: Allows users to ask questions.
     - `status`: Checks system status.
   - Uses `rich` for formatted console output.
   - Interacts with `RAGService` to perform actions.

### 8. Utilities (`utils/`)
   - **`exceptions.py`**: Defines custom exception classes for better error handling.
   - **`helpers.py`**: General utility functions.
   - **`constants.py`**: Project-wide constants.

## Data Flow

### Ingestion Flow
1.  User (via API or CLI) initiates ingestion with a source path.
2.  `RAGService` calls `DocumentLoader` to load files into `Document` objects.
3.  `Document` objects are passed to `TextProcessor` to be cleaned and split into `DocumentChunk`s.
4.  `DocumentChunk`s (content and metadata) are passed to `VectorStoreService`.
5.  `VectorStoreService` (using ChromaDB with its internal SentenceTransformer embedding function) embeds the chunk content and stores the chunk (ID, content, metadata, embedding) in the ChromaDB collection.

### Query Flow
1.  User (via API or CLI) submits a query.
2.  `RAGService` receives the `QueryRequest`.
3.  `EmbeddingService` generates an embedding for the user's `query_text`.
4.  The query embedding is passed to `VectorStoreService.search_similar()`.
5.  `VectorStoreService` queries ChromaDB to find the `top_k` most similar `DocumentChunk`s (returned as `RetrievedChunk`s with scores).
6.  The original query and the list of `RetrievedChunk`s are passed to `LLMService`.
7.  `LLMService` constructs a prompt containing the query and the content of the retrieved chunks.
8.  `LLMService` selects the appropriate LLM client (OpenAI, Anthropic, or Gemini) based on the request or defaults.
9.  The selected LLM client sends the prompt to the respective LLM API and receives the generated answer.
10. `RAGService` packages the answer, source chunks, and other metadata into a `QueryResponse`.

## Frontend (Streamlit - Planned)

-   A separate Streamlit application will reside in the `frontend/` directory.
-   It will interact with the backend FastAPI application via HTTP requests (using `httpx`).
-   **`frontend/utils/api_client.py`**: Will contain functions to make calls to the backend API endpoints.
-   **Pages:**
    -   Querying interface.
    -   System status display.
    -   Document ingestion trigger (server-side path initially, potentially file upload later).

## Testing (`tests/`)

-   Uses `pytest`.
-   **Unit Tests:** For individual components in `core/`, `services/`, `integrations/` (mocking external calls).
-   **API Tests:** For FastAPI endpoints, using `httpx.AsyncClient` and mocking the `RAGService` layer.

## Containerization (Docker)

-   **`Dockerfile`**: Defines the image for the backend FastAPI application.
-   **`docker-compose.yml`**: Orchestrates the `rag-api` service. Mounts volumes for persistent data (ChromaDB, logs) and potentially sample documents.

This architecture aims for modularity, testability, and extensibility.
