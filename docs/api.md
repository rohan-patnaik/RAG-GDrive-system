# RAG GDrive System API Documentation

This document provides examples and details for interacting with the RAG GDrive System REST API.
The API is served using FastAPI and provides endpoints for document ingestion, querying, and system health checks.

**Base URL:** `http://localhost:8000` (or your configured host and port)

**Interactive API Docs (Swagger UI):** [`/docs`](http://localhost:8000/docs)
**Alternative API Docs (ReDoc):** [`/redoc`](http://localhost:8000/redoc)

## Authentication

Currently, the API does not implement authentication. This should be added for production deployments.

## Common Headers

-   `Content-Type: application/json` (for POST/PUT requests with a JSON body)
-   `Accept: application/json` (if you expect a JSON response)

## Endpoints

### 1. Health Check

-   **Endpoint:** `GET /health`
-   **Description:** Checks the operational status of the RAG system and its components.
-   **Success Response (200 OK):**
    ```json
    {
      "system_status": "OK",
      "timestamp": "2025-05-29T10:00:00.123456Z",
      "app_name": "RAG GDrive System",
      "environment": "development",
      "version": "0.1.0",
      "components": [
        {
          "name": "EmbeddingService",
          "status": "OK",
          "message": "Model 'sentence-transformers/all-MiniLM-L6-v2' loaded.",
          "details": {"model_name": "sentence-transformers/all-MiniLM-L6-v2", "dimension": 384}
        },
        {
          "name": "VectorStoreService",
          "status": "OK",
          "message": "Connected to collection 'rag_documents_v1'.",
          "details": {"item_count": 152, "collection_name": "rag_documents_v1"}
        },
        {
          "name": "Gemini LLM",
          "status": "OK",
          "message": "Successfully connected and received test response from model gemini-2.5-flash-preview-05-20.",
          "details": {"default_model_tested": "gemini-2.5-flash-preview-05-20"}
        }
      ]
    }
    ```
-   **Error Response:** The endpoint itself should always return 200 OK if reachable. The `system_status` and `components` fields in the JSON body will indicate issues.

### 2. Document Ingestion

-   **Endpoint:** `POST /documents/ingest`
-   **Description:** Ingests documents from a specified server-side directory.
-   **Request Body:**
    ```json
    {
      "source_directory": "data/sample_documents",
      "file_patterns": ["*.txt", "*.md"], // Optional, defaults to settings
      "recursive": true // Optional, defaults to settings
    }
    ```
-   **Success Response (201 Created):**
    ```json
    {
      "message": "Ingestion completed. Processed 2 documents, added/updated 25 chunks.",
      "documents_processed": 2,
      "chunks_added": 25,
      "errors": []
    }
    ```
-   **Partial Success Response (201 Created or 207 Multi-Status):**
    ```json
    {
      "message": "Ingestion completed. Processed 5 documents, added/updated 40 chunks. Some issues encountered.",
      "documents_processed": 5,
      "chunks_added": 40,
      "errors": ["Failed to process document: data/sample_documents/corrupt.txt - InvalidEncoding"]
    }
    ```
-   **Error Responses:**
    -   `400 Bad Request`: If `source_directory` is invalid or other parameters are malformed.
    -   `404 Not Found`: If `source_directory` does not exist.
    -   `500 Internal Server Error`: For unexpected errors during processing.
    -   `503 Service Unavailable`: If a critical backend service (like VectorDB) is down.

    Example `curl` command:
    ```bash
    curl -X POST "http://localhost:8000/documents/ingest" \
    -H "Content-Type: application/json" \
    -d '{
      "source_directory": "data/sample_documents",
      "file_patterns": ["*.txt"],
      "recursive": true
    }'
    ```

### 3. Query System

-   **Endpoint:** `POST /query/`
-   **Description:** Submits a natural language query to the RAG system.
-   **Request Body:**
    ```json
    {
      "query_text": "What is Retrieval-Augmented Generation?",
      "llm_provider": "gemini", // Optional (openai, anthropic, gemini). Defaults to system setting.
      "llm_model_name": "gemini-1.5-pro-latest", // Optional. Specific model for the provider.
      "top_k": 3, // Optional. Number of chunks to retrieve. Defaults to system setting.
      "similarity_threshold": 0.7 // Optional. Min similarity for chunks. Defaults to system setting.
    }
    ```
-   **Success Response (200 OK):**
    ```json
    {
      "query_text": "What is Retrieval-Augmented Generation?",
      "llm_answer": "Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models by allowing them to access and incorporate information from external knowledge bases...",
      "llm_provider_used": "gemini",
      "llm_model_used": "gemini-1.5-pro-latest",
      "retrieved_chunks": [
        {
          "id": "doc1_chunk_5",
          "content": "RAG combines retrieval with generation...",
          "metadata": {
            "source_id": "data/sample_documents/ai_evolution.txt",
            "filename": "ai_evolution.txt",
            "path": "data/sample_documents",
            "chunk_number": 5,
            "total_chunks": 10
          },
          "score": 0.895
        }
        // ... more chunks
      ]
    }
    ```
-   **Error Responses:**
    -   `422 Unprocessable Entity`: If the request body is invalid (e.g., missing `query_text`).
    -   `500 Internal Server Error`: For unexpected errors during query processing.
    -   `502 Bad Gateway` or `503 Service Unavailable`: If the selected LLM provider or a core component (embedding, vector store) fails.

    Example `curl` command:
    ```bash
    curl -X POST "http://localhost:8000/query/" \
    -H "Content-Type: application/json" \
    -d '{
      "query_text": "Tell me about Python for data science",
      "llm_provider": "openai",
      "top_k": 2
    }'
    ```

### 4. Query System Status

-   **Endpoint:** `GET /query/status`
-   **Description:** Provides the operational status of query-related components. Similar to `/health` but can be more focused or provide different details if needed. (Currently, it might return the same as `/health`).
-   **Success Response (200 OK):** Same format as `/health`.

## Error Handling

The API uses standard HTTP status codes. Error responses will typically include a JSON body with `error`, `message`, and `detail` fields.

```json
// Example Error Response (e.g., 404)
{
  "detail": "Source directory not found: /path/to/nonexistent/docs"
}

// Example Error Response (e.g., 500 from custom exception)
{
  "error": "QueryProcessingError",
  "message": "An error occurred while processing the query.",
  "detail": "Underlying vector store search failed."
}
```

Further details and specific error codes for LLM providers might be included in the `detail` field.
