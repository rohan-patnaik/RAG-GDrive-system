# RAG-GDrive-system

Retrieval-Augmented Generation (RAG) system that allows users to ingest documents, ask natural language questions about their content, and receive answers synthesized by a Large Language Model (LLM), augmented with relevant information retrieved from the ingested documents.

This system supports multiple LLM providers (OpenAI, Anthropic, Google Gemini), offers interaction via a REST API and CLI, and includes a plan for a Streamlit-based frontend.

**Current Date:** 5/29/2025

## Features

*   **Document Ingestion:** Load and process text documents (`.txt` files initially).
*   **Vector Embeddings:** Generate embeddings for document chunks using Sentence Transformers.
*   **Vector Store:** Store and retrieve document chunks using ChromaDB (persistent).
*   **Multi-LLM Support:**
    *   OpenAI (GPT models)
    *   Anthropic (Claude models)
    *   Google Gemini
*   **REST API:** FastAPI-based API for ingestion and querying.
*   **Command-Line Interface (CLI):** `click`-based CLI for easy interaction.
*   **Configurable:** Settings managed via `.env` files.
*   **Containerized:** Docker support for easy deployment.
*   **Testing:** Comprehensive unit and API tests using Pytest.
*   **Logging:** Structured logging to console and files.
*   **Planned Streamlit Frontend:** For rapid prototyping and user interaction.
*   **Planned Google Drive Integration:** Future capability to ingest documents from Google Drive.

## Project Structure

