# Setup and Installation Guide for RAG GDrive System

This guide provides instructions for setting up the RAG GDrive System for development and deployment.

## Prerequisites

*   **Python:** Version 3.9 or higher.
*   **pip:** Python package installer.
*   **Git:** For cloning the repository.
*   **Docker & Docker Compose:** (Optional) For containerized deployment.
*   **API Keys:** You will need API keys for the LLM providers you intend to use:
    *   OpenAI
    *   Anthropic
    *   Google (for Gemini API)

## 1. Clone the Repository

```bash
git clone <your_repository_url>
cd RAG-GDrive-system
```

## 2. Environment Configuration

The system uses a `.env` file to manage environment variables, including API keys and other configurations.

1.  Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
2.  Edit the `.env` file with your actual API keys and desired settings:
    ```bash
    # Example content of .env
    OPENAI_API_KEY="sk-your_openai_api_key"
    ANTHROPIC_API_KEY="sk-ant-your_anthropic_api_key"
    GOOGLE_API_KEY="your_google_gemini_api_key"

    DEFAULT_LLM_PROVIDER="gemini" # or openai, anthropic
    EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_STORE_PATH="./data/vector_store"
    # ... and other settings as defined in .env.example
    ```
    **Important:** Do not commit your actual `.env` file to version control. It's included in `.gitignore`.

## 3. Setup Options

Choose one of the following setup options:

### Option A: Local Python Environment (Recommended for Development)

This method uses a Python virtual environment.

1.  **Run the Setup Script:**
    The `scripts/setup_environment.sh` script automates the creation of a virtual environment and installation of dependencies.
    Make it executable first:
    ```bash
    chmod +x scripts/setup_environment.sh
    ./scripts/setup_environment.sh
    ```
    This script will:
    *   Check your Python version.
    *   Create a virtual environment in `.venv/`.
    *   Activate it.
    *   Install dependencies from `requirements.txt` (which includes main, dev, and Streamlit dependencies).
    *   Install the project in editable mode (`pip install -e .[dev,streamlit]`).
    *   Copy `.env.example` to `.env` if `.env` doesn't exist.

2.  **Manual Setup (if script fails or for understanding):**
    *   Create a virtual environment:
        ```bash
        python3 -m venv .venv
        ```
    *   Activate the virtual environment:
        *   On macOS and Linux: `source .venv/bin/activate`
        *   On Windows: `.venv\Scripts\activate`
    *   Install dependencies:
        ```bash
        pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e ".[dev,streamlit]" # For editable install and dev/streamlit extras
        ```

### Option B: Docker (Recommended for Deployment or Isolated Environment)

This method uses Docker and Docker Compose to run the application in containers.

1.  **Ensure Docker and Docker Compose are installed.**
2.  **Make sure you have a `.env` file** in the project root directory (as described in Step 2). Docker Compose will use this file.
3.  **Build and run the containers:**
    ```bash
    docker-compose up --build
    ```
    This command will:
    *   Build the Docker image for the `rag-api` service as defined in `Dockerfile`.
    *   Start the service.
    *   The API will be accessible at `http://localhost:PORT` (e.g., `http://localhost:8000`, depending on `API_PORT` in your `.env`).
    *   Persistent data for ChromaDB (`data/vector_store/`) and logs (`logs/`) will be mounted as volumes.

    To run in detached mode:
    ```bash
    docker-compose up --build -d
    ```
    To stop the services:
    ```bash
    docker-compose down
    ```

## 4. Prepare Sample Documents (Optional)

The system comes with a `data/sample_documents/` directory. You can place your own `.txt` files there for initial testing.
Two sample files (`sample_doc_1.txt`, `sample_doc_2.txt`) are provided.

## 5. Running the Application

### API Server

*   **If using Local Python Environment:**
    Ensure your virtual environment is activated.
    ```bash
    uvicorn rag_system.api.app:app --host 0.0.0.0 --port 8000 --reload --app-dir backend/
    ```
    (Adjust host and port as needed, or rely on `.env` variables if Uvicorn is configured to read them, though FastAPI settings handle this internally).
*   **If using Docker:**
    The API server is automatically started by `docker-compose up`.

The API documentation (Swagger UI) will be available at `http://localhost:8000/docs`.

### Command-Line Interface (CLI)

*   **If using Local Python Environment (and `pip install -e .` was run):**
    Ensure your virtual environment is activated.
    ```bash
    rag-system --help
    rag-system ingest data/sample_documents/
    rag-system query "What is AI?"
    ```
*   **If using Docker:**
    You can execute CLI commands inside the running `rag-api` container:
    ```bash
    docker-compose exec rag-api rag-system --help
    docker-compose exec rag-api rag-system ingest data/sample_documents/ # Path is relative to inside the container
    docker-compose exec rag-api rag-system query "What is AI?"
    ```
    Alternatively, if you only want to run a CLI command without starting the server:
    ```bash
    docker-compose run --rm rag-api rag-system ingest data/sample_documents/
    ```

### Streamlit Frontend (Planned)

Once the backend API is running:
1.  Navigate to the `frontend/` directory.
2.  Ensure your virtual environment (with Streamlit installed) is active.
3.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
    The Streamlit interface will open in your web browser.

## 6. Testing

To run the automated tests (unit and API tests):

1.  Ensure your development environment is set up and activated (Option A).
2.  Make sure you have a `.env` file, as some test configurations might rely on it (even if external calls are mocked).
3.  Run pytest from the project root directory:
    ```bash
    pytest
    ```
    To run with coverage:
    ```bash
    pytest --cov=rag_system --cov-report=html  # Target backend/rag_system for coverage
    ```
    The coverage report will be in `htmlcov/index.html`.

## Troubleshooting

*   **`ModuleNotFoundError`:** Ensure your virtual environment is activated and all dependencies are installed. If you installed the project with `pip install -e .`, the `rag_system` package should be discoverable. Check your `PYTHONPATH` if issues persist.
*   **API Key Errors:** Double-check that your API keys in the `.env` file are correct and have the necessary permissions for the respective LLM services.
*   **ChromaDB Issues:** Ensure the `VECTOR_STORE_PATH` directory is writable. If you encounter persistent issues, try deleting the contents of this directory (e.g., `data/vector_store/`) and restarting the application to let ChromaDB reinitialize.
*   **Docker Issues:** Check Docker daemon status. Ensure ports are not already in use. View container logs with `docker-compose logs rag-api`.

For further assistance, please refer to the project's `README.md` or open an issue in the repository.
