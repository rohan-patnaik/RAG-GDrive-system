# backend/rag_system/utils/constants.py

# Default RAG parameters (can be overridden by settings)
DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 200
DEFAULT_TOP_K: int = 3
DEFAULT_SIMILARITY_THRESHOLD: float = 0.7 # Example value

# Supported file types for ingestion (initially)
SUPPORTED_FILE_TYPES: list[str] = [".txt"] # Expand as new loaders are added (e.g., .pdf, .md)

# LLM Model Names (examples, actual defaults are in settings.py)
# These can be used for validation or reference if needed.
OPENAI_DEFAULT_MODEL = "gpt-3.5-turbo"
ANTHROPIC_DEFAULT_MODEL = "claude-3-haiku-20240307"
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash-preview-05-20"

# Other project-wide constants can be defined here.
# For example, specific metadata keys used across modules.
METADATA_SOURCE_ID_KEY = "source_id"
METADATA_FILENAME_KEY = "filename"

# Maximum number of characters for a context string to be sent to LLM
# This is a general guideline; specific models have token limits.
MAX_CONTEXT_LENGTH_CHARS = 15000

# Default ChromaDB collection name (also in settings, but can be here for reference)
# DEFAULT_CHROMA_COLLECTION = "rag_documents_v1"

# Log format string (if you want to centralize it, though logging_config.py handles it)
# LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"

# Environment variable names (for reference, primarily used in settings.py)
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
ENV_GOOGLE_API_KEY = "GOOGLE_API_KEY"

# Add more constants as your project grows.
