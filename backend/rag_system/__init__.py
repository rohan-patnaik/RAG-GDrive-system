# backend/rag_system/__init__.py
"""
RAG System - Production-Ready Retrieval-Augmented Generation System

A comprehensive RAG system supporting multiple LLM providers (OpenAI, Anthropic, Google Gemini)
with document ingestion, vector storage, and natural language querying capabilities.
"""

__version__ = "0.1.0"
__author__ = "RAG Development Team"
__email__ = "team@ragdev.com"
__description__ = "Production-Ready RAG Document System with Multi-LLM Support"
__url__ = "https://github.com/yourorg/rag-system"

# Version info tuple for programmatic access
VERSION_INFO = tuple(map(int, __version__.split('.')))

# Package metadata
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "__url__",
    "VERSION_INFO",
]

# Ensure proper imports for type checking
import sys
if sys.version_info < (3, 9):
    raise RuntimeError("RAG System requires Python 3.9 or higher")

# Optional: Set up package-level logging configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
