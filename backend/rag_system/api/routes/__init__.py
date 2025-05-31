# backend/rag_system/api/routes/__init__.py
# This file makes Python treat the directory as a package.

from . import health
from . import documents
from . import query

__all__ = ["health", "documents", "query"]
