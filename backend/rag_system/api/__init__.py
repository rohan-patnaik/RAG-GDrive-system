# backend/rag_system/api/__init__.py
# This file makes Python treat the directory as a package.
from .app import create_app

__all__ = ["create_app"]
