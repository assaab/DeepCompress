"""
External service integrations (cache, vector DB, LLM).
"""

from deepcompress.integrations.cache import CacheManager
from deepcompress.integrations.llm import LLMClient
from deepcompress.integrations.vector_db import VectorDBClient

__all__ = [
    "CacheManager",
    "LLMClient",
    "VectorDBClient",
]

