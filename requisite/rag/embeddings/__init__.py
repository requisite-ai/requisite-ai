"""
Embedding provider implementations.

Lazy-imported SDKs, same pattern as ``requisite.providers`` -- importing
this package never requires ``openai`` or ``google-genai`` to be installed,
only using that specific embedding provider does.
"""

from requisite.rag.embeddings.gemini import GeminiEmbeddingProvider
from requisite.rag.embeddings.openai import OpenAIEmbeddingProvider

__all__ = ["GeminiEmbeddingProvider", "OpenAIEmbeddingProvider"]
