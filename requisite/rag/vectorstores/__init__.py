"""
Vector store implementations.

``InMemoryVectorStore`` ships in core (zero dependencies). Pinecone and
Weaviate integrations are planned (see ``ROADMAP.md``) but not yet
implemented -- ``BaseVectorStore`` (in ``requisite.rag.base``) is the
interface either would implement.
"""

from requisite.rag.vectorstores.in_memory import InMemoryVectorStore

__all__ = ["InMemoryVectorStore"]
