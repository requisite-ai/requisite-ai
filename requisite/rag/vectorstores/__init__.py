"""
Vector store implementations.

``InMemoryVectorStore`` ships in core (zero dependencies) and is the
only one imported eagerly here. ``PineconeVectorStore`` and
``WeaviateVectorStore`` need their optional SDKs installed (``pip
install requisite-ai[pinecone]`` / ``[weaviate]``), so -- like every
optional provider -- import them from their own submodule directly
(``from requisite.rag.vectorstores.pinecone import PineconeVectorStore``)
or via ``default_vector_store_registry.create("pinecone", ...)``, not
from this package's ``__init__``, so simply importing
``requisite.rag.vectorstores`` never requires either SDK to be installed.
"""

from requisite.rag.vectorstores.in_memory import InMemoryVectorStore

__all__ = ["InMemoryVectorStore"]
