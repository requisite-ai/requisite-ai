"""
RAG (Retrieval-Augmented Generation).

Decomposes into independent extension points, per ADR-0001's note that
RAG isn't one interface: :class:`~requisite.rag.base.BaseEmbeddingProvider`,
:class:`~requisite.rag.base.BaseVectorStore`, and
:class:`~requisite.rag.base.BaseRetriever`. :class:`Retriever` is the
concrete, composed dense retriever most applications use.

A retriever is exposed to agents as a capability, not a new ``Agent``
parameter -- see :meth:`~requisite.rag.retriever.Retriever.as_tool`:

>>> from requisite.rag import Retriever
>>> from requisite.rag.embeddings import OpenAIEmbeddingProvider
>>> from requisite.rag.vectorstores import InMemoryVectorStore
>>> retriever = Retriever(
...     embedding_provider=OpenAIEmbeddingProvider(api_key="sk-..."),
...     vector_store=InMemoryVectorStore(),
... )  # doctest: +SKIP
>>> retriever.add_texts(["Paris is the capital of France."])  # doctest: +SKIP
>>> from requisite.capabilities import default_registry as capabilities
>>> capabilities.register("knowledge_base", retriever.as_tool())  # doctest: +SKIP

See ``docs/adr/0005-rag-integration.md`` for the full design.
"""

from requisite.rag.base import (
    BaseEmbeddingProvider,
    BaseRetriever,
    BaseVectorStore,
    Chunk,
    ScoredChunk,
)
from requisite.rag.chunking import chunk_text
from requisite.rag.factory import (
    EmbeddingRegistry,
    VectorStoreRegistry,
    default_embedding_registry,
    default_vector_store_registry,
)
from requisite.rag.retriever import Retriever

__all__ = [
    "Chunk",
    "ScoredChunk",
    "BaseEmbeddingProvider",
    "BaseVectorStore",
    "BaseRetriever",
    "Retriever",
    "chunk_text",
    "EmbeddingRegistry",
    "default_embedding_registry",
    "VectorStoreRegistry",
    "default_vector_store_registry",
]
