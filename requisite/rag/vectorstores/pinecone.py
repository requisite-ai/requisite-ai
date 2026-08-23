"""
Pinecone vector store.

Uses the current ``pinecone`` package (v9+; the older ``pinecone-client``
distribution and its ``environment=`` constructor argument are gone --
confirmed against the installed SDK). Serverless indexes are addressed
by ``cloud``/``region`` (via ``ServerlessSpec``), not the legacy
``environment`` string; ``.env.example``'s ``PINECONE_ENVIRONMENT``
placeholder predates this and is not read by this implementation.

Pinecone vectors carry a flat metadata dict and no native text field, so
a chunk's ``text`` is stored under a reserved ``"_text"`` metadata key
and split back out on read -- callers' own ``Chunk.metadata`` never
needs to know this key exists.

Install with: ``pip install pinecone``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from requisite.core.exceptions import ConfigurationException, VectorStoreException
from requisite.rag.base import BaseVectorStore, Chunk, ScoredChunk

_TEXT_METADATA_KEY = "_text"


class PineconeVectorStore(BaseVectorStore):
    """Vector store backed by a Pinecone serverless index.

    Parameters
    ----------
    api_key:
        Pinecone API key. Falls back to the ``PINECONE_API_KEY``
        environment variable if not provided (handled by the SDK).
    index_name:
        Name of the Pinecone index to use.
    namespace:
        Pinecone namespace to scope reads/writes to. Defaults to the
        account's default namespace.
    dimension:
        Embedding dimensionality. Required only if ``index_name`` doesn't
        exist yet and ``create_if_missing`` is true -- Pinecone indexes
        are created with a fixed dimension.
    metric:
        Similarity metric for a newly created index (``"cosine"``,
        ``"dotproduct"``, or ``"euclidean"``).
    cloud:
        Cloud provider for a newly created serverless index.
    region:
        Cloud region for a newly created serverless index.
    create_if_missing:
        Create ``index_name`` automatically (as a serverless index) if it
        doesn't already exist. Set to ``False`` if the index is
        provisioned out-of-band and this store should only ever attach
        to it, never create it.

    Examples
    --------
    >>> store = PineconeVectorStore(
    ...     api_key="...", index_name="requisite-demo", dimension=1536
    ... )  # doctest: +SKIP
    >>> store.add([Chunk(id="1", text="cats are great", embedding=[...])])  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        index_name: str,
        namespace: str = "",
        dimension: Optional[int] = None,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        create_if_missing: bool = True,
    ) -> None:
        self._api_key = api_key
        self._index_name = index_name
        self._namespace = namespace
        self._dimension = dimension
        self._metric = metric
        self._cloud = cloud
        self._region = region
        self._create_if_missing = create_if_missing
        self._client: Any = None
        self._index: Any = None

    @property
    def name(self) -> str:
        return "pinecone"

    def _get_index(self) -> Any:
        if self._index is not None:
            return self._index

        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ConfigurationException(
                "The 'pinecone' package is required for PineconeVectorStore. "
                "Install it with: pip install pinecone",
            ) from exc

        if self._client is None:
            self._client = Pinecone(api_key=self._api_key)

        if not self._client.has_index(self._index_name):
            if not self._create_if_missing:
                raise ConfigurationException(
                    f"Pinecone index '{self._index_name}' does not exist and "
                    f"create_if_missing=False.",
                )
            if self._dimension is None:
                raise ConfigurationException(
                    f"Pinecone index '{self._index_name}' does not exist and "
                    f"'dimension' was not provided -- required to create it.",
                )
            self._client.create_index(
                name=self._index_name,
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
                dimension=self._dimension,
                metric=self._metric,
            )

        self._index = self._client.index(name=self._index_name)
        return self._index

    def add(self, chunks: Sequence[Chunk]) -> None:
        index = self._get_index()
        vectors = [
            (chunk.id, chunk.embedding, {**chunk.metadata, _TEXT_METADATA_KEY: chunk.text})
            for chunk in chunks
        ]
        try:
            index.upsert(vectors=vectors, namespace=self._namespace)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreException(
                f"Pinecone upsert failed: {exc}",
                store=self.name,
                original_error=exc,
            ) from exc

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        filter: Optional[dict[str, Any]] = None,  # noqa: A002
    ) -> list[ScoredChunk]:
        # Matches InMemoryVectorStore/WeaviateVectorStore's short-circuit
        # (docs/adr/0022-vector-memory.md) -- without it, top_k<=0 goes
        # straight to the live Pinecone service, whose behavior for a
        # non-positive top_k is unspecified here, instead of the
        # locally-guaranteed [] the other two stores promise uniformly.
        if top_k <= 0:
            return []

        index = self._get_index()
        try:
            response = index.query(
                vector=list(query_embedding),
                top_k=top_k,
                namespace=self._namespace,
                include_metadata=True,
                filter=filter,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreException(
                f"Pinecone query failed: {exc}",
                store=self.name,
                original_error=exc,
            ) from exc

        results: list[ScoredChunk] = []
        for match in response.matches:
            metadata = dict(match.metadata or {})
            text = metadata.pop(_TEXT_METADATA_KEY, "")
            results.append(
                ScoredChunk(
                    chunk=Chunk(id=match.id, text=text, metadata=metadata),
                    score=match.score,
                )
            )
        return results

    def delete(self, chunk_ids: Sequence[str]) -> None:
        index = self._get_index()
        try:
            index.delete(ids=list(chunk_ids), namespace=self._namespace)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreException(
                f"Pinecone delete failed: {exc}",
                store=self.name,
                original_error=exc,
            ) from exc
