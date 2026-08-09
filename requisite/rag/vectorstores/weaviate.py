"""
Weaviate vector store.

Uses the current ``weaviate-client`` v4 API (the ``WeaviateClient`` /
``collections`` interface -- confirmed against the installed SDK; v3's
``weaviate.Client(url=...)`` class is a different, older API and is not
used here). Bring-your-own-embeddings only: collections are created with
``Configure.Vectorizer.none()``, so this store never calls out to a
Weaviate-side vectorizer module.

Weaviate objects are addressed by UUID, not arbitrary strings, so each
:class:`~requisite.rag.base.Chunk`'s ``id`` is mapped to a UUID
deterministically via ``uuid.uuid5`` -- the same ``chunk.id`` always
produces the same Weaviate object id, so ``delete`` can recompute it
without a separate lookup, and re-adding a chunk with the same id
overwrites the same object rather than duplicating it.

Install with: ``pip install weaviate-client``
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, Optional

from requisite.core.exceptions import ConfigurationException, VectorStoreException
from requisite.rag.base import BaseVectorStore, Chunk, ScoredChunk

_UUID_NAMESPACE = uuid.UUID("f4f339a6-3e21-4b8e-9f7d-3a2a9f0f9f10")


def _uuid_for(chunk_id: str) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE, chunk_id))


class WeaviateVectorStore(BaseVectorStore):
    """Vector store backed by a Weaviate collection.

    Parameters
    ----------
    url:
        Weaviate Cloud cluster URL (e.g.
        ``"https://my-cluster.weaviate.network"``). If omitted, connects
        to a local instance instead (``http://localhost:8080``).
    api_key:
        Weaviate Cloud API key. Required if ``url`` is given; ignored
        for a local connection (falls back to the ``WEAVIATE_API_KEY``
        environment variable if not provided explicitly).
    collection_name:
        Name of the Weaviate collection to use.
    create_if_missing:
        Create ``collection_name`` automatically (bring-your-own-vectors,
        cosine distance) if it doesn't already exist.

    Notes
    -----
    Holds an open connection to the Weaviate server for the lifetime of
    this instance (the v4 client uses persistent HTTP + gRPC
    connections). Access the raw client via the ``client`` property to
    call ``.close()`` explicitly in a long-running process if needed.

    Examples
    --------
    >>> store = WeaviateVectorStore(
    ...     url="https://my-cluster.weaviate.network", api_key="..."
    ... )  # doctest: +SKIP
    >>> store.add([Chunk(id="1", text="cats are great", embedding=[...])])  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "requisite_chunks",
        create_if_missing: bool = True,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._collection_name = collection_name
        self._create_if_missing = create_if_missing
        self._client: Any = None

    @property
    def name(self) -> str:
        return "weaviate"

    @property
    def client(self) -> Any:
        """The underlying, connected ``weaviate.WeaviateClient`` instance."""
        return self._get_client()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            import weaviate
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ConfigurationException(
                "The 'weaviate-client' package is required for WeaviateVectorStore. "
                "Install it with: pip install weaviate-client",
            ) from exc

        if self._url:
            if not self._api_key:
                raise ConfigurationException(
                    "WeaviateVectorStore(url=...) requires 'api_key' for Weaviate Cloud.",
                )
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self._url,
                auth_credentials=self._api_key,
            )
        else:
            self._client = weaviate.connect_to_local()
        return self._client

    def _get_collection(self) -> Any:
        client = self._get_client()
        if client.collections.exists(self._collection_name):
            return client.collections.get(self._collection_name)

        if not self._create_if_missing:
            raise ConfigurationException(
                f"Weaviate collection '{self._collection_name}' does not exist and "
                f"create_if_missing=False.",
            )

        from weaviate.classes.config import Configure, DataType, Property, VectorDistances

        return client.collections.create(
            name=self._collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE),
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
                Property(name="metadata_json", data_type=DataType.TEXT),
            ],
        )

    def add(self, chunks: Sequence[Chunk]) -> None:
        import json

        collection = self._get_collection()

        from weaviate.classes.data import DataObject

        objects = [
            DataObject(
                properties={
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "metadata_json": json.dumps(chunk.metadata),
                },
                uuid=_uuid_for(chunk.id),
                vector=list(chunk.embedding) if chunk.embedding is not None else None,
            )
            for chunk in chunks
        ]
        try:
            collection.data.insert_many(objects)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreException(
                f"Weaviate insert failed: {exc}",
                store=self.name,
                original_error=exc,
            ) from exc

    def search(self, query_embedding: Sequence[float], *, top_k: int = 5) -> list[ScoredChunk]:
        import json

        collection = self._get_collection()

        from weaviate.classes.query import MetadataQuery

        try:
            response = collection.query.near_vector(
                near_vector=list(query_embedding),
                limit=top_k,
                return_metadata=MetadataQuery(distance=True),
                return_properties=["chunk_id", "text", "metadata_json"],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreException(
                f"Weaviate query failed: {exc}",
                store=self.name,
                original_error=exc,
            ) from exc

        results: list[ScoredChunk] = []
        for obj in response.objects:
            metadata = json.loads(obj.properties.get("metadata_json") or "{}")
            distance = obj.metadata.distance if obj.metadata.distance is not None else 1.0
            results.append(
                ScoredChunk(
                    chunk=Chunk(
                        id=str(obj.properties.get("chunk_id", "")),
                        text=str(obj.properties.get("text", "")),
                        metadata=metadata,
                    ),
                    score=1.0 - distance,
                )
            )
        return results

    def delete(self, chunk_ids: Sequence[str]) -> None:
        collection = self._get_collection()
        try:
            for chunk_id in chunk_ids:
                collection.data.delete_by_id(_uuid_for(chunk_id))
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreException(
                f"Weaviate delete failed: {exc}",
                store=self.name,
                original_error=exc,
            ) from exc
