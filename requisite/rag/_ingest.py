"""
Shared ingest and result-formatting helpers for the three retrievers.

``Retriever``, ``HybridRetriever`` and ``BM25Retriever`` all chunk texts, attach
per-document metadata and assign chunk ids the same way; that logic lives here
once. See ``docs/adr/0042-access-controlled-retrieval.md``.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, Optional

from requisite.core.exceptions import ConfigurationException
from requisite.rag.base import ScoredChunk
from requisite.rag.chunking import chunk_text


def prepare_chunks(
    texts: Sequence[str],
    *,
    metadatas: Optional[Sequence[dict[str, Any]]],
    doc_ids: Optional[Sequence[str]],
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Chunk ``texts`` and return parallel ``(chunk_ids, pieces, metadata)`` lists.

    Without ``doc_ids`` each chunk gets a random ``uuid4`` id and exactly its
    document's metadata. With ``doc_ids`` the id is the deterministic
    ``"<doc_id>:<chunk_index>"`` -- so re-ingesting a document overwrites its
    chunks instead of duplicating them -- and the metadata also gets ``doc_id``
    and ``chunk_index`` (overriding any same-named keys the caller supplied),
    which is what a citation points at.
    """
    if metadatas is not None and len(metadatas) != len(texts):
        raise ConfigurationException("metadatas must be the same length as texts.")
    if doc_ids is not None:
        if len(doc_ids) != len(texts):
            raise ConfigurationException("doc_ids must be the same length as texts.")
        if len(set(doc_ids)) != len(doc_ids):
            raise ConfigurationException(
                "doc_ids must be unique within one add_texts call; a repeated id would "
                "overwrite its own earlier chunks.",
            )

    chunk_ids: list[str] = []
    pieces_out: list[str] = []
    metadata_out: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        base = dict(metadatas[index]) if metadatas is not None else {}
        for chunk_index, piece in enumerate(
            chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            metadata = dict(base)
            if doc_ids is not None:
                metadata["doc_id"] = doc_ids[index]
                metadata["chunk_index"] = chunk_index
                chunk_ids.append(f"{doc_ids[index]}:{chunk_index}")
            else:
                chunk_ids.append(str(uuid.uuid4()))
            pieces_out.append(piece)
            metadata_out.append(metadata)
    return chunk_ids, pieces_out, metadata_out


def format_results(results: Sequence[ScoredChunk], source_key: Optional[str] = None) -> str:
    """Render retrieval results as the text a tool returns to the model.

    Default: ``[score=0.812] text``. With ``source_key`` (a metadata key such as
    ``"doc_id"``) each line also carries the chunk's source so the model can cite
    it: ``[score=0.812 source=handbook-3#2] text`` (``#<chunk_index>`` when the
    chunk has one; ``source=unknown`` when the key is absent).
    """
    lines = []
    for result in results:
        if source_key is None:
            label = f"[score={result.score:.3f}]"
        else:
            source = result.chunk.metadata.get(source_key)
            if source is None:
                shown = "unknown"
            else:
                shown = str(source)
                chunk_index = result.chunk.metadata.get("chunk_index")
                if chunk_index is not None:
                    shown += f"#{chunk_index}"
            label = f"[score={result.score:.3f} source={shown}]"
        lines.append(f"{label} {result.chunk.text}")
    return "\n\n".join(lines)
