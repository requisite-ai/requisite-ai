"""
LLM-based context compression: shrink an already-retrieved candidate
list's text to whatever is relevant to the query, with a single
structured-output call to any already-integrated provider.

Listwise, not pointwise -- one ``AI.chat_response(response_model=...)``
call compresses every candidate at once, mirroring
:class:`~requisite.rag.reranker.LLMReranker`'s same reasoning: reuse the
framework's own ``AI`` facade and ``response_model=`` structured output
instead of a summarization-specific ML dependency -- see
``docs/adr/0010-hybrid-bm25-retrieval-and-reranking.md`` (reranking) and
``docs/adr/0024-rag-context-compression.md`` (this feature).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from pydantic import BaseModel

from requisite.ai import AI
from requisite.rag.base import BaseCompressor, ScoredChunk

_PROMPT_TEMPLATE = """\
Query: {query}

For each of the following passages, extract only the sentences or \
phrases relevant to answering the query. Condense freely -- do not \
copy irrelevant surrounding text. If a passage has nothing relevant to \
the query, return an empty string for it. Return one entry for every \
passage listed below, identified by its id.

{candidates}"""


class _CompressedPassage(BaseModel):
    chunk_id: str
    compressed_text: str


class _CompressionResult(BaseModel):
    compressed: list[_CompressedPassage]


def _build_prompt(query: str, results: Sequence[ScoredChunk]) -> str:
    candidates = "\n\n".join(f"[id={r.chunk.id}]\n{r.chunk.text}" for r in results)
    return _PROMPT_TEMPLATE.format(query=query, candidates=candidates)


def _apply_compression(results: Sequence[ScoredChunk], parsed: object) -> list[ScoredChunk]:
    # ChatResponse.parsed isn't consistently one type across providers
    # (a validated model instance for some, a plain dict for others) --
    # model_validate() accepts either.
    compression_result = _CompressionResult.model_validate(parsed)
    compressed_text_by_id = {p.chunk_id: p.compressed_text for p in compression_result.compressed}

    compressed: list[ScoredChunk] = []
    for result in results:
        text = compressed_text_by_id.get(result.chunk.id)
        if not text:
            continue
        compressed.append(
            ScoredChunk(chunk=result.chunk.model_copy(update={"text": text}), score=result.score)
        )
    return compressed


class LLMContextCompressor(BaseCompressor):
    """Compresses a candidate list's text via one structured-output LLM call.

    Parameters
    ----------
    ai:
        An :class:`~requisite.ai.AI` instance to use. If omitted, one is
        built from ``provider``/``model`` (falling back to the
        framework's normal default-provider resolution if those are
        also omitted).
    provider, model:
        Passed to :class:`~requisite.ai.AI` when ``ai`` isn't given
        directly -- the common case of not wanting to build one by hand.

    Examples
    --------
    >>> compressor = LLMContextCompressor(provider="openai")  # doctest: +SKIP
    >>> candidates = retriever.retrieve(query, top_k=20)  # doctest: +SKIP
    >>> compressed = compressor.compress(query, candidates)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        ai: Optional[AI] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._ai = ai or AI(provider=provider, model=model)

    def compress(self, query: str, results: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        if not results:
            return []
        response = self._ai.chat_response(
            _build_prompt(query, results), response_model=_CompressionResult
        )
        return _apply_compression(results, response.parsed)

    async def acompress(self, query: str, results: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        if not results:
            return []
        response = await self._ai.achat_response(
            _build_prompt(query, results), response_model=_CompressionResult
        )
        return _apply_compression(results, response.parsed)
