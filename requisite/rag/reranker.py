"""
LLM-based re-ranking: re-score an already-retrieved candidate list with a
single structured-output call to any already-integrated provider.

Listwise, not pointwise -- one ``AI.chat_response(response_model=...)``
call scores every candidate at once, rather than one call per candidate.
Deliberately reuses the framework's own ``AI`` facade and
``response_model=`` structured output instead of a cross-encoder ML
dependency (``sentence-transformers`` et al.) -- see
``docs/adr/0010-hybrid-bm25-retrieval-and-reranking.md`` for the full
rationale.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from pydantic import BaseModel, Field

from requisite.ai import AI
from requisite.rag.base import BaseReranker, ScoredChunk

_PROMPT_TEMPLATE = """\
Query: {query}

Rate how relevant each of the following passages is to the query, on a \
scale from 0 (completely irrelevant) to 10 (perfectly relevant). Return \
a score for every passage listed below, identified by its id.

{candidates}"""


class _ChunkRelevance(BaseModel):
    chunk_id: str
    relevance: float = Field(ge=0.0, le=10.0)


class _RerankScores(BaseModel):
    scores: list[_ChunkRelevance]


def _build_prompt(query: str, results: Sequence[ScoredChunk]) -> str:
    candidates = "\n\n".join(f"[id={r.chunk.id}]\n{r.chunk.text}" for r in results)
    return _PROMPT_TEMPLATE.format(query=query, candidates=candidates)


def _apply_scores(
    results: Sequence[ScoredChunk], parsed: object, *, top_k: int
) -> list[ScoredChunk]:
    # ChatResponse.parsed isn't consistently one type across providers
    # (a validated model instance for some, a plain dict for others) --
    # model_validate() accepts either.
    rerank_scores = _RerankScores.model_validate(parsed)
    relevance_by_id = {s.chunk_id: s.relevance for s in rerank_scores.scores}

    rescored = [
        ScoredChunk(chunk=r.chunk, score=relevance_by_id.get(r.chunk.id, 0.0)) for r in results
    ]
    rescored.sort(key=lambda sc: sc.score, reverse=True)
    return rescored[:top_k]


class LLMReranker(BaseReranker):
    """Re-ranks a candidate list via one structured-output LLM call.

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
    >>> reranker = LLMReranker(provider="openai")  # doctest: +SKIP
    >>> candidates = retriever.retrieve(query, top_k=20)  # doctest: +SKIP
    >>> best = reranker.rerank(query, candidates, top_k=5)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        ai: Optional[AI] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._ai = ai or AI(provider=provider, model=model)

    def rerank(
        self, query: str, results: Sequence[ScoredChunk], *, top_k: Optional[int] = None
    ) -> list[ScoredChunk]:
        if not results:
            return []
        response = self._ai.chat_response(
            _build_prompt(query, results), response_model=_RerankScores
        )
        return _apply_scores(results, response.parsed, top_k=top_k or len(results))

    async def arerank(
        self, query: str, results: Sequence[ScoredChunk], *, top_k: Optional[int] = None
    ) -> list[ScoredChunk]:
        if not results:
            return []
        response = await self._ai.achat_response(
            _build_prompt(query, results), response_model=_RerankScores
        )
        return _apply_scores(results, response.parsed, top_k=top_k or len(results))
