
# 0010. Hybrid/BM25 retrieval and re-ranking

Status: Accepted
Date: 2026-08-11

## Context

`ROADMAP.md`/`FEATURES.md` listed "Hybrid / BM25 retriever" and
"Re-ranking" as 📋. `BaseRetriever` (`requisite/rag/base.py`) was
deliberately kept independent of `BaseEmbeddingProvider`/`BaseVectorStore`
at the interface level specifically so a keyword-based retriever could
be added "without interface changes" -- stated in the module docstring,
the class docstring, and ADR-0005's own Follow-ups section. This ADR is
that anticipated follow-up. Re-ranking had no prior design notes
anywhere in the docs (confirmed: no mention in ADR-0005, `ARCHITECTURE.md`
only names it in passing as a future extension point) -- its shape is
designed from scratch here.

## Decision

### BM25 implemented in pure Python, not `rank-bm25`

`requisite/rag/bm25.py`'s `BM25Index` implements Okapi BM25 directly
(~50 lines: tokenize, term frequency, IDF, the standard scoring formula)
rather than depending on the `rank-bm25` PyPI package (which itself
pulls in numpy). This mirrors `InMemoryVectorStore`
(`requisite/rag/vectorstores/in_memory.py`)'s own precedent of a
pure-Python cosine similarity implementation over a numpy dependency --
the algorithm is simple and well-specified enough that a dependency
isn't justified, and it means `BM25Retriever` ships as a zero-dependency
default (importable and usable with nothing beyond the base
`requisite-ai` install), the same tier as `InMemoryVectorStore` and
`SQLiteMemory`.

Tokenization is deliberately simple: lowercase + `\w+` word splitting,
no stemming, no stopword removal. This is a conscious floor, not an
oversight -- `chunk_text()` already defers "token-aware chunking" the
same way (`ROADMAP.md`), and a smarter tokenizer can be added later
without changing `BM25Index`'s public shape.

### `HybridRetriever` holds its own `BM25Index`, not a `BM25Retriever`

`HybridRetriever` (`requisite/rag/hybrid_retriever.py`) composes an
`embedding_provider` + `vector_store` (identical constructor shape to
`Retriever`) plus a private `BM25Index` instance -- not a separate
`BM25Retriever`. Reciprocal Rank Fusion (below) merges the dense and
BM25 result lists **by chunk id**. If `add_texts` delegated to two
independently-chunking retrievers, each would generate its own `uuid4`
chunk ids for what is conceptually the same piece of text, and fusion
would never recognize the same chunk across both lists. Chunking once
in `HybridRetriever.add_texts` and feeding the identical `Chunk` objects
to both `vector_store.add()` and the BM25 index is what makes id-based
fusion correct.

### Reciprocal Rank Fusion, not a normalized weighted-score blend

`_reciprocal_rank_fusion` in `hybrid_retriever.py`:
`score(chunk) = Σ 1/(k + rank)` across every result list the chunk
appears in (1-indexed rank; `k=60`, the constant from the original
Cormack et al. RRF paper, matching its common default use elsewhere).
Chosen specifically because dense cosine similarity (bounded `[-1, 1]`)
and BM25 scores (unbounded, corpus-dependent magnitude) are not on
comparable scales -- any score-based fusion would need its own
normalization scheme (min-max, z-score, ...) with its own edge cases
(what if a result list is empty? what if all scores in a list are
identical?). RRF sidesteps the scale-mismatch problem entirely by using
only rank position, which is directly comparable across arbitrarily
different scoring functions -- the standard technique for exactly this
dense+sparse fusion problem in production hybrid search systems.

`retrieve`/`aretrieve` over-fetch a `candidate_pool_size` (default 20)
from each side before fusing, rather than fusing each side's already-
truncated `top_k` -- so a document ranked, say, 8th by BM25 and 3rd
dense still has a chance to surface in the fused top 5, instead of being
truncated away by one side before fusion even sees it.

### Re-ranking: `BaseReranker` + `LLMReranker`, listwise via the existing `AI` facade

`BaseReranker` (`requisite/rag/base.py`) is a minimal ABC -- one
abstract `rerank(query, results, *, top_k=None) -> list[ScoredChunk]`
plus a thread-wrapped `arerank` default -- the same shape as
`BaseRetriever`. No registry: retrievers don't have one either
(`BaseRetriever` implementations are constructed directly, unlike
`BaseVectorStore`/`BaseEmbeddingProvider`), and there's exactly one
implementation shipping now.

`LLMReranker` (`requisite/rag/reranker.py`) reuses the framework's own
`AI` facade and `response_model=` structured output -- one
`chat_response`/`achat_response(response_model=_RerankScores)` call
scores every candidate at once (**listwise**, not pointwise/one-call-
per-candidate), which is both cheaper (one LLM call regardless of
candidate count) and matches how a reranker is meant to be used: over a
small, already-narrowed candidate pool (10-50 chunks), not the full
corpus. A cross-encoder model (`sentence-transformers` or similar) was
considered and rejected -- it would be the first ML-inference dependency
in the framework (torch, model downloads), disproportionate to what a
"good enough" reranker needs, and the framework already has eight
integrated LLM providers that can do this job with zero new
infrastructure.

`ChatResponse.parsed`'s concrete type isn't fully consistent across
providers today (existing provider tests show a validated model
instance in some cases, a plain dict in others) -- `LLMReranker` calls
`_RerankScores.model_validate(response.parsed)` rather than assuming
either shape, since pydantic v2's `model_validate` accepts both.

### Re-ranking stays a standalone composable step, not a retriever constructor parameter

None of `Retriever`, `BM25Retriever`, or `HybridRetriever` gained a
`reranker=` constructor parameter. Composition happens at the call site:
`reranker.rerank(query, retriever.retrieve(query, top_k=20), top_k=5)`.
This keeps the two halves of this ADR independently testable and usable,
works identically regardless of which retriever produced the
candidates (including a future one), and avoids coupling every
retriever's constructor to a reranker-shaped dependency most callers
won't use.

## Alternatives considered

- **`rank-bm25` + numpy** for the BM25 half. Rejected -- see above;
  the algorithm doesn't need it, and it would be the first non-AI-SDK
  runtime dependency pulled in by a "default" RAG component.
- **Score-normalized weighted-sum fusion** (e.g. min-max normalize each
  side's scores to `[0, 1]`, then `α·dense + (1-α)·bm25`) instead of
  RRF. Rejected -- introduces a tunable `α` with no principled default,
  and normalization has its own degenerate cases (a result list with
  one item, or all-identical scores) that RRF doesn't need to handle at
  all since it never looks at raw scores.
- **`HybridRetriever` composing a `Retriever` + `BM25Retriever` by
  reference** rather than owning its own `BM25Index`. Rejected -- see
  the chunk-id mismatch problem above; would have required threading a
  shared id-generation scheme through two otherwise-independent classes'
  `add_texts` methods, more coupling than just holding the index
  directly.
- **Cross-encoder re-ranking via `sentence-transformers`.** Rejected for
  now -- see above. Revisit if `LLMReranker`'s latency/cost profile
  turns out to be a real problem for a concrete use case (see
  Follow-ups).
- **Pointwise LLM re-ranking** (one call per candidate, asking "is this
  relevant? score 0-10"). Rejected -- `N` LLM calls for `N` candidates
  is materially more expensive and slower than one listwise call, for a
  quality difference that isn't clearly justified at the candidate-pool
  sizes re-ranking is meant to operate on.
- **`reranker=` as a retriever constructor parameter.** Rejected -- see
  "stays a standalone composable step" above.

## Consequences

### Positive

- Both `BM25Retriever` and `HybridRetriever` are zero-dependency,
  exported at the top level (`requisite.rag`/`requisite`) alongside
  `Retriever`, the same tier as the framework's other "just works"
  defaults.
- `LLMReranker` works with any of the framework's 8 already-integrated
  providers with no new setup.
- Re-ranking composes with any retriever, present or future, via one
  call-site pattern.

### Negative / risks

- Pure-Python BM25 recomputes IDF/average-document-length at query time
  rather than maintaining an incrementally-updated inverted index --
  fine at `InMemoryVectorStore`'s target scale (a few thousand chunks),
  not meant for a large corpus.
- `LLMReranker`'s listwise prompt has no explicit token-budget guard --
  a very large or very long-text candidate pool could exceed a model's
  context window or get expensive. Callers are expected to keep the
  candidate pool reasonably small (which over-fetching for fusion/
  re-ranking already implies), but this isn't enforced in code.
- BM25's simple tokenizer (no stemming/stopword removal) means exact
  word-form matches only -- "running" won't match a corpus containing
  only "run".

### Follow-ups

- A cross-encoder or hosted reranker API (Cohere Rerank, Voyage, Jina)
  as a second `BaseReranker` implementation, if `LLMReranker`'s
  cost/latency becomes a real problem for a concrete use case.
- Token-aware/stemming-aware tokenization for `BM25Index`, should exact
  word-form matching prove too limiting in practice.
- A token-budget guard or automatic candidate-pool truncation in
  `LLMReranker` if it proves necessary against very large candidate
  pools.
