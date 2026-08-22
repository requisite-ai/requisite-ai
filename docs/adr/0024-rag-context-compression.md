# 0024. RAG context compression

Status: Accepted
Date: 2026-08-21

## Context

`ROADMAP.md`'s RAG section has one remaining 📋 line: *"Context
compression."* It is the only gap left in that section -- retrieval
(dense, BM25, hybrid) and re-ranking (ADR-0010) both already shipped.

`BaseReranker`/`LLMReranker` (`requisite/rag/base.py`,
`requisite/rag/reranker.py`) already establish the exact shape this
problem needs: *"a standalone, composable post-processing step over an
already-retrieved candidate list,"* deliberately not wired into any
retriever's constructor, implemented as one listwise structured-output
`AI.chat_response(response_model=...)` call rather than a new ML
dependency. Context compression is the same class of problem -- reduce
what an already-retrieved candidate list costs the model to consume,
without touching retrieval or storage -- so this ADR gives it the same
treatment rather than inventing a new shape.

## Decision

### `BaseCompressor` in `requisite/rag/base.py`, mirroring `BaseReranker`

```python
class BaseCompressor(ABC):
    @abstractmethod
    def compress(self, query: str, results: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        ...

    async def acompress(self, query: str, results: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        return await asyncio.to_thread(self.compress, query, results)
```

Same placement (directly after `BaseReranker`, same file), same
thread-wrapped async default every other `Base*` interface in this
module uses, same "not a retriever constructor parameter" stance --
composition happens at the call site:

```python
compressed = compressor.compress(query, reranker.rerank(query, retriever.retrieve(query, top_k=20), top_k=5))
```

### No `top_k` parameter

`rerank`'s `top_k` selects *how many* results to keep. Compression's job
is shrinking *content*, not selecting count -- each input result maps to
at most one output result (dropped if nothing relevant remains; see
below), never truncated by an arbitrary count. An application that wants
both count-limiting and content-shrinking gets it by composing the two
directly, in either order, as shown above -- no new combined parameter
needed.

### Empty compression means *drop*, not zero-score

`LLMReranker`'s `_apply_scores` gives a candidate missing from the LLM's
response a score of `0.0` and keeps it (reranking never changes *which*
chunks exist, only their order/count). Compression is different: if
nothing in a passage is relevant to the query, there is no compressed
text left to contribute to the prompt, so keeping a `ScoredChunk` with
empty `chunk.text` around would be dead weight for whatever consumes the
list next. `_apply_compression` (`requisite/rag/compressor.py`) excludes
any result whose `compressed_text` is empty or absent from the response
entirely, rather than returning it with blank content.

### `LLMContextCompressor` in new `requisite/rag/compressor.py`

Structurally a near-copy of `reranker.py`, for the same reasons ADR-0010
already gave for `LLMReranker` and re-affirmed here rather than
re-litigated: reuse the framework's own `AI` facade and
`response_model=` structured output instead of adding a
summarization-specific ML dependency (e.g. a local extractive-summary
model). One listwise call compresses every candidate at once:

```python
class _CompressedPassage(BaseModel):
    chunk_id: str
    compressed_text: str  # "" if nothing in this passage is relevant

class _CompressionResult(BaseModel):
    compressed: list[_CompressedPassage]
```

`chunk.text` is replaced via `chunk.model_copy(update={"text": ...})` --
`Chunk` is an immutable-by-convention pydantic model elsewhere in the
framework (the same `model_copy(update=...)` pattern `Agent.requires()`
uses on `Tool`), so a new `Chunk`/`ScoredChunk` is constructed rather
than mutating fields in place. `score` is carried through unchanged --
compression doesn't re-judge relevance ranking, a prior retrieval/rerank
step already did that; it only reduces text.

Same `ChatResponse.parsed`-isn't-one-type-across-providers handling as
`LLMReranker`: `_CompressionResult.model_validate(response.parsed)`
rather than assuming a concrete type.

### Same constructor shape as `LLMReranker`

`LLMContextCompressor(*, ai=None, provider=None, model=None)` -- an
`AI` instance or `provider`/`model` passthrough, identical to
`LLMReranker`'s constructor, so an application already using
`LLMReranker` needs zero new mental model to add compression.

## Alternatives considered

- **A `top_k` parameter on `compress`, mirroring `rerank`.** Rejected --
  see "No `top_k` parameter" above; conflating count-selection with
  content-reduction would blur what each step is responsible for, and
  composing the two existing primitives already covers the combined
  case.
- **Keep an empty-compression result in the output with blank text,
  scored `0.0` (mirroring `LLMReranker`'s missing-score handling).**
  Rejected -- see "Empty compression means drop, not zero-score" above;
  an empty-text `ScoredChunk` has nothing left to offer a downstream
  prompt, unlike a low-relevance-but-still-present reranked result.
- **An embeddings-based sentence filter** (embed each sentence in a
  passage, keep only those above a similarity threshold to the query
  embedding) instead of an LLM call. Rejected for the first
  implementation -- it would require chunking passages down to
  sentence-level and re-embedding at compression time (a second,
  finer-grained embedding pass beyond what retrieval already did), for
  a lower-quality result than an LLM's actual reading comprehension of
  what's relevant; `BaseCompressor` doesn't preclude adding this as a
  second implementation later, the same way ADR-0010 left the door open
  for a cross-encoder reranker.
- **Pointwise LLM compression** (one call per candidate). Rejected for
  the same reason ADR-0010 rejected pointwise reranking: `N` calls for
  `N` candidates is materially more expensive for no clear quality gain
  at the candidate-pool sizes this operates on (already-narrowed by
  retrieval, often already re-ranked).
- **A `compress=` parameter wired into `Retriever`/`HybridRetriever`/
  `BM25Retriever`'s constructors.** Rejected -- see "not a retriever
  constructor parameter" above; same reasoning ADR-0010 already applied
  to `reranker=`.

## Consequences

### Positive

- Closes the last 📋 line in `ROADMAP.md`'s RAG section.
- Purely additive: one new interface, one new module, no changes to
  `BaseRetriever`, `BaseReranker`, `Retriever`, `HybridRetriever`,
  `BM25Retriever`, or `Chunk`/`ScoredChunk`'s own fields.
- Works with any of the framework's already-integrated providers, same
  as `LLMReranker`, with zero new setup.
- Composes with re-ranking (in either order) and with any retriever,
  present or future, via the same call-site pattern ADR-0010 already
  established.

### Negative / risks

- Same token-budget caveat ADR-0010 already noted for `LLMReranker`'s
  listwise prompt: no explicit guard against an oversized candidate
  pool exceeding a model's context window. Compression is typically
  applied to an already-narrowed pool (post-retrieval, often
  post-rerank), which mitigates but doesn't eliminate this.
- Compression quality depends entirely on the underlying model's
  extraction faithfulness -- there is no verification that
  `compressed_text` is actually a subset/faithful condensation of the
  original passage rather than a paraphrase that drifts from the
  source. Not enforced in code, same class of risk `LLMReranker`
  already carries with its relevance scoring.
- An extra LLM call in the retrieval pipeline (on top of any reranking
  call) adds latency and cost to every query that uses it -- an
  explicit, opt-in trade-off (nothing is wired in by default), but
  worth stating plainly.

### Follow-ups

- An embeddings-based or extractive (non-LLM) `BaseCompressor`
  implementation, if `LLMContextCompressor`'s latency/cost profile
  becomes a real problem for a concrete use case -- mirrors ADR-0010's
  own cross-encoder-reranker follow-up.
- A token-budget guard or automatic candidate-pool truncation, if it
  proves necessary against very large candidate pools -- same follow-up
  ADR-0010 already left open for `LLMReranker`, now applying equally
  here.
