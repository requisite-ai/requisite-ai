
# 0009. Streaming + tool calls together

Status: Accepted
Date: 2026-08-11

## Context

Streaming (`AI.stream`/`.astream`) and tool calling (`AI.chat_response`
`tools=`) were each fully shipped but never composed. `StreamChunk`
(`requisite/core/interfaces.py`) had exactly three fields -- `delta: str`,
`is_final: bool`, `raw: Any` -- with nowhere to carry a tool call.
`BaseProvider.stream`/`.astream` didn't declare a `tools=` parameter at
all (unlike `chat`/`achat`, which do). Every provider's streaming
implementation silently dropped any tool-call data the underlying SDK
returned mid-stream, even where `tools=` had been passed straight
through to the wire call (Gemini and Anthropic both already forwarded
`tools` into their streaming request, then ignored any tool call in the
response).

Fixing this required first establishing what each provider's SDK
actually supports mid-stream, rather than assuming a single shape. Read
directly against the pinned SDK versions:

- **OpenAI-wire-compatible family** (OpenAI, Groq, Azure OpenAI,
  OpenRouter, Together -- all via `OpenAIProvider`) streams tool-call
  *arguments* as JSON-string fragments (`delta.tool_calls[i].function.arguments`),
  keyed by `delta.tool_calls[i].index`, to be concatenated across chunks.
- **Anthropic** streams the same way at the raw-event level:
  `content_block_start` (opens a `tool_use` block, empty `input`) →
  `content_block_delta` events of type `input_json_delta` carrying
  `partial_json` fragments → `content_block_stop`.
- **Gemini** (Developer API, non-Vertex) explicitly documents
  `FunctionCall.partial_args`/`.will_continue` as "not supported in
  Gemini API" (`google/genai/types.py`) -- function calls always arrive
  whole, in one chunk.
- **Ollama** has no delta/partial type anywhere in its SDK
  (`ollama/_types.py`) -- `Message.ToolCall.Function.arguments` is
  always a fully-parsed `Mapping`, never a partial string.

So of the four provider implementations, only two (OpenAI-family,
Anthropic) can genuinely stream tool-call arguments incrementally; the
other two only ever deliver a tool call complete.

## Decision

### `StreamChunk` reports only complete tool calls, never partial

`StreamChunk` gains one field: `tool_calls: list[ToolCall] = Field(default_factory=list)`,
plus a `has_tool_calls` property mirroring `ChatResponse.has_tool_calls`.
Every provider **accumulates any incremental/fragmented data internally**
and only attaches fully-assembled `ToolCall` objects once known-complete
-- typically on the terminal chunk (`is_final=True`, or for the
OpenAI-family, the chunk carrying `finish_reason == "tool_calls"`). The
public contract is uniform across all 8 providers regardless of what
each SDK does under the hood: **a non-empty `StreamChunk.tool_calls`
means those tool calls are complete and safe to execute.** No
partial-JSON handling, and no per-provider "does this one stream deltas
or not" branching, ever reaches calling code.

Each provider's implementation:

- `OpenAIProvider.stream`/`.astream` (`requisite/providers/openai_provider.py`):
  new `_accumulate_tool_call_deltas`/`_finalize_tool_calls` module
  functions, accumulator keyed by `delta.index`. Fixes `GroqProvider`,
  `AzureOpenAIProvider`, `OpenRouterProvider`, `TogetherProvider` for
  free -- none override `stream`/`astream`.
- `AnthropicProvider.stream`/`.astream` (`requisite/providers/anthropic_provider.py`):
  switched from the convenience `stream.text_stream` helper (text only)
  to raw event iteration (`for event in stream`), since only the raw
  event stream carries `content_block_start`/`input_json_delta`. New
  `_finalize_streamed_tool_calls` module function.
- `GeminiProvider.stream`/`.astream` (`requisite/providers/gemini_provider.py`):
  already accepted `tools=`; added `_extract_tool_calls_from_parts`,
  reusing `_to_chat_response`'s per-part parsing, called per chunk with a
  running index counter (each chunk's own `parts` list restarts at 0).
- `OllamaProvider.stream`/`.astream` (`requisite/providers/ollama_provider.py`):
  previously the only provider whose streaming methods didn't even
  accept `tools=` -- added it, reusing the existing `_to_ollama_tools`
  converter. New `_extract_ollama_tool_calls` module function.

### `AI` gains `stream_response`/`astream_response`; `stream`/`astream` gain `tools=` but stay text-only

`AI.stream`/`.astream` now accept `tools=`, forwarded to the provider,
but still return `Iterator[str]`/`AsyncIterator[str]` (text deltas only)
-- mirroring the *existing* precedent that `AI.chat(tools=...)` already
accepts tools but only returns `response.content`, documented as
"inspect `chat_response(...).tool_calls`" for the structured view. New
`AI.stream_response`/`.astream_response` mirror `chat_response`/
`achat_response` exactly, yielding the full `StreamChunk` sequence.

### `BaseProvider.stream`/`.astream` gain `tools=` in the abstract signature

Matching `chat`/`achat` exactly, so every provider implementation is
statically required to accept it going forward.

## Alternatives considered

- **Expose raw partial-JSON deltas as public API** (e.g. a
  `StreamChunk.tool_call_deltas` field carrying `index`/`id`/`name`/
  `arguments_delta` fragments). Rejected: this only has real content for
  2 of 4 provider implementations (OpenAI-family, Anthropic) -- Gemini
  and Ollama would always deliver a single "fragment" that's already the
  whole thing. Any caller consuming this field would have to branch on
  "does my provider actually fragment this or not," which the
  accumulate-internally design avoids entirely. Revisit if a real,
  specific use case for incremental argument display emerges (see
  Follow-ups).
- **Only support streaming + tools for the two providers that can
  genuinely stream arguments incrementally**, leaving the others
  unchanged (still 📋). Rejected: `ROADMAP.md`'s framing (and the
  project's own "every layer is an interface + implementation(s)"
  philosophy) treats provider parity as the default expectation --
  Gemini/Ollama not supporting *incremental delivery* doesn't mean they
  can't support the *feature* (tool calls surfaced during a streamed
  call at all), just that their existing per-chunk-whole delivery
  already satisfies the same completion contract other providers reach
  via accumulation.

## Consequences

### Positive

- One completion contract (`tool_calls` non-empty ⇒ complete) works
  identically across all 8 providers -- callers never need
  provider-specific branching.
- `AnthropicProvider.stream`/`.astream` now also correctly emit
  `is_final`'s tool calls even for turns that mix text and tool use in
  one response, which `stream.text_stream` alone could never have
  surfaced regardless of this feature.
- `OllamaProvider` streaming gained `tools=` support it never had before
  at all -- previously the only way to get tool calls from Ollama was
  the non-streaming `chat()`.

### Negative / risks

- Anthropic's `stream()`/`.astream()` no longer use the SDK's
  `text_stream` convenience helper, so any future SDK-side improvements
  specific to that helper (rather than the raw event stream) won't be
  picked up automatically.
- A caller who wants to show incremental tool-call-argument progress in
  a UI (e.g. "building the search query...") has no way to do that
  through this API for any provider -- they'd see the whole call appear
  at once, same as a non-streaming response, just interleaved with
  earlier text deltas.

### Follow-ups

- `Agent.stream`/`.astream`: `Agent` still has zero streaming capability
  (`run`/`arun` only, built on `chat_response`/`achat_response`). Making
  an agent stream across its own multi-round tool-calling loop is a
  materially bigger design question (does it stream intermediate
  "thinking" text before a tool call too? how does the concurrent
  tool-call execution from the `arun()` fix interact with a live
  stream?) -- not scoped here, not currently listed elsewhere in
  `ROADMAP.md`.
- If a real use case for incremental tool-call-argument display emerges,
  consider an opt-in raw-delta field for just the two providers that can
  support it, rather than retrofitting it into the uniform contract above.
