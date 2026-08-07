
# 0006. Gemini thought_signature echoing

Status: Accepted
Date: 2026-08-07

## Context

`examples/agent_example.py` run against a live Gemini model started
failing multi-turn tool-calling conversations with:

    400 INVALID_ARGUMENT: Function call is missing a thought_signature

Gemini's API now requires `thought_signature` (opaque `bytes`, carried on
`google.genai.types.Part` as a sibling field to `function_call`) to be
echoed back verbatim on `function_call` parts across turns. A second,
non-fatal symptom showed up alongside it: a noisy `"there are non-text
parts in the response"` warning logged whenever code called
`response.text` on a response containing a function call -- because that
convenience property silently discards `thought_signature` (and the
warning is the SDK's way of flagging the discard).

`GeminiProvider._to_chat_response` was built against `response.text` and
`response.function_calls`, both SDK convenience properties, before this
requirement existed. Fixing this meant deciding (a) where in the
framework's provider-agnostic `ToolCall` model an opaque, Gemini-specific
value should live, and (b) how `GeminiProvider` extracts a response's
content and tool calls without going through the properties that drop it.

The following was verified live against the installed `google-genai`
2.15.0 SDK and Gemini's docs before writing this down, not assumed:

- `thought_signature` lives on `Part`, not on `FunctionCall` itself.
- `response.function_calls` discards it.
- `response.text` triggers the noisy warning and also isn't needed once
  parts are walked directly.
- `types.Part.from_function_call(name=, args=)` has no `thought_signature`
  kwarg -- `Part` is mutable, so it must be set as a post-construction
  attribute assignment (`part.thought_signature = b"..."`), confirmed to
  work against the real SDK.

## Decision

### `ToolCall.provider_data`: an opaque per-provider escape hatch

`requisite/core/interfaces.py`'s `ToolCall` gains one new optional field:

    provider_data: Optional[Any] = None

Documented as opaque, provider-specific data required to correctly replay
this exact tool call on a later turn. Populated only by the provider that
produced the `ToolCall`; every other provider ignores it; framework code
never constructs or interprets it. This mirrors `ChatResponse.raw` --
already an established pattern in this codebase for "provider-native
escape hatch, not part of the stable contract" -- rather than inventing a
new mechanism. A generic `Any`-typed field was chosen over a
Gemini-specific field on `ToolCall` (e.g. `thought_signature: bytes`)
because the framework's `core/` layer has zero provider knowledge by
design (ADR-0001) and must stay that way even for a single-provider
quirk.

### `GeminiProvider` walks `candidates[0].content.parts` directly

`_to_chat_response` no longer touches `response.text` or
`response.function_calls`. It reads `response.candidates[0].content.parts`
and, per part, appends `part.text` to the response's text and, for a
`part.function_call`, builds a `ToolCall` with
`provider_data=part.thought_signature`. This removes the discard at the
source rather than trying to recover the signature after the fact.

### The signature is echoed back on the next turn, not held elsewhere

`_build_contents_and_system` -- which reconstructs Gemini `Content`/`Part`
objects from framework `Message`/`ToolCall` objects when building a
follow-up request -- now does:

    function_call_part = types.Part.from_function_call(name=call.name, args=call.arguments)
    if isinstance(call.provider_data, bytes):
        function_call_part.thought_signature = call.provider_data
    parts.append(function_call_part)

The `isinstance(..., bytes)` guard is deliberate: a `ToolCall` with no
`provider_data` (hand-built by application code, or produced by a
different provider entirely -- e.g. an agent that swaps providers
mid-conversation) must not fabricate a signature. It's left unset and
Gemini is given the chance to either accept the call without one or
surface its own error, rather than the framework guessing.

## Alternatives considered

- **Store `thought_signature` on `Message` instead of `ToolCall`.**
  Rejected: the signature is a property of one specific tool call, not of
  the message as a whole; a message can carry multiple tool calls, each
  needing its own signature.
- **A Gemini-specific field name (`thought_signature: Optional[bytes]`)
  directly on `ToolCall`.** Rejected: leaks a single provider's wire
  format into the provider-agnostic core model. The next provider with a
  similar "opaque replay token" requirement (plausible -- several
  providers are moving toward signed/verified tool-call chains) would
  need its own field, and `ToolCall` would accumulate one optional field
  per provider quirk indefinitely.
- **Keep using `response.text` / `response.function_calls` and recover
  the signature via a second pass over `response.candidates`.** Rejected
  as needless complexity -- once the parts have to be walked anyway to
  find the signature, walking them for text and tool calls too is less
  code, not more, and it removes the noisy warning as a side effect.

## Consequences

### Positive

- Multi-turn Gemini tool-calling conversations work again;
  `examples/agent_example.py` -- the script that surfaced the bug --
  completes without the `400 INVALID_ARGUMENT` error or the
  non-text-parts warning.
- `provider_data` is available to any future provider with a similar
  requirement without another core-model change.
- No other provider's code path is touched -- `provider_data` defaults to
  `None` and every non-Gemini provider simply never sets or reads it.

### Negative / risks

- `ToolCall.provider_data` is untyped (`Any`) by necessity, so nothing
  stops application code from reading or setting it directly. The
  docstring says not to; there's no runtime enforcement. Acceptable since
  the same trust boundary already exists for `ChatResponse.raw`.
- If a future provider needs *multiple* opaque replay values per tool
  call (not just one blob), a single `provider_data: Any` won't cleanly
  hold more than one without that provider inventing its own internal
  structure inside it. Not a problem today; revisit if it happens.

### Follow-ups

- None required for this fix. If a second provider needs the same
  mechanism, confirm `provider_data: Any` is still the right shape (a
  dict-of-provider-name-to-value might be better with two data points
  instead of one) before extending it further.
