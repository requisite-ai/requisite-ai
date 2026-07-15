
# 0002. Provider-specific configuration, OpenAI-compatible providers, and Memory integration

Status: Accepted
Date: 2026-07-12

## Context

Implementing Anthropic, Groq, and Azure OpenAI as the next three providers,
plus `BaseMemory`'s first real implementation, surfaced three concrete
design questions ADR-0001 either flagged as a follow-up or specified only
at the interface level:

1. Azure OpenAI's constructor needs configuration (`azure_endpoint`) that
   no other provider needs. ADR-0001 flagged this exact scenario as the
   trigger for revisiting the single-`Settings`-object model.
2. Groq and (as of its v1 GA API) Azure OpenAI both turned out to be
   fully OpenAI-wire-compatible -- raising the question of whether to
   duplicate `OpenAIProvider`'s ~250 lines of translation logic per
   provider, or share it.
3. `BaseMemory` was specified in ADR-0001 but not wired into `Agent`.
   Doing so requires deciding what "an agent with memory" actually means
   operationally: what's persisted, when, and what happens if the caller
   passes a full message history instead of a single new turn.

## Decision

### Provider-specific configuration: `Settings.provider_kwargs(name)`

Rather than expanding `AI._resolve_provider` with per-provider
special-casing (`if provider_name == "azure_openai": ...`), `Settings`
gained one small method:

```python
def provider_kwargs(self, provider: str) -> dict[str, Any]:
    if provider.lower() == "azure_openai":
        return {"azure_endpoint": self.azure_openai_endpoint}
    return {}
```

`AI._resolve_provider` merges this in unconditionally:

```python
extra_kwargs = self._settings.provider_kwargs(provider_name)
self._registry.create(provider_name, api_key=..., model=..., **extra_kwargs)
```

`AI` itself still knows nothing about Azure specifically -- it just always
asks `Settings` "does this provider need anything extra?" This keeps the
provider-agnostic promise intact (`AI`'s code path is identical regardless
of which provider needs extra config) while giving `Settings` a place to
own that knowledge, which is where configuration concerns already live.

**Not chosen:** a `ProviderSettings` sub-model per provider (e.g.
`Settings.azure = AzureSettings(endpoint=..., ...)`). That's a reasonable
design and may still be worth it if a *third* provider needs multiple
extra fields with validation between them -- but for one provider needing
one extra string, it would be more structure than the problem currently
justifies. Revisit if that changes (see Follow-ups).

### OpenAI-wire-compatible providers subclass `OpenAIProvider`

Confirmed against both providers' current documentation before deciding:

- Groq's chat completions endpoint is deliberately OpenAI-request/response
  compatible; Groq's own quickstart shows using the `openai` package
  directly with a swapped `base_url`.
- Azure OpenAI's **v1 GA API** (generally available since August 2025)
  works the same way: the plain `OpenAI` client pointed at
  `{endpoint}/openai/v1/`, no `AzureOpenAI`-specific client class and no
  dated `api-version` string required. This superseded the older
  `AzureOpenAI`/`AsyncAzureOpenAI` + monthly `api-version` pattern, which
  we deliberately did not implement.

Given that, `GroqProvider` and `AzureOpenAIProvider` are implemented as
subclasses of `OpenAIProvider`, overriding only `__init__` (to fix/require
a `base_url`) and `name`:

```python
class GroqProvider(OpenAIProvider):
    def __init__(self, *, api_key=None, model="llama-3.3-70b-versatile", **kwargs):
        super().__init__(api_key=api_key, model=model, base_url="https://api.groq.com/openai/v1", **kwargs)

    @property
    def name(self) -> str:
        return "groq"
```

This is a deliberate, narrow exception to ADR-0001's "composition over
inheritance, no subclassing between concrete implementations" guidance --
justified specifically because the *wire format itself* (not just
convenient code reuse) is identical, confirmed against both vendors' own
documentation rather than assumed. Any future OpenAI-wire-compatible
provider (OpenRouter, Together AI, and others in `ROADMAP.md`'s table)
should follow this same pattern: verify wire compatibility against
current docs first, then subclass `OpenAIProvider` with a `base_url`
override, rather than reimplementing the same translation logic again.

**Anthropic is not part of this family.** Its wire format differs
meaningfully (system prompt as a separate parameter, required
`max_tokens`, `input_schema` instead of `parameters` for tools, a
`content` block list instead of `choices[0].message`), so
`AnthropicProvider` is a standalone `BaseProvider` implementation, using
the SDK's native `messages.parse(output_format=...)` for structured
output rather than the more common "force a single tool call" workaround
-- confirmed to exist in the current `anthropic` SDK before choosing it.

### Memory integration in `Agent`

`Agent` gained two constructor parameters: `memory: Optional[BaseMemory]`
and `session_id: Optional[str]`. Three decisions were made, none of which
were fully specified in ADR-0001:

**1. `session_id` is required if `memory` is set, checked at construction
time, not at call time.** An agent with memory but no session_id has no
sensible default (there's no "current user" concept in `Agent` itself),
so failing fast at `Agent(...)` construction is more useful than failing
on the first `run()` call, or silently using some placeholder session key.

**2. `run(prompt)`/`arun(prompt)` must be called with a plain string when
memory is configured -- passing a full `Sequence[Message]` raises
`ConfigurationException`.** The alternative (merging caller-supplied
history with memory-loaded history) has no non-arbitrary reconciliation
rule: if they disagree, which wins? Refusing the ambiguous case outright
is more honest than picking a resolution rule that would surprise someone
in the other case.

**3. Only the user's new turn and the agent's final answer are persisted
-- not the intermediate tool-call round-trip.** An agent with memory that
called three tools before answering doesn't write six extra messages
into the stored session; it writes two (the user's question, the final
answer). Rationale: tool-call messages carry provider-specific IDs
(`ToolCall.id`) that were only ever meaningful within that one call to
that one provider -- replaying them into a later conversation (possibly
with a different provider, if the agent's `provider=` changes between
sessions) has no clear semantics. The *content* of what happened (what
the agent ultimately said) is what's meaningful to recall; the mechanism
by which it got there is not.

## Alternatives considered

- **A `ProviderSettings` sub-model now, instead of `provider_kwargs`.**
  Rejected for now as premature structure for a single string field --
  see [Follow-ups](#follow-ups).
- **A generic `openai_compatible=True` flag on `BaseProvider`** instead of
  subclassing. Rejected: it would still need `OpenAIProvider`'s actual
  translation methods to be reachable, which subclassing already gives
  for free, without inventing a new flag-based composition mechanism for
  a two-provider (so far) need.
- **Persisting the full tool-call round-trip in memory.** Rejected -- see
  point 3 above. Revisit if a real use case needs to resume a
  conversation mid-tool-call, which no current use case does.
- **A default, magic `session_id` (e.g. derived from `Agent.name`) instead
  of requiring one explicitly.** Rejected: it would make every agent with
  the same name silently share one conversation, which is a much worse
  failure mode (cross-user data leakage in a multi-tenant app) than
  requiring an explicit, if slightly more verbose, `session_id`.

## Consequences

### Positive

- Adding Anthropic, Groq, and Azure OpenAI required zero changes to `AI`,
  `Agent`, `Workflow`, or any registry beyond the expected
  `registry.register(...)` call -- confirming ADR-0001's extension model
  holds up against a real, less-trivial addition (provider-specific
  config, wire-format reuse).
- Groq and Azure OpenAI's provider files are each under 40 lines because
  they reuse `OpenAIProvider` -- a concrete, measurable payoff from
  verifying wire compatibility before choosing standalone implementations.
- Memory's three design decisions (required session_id, string-only
  prompt, partial persistence) each fail loudly and specifically rather
  than silently doing something surprising -- consistent with the
  framework's "never swallow, always wrap with context" error-handling
  convention.

### Negative / risks

- `provider_kwargs` is a slightly awkward escape hatch -- a `dict[str, Any]`
  with no schema. It's deliberately minimal, not deliberately final; see
  Follow-ups for when to replace it.
- The `OpenAIProvider` subclassing pattern only works because we checked
  wire compatibility live against current docs. A future OpenAI-compatible
  provider that later *diverges* in some subtle way (e.g. different tool
  schema quirks) would silently misbehave through inherited code rather
  than failing loudly. Anyone adding a new subclass should re-verify wire
  compatibility for the specific features they're relying on (tool
  calling, structured output), not just chat completions.
- `Agent`'s memory persistence (turn + final answer only) means a
  paused-and-resumed agent mid-tool-call cannot currently be resumed
  faithfully from memory -- acceptable today since nothing resumes
  mid-run, but worth remembering if that changes.

### Follow-ups

- Revisit `provider_kwargs` -> a real `ProviderSettings` sub-model
  (per-provider, validated) once a second provider needs more than one
  extra field, or once any extra field needs its own validation beyond
  "is it set."
- When a second OpenAI-wire-compatible provider (OpenRouter, Together AI)
  is added, confirm this ADR's subclassing guidance still holds, or
  extract a smaller shared base (e.g. `OpenAICompatibleProvider`) if three
  providers reveals it's not just `OpenAIProvider` with a different
  `base_url`.
- If a future use case needs to resume an agent mid-tool-call, revisit
  memory's "final answer only" persistence rule -- likely needs a
  separate, explicit "checkpoint" concept rather than changing what
  `Agent.run` persists by default.
