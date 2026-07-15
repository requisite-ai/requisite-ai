
# 0003. Prompt templates, structured logging, and conversation policies

Status: Accepted
Date: 2026-07-13

## Context

`FEATURES.md`'s traceability against the original project vision flagged
three gaps: prompt templates (📋), structured/JSON logging (📋), and
conversation management lacking any summarization/truncation policy
(🚧 — storage existed via `BaseMemory`, but nothing governed how much
history actually gets sent to the model). Closing all three raised
design questions worth recording, since each has more than one reasonable
shape.

## Decision

### Prompt templates: `str.format`, not a templating engine

`PromptTemplate` (a single string) and `ChatPromptTemplate` (role-tagged
message sequence) use Python's own `str.format` with named fields for
variable substitution — not Jinja2, not a custom mini-language. Rationale:
in practice, prompt templating is almost always flat named substitution
(`"Translate to {language}: {text}"`), and `str.format` already does that
correctly with zero new dependencies. `string.Formatter().parse()`
auto-derives `input_variables` from the template text, so callers don't
maintain that list by hand.

`partial(**kwargs)` (pre-filling some variables, leaving others as
placeholders for later) is implemented via a custom `dict` subclass whose
`__missing__` echoes `"{key}"` back into `format_map`, rather than
tracking partial state as a separate concept — the "partially filled"
result is just another valid template string.

**`ChatPromptTemplate.format_messages(**kwargs)` returns a plain
`list[Message]`, and that's the entire integration surface with the rest
of the framework.** It was deliberately *not* wired into `AI.chat`'s
signature (e.g. no `AI.chat(template=..., **variables)` overload).
`AI.chat` already accepts `Union[str, Sequence[Message]]`; a rendered
template is just a `Sequence[Message]`, so no new parameter was needed —
consistent with ADR-0001's public API principle of not changing method
signatures for what's really just a different way to produce an existing
input type.

### Structured logging: opt-in, never automatic

`requisite.telemetry.configure_logging(json_format=True)` attaches a
`JSONFormatter` to the `"requisite"` logger tree. It is **never called by
the framework itself** — no module-level `configure_logging()` call
anywhere, no `Settings.__init__` side effect. This follows the same rule
already in place for `Settings` generally (ADR-0001's Configuration
Model): the framework reads/stores preferences, applications decide when
to act on them. `Settings.log_format` exists purely so an application
*can* wire `configure_logging(json_format=settings.log_format == "json")`
itself if it wants to — Settings doesn't do it automatically, for the
same reason `Settings` doesn't call `os.environ` deep in the stack: an
import-time or construction-time side effect that reconfigures Python's
global logging state would surprise a host application that has its own
logging setup.

`JSONFormatter` merges any `extra={...}` fields from a log call directly
into the JSON payload, rather than requiring a different logging API for
"structured" vs. "plain" calls — the same `logger.debug(...)` call
produces a readable string with the plain formatter and a structured
payload with the JSON one. A representative set of existing registration/
resolution log calls (provider, orchestrator, memory, tool, skill, agent,
prompt-template registration; capability resolution) were updated to pass
relevant fields via `extra=` as worked examples — not an exhaustive pass
over every log call in the codebase. New log call sites should follow the
same pattern when the logged event has an obvious structured shape (an
entity name, a count, a decision); a log call that's genuinely just prose
doesn't need `extra=` invented for it.

### Conversation policies: applied once, at read time, independent of storage

`BaseConversationPolicy.apply(messages) -> messages` is a pure function
over a message list — it doesn't know about `BaseMemory`, sessions, or
persistence. `Agent` applies it (if configured) exactly once, to the
initial message list, before the tool-calling loop starts:

```
load history (if memory)  ->  apply conversation_policy  ->  tool-calling loop  ->  persist (if memory)
```

Three consequences of this placement, each deliberate:

1. **It works identically with or without `memory` configured** — a
   long `Sequence[Message]` passed directly to `agent.run(history)` gets
   the same trimming/summarization as a long history loaded from memory.
   Conversation management isn't a memory-only feature.
2. **It's applied once per run, not per tool-call iteration.** Summarizing
   or trimming mid-tool-loop would risk cutting a tool's own result out
   of context before the model has used it — the loop's internal
   back-and-forth is exempt by construction, not by a special case.
3. **What gets persisted to `memory` is the *original* turn and answer,
   not the policy-adjusted history** (unchanged from ADR-0002 — this ADR
   doesn't revisit that). Each future `run()` call re-applies the policy
   fresh over the then-current stored history, so a policy change (e.g.
   tightening `max_messages`) takes effect on the very next call, with no
   migration needed for already-stored sessions.

`SummarizingPolicy` takes its own `AI` instance, independent of the
agent's. This is intentional, not an oversight: summarization quality
requirements are usually lower than the agent's main task, so a
cheaper/faster model (`AI(provider="groq", model="llama-3.3-70b-versatile")`
alongside a GPT-4-class agent, for instance) is a reasonable and common
choice — hard-coding the agent's own provider for summarization would
remove that option.

## Alternatives considered

- **Jinja2 (or another templating engine) for `PromptTemplate`.**
  Rejected: adds a dependency for conditionals/loops that essentially no
  real prompt template in the wild actually needs beyond named
  substitution; anyone who does need that expressiveness can build the
  final string themselves before handing it to `AI.chat`.
- **A `AI.chat(prompt_template=..., **variables)` convenience overload.**
  Rejected — see the "entire integration surface" note above. Two ways to
  call `chat` for what's fundamentally the same `Sequence[Message]` input
  would be surface area with no corresponding capability gain.
- **`Settings` calling `configure_logging()` automatically based on
  `log_format`.** Rejected — see Structured Logging above. This was the
  most tempting shortcut (it would make JSON logging "just work" from
  `.env`) and the most clearly wrong one on reflection: it's exactly the
  kind of implicit global side effect the framework avoids everywhere
  else.
- **Rewriting every existing log call site to use `extra=`.** Rejected
  as scope creep for this change — done for a representative set (see
  above) with the pattern documented in `DEVELOPMENT.md` for new code,
  rather than churning the whole codebase in one PR.
- **A single `TruncationPolicy` covering both trimming and summarization**
  via a strategy flag, instead of two classes (`MessageCountPolicy`,
  `SummarizingPolicy`). Rejected: they have different constructor
  requirements (`SummarizingPolicy` needs an `AI`; `MessageCountPolicy`
  needs nothing), and a shared base with optional fields for
  strategy-specific config is worse than two small, fully-typed classes
  under one shared `BaseConversationPolicy` interface.

## Consequences

### Positive

- Prompt templates, structured logging, and conversation policies each
  integrate through an existing seam (`list[Message]`, a log call's
  `extra=` kwarg, `Agent`'s existing message-list construction) rather
  than requiring a new integration point on `AI`, `Settings`, or `Agent`'s
  core loop shape.
- `FEATURES.md`'s three flagged gaps are now ✅ / 🚧-resolved without any
  change to previously-shipped public APIs — `AI.chat`, `Agent.run`, and
  `Settings`'s existing fields are all unchanged.

### Negative / risks

- `PromptTemplate`'s `str.format`-based approach cannot express
  conditionals or loops in a template. Acceptable for the stated goal
  (named substitution); would need revisiting if a real use case needs
  more (see Follow-ups).
- The `extra=`-based structured logging convention depends on
  contributors remembering to use it for new structured-shaped log
  events; nothing enforces it. A lint rule could catch obviously
  structured f-string-only log calls later, but doesn't exist today.
- `SummarizingPolicy.apply` makes a real LLM call synchronously inside
  what's otherwise a fast, local operation (`Agent.run`'s setup). This is
  inherent to what summarization is, not a bug, but worth knowing:
  attaching a `SummarizingPolicy` adds one extra provider round-trip
  (with its own latency and cost) to any `run()` call where history
  exceeds `max_messages`.

### Follow-ups

- If a real use case needs conditional/loop logic in a prompt template,
  reconsider Jinja2 (or a minimal subset) as an opt-in extra rather than
  a core dependency — don't add it speculatively.
- Consider a lint rule (or a `DEVELOPMENT.md`-documented review checklist
  item) for "this log call has an obvious entity/count — should it use
  `extra=`?" once there's a large enough set of real structured-logging
  consumers to justify the tooling investment.
- `SummarizingPolicy`'s summary is a single flat string reinserted as one
  `Message.system(...)`. If a future use case needs recursive/hierarchical
  summarization (summarizing summaries as a conversation keeps growing
  indefinitely), that's a new policy, not a change to this one.
