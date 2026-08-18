
# 0020. `github` default capability resolver

Status: Accepted
Date: 2026-08-19

## Context

With general graph execution shipped (ADR-0019), `ROADMAP.md`'s
Capabilities section has one remaining 📋 line: *"`github` default
resolver (public, unauthenticated REST API)."* This is a small, additive
feature -- ADR-0001 already settled the shape question when
`CapabilityRegistry` was designed: *"a capability *is* a `Tool` (or
resolves to one) -- it's a naming layer over `ToolRegistry`, not a new
kind of thing."* So this isn't a new interface or class hierarchy, it's
one more plain function in `requisite/capabilities/resolvers.py`,
registered the same way `read_file`/`get_weather`/`search_web` already
are.

This is a genuinely distinct feature from the separately-tracked
*"First-party MCP servers as default capability providers (GitHub,
databases)"* line in the MCP section, which reserves `GITHUB_TOKEN` in
`.env.example` for a future authenticated integration. `ROADMAP.md`
itself already draws this line explicitly (capability-section row: no
token; MCP-section row: `GITHUB_TOKEN` reserved) -- this ADR is only
about the former, and deliberately does not read `GITHUB_TOKEN` at all.

## Decision

### `search_github(query: str) -> str`, not `get_repo(owner, repo)`

Every existing default resolver takes one narrow, fuzzy input rather
than exact identifiers: `get_weather(city)` geocodes a name, `search_web
(query)` does a keyless search, `read_file(path)` is the one exception
(exact by nature -- there's no "fuzzy" file). An agent asking about
GitHub is, in the common case, in the same position as an agent asking
`search_web` a question: it has a topic, not a known `owner/repo`. A
`search_repositories` op (`GET /search/repositories?q=...`) matches that
existing pattern directly; an exact-lookup `get_repo(owner, repo)` would
be the odd one out, forcing the agent to already know something it
usually doesn't.

The alternative of one `action`-parameterized function
(`github(action: Literal["repo", "search", "issues"], ...)`, dispatching
several GitHub REST operations through a single tool) was considered and
rejected: no existing precedent for that shape exists anywhere in
`requisite/capabilities/` or `requisite/tools/` (checked directly), and
`Agent.requires("github")` resolving to exactly *one* `Tool` under the
capability name is the established, simple mental model every other
resolver follows. Introducing a new "multi-op tool" convention for the
sake of API completeness isn't worth the departure -- register a richer,
multi-op GitHub integration (e.g. the future MCP server) at a higher
priority instead; that's exactly the override mechanism
`CapabilityRegistry` already provides for this.

### stdlib `urllib`, not `httpx`/`requests`

Matches `get_weather`/`search_web` exactly, and keeps this resolver
"core" rather than "optional integration" per ADR-0001's dependency-flow
table, which draws that line at "requires an extra pip package," not "makes
a network call." GitHub's REST API is plain JSON over HTTPS -- no SDK
needed, so there's no reason to cross that line here either.

### Two deliberate deviations from `get_weather`/`search_web`'s pattern

Both are GitHub-specific necessities, not stylistic choices:

1. **A `User-Agent` header is required.** GitHub's REST API returns 403
   for any request without one -- neither `get_weather` nor `search_web`
   needed a custom header, so this is new code: a
   `urllib.request.Request(url, headers={"User-Agent": "requisite-ai",
   "Accept": "application/vnd.github+json"})` instead of passing a bare
   URL string to `urlopen`.
2. **`urllib.error.HTTPError` (a subclass of `URLError`) is caught
   before the shared generic exception tuple**, specifically to give an
   actionable message on HTTP 403 -- "rate limited (10 searches/minute),
   try again shortly" -- rather than surfacing GitHub's raw error body.
   An agent hitting this needs to know "wait and retry" is the right
   move, not "the query was malformed." Every other HTTP status still
   falls through to a generic `"HTTP {code}"` message; the generic
   `URLError`/`TimeoutError`/`JSONDecodeError` tuple stays exactly as it
   was in `get_weather`/`search_web` for connection-level failures.

### Result formatting: short strings, not raw JSON

Same convention as every existing resolver -- the top 5 results (`sort by
stars desc`, matching what a agent asking "find X" usually wants first),
each rendered as `"{full_name} ({stars} stars) - {description}
{html_url}"`, joined by newlines. Zero results returns a plain, non-error
string ("No GitHub repositories found for 'X'.") -- matching
`get_weather`'s "could not find a location" and `search_web`'s "no quick
summary found" precedent: an empty result isn't a failure state.

Deliberately plain ASCII (`"{n} stars"`, not a `★` glyph): the real-
network verification pass for this ADR hit a live `UnicodeEncodeError`
printing a `★`-containing result on a default Windows console (`cp1252`
codepage) -- a real crash risk for any Windows user running
`examples/capability_example.py` unmodified if a model echoes the tool
output verbatim. No other resolver in this module emits non-ASCII
characters; this one now doesn't either.

### Test coverage: monkeypatched `urllib.request.urlopen`, no new dependency

No existing resolver test exercises the actual HTTP call (only the
registry/resolution machinery around it -- `get_weather`/`search_web`
have zero direct tests of their own network logic). `search_github` has
meaningfully more branching than either (multi-result formatting,
zero-results, 403 special-case, generic HTTP error, generic connection
error), so it gets direct coverage via `unittest.mock.patch("urllib
.request.urlopen", ...)` -- stdlib only, no new test dependency, and no
new precedent being established beyond "this one resolver's tests
happen to monkeypatch its own HTTP call," which doesn't obligate
`get_weather`/`search_web` to be retrofitted.

## Alternatives considered

- **`get_repo(owner: str, repo: str) -> str`** (exact lookup instead of
  search). Rejected -- see "Decision" above; breaks the
  fuzzy-input pattern every other resolver (except the necessarily-exact
  `read_file`) follows.
- **`action`-parameterized multi-op tool.** Rejected -- no precedent
  anywhere in the framework for one capability resolving to a
  multi-operation dispatch tool; `CapabilityRegistry`'s priority-override
  mechanism already exists for adding a richer integration later.
- **Reading `GITHUB_TOKEN` if present, for higher rate limits.**
  Rejected for this resolver: `ROADMAP.md` explicitly scopes this line
  as "public, unauthenticated," and `GITHUB_TOKEN` is already earmarked
  for the separate, future first-party MCP GitHub server (ADR-0001's own
  `agent.requires("github")` example describes that MCP-backed provider
  registering at a *higher priority* than this one, the same
  `weather`/`acme-weather` pattern `examples/capability_example.py`
  already demonstrates) -- conflating the two would blur that boundary
  for no real benefit today.

## Consequences

### Positive

- Closes the last 📋 line in `ROADMAP.md`'s Capabilities section.
- Zero-config, zero-dependency, matching every other default resolver's
  bar exactly -- `agent.requires("github")` works out of the box.
- Purely additive: one new function, one new registration line, no
  changes to `CapabilityRegistry`, `CapabilityProvider`, or `Agent
  .requires(...)`.

### Negative / risks

- Unauthenticated GitHub Search API is rate-limited to 10 requests/minute
  per IP -- a busy agent (or shared dev machine hitting it repeatedly)
  will see the 403 message more often than `weather`/`internet_search`'s
  more generous free tiers. Mitigated by the actionable rate-limit
  message, not eliminated; register a token-backed provider at higher
  priority for production use.
- Search relevance is whatever GitHub's own `sort=stars` ranking
  produces -- no query refinement/filtering beyond what's passed through
  directly. Same class of "good enough default, not production-grade"
  limitation the module's own docstring already states applies to all
  four resolvers.

### Follow-ups

- The separate first-party GitHub MCP server (`GITHUB_TOKEN`-backed,
  registered as a default capability provider) remains 📋 on
  `ROADMAP.md`'s MCP section -- not scoped here, and not blocked by this
  change; it would simply register `"github"` at a higher priority than
  `search_github` once built.
