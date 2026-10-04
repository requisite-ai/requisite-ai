"""Request-scoped context: the carrier, tool injection, Agent/Workflow entry points,
isolation between concurrent requests, and propagation through every execution path."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import pytest

from requisite import (
    Agent,
    RequestContext,
    Tool,
    ToolException,
    current_context,
    request_context,
    require_context,
    submit_with_context,
    tool,
)
from requisite.config.settings import Settings
from requisite.core.exceptions import ConfigurationException
from requisite.core.interfaces import ChatResponse, Role, ToolCall
from requisite.core.rate_limiter import RateLimiter
from requisite.core.sync_bridge import run_sync
from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry

# ---------------------------------------------------------------------------
# The carrier
# ---------------------------------------------------------------------------


def test_no_context_by_default() -> None:
    assert current_context() is None


def test_request_context_sets_and_restores() -> None:
    ctx = RequestContext(user="op-1")
    with request_context(ctx) as inside:
        assert inside is ctx
        assert current_context() is ctx
    assert current_context() is None


def test_request_context_nests_and_inner_overrides() -> None:
    outer, inner = RequestContext(user="outer"), RequestContext(user="inner")
    with request_context(outer):
        with request_context(inner):
            assert current_context() is inner
        assert current_context() is outer


def test_request_context_none_inherits_the_ambient_context() -> None:
    ctx = RequestContext(user="op-1")
    with request_context(ctx):
        with request_context(None):
            assert current_context() is ctx


def test_request_context_restores_after_exception() -> None:
    with pytest.raises(RuntimeError):
        with request_context(RequestContext(user="x")):
            raise RuntimeError("boom")
    assert current_context() is None


def test_context_is_immutable_and_evolve_returns_a_copy() -> None:
    ctx = RequestContext(user="a", tenant="t", correlation_id="c", attributes={"k": 1})
    changed = ctx.evolve(user="b")

    assert (ctx.user, changed.user, changed.tenant) == ("a", "b", "t")
    with pytest.raises(Exception):
        ctx.user = "z"  # type: ignore[misc]


def test_require_context_raises_a_helpful_error() -> None:
    with pytest.raises(ConfigurationException, match="context="):
        require_context()
    with request_context(RequestContext(user="u")):
        assert require_context().user == "u"


# ---------------------------------------------------------------------------
# Propagation primitives the feature relies on
# ---------------------------------------------------------------------------


def test_context_flows_through_the_sync_bridge() -> None:
    async def read() -> Optional[str]:
        ctx = current_context()
        return ctx.user if ctx else None

    with request_context(RequestContext(user="via-bridge")):
        assert run_sync(read()) == "via-bridge"


@pytest.mark.asyncio
async def test_context_flows_through_to_thread_and_tasks() -> None:
    def read() -> Optional[str]:
        ctx = current_context()
        return ctx.user if ctx else None

    with request_context(RequestContext(user="u")):
        assert await asyncio.to_thread(read) == "u"
        assert await asyncio.create_task(asyncio.to_thread(read)) == "u"


def test_plain_executor_submit_loses_context_but_submit_with_context_keeps_it() -> None:
    """The one gap: ThreadPoolExecutor.submit starts workers with an empty context."""

    def read() -> Optional[str]:
        ctx = current_context()
        return ctx.user if ctx else None

    with request_context(RequestContext(user="u")):
        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(read).result() is None
            assert submit_with_context(pool, read).result() == "u"


def test_submit_with_context_passes_arguments() -> None:
    with ThreadPoolExecutor(1) as pool:
        assert submit_with_context(pool, lambda a, b=0: a + b, 1, b=2).result() == 3


# ---------------------------------------------------------------------------
# Tool injection
# ---------------------------------------------------------------------------


@tool
def whoami(ctx: RequestContext) -> str:
    """Return the operator."""
    return f"{ctx.user}@{ctx.tenant}"


@tool
async def awhoami(ctx: RequestContext, note: str = "") -> str:
    """Return the operator (async)."""
    return f"{ctx.user}:{note}"


@tool
def maybe_whoami(ctx: Optional[RequestContext] = None) -> str:
    """Return the operator if there is one."""
    return ctx.user if ctx else "anonymous"


@tool
def optional_no_default(ctx: Optional[RequestContext]) -> str:
    """Optional with no default."""
    return ctx.user if ctx else "none"


def test_context_parameter_is_hidden_from_the_model_schema() -> None:
    assert whoami.tool.parameters_schema["properties"] == {}
    assert whoami.tool.parameters_schema["required"] == []
    assert list(awhoami.tool.parameters_schema["properties"]) == ["note"]


def test_context_is_injected_into_a_sync_tool() -> None:
    with request_context(RequestContext(user="op-7", tenant="acme")):
        assert whoami.tool.execute() == "op-7@acme"


@pytest.mark.asyncio
async def test_context_is_injected_into_async_and_threaded_tools() -> None:
    with request_context(RequestContext(user="op-8")):
        assert await awhoami.tool.aexecute(note="n") == "op-8:n"
        assert await whoami.tool.aexecute() == "op-8@None"  # sync tool, run via to_thread


def test_async_tool_executed_synchronously_sees_the_context() -> None:
    with request_context(RequestContext(user="op-9")):
        assert awhoami.tool.execute() == "op-9:"


def test_model_cannot_spoof_the_context_argument() -> None:
    forged = RequestContext(user="admin")
    with request_context(RequestContext(user="real")):
        assert whoami.tool.execute(ctx=forged) == "real@None"


def test_missing_context_is_a_clear_error_for_a_required_parameter() -> None:
    with pytest.raises(ToolException, match=r"whoami.*context="):
        whoami.tool.execute()


def test_missing_context_uses_the_default_for_an_optional_parameter() -> None:
    assert maybe_whoami.tool.execute() == "anonymous"
    assert optional_no_default.tool.execute() == "none"


def test_tools_without_a_context_parameter_are_unchanged() -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    assert add.tool.context_param is None
    assert add.tool.execute(a=1, b=2) == 3


def test_injection_survives_a_rename_copy() -> None:
    renamed = whoami.tool.model_copy(update={"name": "who"})
    with request_context(RequestContext(user="u")):
        assert renamed.execute() == "u@None"


def test_string_annotations_are_recognised() -> None:
    def handler(ctx: "RequestContext") -> str:
        """Stringly typed."""
        return ctx.user or ""

    built = Tool.from_function(handler)
    assert built.context_param == "ctx"
    assert built.parameters_schema["properties"] == {}


# ---------------------------------------------------------------------------
# Agent: context= reaches tools and providers; concurrent requests stay isolated
# ---------------------------------------------------------------------------


class _CallsWhoamiThenAnswers(BaseProvider):
    """Asks for the ``whoami`` tool, then answers with the tool result and the
    context its own provider call saw (proving providers can read it too)."""

    seen_by_provider: list[Optional[str]] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="k", model="m")

    @property
    def name(self) -> str:
        return "ctxprov"

    def chat(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> ChatResponse:
        ctx = current_context()
        type(self).seen_by_provider.append(ctx.user if ctx else None)
        last = messages[-1]
        if last.role == Role.TOOL:
            return ChatResponse(content=f"tool said {last.content}", model="m", provider=self.name)
        call = ToolCall(id="1", name="whoami", arguments={})
        return ChatResponse(content="", model="m", provider=self.name, tool_calls=[call])

    async def achat(self, messages: Any, **kwargs: Any) -> ChatResponse:
        await asyncio.sleep(0)  # yield so concurrent requests interleave
        return self.chat(messages, **kwargs)

    def stream(self, messages: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def astream(self, messages: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError
        yield


_FAST = RateLimiter(requests_per_minute=10**6)


def _agent(name: str = "A") -> Agent:
    registry = ProviderRegistry()
    registry.register("ctxprov", _CallsWhoamiThenAnswers)
    return Agent(
        name=name,
        provider="ctxprov",
        settings=Settings(default_provider="ctxprov", model="m"),
        registry=registry,
        tools=[whoami],
        rate_limiter=_FAST,
    )


def test_agent_run_passes_context_to_tools_and_providers() -> None:
    _CallsWhoamiThenAnswers.seen_by_provider = []
    result = _agent().run("go", context=RequestContext(user="op-1", tenant="acme"))

    assert result.content == "tool said op-1@acme"
    assert _CallsWhoamiThenAnswers.seen_by_provider == ["op-1", "op-1"]
    assert current_context() is None  # restored afterwards


@pytest.mark.asyncio
async def test_agent_arun_passes_context_to_tools_and_providers() -> None:
    result = await _agent().arun("go", context=RequestContext(user="op-2", tenant="acme"))
    assert result.content == "tool said op-2@acme"
    assert current_context() is None


def test_agent_context_is_not_forwarded_to_the_provider_as_a_kwarg() -> None:
    class StrictProvider(_CallsWhoamiThenAnswers):
        def chat(self, messages: Any, *, tools: Any = None, **kwargs: Any) -> ChatResponse:
            assert "context" not in kwargs
            return super().chat(messages, tools=tools, **kwargs)

    registry = ProviderRegistry()
    registry.register("strict", StrictProvider)
    agent = Agent(
        name="A",
        provider="strict",
        settings=Settings(default_provider="strict", model="m"),
        registry=registry,
        tools=[whoami],
        rate_limiter=_FAST,
    )
    assert agent.run("go", context=RequestContext(user="u")).content == "tool said u@None"


def test_agent_without_context_still_runs_for_tools_that_do_not_need_it() -> None:
    @tool
    def whoami() -> str:  # noqa: F811 - same tool name, no context parameter
        """Anonymous."""
        return "nobody"

    registry = ProviderRegistry()
    registry.register("ctxprov", _CallsWhoamiThenAnswers)
    agent = Agent(
        name="A",
        provider="ctxprov",
        settings=Settings(default_provider="ctxprov", model="m"),
        registry=registry,
        tools=[whoami],
        rate_limiter=_FAST,
    )
    assert agent.run("go").content == "tool said nobody"


@pytest.mark.asyncio
async def test_concurrent_requests_for_different_operators_stay_isolated() -> None:
    """The lab's one-ticket-at-a-time limit: many tickets in flight at once, each
    tool call must see its own operator."""
    agent = _agent()
    operators = [f"op-{i}" for i in range(25)]

    results = await asyncio.gather(
        *(agent.arun("go", context=RequestContext(user=op, tenant="acme")) for op in operators)
    )

    assert [r.content for r in results] == [f"tool said {op}@acme" for op in operators]


def test_concurrent_sync_runs_from_threads_stay_isolated() -> None:
    agent = _agent()
    operators = [f"op-{i}" for i in range(12)]
    out: dict[str, str] = {}

    def work(op: str) -> None:
        out[op] = agent.run("go", context=RequestContext(user=op)).content

    threads = [threading.Thread(target=work, args=(op,)) for op in operators]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert out == {op: f"tool said {op}@None" for op in operators}


def test_context_sets_correlation_id_on_the_run_span(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import requisite.agents.agent as agent_module

    exporter = InMemorySpanExporter()
    provider = TracerProvider()  # local only: never installed globally
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(agent_module, "_tracer", provider.get_tracer("requisite.agent"))

    _agent().run("go", context=RequestContext(user="u", correlation_id="t-1001"))

    spans = [s for s in exporter.get_finished_spans() if s.name == "requisite.agent.run"]
    assert spans[0].attributes["requisite.correlation_id"] == "t-1001"
    assert "requisite.user" not in spans[0].attributes  # identifiers stay off spans


# ---------------------------------------------------------------------------
# Workflow: context= and propagation through every native thread-pool strategy
# ---------------------------------------------------------------------------

from requisite import Workflow  # noqa: E402


def _workflow_agents(n: int) -> list[Agent]:
    return [_agent(f"agent-{i}") for i in range(n)]


_THREADED = {
    "parallel": (lambda w: w.parallel(), {}),
    "consensus": (lambda w: w.consensus(), {}),
    "map_reduce": (lambda w: w.map_reduce(), {"map_items": ["a", "b", "c"]}),
    "debate": (lambda w: w.debate(), {"max_rounds": 1}),
    "tree_of_thoughts": (lambda w: w.tree_of_thoughts(), {}),
}


@pytest.mark.parametrize("strategy", list(_THREADED))
def test_native_threaded_strategies_deliver_the_context_to_every_agent(strategy: str) -> None:
    """These strategies run agents on a ThreadPoolExecutor, which starts workers with an
    empty context unless the submit copies it."""
    configure, kwargs = _THREADED[strategy]
    _CallsWhoamiThenAnswers.seen_by_provider = []
    workflow = Workflow()
    for agent in _workflow_agents(3):
        workflow.add(agent)
    configure(workflow)

    try:
        workflow.run("go", context=RequestContext(user="op-1"), **kwargs)
    except AttributeError:
        # tree_of_thoughts' evaluator needs structured output, which this text-only fake
        # cannot produce. The thread-pool step under test ran before it.
        assert strategy == "tree_of_thoughts"

    assert _CallsWhoamiThenAnswers.seen_by_provider
    assert set(_CallsWhoamiThenAnswers.seen_by_provider) == {"op-1"}


@pytest.mark.asyncio
async def test_workflow_arun_context_reaches_tools_and_is_not_forwarded() -> None:
    workflow = Workflow().add(_agent()).sequential()
    result = await workflow.arun("go", context=RequestContext(user="op-3", tenant="acme"))
    assert result.content == "tool said op-3@acme"
    assert current_context() is None


def test_workflow_run_context_reaches_tools() -> None:
    workflow = Workflow().add(_agent()).sequential()
    assert workflow.run("go", context=RequestContext(user="op-4")).content == "tool said op-4@None"


def test_inner_agent_context_overrides_the_workflow_context() -> None:
    inner = _agent()

    class Wrapper:
        name = "wrapper"

        def run(self, prompt: str, **kwargs: Any) -> Any:
            return inner.run(prompt, context=RequestContext(user="inner"), **kwargs)

    seen = Workflow().add(Wrapper()).sequential().run("go", context=RequestContext(user="outer"))
    assert "inner" in seen.content


def test_sync_bridge_throwaway_thread_branch_keeps_the_context() -> None:
    """run_sync called from the bridge thread itself falls back to a helper thread."""

    async def read() -> Optional[str]:
        ctx = current_context()
        return ctx.user if ctx else None

    async def outer() -> Optional[str]:
        return run_sync(read())  # called ON the bridge thread

    with request_context(RequestContext(user="nested")):
        assert run_sync(outer()) == "nested"


# ---------------------------------------------------------------------------
# Third-party orchestrator backends deliver the context (each skipped if its SDK is absent)
# ---------------------------------------------------------------------------

_BACKENDS = [
    ("langgraph", "langgraph", "use_langgraph"),
    ("crewai", "crewai", "use_crewai"),
    ("autogen", "autogen_agentchat", "use_autogen"),
    ("adk", "google.adk", "use_adk"),
    ("openai_agents", "agents", "use_openai_agents"),
    ("strands", "strands", "use_strands"),
    ("agent_framework", "agent_framework", "use_agent_framework"),
]


@pytest.mark.parametrize(("label", "module", "use"), _BACKENDS, ids=[b[0] for b in _BACKENDS])
def test_backend_delivers_context_on_sync_run(label: str, module: str, use: str) -> None:
    pytest.importorskip(module)
    _CallsWhoamiThenAnswers.seen_by_provider = []
    workflow = getattr(Workflow().add(_agent()).sequential(), use)()

    result = workflow.run("go", context=RequestContext(user="op-5", tenant="acme"))

    assert "op-5@acme" in result.content
    assert set(_CallsWhoamiThenAnswers.seen_by_provider) == {"op-5"}


@pytest.mark.asyncio
@pytest.mark.parametrize(("label", "module", "use"), _BACKENDS, ids=[b[0] for b in _BACKENDS])
async def test_backend_delivers_context_on_async_run(label: str, module: str, use: str) -> None:
    pytest.importorskip(module)
    workflow = getattr(Workflow().add(_agent()).sequential(), use)()

    result = await workflow.arun("go", context=RequestContext(user="op-6", tenant="acme"))

    assert "op-6@acme" in result.content
