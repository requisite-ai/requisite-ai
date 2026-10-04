"""
Request-scoped context: who a run is for, carried to tools and providers.

Pass a :class:`RequestContext` to ``Agent.run(..., context=...)`` (or
``Workflow.run``) and every tool and provider call made during that run can
read it, with no global state and no per-tool bookkeeping -- so concurrent
requests for different users stay isolated. The carrier is one
:class:`contextvars.ContextVar`, which already flows through ``await``,
``asyncio.create_task``, ``asyncio.to_thread`` and the sync bridge
(:func:`~requisite.core.sync_bridge.run_sync`). It does **not** flow through
``ThreadPoolExecutor.submit``; use :func:`submit_with_context` there.

The context is trusted, application-supplied data. The model never sees it
and cannot set it (a tool parameter annotated :class:`RequestContext` is
excluded from the schema and always overwritten). Derive access decisions from
it, never from message text. See
``docs/adr/0043-request-scoped-context.md``.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, Future
from contextlib import contextmanager
from typing import Any, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from requisite.core.exceptions import ConfigurationException

_T = TypeVar("_T")


class RequestContext(BaseModel):
    """Immutable, request-scoped identity and tracing data.

    Parameters
    ----------
    user:
        Who the request is on behalf of (an id, never a credential).
    tenant:
        The tenant/organisation the request belongs to.
    correlation_id:
        An id tying every span and log of one request together. Also set on
        the ``requisite.agent.run`` span when present.
    attributes:
        Free-form, application-defined data (a role, a ticket id, ...).
        Treat it as read-only: the context is shared by every tool and
        provider call in the run.

    Examples
    --------
    >>> ctx = RequestContext(user="op-42", tenant="acme", correlation_id="t-1001")
    >>> ctx.evolve(user="op-43").user
    'op-43'
    """

    model_config = ConfigDict(frozen=True)

    user: Optional[str] = None
    tenant: Optional[str] = None
    correlation_id: Optional[str] = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    def evolve(self, **changes: Any) -> "RequestContext":
        """Return a copy with ``changes`` applied (the original is frozen)."""
        return self.model_copy(update=changes)


_current: contextvars.ContextVar[Optional[RequestContext]] = contextvars.ContextVar(
    "requisite_request_context", default=None
)


def current_context() -> Optional[RequestContext]:
    """The :class:`RequestContext` of the run in progress, or ``None``."""
    return _current.get()


def require_context() -> RequestContext:
    """Like :func:`current_context` but raise if no context is in scope."""
    context = _current.get()
    if context is None:
        raise ConfigurationException(
            "No request context is in scope. Pass context=RequestContext(...) to "
            "Agent.run()/arun() or Workflow.run()/arun(), or wrap the call in "
            "'with request_context(...)'.",
        )
    return context


@contextmanager
def request_context(context: Optional[RequestContext]) -> Iterator[Optional[RequestContext]]:
    """Make ``context`` current for the ``with`` block, then restore the previous one.

    Nests: an inner context overrides the outer one for its duration. ``None``
    leaves whatever is already current untouched.
    """
    if context is None:
        yield _current.get()
        return
    token = _current.set(context)
    try:
        yield context
    finally:
        _current.reset(token)


def submit_with_context(
    executor: Executor, fn: Callable[..., _T], /, *args: Any, **kwargs: Any
) -> "Future[_T]":
    """``executor.submit(fn, ...)`` that runs ``fn`` in a copy of the caller's context.

    A plain ``ThreadPoolExecutor.submit`` starts the worker with an empty
    context, so :func:`current_context` would be ``None`` inside it.
    """
    return executor.submit(contextvars.copy_context().run, fn, *args, **kwargs)
