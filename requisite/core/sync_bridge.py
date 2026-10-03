"""
Run a coroutine to completion from synchronous code, on one long-lived loop.

Several orchestrator backends (AutoGen, ADK, the OpenAI Agents SDK,
Strands, Microsoft Agent Framework) are async-only underneath, so their
synchronous ``run()`` has to drive an async ``arun()``. Doing that with
``asyncio.run()`` creates -- and then closes -- a brand-new event loop per
call. That breaks any agent reused across calls: a provider caches its
async HTTP client, the client's connection pool binds to the loop it was
first used on, and the next ``asyncio.run()`` hands it a different loop
while the old one is closed, so the call dies with ``RuntimeError: Event
loop is closed``. Reproduced against the bare Gemini provider (an
httpx-only ``google-genai`` install: the second of two consecutive
``asyncio.run(agent.arun(...))`` calls fails) -- not specific to any one
backend.

:func:`run_sync` instead runs every coroutine on a single background event
loop that lives for the process, so cached async clients always see the
loop they were created on. As a side effect it also works when called from
a thread that already has a running loop (e.g. a notebook), where
``asyncio.run()`` raises.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, TypeVar

_T = TypeVar("_T")

_lock = threading.Lock()
_loop: Optional[asyncio.AbstractEventLoop] = None
_thread: Optional[threading.Thread] = None


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop, _thread
    with _lock:
        if _loop is None or _loop.is_closed():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, name="requisite-sync-bridge", daemon=True
            )
            thread.start()
            _loop, _thread = loop, thread
        return _loop


def run_sync(coroutine: Coroutine[Any, Any, _T]) -> _T:
    """Run ``coroutine`` to completion and return its result (or raise its exception).

    The coroutine always runs on the same background event loop, so state
    bound to that loop (async HTTP clients, connection pools) stays valid
    across calls. Exceptions propagate with their original type.
    """
    loop = _ensure_loop()
    if threading.current_thread() is _thread:
        # Called from code already running *on* the bridge loop (a sync tool
        # that itself starts a workflow, say). Blocking this thread on a
        # future scheduled on its own loop would deadlock, so run the
        # coroutine on a throwaway loop in a helper thread instead.
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coroutine).result()

    future = asyncio.run_coroutine_threadsafe(coroutine, loop)
    try:
        return future.result()
    except BaseException:
        future.cancel()
        raise
