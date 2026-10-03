"""Unit tests for :mod:`requisite.core.sync_bridge`.

The bug it exists for: a provider caches an async HTTP client whose
connection pool binds to the event loop it was first used on, so calling
``asyncio.run()`` twice (a fresh loop each time, the first one closed)
makes the second call fail with ``RuntimeError: Event loop is closed``.
``_LoopBoundClient`` reproduces that behavior without any network.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from requisite.core.sync_bridge import run_sync


class _LoopBoundClient:
    """Like a cached httpx.AsyncClient: usable only on the loop it first ran on."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None

    async def call(self) -> str:
        running = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = running
        elif self._loop is not running:
            raise RuntimeError("Event loop is closed")
        await asyncio.sleep(0)
        return "ok"


def test_asyncio_run_twice_breaks_a_loop_bound_client() -> None:
    """Control: this is the failure the bridge fixes."""
    client = _LoopBoundClient()

    assert asyncio.run(client.call()) == "ok"
    with pytest.raises(RuntimeError, match="Event loop is closed"):
        asyncio.run(client.call())


def test_run_sync_keeps_a_loop_bound_client_working_across_calls() -> None:
    client = _LoopBoundClient()

    assert [run_sync(client.call()) for _ in range(3)] == ["ok", "ok", "ok"]


def test_run_sync_returns_the_coroutines_value() -> None:
    async def compute() -> int:
        return 21 * 2

    assert run_sync(compute()) == 42


def test_run_sync_propagates_the_original_exception_type() -> None:
    class CustomError(Exception):
        pass

    async def boom() -> None:
        raise CustomError("exploded")

    with pytest.raises(CustomError, match="exploded"):
        run_sync(boom())


def test_run_sync_runs_off_the_calling_thread_on_one_shared_loop() -> None:
    async def where() -> tuple[int, str]:
        return id(asyncio.get_running_loop()), threading.current_thread().name

    first_loop, first_thread = run_sync(where())
    second_loop, _ = run_sync(where())

    assert first_loop == second_loop
    assert first_thread == "requisite-sync-bridge"
    assert first_thread != threading.current_thread().name


@pytest.mark.asyncio
async def test_run_sync_works_from_inside_a_running_event_loop() -> None:
    """asyncio.run() raises here ('cannot be called from a running event loop')."""

    async def compute() -> str:
        return "from inside"

    assert run_sync(compute()) == "from inside"


def test_run_sync_called_from_the_bridge_loop_itself_does_not_deadlock() -> None:
    """A sync callable running on the bridge loop that starts another
    run_sync (e.g. a tool that runs a workflow) must not block forever."""

    async def inner() -> str:
        return "inner"

    async def outer() -> str:
        return run_sync(inner())

    assert run_sync(outer()) == "inner"


def test_run_sync_is_safe_under_concurrent_callers() -> None:
    results: list[int] = []

    async def double(n: int) -> int:
        await asyncio.sleep(0.01)
        return n * 2

    def worker(n: int) -> None:
        results.append(run_sync(double(n)))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == [n * 2 for n in range(8)]
