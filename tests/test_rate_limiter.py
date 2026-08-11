"""Unit tests for :class:`requisite.core.rate_limiter.RateLimiter`.

Tests monkeypatch the module's ``_WINDOW_SECONDS`` constant down to a
fraction of a second rather than mocking the clock, so they exercise the
real ``time.sleep``/``asyncio.sleep`` blocking behavior without the
suite actually waiting a full 60-second window.
"""

from __future__ import annotations

import threading
import time

import pytest

from requisite.core import rate_limiter as rate_limiter_module
from requisite.core.exceptions import RateLimitException
from requisite.core.rate_limiter import RateLimiter


def test_rate_limiter_requires_positive_requests_per_minute() -> None:
    with pytest.raises(ValueError):
        RateLimiter(requests_per_minute=0)


def test_rate_limiter_allows_calls_up_to_the_limit_without_blocking() -> None:
    limiter = RateLimiter(requests_per_minute=3)
    start = time.monotonic()
    for _ in range(3):
        limiter.acquire()
    assert time.monotonic() - start < 0.5


def test_rate_limiter_blocks_until_window_frees_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rate_limiter_module, "_WINDOW_SECONDS", 0.2)
    limiter = RateLimiter(requests_per_minute=1)

    limiter.acquire()
    start = time.monotonic()
    limiter.acquire()
    elapsed = time.monotonic() - start

    assert elapsed >= 0.1


def test_rate_limiter_raises_when_wait_exceeds_max_wait_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rate_limiter_module, "_WINDOW_SECONDS", 5.0)
    limiter = RateLimiter(requests_per_minute=1, max_wait_seconds=0.05)

    limiter.acquire()
    with pytest.raises(RateLimitException, match="max_wait_seconds"):
        limiter.acquire()


@pytest.mark.asyncio
async def test_rate_limiter_aacquire_blocks_until_window_frees_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rate_limiter_module, "_WINDOW_SECONDS", 0.2)
    limiter = RateLimiter(requests_per_minute=1)

    await limiter.aacquire()
    start = time.monotonic()
    await limiter.aacquire()
    elapsed = time.monotonic() - start

    assert elapsed >= 0.1


@pytest.mark.asyncio
async def test_rate_limiter_aacquire_raises_when_wait_exceeds_max_wait_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rate_limiter_module, "_WINDOW_SECONDS", 5.0)
    limiter = RateLimiter(requests_per_minute=1, max_wait_seconds=0.05)

    await limiter.aacquire()
    with pytest.raises(RateLimitException, match="max_wait_seconds"):
        await limiter.aacquire()


def test_rate_limiter_shared_across_threads_never_exceeds_limit_in_any_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The invariant that matters: no more than `requests_per_minute` calls
    ever land within any trailing window, even under concurrent callers.

    Window is 1.0s, not a smaller fraction -- this test measures wall-clock
    timestamps recorded *after* `acquire()` returns (plus a separate lock +
    list-append), not the limiter's own internal claim times, so it's
    inherently sensitive to real OS thread-scheduling jitter (GIL
    contention, a loaded/noisy CI runner). A too-tight window turns
    ordinary scheduling delay into a false failure even when `acquire()`
    itself claimed slots correctly; 1.0s gives enough headroom to absorb
    that jitter while still keeping the test fast (worst case ~3s for 6
    threads at a limit of 2).
    """
    monkeypatch.setattr(rate_limiter_module, "_WINDOW_SECONDS", 1.0)
    limiter = RateLimiter(requests_per_minute=2)
    call_times: list[float] = []
    lock = threading.Lock()

    def worker() -> None:
        limiter.acquire()
        with lock:
            call_times.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(call_times) == 6
    call_times.sort()
    for call_time in call_times:
        calls_in_window = sum(1 for other in call_times if call_time - 1.0 < other <= call_time)
        assert calls_in_window <= 2
