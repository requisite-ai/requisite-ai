"""
Proactive rate limiting for provider calls.

A single, dependency-free :class:`RateLimiter` shared across one or more
:class:`~requisite.ai.AI` / :class:`~requisite.agents.agent.Agent`
instances keeps their *combined* call rate under a real API quota (e.g.
a free-tier "N requests per minute" limit), by blocking until capacity
is available instead of letting the provider reject the call.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Optional

from requisite.core.exceptions import RateLimitException

_WINDOW_SECONDS = 60.0


class RateLimiter:
    """Blocks calls as needed to stay under ``requests_per_minute``.

    Uses a sliding-window log, not a token bucket: a call is allowed
    immediately if fewer than ``requests_per_minute`` calls happened in
    the trailing 60 seconds; otherwise it waits until the oldest call in
    that window ages out. This matches how providers enforce "N requests
    per minute" quotas -- e.g. Gemini's free tier -- more precisely than
    a bucket-refill approximation would.

    Share **one instance** across every :class:`~requisite.ai.AI` /
    :class:`~requisite.agents.agent.Agent` that draws on the same
    underlying API key/quota. A limiter scoped to a single instance only
    prevents that instance from exceeding the limit on its own -- it
    does nothing for the *combined* rate of several instances hitting
    the same real quota (e.g. several agents in one
    :class:`~requisite.workflows.workflow.Workflow`).

    Thread-safe for concurrent :meth:`acquire` calls (guarded by a
    ``threading.Lock``) and separately safe for concurrent :meth:`aacquire`
    calls (guarded by an ``asyncio.Lock``). A single instance used
    concurrently from *both* the sync and async path at once is not
    cross-safe between those two locks -- not a supported usage pattern.

    Parameters
    ----------
    requests_per_minute:
        Maximum number of calls allowed in any trailing 60-second window.
    max_wait_seconds:
        If waiting for capacity would take longer than this, raise
        :class:`~requisite.core.exceptions.RateLimitException` instead
        of waiting. ``None`` (default) waits as long as necessary.

    Examples
    --------
    >>> limiter = RateLimiter(requests_per_minute=15)
    >>> research = Agent(name="Researcher", provider="gemini", rate_limiter=limiter)  # doctest: +SKIP
    >>> writer = Agent(name="Writer", provider="gemini", rate_limiter=limiter)  # doctest: +SKIP
    """

    def __init__(
        self, *, requests_per_minute: int, max_wait_seconds: Optional[float] = None
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive.")
        self._limit = requests_per_minute
        self._max_wait_seconds = max_wait_seconds
        self._timestamps: "deque[float]" = deque()
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    def _claim_or_wait_time(self, now: float) -> Optional[float]:
        """Must be called with the relevant lock held.

        Evicts expired timestamps, then either claims a slot (returning
        ``None``) or reports how many seconds until one frees up.
        """
        while self._timestamps and self._timestamps[0] <= now - _WINDOW_SECONDS:
            self._timestamps.popleft()
        if len(self._timestamps) < self._limit:
            self._timestamps.append(now)
            return None
        return self._timestamps[0] + _WINDOW_SECONDS - now

    def _check_max_wait(self, wait_seconds: float) -> None:
        if self._max_wait_seconds is not None and wait_seconds > self._max_wait_seconds:
            raise RateLimitException(
                f"Rate limit exceeded: would need to wait {wait_seconds:.1f}s for "
                f"capacity, more than max_wait_seconds={self._max_wait_seconds}.",
            )

    def acquire(self) -> None:
        """Block (synchronously) until a call is allowed under the limit."""
        while True:
            with self._lock:
                wait_seconds = self._claim_or_wait_time(time.monotonic())
            if wait_seconds is None:
                return
            self._check_max_wait(wait_seconds)
            time.sleep(wait_seconds)

    async def aacquire(self) -> None:
        """Async counterpart to :meth:`acquire`."""
        while True:
            async with self._async_lock:
                wait_seconds = self._claim_or_wait_time(time.monotonic())
            if wait_seconds is None:
                return
            self._check_max_wait(wait_seconds)
            await asyncio.sleep(wait_seconds)
