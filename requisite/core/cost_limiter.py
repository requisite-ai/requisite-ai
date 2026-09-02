"""
Reactive dollar-budget limiting for provider calls.

Unlike :class:`~requisite.core.rate_limiter.RateLimiter`, which can gate
a call *before* it happens because request count is fully knowable in
advance, a dollar-cost budget cannot be enforced purely proactively:
prompt-token cost is knowable before a call, but completion-token cost
only exists once the provider responds. :class:`CostLimiter` is
therefore reactive -- :meth:`CostLimiter.check` raises before a call
only once *already-recorded* spend has reached the budget;
:meth:`CostLimiter.record` updates cumulative spend after each call
completes. One call can still push spend over budget; every call after
that raises immediately. This is a real, honest limitation of dollar
budgeting, not an oversight -- see docs/adr/0038-cost-based-rate-limiting.md.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Callable

from requisite.core.exceptions import CostLimitException

if TYPE_CHECKING:
    from requisite.core.interfaces import Usage

#: ``(usage, model) -> dollars`` for one completed call. Callers own
#: pricing -- see :func:`cost_per_token` for the common flat-rate case.
CostFn = Callable[["Usage", str], float]


def cost_per_token(*, prompt_rate_per_1k: float, completion_rate_per_1k: float) -> CostFn:
    """Build a flat, per-model :data:`CostFn` from $/1K-token rates.

    Parameters
    ----------
    prompt_rate_per_1k:
        Dollars per 1,000 prompt (input) tokens.
    completion_rate_per_1k:
        Dollars per 1,000 completion (output) tokens.

    Examples
    --------
    >>> cost_fn = cost_per_token(prompt_rate_per_1k=0.15, completion_rate_per_1k=0.60)
    >>> limiter = CostLimiter(budget_usd=10.0, cost_fn=cost_fn)  # doctest: +SKIP
    """

    def _cost(usage: "Usage", model: str) -> float:  # noqa: ARG001 - model unused in the flat-rate case
        return (
            usage.prompt_tokens / 1000 * prompt_rate_per_1k
            + usage.completion_tokens / 1000 * completion_rate_per_1k
        )

    return _cost


class CostLimiter:
    """Raises once cumulative recorded spend reaches ``budget_usd``.

    Reactive, not proactive -- see module docstring. Share **one
    instance** across every :class:`~requisite.ai.AI` /
    :class:`~requisite.agents.agent.Agent` that draws on the same
    budget, the same sharing pattern
    :class:`~requisite.core.rate_limiter.RateLimiter` already
    establishes for request-rate quotas.

    Never blocks or waits (unlike ``RateLimiter.acquire``, which can
    sleep), so a single ``threading.Lock`` is sufficient for both sync
    and async call sites -- no separate async lock is needed.

    Parameters
    ----------
    budget_usd:
        Total dollars allowed before further calls raise
        :class:`~requisite.core.exceptions.CostLimitException`.
    cost_fn:
        ``(usage, model) -> dollars`` for one completed call. Callers
        own pricing; see :func:`cost_per_token` for the common
        flat-rate case.

    Examples
    --------
    >>> limiter = CostLimiter(
    ...     budget_usd=10.0,
    ...     cost_fn=cost_per_token(prompt_rate_per_1k=0.15, completion_rate_per_1k=0.60),
    ... )
    >>> agent = Agent(name="Assistant", provider="openai", cost_limiter=limiter)  # doctest: +SKIP
    """

    def __init__(self, *, budget_usd: float, cost_fn: CostFn) -> None:
        if budget_usd <= 0:
            raise ValueError("budget_usd must be positive.")
        self._budget_usd = budget_usd
        self._cost_fn = cost_fn
        self._spent_usd = 0.0
        self._lock = threading.Lock()

    @property
    def spent_usd(self) -> float:
        """Total dollars recorded so far, since construction or the last :meth:`reset`."""
        with self._lock:
            return self._spent_usd

    @property
    def remaining_usd(self) -> float:
        """Dollars left before the next :meth:`check` raises, never negative."""
        with self._lock:
            return max(0.0, self._budget_usd - self._spent_usd)

    def check(self) -> None:
        """Raise :class:`CostLimitException` if the budget is already exhausted."""
        with self._lock:
            spent, budget = self._spent_usd, self._budget_usd
        if spent >= budget:
            raise CostLimitException(
                f"Cost budget exhausted: spent ${spent:.4f} of ${budget:.4f}.",
            )

    def record(self, usage: "Usage", model: str) -> float:
        """Record a completed call's cost against the budget.

        Returns
        -------
        float
            The dollar cost this call added, as computed by ``cost_fn``.
        """
        cost = self._cost_fn(usage, model)
        with self._lock:
            self._spent_usd += cost
        return cost

    def reset(self) -> None:
        """Zero out recorded spend, e.g. at the start of a new budget period."""
        with self._lock:
            self._spent_usd = 0.0
