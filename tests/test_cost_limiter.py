"""Unit tests for :class:`requisite.core.cost_limiter.CostLimiter`.

Unlike :class:`~requisite.core.rate_limiter.RateLimiter`, ``CostLimiter``
never blocks/waits, so these tests need no clock monkeypatching -- they
exercise real, instant accrual/exhaustion/reset behavior directly.
"""

from __future__ import annotations

import threading

import pytest

from requisite.core.cost_limiter import CostLimiter, cost_per_token
from requisite.core.exceptions import CostLimitException
from requisite.core.interfaces import Usage


def test_cost_limiter_requires_positive_budget() -> None:
    with pytest.raises(ValueError):
        CostLimiter(
            budget_usd=0, cost_fn=cost_per_token(prompt_rate_per_1k=1.0, completion_rate_per_1k=1.0)
        )


def test_cost_limiter_allows_calls_under_budget() -> None:
    limiter = CostLimiter(
        budget_usd=5.0, cost_fn=cost_per_token(prompt_rate_per_1k=1.0, completion_rate_per_1k=1.0)
    )
    usage = Usage(prompt_tokens=1000, completion_tokens=0, total_tokens=1000)  # $1/record

    limiter.check()  # should not raise, nothing spent yet
    limiter.record(usage, "fake-model")
    limiter.check()  # still under budget ($1 of $5)
    assert limiter.spent_usd == 1.0
    assert limiter.remaining_usd == 4.0


def test_cost_limiter_raises_once_budget_exhausted() -> None:
    limiter = CostLimiter(
        budget_usd=2.0, cost_fn=cost_per_token(prompt_rate_per_1k=1.0, completion_rate_per_1k=0.0)
    )
    usage = Usage(prompt_tokens=1000, completion_tokens=0, total_tokens=1000)  # $1/record

    limiter.record(usage, "fake-model")  # spent $1
    limiter.check()  # $1 < $2, fine
    limiter.record(usage, "fake-model")  # spent $2, exactly at budget

    with pytest.raises(CostLimitException, match=r"\$2\.0000 of \$2\.0000"):
        limiter.check()
    assert limiter.remaining_usd == 0.0


def test_cost_limiter_reset_clears_spend() -> None:
    limiter = CostLimiter(
        budget_usd=1.0, cost_fn=cost_per_token(prompt_rate_per_1k=1.0, completion_rate_per_1k=0.0)
    )
    usage = Usage(prompt_tokens=1000, completion_tokens=0, total_tokens=1000)  # $1/record

    limiter.record(usage, "fake-model")
    with pytest.raises(CostLimitException):
        limiter.check()

    limiter.reset()
    assert limiter.spent_usd == 0.0
    limiter.check()  # should not raise anymore


def test_cost_per_token_computes_expected_dollar_amount() -> None:
    cost_fn = cost_per_token(prompt_rate_per_1k=0.15, completion_rate_per_1k=0.60)
    usage = Usage(prompt_tokens=2000, completion_tokens=1000, total_tokens=3000)

    cost = cost_fn(usage, "gpt-4o-mini")

    assert cost == pytest.approx(2 * 0.15 + 1 * 0.60)


def test_cost_limiter_record_returns_the_dollar_amount_recorded() -> None:
    limiter = CostLimiter(
        budget_usd=100.0, cost_fn=cost_per_token(prompt_rate_per_1k=1.0, completion_rate_per_1k=2.0)
    )
    usage = Usage(prompt_tokens=1000, completion_tokens=1000, total_tokens=2000)

    cost = limiter.record(usage, "fake-model")

    assert cost == pytest.approx(3.0)
    assert limiter.spent_usd == pytest.approx(3.0)


def test_cost_limiter_shared_across_threads_tracks_spend_correctly() -> None:
    """No lost updates: N threads each recording M calls of known cost
    must sum to exactly N * M * cost, the cost-budget analogue of
    RateLimiter's own concurrent-sharing invariant test."""
    limiter = CostLimiter(
        budget_usd=1_000_000.0,
        cost_fn=cost_per_token(prompt_rate_per_1k=1.0, completion_rate_per_1k=0.0),
    )
    usage = Usage(prompt_tokens=1000, completion_tokens=0, total_tokens=1000)  # $1/record

    def worker() -> None:
        for _ in range(200):
            limiter.record(usage, "fake-model")

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert limiter.spent_usd == pytest.approx(10 * 200 * 1.0)
