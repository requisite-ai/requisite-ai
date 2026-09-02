"""
CostLimiter example: capping real dollar spend, independent of and
composable with RateLimiter (one paces call rate, the other caps call
cost). Pricing is a caller-supplied cost_fn -- no price table is
shipped or maintained by the framework.

The budget below is set deliberately low so the exhaustion path
actually triggers against real usage numbers, not just a scripted one.

Run with:
    GEMINI_API_KEY=... python examples/cost_limiter_example.py
"""

from requisite import Agent, CostLimitException, CostLimiter, cost_per_token

# Illustrative Gemini 2.5 Flash-ish rates, not guaranteed current --
# always confirm against your provider's own current pricing page.
cost_fn = cost_per_token(prompt_rate_per_1k=0.000075, completion_rate_per_1k=0.0003)

# Deliberately tiny: even a couple of short exchanges should exhaust it,
# so this example's exhaustion path is exercised against real usage
# numbers rather than requiring a long, expensive run to demonstrate.
budget = CostLimiter(budget_usd=0.000005, cost_fn=cost_fn)

agent = Agent(
    name="Assistant",
    provider="gemini",
    system_prompt="You are a concise assistant. Answer in one short sentence.",
    cost_limiter=budget,
)


def main() -> None:
    prompts = [
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the capital of Peru?",
    ]

    for prompt in prompts:
        try:
            result = agent.run(prompt)
        except CostLimitException as exc:
            print(f"\nBudget exhausted, call blocked before reaching the provider: {exc}")
            break
        print(f"\n{prompt}\n{result.content}")
        print(f"(spent so far: ${budget.spent_usd:.6f}, remaining: ${budget.remaining_usd:.6f})")

    print(f"\nFinal: spent ${budget.spent_usd:.6f}, remaining ${budget.remaining_usd:.6f}")

    # A caller in control of the budget period resets it explicitly --
    # e.g. at the start of a new day/month, on their own schedule.
    budget.reset()
    print(f"After reset(): spent ${budget.spent_usd:.6f}")


if __name__ == "__main__":
    main()
