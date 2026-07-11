"""
Capability resolution example.

Shows `agent.requires(...)` -- declaring *what* an agent needs rather
than binding it to one specific tool -- plus how a higher-priority
provider automatically takes over for a capability.

Run with:
    OPENAI_API_KEY=sk-... python examples/capability_example.py
"""

from requisite import Agent


def main() -> None:
    # The three built-in capabilities work with zero setup: "weather" and
    # "internet_search" call free, keyless public APIs; "filesystem" reads
    # local files. See requisite/capabilities/resolvers.py.
    agent = Agent(name="Assistant", provider="openai")
    agent.requires("weather", "internet_search", "filesystem")

    result = agent.run("What's the weather in Tokyo right now?")
    print(result.content)
    print(f"Tools used: {result.tool_calls_executed}")

    # --- Overriding a capability with a better implementation ---
    #
    # Register a higher-priority provider for "weather" (e.g. a paid API),
    # gated on an API key being configured. Agent code that already calls
    # `.requires("weather")` picks it up automatically -- no code changes.
    from requisite.capabilities import default_registry
    import os

    def paid_weather_api(city: str) -> str:
        """Paid, more accurate weather provider."""
        return f"[precise forecast for {city}]"

    default_registry.register(
        "weather",
        paid_weather_api,
        provider_name="acme-weather",
        priority=10,
        is_available=lambda: bool(os.environ.get("ACME_WEATHER_API_KEY")),
    )

    agent2 = Agent(name="Assistant2", provider="openai")
    agent2.requires("weather")  # uses acme-weather if ACME_WEATHER_API_KEY is set, else falls back
    print(agent2.run("Weather in Paris?").content)


if __name__ == "__main__":
    main()
