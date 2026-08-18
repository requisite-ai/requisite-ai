"""Unit tests for :class:`requisite.capabilities.registry.CapabilityRegistry`
and :meth:`requisite.agents.agent.Agent.requires`.
"""

from __future__ import annotations

import io
import json
import urllib.error
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any
from unittest.mock import patch

import pytest

from requisite.agents.agent import Agent
from requisite.capabilities.registry import CapabilityRegistry
from requisite.capabilities.resolvers import register_default_capabilities, search_github
from requisite.config.settings import Settings
from requisite.core.exceptions import CapabilityException
from requisite.core.interfaces import ChatResponse, Message, StreamChunk
from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry


def stub_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"stub-sunny in {city}"


def pro_weather(city: str) -> str:
    """Paid weather provider."""
    return f"pro-forecast for {city}"


def test_register_and_resolve_single_provider() -> None:
    registry = CapabilityRegistry()
    registry.register("weather", stub_weather, provider_name="stub")
    tool = registry.resolve("weather")
    assert tool.execute(city="Paris") == "stub-sunny in Paris"


def test_resolve_prefers_higher_priority_when_available() -> None:
    registry = CapabilityRegistry()
    registry.register("weather", stub_weather, provider_name="stub", priority=0)
    registry.register(
        "weather", pro_weather, provider_name="pro", priority=10, is_available=lambda: True
    )
    tool = registry.resolve("weather")
    assert tool.execute(city="Paris") == "pro-forecast for Paris"


def test_resolve_falls_back_when_higher_priority_unavailable() -> None:
    registry = CapabilityRegistry()
    registry.register("weather", stub_weather, provider_name="stub", priority=0)
    registry.register(
        "weather", pro_weather, provider_name="pro", priority=10, is_available=lambda: False
    )
    tool = registry.resolve("weather")
    assert tool.execute(city="Paris") == "stub-sunny in Paris"


def test_resolve_unknown_capability_raises() -> None:
    registry = CapabilityRegistry()
    with pytest.raises(CapabilityException):
        registry.resolve("does-not-exist")


def test_resolve_raises_when_all_providers_unavailable() -> None:
    registry = CapabilityRegistry()
    registry.register("weather", stub_weather, provider_name="stub", is_available=lambda: False)
    with pytest.raises(CapabilityException):
        registry.resolve("weather")


def test_providers_for_returns_all_ranked_by_priority() -> None:
    registry = CapabilityRegistry()
    registry.register("weather", stub_weather, provider_name="stub", priority=0)
    registry.register("weather", pro_weather, provider_name="pro", priority=10)
    providers = registry.providers_for("weather")
    assert [p.provider_name for p in providers] == ["pro", "stub"]


def test_list_capabilities_and_contains() -> None:
    registry = CapabilityRegistry()
    registry.register("weather", stub_weather)
    assert "weather" in registry
    assert registry.list_capabilities() == ["weather"]


def test_unregister_removes_specific_provider() -> None:
    registry = CapabilityRegistry()
    registry.register("weather", stub_weather, provider_name="stub")
    registry.register("weather", pro_weather, provider_name="pro")
    registry.unregister("weather", "pro")
    assert [p.provider_name for p in registry.providers_for("weather")] == ["stub"]


def test_default_capabilities_are_registered() -> None:
    registry = CapabilityRegistry()
    register_default_capabilities(registry)
    assert set(registry.list_capabilities()) == {
        "filesystem",
        "weather",
        "internet_search",
        "github",
    }
    for capability in registry.list_capabilities():
        # All four ship always-available (no API key required).
        assert registry.resolve(capability) is not None


# ---------------------------------------------------------------------------
# search_github (the "github" capability's implementation)
# ---------------------------------------------------------------------------


def _github_search_response(payload: dict[str, Any]) -> io.BytesIO:
    return io.BytesIO(json.dumps(payload).encode("utf-8"))


def test_search_github_formats_results() -> None:
    payload = {
        "items": [
            {
                "full_name": "octocat/Hello-World",
                "stargazers_count": 42,
                "description": "My first repository on GitHub!",
                "html_url": "https://github.com/octocat/Hello-World",
            }
        ]
    }
    with patch(
        "urllib.request.urlopen", return_value=_github_search_response(payload)
    ) as mock_urlopen:
        result = search_github("hello world")

    assert "octocat/Hello-World" in result
    assert "42 stars" in result
    assert "My first repository on GitHub!" in result
    assert "https://github.com/octocat/Hello-World" in result
    # GitHub's API rejects requests with no User-Agent header -- confirm we send one.
    request = mock_urlopen.call_args[0][0]
    assert request.get_header("User-agent") == "requisite-ai"


def test_search_github_no_results() -> None:
    with patch("urllib.request.urlopen", return_value=_github_search_response({"items": []})):
        result = search_github("zzzznonexistentrepoquery")

    assert "No GitHub repositories found" in result


def test_search_github_rate_limited_returns_actionable_message() -> None:
    error = urllib.error.HTTPError(
        url="https://api.github.com/search/repositories",
        code=403,
        msg="Forbidden",
        hdrs=None,  # type: ignore[arg-type]
        fp=None,
    )
    with patch("urllib.request.urlopen", side_effect=error):
        result = search_github("query")

    assert "rate limited" in result


def test_search_github_generic_http_error() -> None:
    error = urllib.error.HTTPError(
        url="https://api.github.com/search/repositories",
        code=500,
        msg="Server Error",
        hdrs=None,  # type: ignore[arg-type]
        fp=None,
    )
    with patch("urllib.request.urlopen", side_effect=error):
        result = search_github("query")

    assert "HTTP 500" in result


def test_search_github_network_error() -> None:
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("boom")):
        result = search_github("query")

    assert "Error searching GitHub" in result


# ---------------------------------------------------------------------------
# Agent.requires() integration
# ---------------------------------------------------------------------------


class EchoProvider(BaseProvider):
    """A minimal fake provider used to construct an Agent for these tests."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model="fake-model")

    @property
    def name(self) -> str:
        return "echo"

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model=None,
        temperature=None,
        tools=None,
        response_model=None,
        **kwargs,
    ) -> ChatResponse:
        return ChatResponse(content="ok", model=self._model, provider=self.name)

    async def achat(
        self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kwargs
    ) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def stream(
        self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kwargs
    ) -> Iterator[StreamChunk]:  # pragma: no cover
        raise NotImplementedError

    async def astream(
        self, messages, *, model=None, temperature=None, tools=None, response_model=None, **kwargs
    ) -> AsyncIterator[StreamChunk]:  # pragma: no cover
        raise NotImplementedError
        yield  # pragma: no cover


@pytest.fixture
def provider_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("echo", EchoProvider)
    return registry


@pytest.fixture
def settings() -> Settings:
    return Settings(default_provider="echo", model="fake-model")


@pytest.fixture
def capability_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register("weather", stub_weather, provider_name="stub")
    return registry


def test_agent_requires_via_constructor(
    provider_registry: ProviderRegistry, settings: Settings, capability_registry: CapabilityRegistry
) -> None:
    agent = Agent(
        name="Assistant",
        provider="echo",
        settings=settings,
        registry=provider_registry,
        requires=["weather"],
        capability_registry=capability_registry,
    )
    assert "weather" in agent.tools
    assert agent.tools.get("weather").execute(city="Rome") == "stub-sunny in Rome"


def test_agent_requires_fluent_method_returns_self(
    provider_registry: ProviderRegistry, settings: Settings, capability_registry: CapabilityRegistry
) -> None:
    agent = Agent(
        name="Assistant",
        provider="echo",
        settings=settings,
        registry=provider_registry,
        capability_registry=capability_registry,
    )
    result = agent.requires("weather")
    assert result is agent
    assert "weather" in agent.tools


def test_agent_requires_unknown_capability_raises(
    provider_registry: ProviderRegistry, settings: Settings, capability_registry: CapabilityRegistry
) -> None:
    agent = Agent(
        name="Assistant",
        provider="echo",
        settings=settings,
        registry=provider_registry,
        capability_registry=capability_registry,
    )
    with pytest.raises(CapabilityException):
        agent.requires("does-not-exist")


def test_agent_defaults_to_the_builtin_capability_registry(
    provider_registry: ProviderRegistry, settings: Settings
) -> None:
    agent = Agent(name="Assistant", provider="echo", settings=settings, registry=provider_registry)
    # No explicit capability_registry passed -- falls back to the
    # framework's default, pre-populated registry.
    agent.requires("weather")
    assert "weather" in agent.tools
