"""Unit tests for the provider registry, plus OpenAI/Gemini providers.

The OpenAI and Gemini SDKs are faked out via ``sys.modules`` injection so
these tests run without the real ``openai`` / ``google-genai`` packages
installed, and without any network access -- consistent with the
"mock providers, mock LLM responses" testing standard.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from requisite.core.exceptions import ConfigurationException, ProviderException
from requisite.core.interfaces import Message
from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry, default_registry


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class DummyProvider(BaseProvider):
    @property
    def name(self) -> str:
        return "dummy"

    def chat(self, messages, *, model=None, temperature=None, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def achat(self, messages, *, model=None, temperature=None, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def stream(self, messages, *, model=None, temperature=None, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def astream(
        self, messages, *, model=None, temperature=None, **kwargs
    ):  # pragma: no cover
        raise NotImplementedError


def test_default_registry_knows_builtin_providers() -> None:
    assert "openai" in default_registry.available()
    assert "gemini" in default_registry.available()


def test_register_and_create_custom_provider() -> None:
    registry = ProviderRegistry()
    registry.register("dummy", DummyProvider)
    provider = registry.create("dummy", api_key="k", model="m")
    assert isinstance(provider, DummyProvider)
    assert provider.name == "dummy"


def test_create_unknown_provider_raises() -> None:
    registry = ProviderRegistry()
    with pytest.raises(ConfigurationException):
        registry.create("nonexistent")


def test_register_empty_name_raises() -> None:
    registry = ProviderRegistry()
    with pytest.raises(ConfigurationException):
        registry.register("", DummyProvider)


def test_unregister_is_idempotent() -> None:
    registry = ProviderRegistry()
    registry.register("dummy", DummyProvider)
    registry.unregister("dummy")
    registry.unregister("dummy")  # no error
    assert not registry.is_registered("dummy")


def test_validate_config_requires_api_key() -> None:
    provider = DummyProvider(model="m")
    with pytest.raises(ConfigurationException):
        provider.validate_config()


# ---------------------------------------------------------------------------
# OpenAI provider tests (SDK faked via sys.modules)
# ---------------------------------------------------------------------------


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = types.SimpleNamespace(content=content)
        self.finish_reason = "stop"


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.model = "gpt-4o-mini"
        self.usage = types.SimpleNamespace(prompt_tokens=5, completion_tokens=7, total_tokens=12)


class _FakeOpenAIClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> _FakeCompletion:
        return _FakeCompletion(f"echo: {kwargs['messages'][-1]['content']}")


@pytest.fixture
def fake_openai_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAIClient  # type: ignore[attr-defined]
    fake_module.AsyncOpenAI = _FakeOpenAIClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return fake_module


def test_openai_provider_chat(fake_openai_module: types.ModuleType) -> None:
    from requisite.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini")
    response = provider.chat([Message.user("hi")])
    assert response.content == "echo: hi"
    assert response.provider == "openai"
    assert response.usage.total_tokens == 12


def test_openai_provider_missing_sdk_raises_configuration_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)  # simulate ImportError
    from requisite.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini")
    with pytest.raises(ConfigurationException):
        provider.chat([Message.user("hi")])


def test_openai_provider_wraps_sdk_errors(fake_openai_module: types.ModuleType) -> None:
    def _boom(**kwargs: Any) -> Any:
        raise RuntimeError("rate limited")

    fake_openai_module.OpenAI = lambda **kwargs: types.SimpleNamespace(  # type: ignore[attr-defined]
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_boom))
    )

    from requisite.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini")
    with pytest.raises(ProviderException):
        provider.chat([Message.user("hi")])


class _FakeFunctionCall:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = _FakeFunctionCall(name, arguments)


def test_openai_provider_parses_tool_calls(fake_openai_module: types.ModuleType) -> None:
    def _create_with_tool_call(**kwargs: Any) -> _FakeCompletion:
        completion = _FakeCompletion("")
        completion.choices[0].message.tool_calls = [
            _FakeToolCall("call_1", "get_weather", '{"city": "Paris"}')
        ]
        return completion

    fake_openai_module.OpenAI = lambda **kwargs: types.SimpleNamespace(  # type: ignore[attr-defined]
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_create_with_tool_call))
    )

    from requisite.providers.openai_provider import OpenAIProvider
    from requisite.tools.base import Tool

    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini")
    response = provider.chat(
        [Message.user("weather in Paris?")], tools=[Tool.from_function(get_weather)]
    )
    assert response.has_tool_calls
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "Paris"}


# ---------------------------------------------------------------------------
# Gemini provider tests (SDK faked via sys.modules)
# ---------------------------------------------------------------------------


class _FakePart:
    def __init__(self, text: str) -> None:
        self.text = text

    @classmethod
    def from_text(cls, text: str) -> "_FakePart":
        return cls(text)


class _FakeContent:
    def __init__(self, role: str, parts: list[_FakePart]) -> None:
        self.role = role
        self.parts = parts


class _FakeGenerateContentConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeGeminiResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = types.SimpleNamespace(
            prompt_token_count=3, candidates_token_count=4, total_token_count=7
        )
        self.candidates = [types.SimpleNamespace(finish_reason="STOP")]


class _FakeGeminiModels:
    def generate_content(
        self, *, model: str, contents: list[Any], config: Any
    ) -> _FakeGeminiResponse:
        last_text = contents[-1].parts[0].text if contents else ""
        return _FakeGeminiResponse(f"echo: {last_text}")


class _FakeGeminiClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.models = _FakeGeminiModels()


@pytest.fixture
def fake_genai_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake_types_module = types.ModuleType("google.genai.types")
    fake_types_module.Content = _FakeContent  # type: ignore[attr-defined]
    fake_types_module.Part = _FakePart  # type: ignore[attr-defined]
    fake_types_module.GenerateContentConfig = _FakeGenerateContentConfig  # type: ignore[attr-defined]

    fake_genai_module = types.ModuleType("google.genai")
    fake_genai_module.Client = _FakeGeminiClient  # type: ignore[attr-defined]
    fake_genai_module.types = fake_types_module  # type: ignore[attr-defined]

    fake_google_module = types.ModuleType("google")
    fake_google_module.genai = fake_genai_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", fake_google_module)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types_module)
    return fake_genai_module


def test_gemini_provider_chat(fake_genai_module: types.ModuleType) -> None:
    from requisite.providers.gemini_provider import GeminiProvider

    provider = GeminiProvider(api_key="g-test", model="gemini-2.5-flash")
    response = provider.chat([Message.user("hi")])
    assert response.content == "echo: hi"
    assert response.provider == "gemini"
    assert response.usage.total_tokens == 7


def test_gemini_provider_missing_sdk_raises_configuration_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "google", None)
    monkeypatch.setitem(sys.modules, "google.genai", None)
    from requisite.providers.gemini_provider import GeminiProvider

    provider = GeminiProvider(api_key="g-test", model="gemini-2.5-flash")
    with pytest.raises(ConfigurationException):
        provider.chat([Message.user("hi")])
