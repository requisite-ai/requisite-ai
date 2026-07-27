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
from requisite.core.interfaces import Message, ToolCall
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
    def from_text(cls, text: str) -> _FakePart:
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


# ---------------------------------------------------------------------------
# Anthropic provider tests
# ---------------------------------------------------------------------------


class _FakeAnthropicTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text
        self.parsed_output = None


class _FakeAnthropicToolUseBlock:
    def __init__(self, block_id: str, name: str, tool_input: dict[str, Any]) -> None:
        self.type = "tool_use"
        self.id = block_id
        self.name = name
        self.input = tool_input


class _FakeAnthropicMessage:
    def __init__(self, content: list[Any], model: str = "claude-sonnet-4-6") -> None:
        self.content = content
        self.model = model
        self.stop_reason = "end_turn"
        self.usage = types.SimpleNamespace(input_tokens=5, output_tokens=7)


class _FakeAnthropicClient:
    def __init__(self, **kwargs: Any) -> None:
        self.messages = types.SimpleNamespace(
            create=self._create, parse=self._parse, stream=self._stream
        )

    def _create(self, **kwargs: Any) -> _FakeAnthropicMessage:
        last_user = kwargs["messages"][-1]["content"]
        text = last_user if isinstance(last_user, str) else str(last_user)
        return _FakeAnthropicMessage([_FakeAnthropicTextBlock(f"echo: {text}")])

    def _parse(self, **kwargs: Any) -> _FakeAnthropicMessage:
        block = _FakeAnthropicTextBlock('{"answer": "ok"}')
        block.parsed_output = {"answer": "ok"}
        return _FakeAnthropicMessage([block])

    def _stream(self, **kwargs: Any) -> Any:  # pragma: no cover - exercised indirectly if needed
        raise NotImplementedError


def test_anthropic_provider_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropicClient)
    from requisite.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="sk-ant-test")
    response = provider.chat([Message.user("hi")])
    assert response.content == "echo: hi"
    assert response.provider == "anthropic"
    assert response.usage.total_tokens == 12


def test_anthropic_provider_parses_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic

    class _ToolCallingClient(_FakeAnthropicClient):
        def _create(self, **kwargs: Any) -> _FakeAnthropicMessage:
            return _FakeAnthropicMessage(
                [_FakeAnthropicToolUseBlock("call_1", "get_weather", {"city": "Paris"})]
            )

    monkeypatch.setattr(anthropic, "Anthropic", _ToolCallingClient)
    from requisite.providers.anthropic_provider import AnthropicProvider
    from requisite.tools.base import Tool

    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    provider = AnthropicProvider(api_key="sk-ant-test")
    response = provider.chat(
        [Message.user("weather in Paris?")], tools=[Tool.from_function(get_weather)]
    )
    assert response.has_tool_calls
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "Paris"}


def test_anthropic_provider_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic
    from pydantic import BaseModel

    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropicClient)
    from requisite.providers.anthropic_provider import AnthropicProvider

    class Answer(BaseModel):
        answer: str

    provider = AnthropicProvider(api_key="sk-ant-test")
    response = provider.chat([Message.user("...")], response_model=Answer)
    assert response.parsed == {"answer": "ok"}


def test_anthropic_provider_missing_sdk_raises_configuration_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    from requisite.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="sk-ant-test")
    with pytest.raises(ConfigurationException):
        provider.chat([Message.user("hi")])


def test_anthropic_provider_wraps_sdk_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic

    class _BoomClient(_FakeAnthropicClient):
        def _create(self, **kwargs: Any) -> Any:
            raise RuntimeError("rate limited")

    monkeypatch.setattr(anthropic, "Anthropic", _BoomClient)
    from requisite.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="sk-ant-test")
    with pytest.raises(ProviderException):
        provider.chat([Message.user("hi")])


def test_anthropic_message_conversion_merges_tool_results() -> None:
    from requisite.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(api_key="sk-ant-test")
    messages = [
        Message.system("be terse"),
        Message.user("weather in Paris?"),
        Message.assistant_tool_calls(
            [ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})]
        ),
        Message.tool_result("sunny", tool_call_id="call_1", name="get_weather"),
    ]
    converted, system = provider._to_anthropic_messages(messages)
    assert system == "be terse"
    assert converted[0]["role"] == "user"
    assert converted[1]["role"] == "assistant"
    assert converted[1]["content"][0]["type"] == "tool_use"
    assert converted[2]["role"] == "user"
    assert converted[2]["content"][0]["type"] == "tool_result"
    assert converted[2]["content"][0]["tool_use_id"] == "call_1"


# ---------------------------------------------------------------------------
# Groq provider tests (reuses the OpenAI-compatible fake client)
# ---------------------------------------------------------------------------


def test_groq_provider_uses_groq_base_url(fake_openai_module: types.ModuleType) -> None:
    captured_kwargs: dict[str, Any] = {}

    class _CapturingClient(_FakeOpenAIClient):
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)
            super().__init__(**kwargs)

    fake_openai_module.OpenAI = _CapturingClient  # type: ignore[attr-defined]

    from requisite.providers.groq_provider import GroqProvider

    provider = GroqProvider(api_key="gsk-test")
    response = provider.chat([Message.user("hi")])
    assert response.content == "echo: hi"
    assert response.provider == "groq"
    assert captured_kwargs["base_url"] == "https://api.groq.com/openai/v1"


# ---------------------------------------------------------------------------
# Azure OpenAI provider tests (reuses the OpenAI-compatible fake client)
# ---------------------------------------------------------------------------


def test_azure_openai_provider_builds_v1_base_url(fake_openai_module: types.ModuleType) -> None:
    captured_kwargs: dict[str, Any] = {}

    class _CapturingClient(_FakeOpenAIClient):
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)
            super().__init__(**kwargs)

    fake_openai_module.OpenAI = _CapturingClient  # type: ignore[attr-defined]

    from requisite.providers.azure_openai_provider import AzureOpenAIProvider

    provider = AzureOpenAIProvider(
        api_key="az-test",
        azure_endpoint="https://my-resource.openai.azure.com",
        model="gpt-4.1-nano",
    )
    response = provider.chat([Message.user("hi")])
    assert response.content == "echo: hi"
    assert response.provider == "azure_openai"
    assert captured_kwargs["base_url"] == "https://my-resource.openai.azure.com/openai/v1/"


def test_azure_openai_provider_requires_endpoint() -> None:
    from requisite.providers.azure_openai_provider import AzureOpenAIProvider

    with pytest.raises(ConfigurationException, match="azure_endpoint"):
        AzureOpenAIProvider(api_key="az-test")
