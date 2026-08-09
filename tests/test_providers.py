"""Unit tests for the provider registry, plus OpenAI/Gemini providers.

The OpenAI and Gemini SDKs are faked out via ``sys.modules`` injection so
these tests run without the real ``openai`` / ``google-genai`` packages
installed, and without any network access -- consistent with the
"mock providers, mock LLM responses" testing standard.
"""

from __future__ import annotations

import sys
import types
from typing import Any, ClassVar, Optional

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
    assert "anthropic" in default_registry.available()
    assert "groq" in default_registry.available()
    assert "azure_openai" in default_registry.available()
    assert "openrouter" in default_registry.available()
    assert "together" in default_registry.available()
    assert "ollama" in default_registry.available()


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


class _FakeGeminiFunctionCall:
    def __init__(self, name: str, args: dict[str, Any]) -> None:
        self.name = name
        self.args = args


class _FakePart:
    def __init__(
        self,
        *,
        text: Optional[str] = None,
        function_call: Optional[_FakeGeminiFunctionCall] = None,
        thought_signature: Optional[bytes] = None,
    ) -> None:
        self.text = text
        self.function_call = function_call
        self.thought_signature = thought_signature

    @classmethod
    def from_text(cls, text: str) -> "_FakePart":
        return cls(text=text)

    @classmethod
    def from_function_call(cls, name: str, args: dict[str, Any]) -> "_FakePart":
        return cls(function_call=_FakeGeminiFunctionCall(name, args))

    @classmethod
    def from_function_response(cls, name: str, response: dict[str, Any]) -> "_FakePart":
        return cls(text=None)


class _FakeContent:
    def __init__(self, role: str, parts: list[_FakePart]) -> None:
        self.role = role
        self.parts = parts


class _FakeCandidate:
    def __init__(self, content: _FakeContent, *, finish_reason: str = "STOP") -> None:
        self.content = content
        self.finish_reason = finish_reason


class _FakeGenerateContentConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeGeminiResponse:
    def __init__(self, parts: list[_FakePart], *, finish_reason: str = "STOP") -> None:
        self.usage_metadata = types.SimpleNamespace(
            prompt_token_count=3, candidates_token_count=4, total_token_count=7
        )
        self.candidates = [
            _FakeCandidate(_FakeContent("model", parts), finish_reason=finish_reason)
        ]


class _FakeGeminiModels:
    """Real ``.models.generate_content``, plus a test hook.

    Tests that need a specific response shape (e.g. a function call with a
    ``thought_signature``) set ``next_response`` via ``monkeypatch`` before
    calling into the provider; it's consumed once and cleared. Absent that,
    ``generate_content`` echoes the last message's text, matching the
    original fixture's behavior.
    """

    next_response: ClassVar[Optional[_FakeGeminiResponse]] = None

    def generate_content(
        self, *, model: str, contents: list[Any], config: Any
    ) -> _FakeGeminiResponse:
        if _FakeGeminiModels.next_response is not None:
            response = _FakeGeminiModels.next_response
            _FakeGeminiModels.next_response = None
            return response
        last_text = contents[-1].parts[0].text if contents else ""
        return _FakeGeminiResponse([_FakePart.from_text(f"echo: {last_text}")])


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


def test_gemini_provider_captures_thought_signature_on_tool_call(
    fake_genai_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A function-call part's thought_signature is captured into ToolCall.provider_data."""
    from requisite.providers.gemini_provider import GeminiProvider

    part = _FakePart.from_function_call(name="search_weather", args={"city": "Paris"})
    part.thought_signature = b"sig-bytes"
    monkeypatch.setattr(_FakeGeminiModels, "next_response", _FakeGeminiResponse([part]))

    provider = GeminiProvider(api_key="g-test", model="gemini-2.5-flash")
    response = provider.chat([Message.user("weather in paris")])

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "search_weather"
    assert response.tool_calls[0].arguments == {"city": "Paris"}
    assert response.tool_calls[0].provider_data == b"sig-bytes"


def test_gemini_provider_echoes_thought_signature_on_next_turn(
    fake_genai_module: types.ModuleType,
) -> None:
    """A ToolCall.provider_data captured from a prior turn is echoed back verbatim."""
    from requisite.providers.gemini_provider import GeminiProvider

    provider = GeminiProvider(api_key="g-test", model="gemini-2.5-flash")
    tool_call = ToolCall(
        id="search_weather-0",
        name="search_weather",
        arguments={"city": "Paris"},
        provider_data=b"sig-bytes",
    )
    message = Message.assistant_tool_calls([tool_call])

    contents, _ = provider._build_contents_and_system([message])

    reconstructed_part = contents[0].parts[0]
    assert reconstructed_part.function_call.name == "search_weather"
    assert reconstructed_part.thought_signature == b"sig-bytes"


def test_gemini_provider_leaves_thought_signature_unset_without_provider_data(
    fake_genai_module: types.ModuleType,
) -> None:
    """A ToolCall with no provider_data (e.g. hand-built) doesn't fabricate a signature."""
    from requisite.providers.gemini_provider import GeminiProvider

    provider = GeminiProvider(api_key="g-test", model="gemini-2.5-flash")
    tool_call = ToolCall(id="search_weather-0", name="search_weather", arguments={"city": "Paris"})
    message = Message.assistant_tool_calls([tool_call])

    contents, _ = provider._build_contents_and_system([message])

    reconstructed_part = contents[0].parts[0]
    assert reconstructed_part.thought_signature is None


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


# ---------------------------------------------------------------------------
# OpenRouter provider tests (reuses the OpenAI-compatible fake client)
# ---------------------------------------------------------------------------


def test_openrouter_provider_uses_openrouter_base_url(
    fake_openai_module: types.ModuleType,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    class _CapturingClient(_FakeOpenAIClient):
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)
            super().__init__(**kwargs)

    fake_openai_module.OpenAI = _CapturingClient  # type: ignore[attr-defined]

    from requisite.providers.openrouter_provider import OpenRouterProvider

    provider = OpenRouterProvider(api_key="sk-or-test")
    response = provider.chat([Message.user("hi")])
    assert response.content == "echo: hi"
    assert response.provider == "openrouter"
    assert captured_kwargs["base_url"] == "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Together AI provider tests (reuses the OpenAI-compatible fake client)
# ---------------------------------------------------------------------------


def test_together_provider_uses_together_base_url(fake_openai_module: types.ModuleType) -> None:
    captured_kwargs: dict[str, Any] = {}

    class _CapturingClient(_FakeOpenAIClient):
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)
            super().__init__(**kwargs)

    fake_openai_module.OpenAI = _CapturingClient  # type: ignore[attr-defined]

    from requisite.providers.together_provider import TogetherProvider

    provider = TogetherProvider(api_key="together-test")
    response = provider.chat([Message.user("hi")])
    assert response.content == "echo: hi"
    assert response.provider == "together"
    assert captured_kwargs["base_url"] == "https://api.together.ai/v1"


# ---------------------------------------------------------------------------
# Ollama provider tests (SDK faked via sys.modules)
# ---------------------------------------------------------------------------


class _FakeOllamaFunction:
    def __init__(self, name: str, arguments: dict[str, Any]) -> None:
        self.name = name
        self.arguments = arguments


class _FakeOllamaToolCall:
    def __init__(self, name: str, arguments: dict[str, Any]) -> None:
        self.function = _FakeOllamaFunction(name, arguments)


class _FakeOllamaMessage:
    def __init__(
        self, content: str, *, tool_calls: Optional[list[_FakeOllamaToolCall]] = None
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _FakeOllamaChatResponse:
    def __init__(
        self,
        content: str,
        *,
        tool_calls: Optional[list[_FakeOllamaToolCall]] = None,
        done_reason: str = "stop",
        done: bool = True,
    ) -> None:
        self.message = _FakeOllamaMessage(content, tool_calls=tool_calls)
        self.done_reason = done_reason
        self.done = done
        self.prompt_eval_count = 3
        self.eval_count = 4


class _FakeOllamaClient:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def chat(
        self, *, model: str, messages: list[dict[str, Any]], stream: bool = False, **kwargs: Any
    ) -> Any:
        if stream:
            return iter(
                [
                    _FakeOllamaChatResponse("echo: ", done=False),
                    _FakeOllamaChatResponse("hi", done=True),
                ]
            )
        last_content = messages[-1]["content"] if messages else ""
        return _FakeOllamaChatResponse(f"echo: {last_content}")


@pytest.fixture
def fake_ollama_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake_module = types.ModuleType("ollama")
    fake_module.Client = _FakeOllamaClient  # type: ignore[attr-defined]
    fake_module.AsyncClient = _FakeOllamaClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ollama", fake_module)
    return fake_module


def test_ollama_provider_chat(fake_ollama_module: types.ModuleType) -> None:
    from requisite.providers.ollama_provider import OllamaProvider

    provider = OllamaProvider(model="llama3.2")
    response = provider.chat([Message.user("hi")])
    assert response.content == "echo: hi"
    assert response.provider == "ollama"
    assert response.usage.total_tokens == 7


def test_ollama_provider_parses_tool_calls(fake_ollama_module: types.ModuleType) -> None:
    class _ToolCallClient(_FakeOllamaClient):
        def chat(self, *, model, messages, stream=False, **kwargs):  # noqa: ANN001
            return _FakeOllamaChatResponse(
                "", tool_calls=[_FakeOllamaToolCall("get_weather", {"city": "Paris"})]
            )

    fake_ollama_module.Client = _ToolCallClient  # type: ignore[attr-defined]

    from requisite.providers.ollama_provider import OllamaProvider
    from requisite.tools.base import Tool

    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    provider = OllamaProvider(model="llama3.2")
    response = provider.chat(
        [Message.user("weather in Paris?")], tools=[Tool.from_function(get_weather)]
    )
    assert response.has_tool_calls
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"city": "Paris"}


def test_ollama_provider_stream(fake_ollama_module: types.ModuleType) -> None:
    from requisite.providers.ollama_provider import OllamaProvider

    provider = OllamaProvider(model="llama3.2")
    chunks = list(provider.stream([Message.user("hi")]))
    assert [c.delta for c in chunks] == ["echo: ", "hi"]
    assert chunks[-1].is_final


def test_ollama_provider_missing_sdk_raises_configuration_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "ollama", None)
    from requisite.providers.ollama_provider import OllamaProvider

    provider = OllamaProvider(model="llama3.2")
    with pytest.raises(ConfigurationException):
        provider.chat([Message.user("hi")])


def test_ollama_provider_does_not_require_api_key() -> None:
    from requisite.providers.ollama_provider import OllamaProvider

    provider = OllamaProvider(model="llama3.2")
    provider.validate_config()  # must not raise, unlike the BaseProvider default
