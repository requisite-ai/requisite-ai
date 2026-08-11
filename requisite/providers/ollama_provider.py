"""
Ollama provider.

Uses the native ``ollama`` Python package (``Client`` / ``AsyncClient``),
not Ollama's OpenAI-compatible ``/v1`` endpoint. That compat endpoint is
explicitly documented by Ollama itself as "experimental ... subject to
major adjustments including breaking changes," with the native library
recommended for production use (confirmed against Ollama's own docs and
the installed SDK) -- so, unlike Groq/OpenRouter/Together, this provider
is a full translation layer, not a thin ``OpenAIProvider`` subclass.

Install with: ``pip install ollama``
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Optional

from pydantic import BaseModel

from requisite.core.exceptions import ConfigurationException, ProviderException
from requisite.core.interfaces import ChatResponse, Message, Role, StreamChunk, ToolCall, Usage
from requisite.providers.base import BaseProvider
from requisite.tools.base import Tool

logger = logging.getLogger("requisite.providers.ollama")


def _to_ollama_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Convert framework :class:`Message` objects to Ollama's wire format.

    Ollama has no per-call correlation id for tool results (unlike
    OpenAI/Anthropic's ``tool_call_id``) -- a tool-result message is
    matched to its call by name and turn order only, via ``tool_name``.
    """
    converted: list[dict[str, Any]] = []
    for m in messages:
        if m.role == Role.ASSISTANT and m.tool_calls:
            converted.append(
                {
                    "role": "assistant",
                    "content": m.content,
                    "tool_calls": [
                        {"function": {"name": call.name, "arguments": call.arguments}}
                        for call in m.tool_calls
                    ],
                }
            )
        elif m.role == Role.TOOL:
            converted.append({"role": "tool", "content": m.content, "tool_name": m.name or ""})
        else:
            converted.append({"role": m.role.value, "content": m.content})
    return converted


def _to_ollama_tools(tools: Sequence[Tool]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters_schema or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def _extract_ollama_tool_calls(chunk: Any, start_index: int) -> list[ToolCall]:
    """Build :class:`ToolCall`s for a streamed chunk's ``message.tool_calls``,
    mirroring :meth:`OllamaProvider._to_chat_response`'s parsing.

    ``start_index`` lets streaming callers keep a running counter across
    chunks so synthesized ids stay unique -- Ollama has no incremental
    tool-call streaming (``Function.arguments`` is always a fully-parsed
    mapping, never a partial string), so whichever chunk carries
    ``message.tool_calls`` already has them complete.
    """
    tool_calls: list[ToolCall] = []
    for offset, call in enumerate(chunk.message.tool_calls or []):
        tool_calls.append(
            ToolCall(
                id=f"{call.function.name}-{start_index + offset}",
                name=call.function.name,
                arguments=dict(call.function.arguments),
            )
        )
    return tool_calls


class OllamaProvider(BaseProvider):
    """Provider implementation backed by a local (or remote) Ollama server.

    Parameters
    ----------
    api_key:
        Optional bearer token, only needed for Ollama Cloud or a
        proxied/authenticated Ollama deployment -- a purely local server
        needs none. Falls back to the ``OLLAMA_API_KEY`` environment
        variable if not provided.
    model:
        Default model, e.g. ``"llama3.2"``, ``"qwen3"`` -- must already
        be pulled on the target server (``ollama pull <model>``).
    host:
        Ollama server URL. Defaults to ``http://localhost:11434``
        (or the ``OLLAMA_HOST`` environment variable) via the underlying
        SDK's own resolution -- pass explicitly for a remote server.
    timeout:
        Per-request timeout, in seconds.
    max_retries:
        Accepted for interface symmetry with other providers; the
        underlying ``ollama`` client has no built-in retry mechanism to
        delegate to, so this is currently informational only.

    Examples
    --------
    >>> provider = OllamaProvider(model="llama3.2")
    >>> response = provider.chat([Message.user("Say hi in one word.")])  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "llama3.2",
        host: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key, model=model, timeout=timeout, max_retries=max_retries, **kwargs
        )
        self._host = host
        self._client: Any = None
        self._async_client: Any = None

    @property
    def name(self) -> str:
        return "ollama"

    def validate_config(self) -> None:
        """Ollama needs no API key for a local server -- override the
        default (which requires one) to do nothing."""

    def _headers(self) -> Optional[dict[str, str]]:
        return {"authorization": f"Bearer {self._api_key}"} if self._api_key else None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._build_client(is_async=False)
        return self._client

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            self._async_client = self._build_client(is_async=True)
        return self._async_client

    def _build_client(self, *, is_async: bool) -> Any:
        try:
            from ollama import AsyncClient, Client
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ConfigurationException(
                "The 'ollama' package is required for OllamaProvider. "
                "Install it with: pip install ollama",
            ) from exc

        client_cls = AsyncClient if is_async else Client
        return client_cls(host=self._host, timeout=self._timeout, headers=self._headers())

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence[Tool]] = None,
        response_model: Optional[type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        client = self._get_client()
        resolved_model = model or self._model
        try:
            response = client.chat(
                model=resolved_model,
                messages=_to_ollama_messages(messages),
                tools=_to_ollama_tools(tools) if tools else None,
                format=response_model.model_json_schema() if response_model is not None else None,
                options=self._options(temperature, kwargs),
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderException(
                f"Ollama chat failed: {exc}",
                provider=self.name,
                original_error=exc,
            ) from exc

        return self._to_chat_response(response, resolved_model, response_model=response_model)

    async def achat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence[Tool]] = None,
        response_model: Optional[type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        client = self._get_async_client()
        resolved_model = model or self._model
        try:
            response = await client.chat(
                model=resolved_model,
                messages=_to_ollama_messages(messages),
                tools=_to_ollama_tools(tools) if tools else None,
                format=response_model.model_json_schema() if response_model is not None else None,
                options=self._options(temperature, kwargs),
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderException(
                f"Ollama async chat failed: {exc}",
                provider=self.name,
                original_error=exc,
            ) from exc

        return self._to_chat_response(response, resolved_model, response_model=response_model)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence[Tool]] = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        client = self._get_client()
        tool_calls: list[ToolCall] = []
        try:
            for chunk in client.chat(
                model=model or self._model,
                messages=_to_ollama_messages(messages),
                tools=_to_ollama_tools(tools) if tools else None,
                stream=True,
                options=self._options(temperature, kwargs),
            ):
                tool_calls.extend(_extract_ollama_tool_calls(chunk, len(tool_calls)))
                yield StreamChunk(
                    delta=chunk.message.content or "",
                    is_final=chunk.done,
                    raw=chunk,
                    tool_calls=tool_calls if chunk.done else [],
                )
        except Exception as exc:  # noqa: BLE001
            raise ProviderException(
                f"Ollama streaming failed: {exc}",
                provider=self.name,
                original_error=exc,
            ) from exc

    async def astream(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence[Tool]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        client = self._get_async_client()
        tool_calls: list[ToolCall] = []
        try:
            async for chunk in await client.chat(
                model=model or self._model,
                messages=_to_ollama_messages(messages),
                tools=_to_ollama_tools(tools) if tools else None,
                stream=True,
                options=self._options(temperature, kwargs),
            ):
                tool_calls.extend(_extract_ollama_tool_calls(chunk, len(tool_calls)))
                yield StreamChunk(
                    delta=chunk.message.content or "",
                    is_final=chunk.done,
                    raw=chunk,
                    tool_calls=tool_calls if chunk.done else [],
                )
        except Exception as exc:  # noqa: BLE001
            raise ProviderException(
                f"Ollama async streaming failed: {exc}",
                provider=self.name,
                original_error=exc,
            ) from exc

    @staticmethod
    def _options(temperature: Optional[float], kwargs: dict[str, Any]) -> dict[str, Any]:
        options = dict(kwargs)
        if temperature is not None:
            options["temperature"] = temperature
        return options

    def _to_chat_response(
        self, response: Any, model: str, *, response_model: Optional[type[BaseModel]] = None
    ) -> ChatResponse:
        message = response.message
        tool_calls: list[ToolCall] = []
        for index, call in enumerate(message.tool_calls or []):
            tool_calls.append(
                ToolCall(
                    id=f"{call.function.name}-{index}",
                    name=call.function.name,
                    arguments=dict(call.function.arguments),
                )
            )

        parsed = None
        if response_model is not None and message.content:
            parsed = response_model.model_validate_json(message.content)

        prompt_tokens = response.prompt_eval_count or 0
        completion_tokens = response.eval_count or 0

        return ChatResponse(
            content=message.content or "",
            model=model,
            provider=self.name,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            raw=response,
            finish_reason=response.done_reason,
            tool_calls=tool_calls,
            parsed=parsed,
        )
