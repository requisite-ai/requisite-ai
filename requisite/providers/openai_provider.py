"""
OpenAI provider.

Uses the current ``openai`` Python SDK (v1.x, the client-based
``OpenAI`` / ``AsyncOpenAI`` interface). The legacy ``openai.ChatCompletion``
module-level API (pre-1.0) is deprecated and intentionally not used here.

Install with: ``pip install openai``
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Optional

from pydantic import BaseModel

from requisite.core.exceptions import ConfigurationException, ProviderException
from requisite.core.interfaces import ChatResponse, Message, Role, StreamChunk, ToolCall, Usage
from requisite.providers.base import BaseProvider
from requisite.tools.base import Tool

logger = logging.getLogger("requisite.providers.openai")


def _to_openai_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Convert framework :class:`Message` objects to the OpenAI wire format.

    Handles the two tool-calling special cases OpenAI requires:
    an assistant message that requested tool calls must carry a
    ``tool_calls`` array, and a message reporting a tool's result must
    be role ``"tool"`` with a matching ``tool_call_id``.
    """
    converted: list[dict[str, Any]] = []
    for m in messages:
        if m.role == Role.ASSISTANT and m.tool_calls:
            converted.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in m.tool_calls
                    ],
                }
            )
        elif m.role == Role.TOOL:
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id or "",
                    "content": m.content,
                }
            )
        else:
            converted.append({"role": m.role.value, "content": m.content})
    return converted


class OpenAIProvider(BaseProvider):
    """Provider implementation backed by the OpenAI Chat Completions API.

    Parameters
    ----------
    api_key:
        OpenAI API key. Falls back to the ``OPENAI_API_KEY`` environment
        variable if not provided (handled by the underlying SDK).
    model:
        Default model, e.g. ``"gpt-4o-mini"``, ``"gpt-4o"``.
    timeout:
        Per-request timeout, in seconds.
    max_retries:
        Number of retries the SDK performs on transient errors.
    base_url:
        Optional custom endpoint, useful for OpenAI-compatible proxies
        or Azure-style gateways that speak the same wire format.

    Examples
    --------
    >>> provider = OpenAIProvider(api_key="sk-...", model="gpt-4o-mini")
    >>> response = provider.chat([Message.user("Say hi in one word.")])  # doctest: +SKIP
    >>> response.content  # doctest: +SKIP
    'Hi'
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
        max_retries: int = 2,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key, model=model, timeout=timeout, max_retries=max_retries, **kwargs
        )
        self._base_url = base_url
        self._client: Any = None
        self._async_client: Any = None

    @property
    def name(self) -> str:
        return "openai"

    def _get_client(self) -> Any:
        """Lazily construct (and cache) the synchronous SDK client."""
        if self._client is None:
            self._client = self._build_client(is_async=False)
        return self._client

    def _get_async_client(self) -> Any:
        """Lazily construct (and cache) the asynchronous SDK client."""
        if self._async_client is None:
            self._async_client = self._build_client(is_async=True)
        return self._async_client

    def _build_client(self, *, is_async: bool) -> Any:
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ConfigurationException(
                f"The 'openai' package is required for {type(self).__name__}. "
                "Install it with: pip install openai",
            ) from exc

        self.validate_config()
        client_cls = AsyncOpenAI if is_async else OpenAI
        return client_cls(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

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
            if response_model is not None:
                completion = client.chat.completions.parse(
                    model=resolved_model,
                    messages=_to_openai_messages(messages),
                    temperature=temperature,
                    response_format=response_model,
                    **kwargs,
                )
            else:
                if tools:
                    kwargs["tools"] = [t.to_openai_schema() for t in tools]
                completion = client.chat.completions.create(
                    model=resolved_model,
                    messages=_to_openai_messages(messages),
                    temperature=temperature,
                    **kwargs,
                )
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            raise ProviderException(
                f"OpenAI chat completion failed: {exc}",
                provider=self.name,
                original_error=exc,
            ) from exc

        return self._to_chat_response(completion, resolved_model)

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
            if response_model is not None:
                completion = await client.chat.completions.parse(
                    model=resolved_model,
                    messages=_to_openai_messages(messages),
                    temperature=temperature,
                    response_format=response_model,
                    **kwargs,
                )
            else:
                if tools:
                    kwargs["tools"] = [t.to_openai_schema() for t in tools]
                completion = await client.chat.completions.create(
                    model=resolved_model,
                    messages=_to_openai_messages(messages),
                    temperature=temperature,
                    **kwargs,
                )
        except Exception as exc:  # noqa: BLE001
            raise ProviderException(
                f"OpenAI async chat completion failed: {exc}",
                provider=self.name,
                original_error=exc,
            ) from exc

        return self._to_chat_response(completion, resolved_model)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        client = self._get_client()
        try:
            stream = client.chat.completions.create(
                model=model or self._model,
                messages=_to_openai_messages(messages),
                temperature=temperature,
                stream=True,
                **kwargs,
            )
            for event in stream:
                delta = ""
                if event.choices:
                    delta = event.choices[0].delta.content or ""
                is_final = bool(event.choices) and event.choices[0].finish_reason is not None
                yield StreamChunk(delta=delta, is_final=is_final, raw=event)
        except Exception as exc:  # noqa: BLE001
            raise ProviderException(
                f"OpenAI streaming failed: {exc}",
                provider=self.name,
                original_error=exc,
            ) from exc

    async def astream(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        client = self._get_async_client()
        try:
            stream = await client.chat.completions.create(
                model=model or self._model,
                messages=_to_openai_messages(messages),
                temperature=temperature,
                stream=True,
                **kwargs,
            )
            async for event in stream:
                delta = ""
                if event.choices:
                    delta = event.choices[0].delta.content or ""
                is_final = bool(event.choices) and event.choices[0].finish_reason is not None
                yield StreamChunk(delta=delta, is_final=is_final, raw=event)
        except Exception as exc:  # noqa: BLE001
            raise ProviderException(
                f"OpenAI async streaming failed: {exc}",
                provider=self.name,
                original_error=exc,
            ) from exc

    def _to_chat_response(self, completion: Any, model: str) -> ChatResponse:
        """Convert an OpenAI SDK completion object into a :class:`ChatResponse`."""
        choice = completion.choices[0]
        usage_obj = getattr(completion, "usage", None)
        usage = Usage(
            prompt_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage_obj, "total_tokens", 0) or 0,
        )

        tool_calls: list[ToolCall] = []
        raw_tool_calls = getattr(choice.message, "tool_calls", None) or []
        for raw_call in raw_tool_calls:
            try:
                arguments = json.loads(raw_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ToolCall(id=raw_call.id, name=raw_call.function.name, arguments=arguments)
            )

        parsed = getattr(choice.message, "parsed", None)

        return ChatResponse(
            content=choice.message.content or "",
            model=getattr(completion, "model", model),
            provider=self.name,
            usage=usage,
            raw=completion,
            finish_reason=choice.finish_reason,
            tool_calls=tool_calls,
            parsed=parsed,
        )
