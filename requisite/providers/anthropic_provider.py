"""
Anthropic (Claude) provider.

Uses the current ``anthropic`` Python SDK (``Anthropic`` / ``AsyncAnthropic``
clients, ``client.messages.create`` / ``client.messages.parse`` /
``client.messages.stream``). Verified against ``anthropic`` 0.116.

Key differences from OpenAI/Gemini this provider bridges:

- System prompts are a separate top-level ``system`` parameter, not a
  message in the ``messages`` list.
- ``max_tokens`` is a *required* parameter (no server-side default).
- Tool schemas use ``input_schema`` rather than ``parameters``.
- Structured output uses the SDK's native ``messages.parse(output_format=...)``
  helper, which returns a validated instance on the response's text content
  block (``content[0].parsed_output``) -- not a client-side "force a tool
  call" workaround.

Install with: ``pip install anthropic``
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

logger = logging.getLogger("requisite.providers.anthropic")


class AnthropicProvider(BaseProvider):
    """Provider implementation backed by Anthropic's Messages API.

    Parameters
    ----------
    api_key:
        Anthropic API key. Falls back to the ``ANTHROPIC_API_KEY``
        environment variable if not provided (handled by the SDK).
    model:
        Default model, e.g. ``"claude-sonnet-4-6"``, ``"claude-opus-4-8"``.
    timeout:
        Per-request timeout, in seconds.
    max_retries:
        Number of retries the SDK performs on transient errors.
    max_tokens:
        Default max output tokens. Anthropic requires this on every
        request (unlike OpenAI/Gemini, which have server-side defaults);
        override per-call via the ``max_tokens`` keyword argument to
        :meth:`chat`.

    Examples
    --------
    >>> provider = AnthropicProvider(api_key="sk-ant-...", model="claude-sonnet-4-6")
    >>> response = provider.chat([Message.user("Say hi in one word.")])  # doctest: +SKIP
    >>> response.content  # doctest: +SKIP
    'Hi'
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        timeout: float = 60.0,
        max_retries: int = 2,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key, model=model, timeout=timeout, max_retries=max_retries, **kwargs
        )
        self._max_tokens = max_tokens
        self._client: Any = None
        self._async_client: Any = None

    @property
    def name(self) -> str:
        return "anthropic"

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
            from anthropic import Anthropic, AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise ConfigurationException(
                "The 'anthropic' package is required for AnthropicProvider. "
                "Install it with: pip install anthropic",
            ) from exc

        self.validate_config()
        client_cls = AsyncAnthropic if is_async else Anthropic
        return client_cls(
            api_key=self._api_key, timeout=self._timeout, max_retries=self._max_retries
        )

    def _to_anthropic_messages(
        self, messages: Sequence[Message]
    ) -> tuple[list[dict[str, Any]], Optional[str]]:
        """Convert framework messages to Anthropic's wire format.

        Returns ``(messages, system)`` -- system messages are extracted
        out of the list entirely, since Anthropic takes them as a separate
        top-level parameter. Consecutive tool-result messages are merged
        into a single ``user`` turn with multiple ``tool_result`` blocks,
        since Anthropic expects one user turn per assistant tool-call turn.
        """
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []
        pending_tool_results: Optional[list[dict[str, Any]]] = None

        def _flush_tool_results() -> None:
            nonlocal pending_tool_results
            if pending_tool_results is not None:
                anthropic_messages.append({"role": "user", "content": pending_tool_results})
                pending_tool_results = None

        for message in messages:
            if message.role == Role.SYSTEM:
                _flush_tool_results()
                system_parts.append(message.content)
                continue

            if message.role == Role.ASSISTANT and message.tool_calls:
                _flush_tool_results()
                content: list[dict[str, Any]] = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                for call in message.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                anthropic_messages.append({"role": "assistant", "content": content})
                continue

            if message.role == Role.TOOL:
                if pending_tool_results is None:
                    pending_tool_results = []
                pending_tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id or "",
                        "content": message.content,
                    }
                )
                continue

            _flush_tool_results()
            role = "assistant" if message.role == Role.ASSISTANT else "user"
            anthropic_messages.append({"role": role, "content": message.content})

        _flush_tool_results()
        system = "\n".join(system_parts) if system_parts else None
        return anthropic_messages, system

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
        anthropic_messages, system = self._to_anthropic_messages(messages)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)

        try:
            if response_model is not None:
                result = client.messages.parse(
                    model=resolved_model,
                    max_tokens=max_tokens,
                    messages=anthropic_messages,
                    system=system or "",
                    temperature=temperature,
                    output_format=response_model,
                    **kwargs,
                )
            else:
                if tools:
                    kwargs["tools"] = [t.to_anthropic_schema() for t in tools]
                result = client.messages.create(
                    model=resolved_model,
                    max_tokens=max_tokens,
                    messages=anthropic_messages,
                    system=system or "",
                    temperature=temperature,
                    **kwargs,
                )
        except Exception as exc:  # noqa: BLE001
            raise ProviderException(
                f"Anthropic messages.create failed: {exc}",
                provider=self.name,
                original_error=exc,
            ) from exc

        return self._to_chat_response(result, resolved_model, response_model=response_model)

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
        anthropic_messages, system = self._to_anthropic_messages(messages)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)

        try:
            if response_model is not None:
                result = await client.messages.parse(
                    model=resolved_model,
                    max_tokens=max_tokens,
                    messages=anthropic_messages,
                    system=system or "",
                    temperature=temperature,
                    output_format=response_model,
                    **kwargs,
                )
            else:
                if tools:
                    kwargs["tools"] = [t.to_anthropic_schema() for t in tools]
                result = await client.messages.create(
                    model=resolved_model,
                    max_tokens=max_tokens,
                    messages=anthropic_messages,
                    system=system or "",
                    temperature=temperature,
                    **kwargs,
                )
        except Exception as exc:  # noqa: BLE001
            raise ProviderException(
                f"Anthropic async messages.create failed: {exc}",
                provider=self.name,
                original_error=exc,
            ) from exc

        return self._to_chat_response(result, resolved_model, response_model=response_model)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence[Tool]] = None,
        response_model: Optional[type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        client = self._get_client()
        resolved_model = model or self._model
        anthropic_messages, system = self._to_anthropic_messages(messages)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)
        if tools:
            kwargs["tools"] = [t.to_anthropic_schema() for t in tools]

        try:
            with client.messages.stream(
                model=resolved_model,
                max_tokens=max_tokens,
                messages=anthropic_messages,
                system=system or "",
                temperature=temperature,
                **kwargs,
            ) as stream:
                for text in stream.text_stream:
                    yield StreamChunk(delta=text, is_final=False)
                yield StreamChunk(delta="", is_final=True)
        except Exception as exc:  # noqa: BLE001
            raise ProviderException(
                f"Anthropic streaming failed: {exc}",
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
        response_model: Optional[type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        client = self._get_async_client()
        resolved_model = model or self._model
        anthropic_messages, system = self._to_anthropic_messages(messages)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)
        if tools:
            kwargs["tools"] = [t.to_anthropic_schema() for t in tools]

        try:
            async with client.messages.stream(
                model=resolved_model,
                max_tokens=max_tokens,
                messages=anthropic_messages,
                system=system or "",
                temperature=temperature,
                **kwargs,
            ) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(delta=text, is_final=False)
                yield StreamChunk(delta="", is_final=True)
        except Exception as exc:  # noqa: BLE001
            raise ProviderException(
                f"Anthropic async streaming failed: {exc}",
                provider=self.name,
                original_error=exc,
            ) from exc

    def _to_chat_response(
        self, result: Any, model: str, *, response_model: Optional[type[BaseModel]] = None
    ) -> ChatResponse:
        """Convert an Anthropic ``Message`` (or ``ParsedMessage``) into a :class:`ChatResponse`."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        parsed: Any = None

        for block in result.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(block.text)
                if response_model is not None:
                    parsed = getattr(block, "parsed_output", None)
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )

        usage_obj = getattr(result, "usage", None)
        input_tokens = getattr(usage_obj, "input_tokens", 0) or 0
        output_tokens = getattr(usage_obj, "output_tokens", 0) or 0
        usage = Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

        return ChatResponse(
            content="".join(text_parts),
            model=getattr(result, "model", model),
            provider=self.name,
            usage=usage,
            raw=result,
            finish_reason=getattr(result, "stop_reason", None),
            tool_calls=tool_calls,
            parsed=parsed,
        )
