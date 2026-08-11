"""
Google Gemini provider.

Uses the current, unified ``google-genai`` SDK (``from google import genai``),
which replaced the now-deprecated ``google-generativeai`` package. Do not
reintroduce ``google.generativeai`` / ``genai.configure`` / ``genai.GenerativeModel``
anywhere in this module -- that surface is legacy and unsupported going
forward.

Install with: ``pip install google-genai``
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

logger = logging.getLogger("requisite.providers.gemini")


def _extract_tool_calls_from_parts(parts: Sequence[Any], start_index: int) -> list[ToolCall]:
    """Build :class:`ToolCall`s for any ``function_call`` parts, mirroring
    :meth:`GeminiProvider._to_chat_response`'s per-part parsing.

    ``start_index`` lets streaming callers keep a running counter across
    chunks (each chunk's own ``parts`` list restarts at 0) so synthesized
    ids stay unique across the whole stream -- the Gemini Developer API
    (non-Vertex) never streams function-call arguments incrementally, so
    each part here is already complete.
    """
    tool_calls: list[ToolCall] = []
    for offset, part in enumerate(parts):
        function_call = getattr(part, "function_call", None)
        if function_call is None:
            continue
        args = dict(getattr(function_call, "args", None) or {})
        call_name = getattr(function_call, "name", "call")
        tool_calls.append(
            ToolCall(
                id=f"{call_name}-{start_index + offset}",
                name=call_name,
                arguments=args,
                provider_data=getattr(part, "thought_signature", None),
            )
        )
    return tool_calls


class GeminiProvider(BaseProvider):
    """Provider implementation backed by Google's Gemini API via ``google-genai``.

    Parameters
    ----------
    api_key:
        Gemini API key (from Google AI Studio). Falls back to the
        ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` environment variables if
        not provided (handled by the underlying SDK).
    model:
        Default model, e.g. ``"gemini-2.5-flash"``, ``"gemini-2.5-pro"``.
        Prefer versioned model IDs in production, since preview models
        can change behavior between updates.
    timeout:
        Per-request timeout, in seconds.
    max_retries:
        Number of retries for transient failures. Currently informational;
        surfaced for interface symmetry with other providers and for
        future use once the SDK exposes granular retry configuration.

    Examples
    --------
    >>> provider = GeminiProvider(api_key="...", model="gemini-2.5-flash")
    >>> response = provider.chat([Message.user("Say hi in one word.")])  # doctest: +SKIP
    >>> response.content  # doctest: +SKIP
    'Hi'
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        timeout: float = 60.0,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key, model=model, timeout=timeout, max_retries=max_retries, **kwargs
        )
        self._client: Any = None

    @property
    def name(self) -> str:
        return "gemini"

    def _get_client(self) -> Any:
        """Lazily construct (and cache) the ``google-genai`` client.

        The same client instance exposes both the sync API directly and
        the async API via its ``.aio`` namespace, so a single client is
        sufficient for sync, async, and streaming use.
        """
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - exercised only without the dep
                raise ConfigurationException(
                    "The 'google-genai' package is required for GeminiProvider. "
                    "Install it with: pip install google-genai "
                    "(do not install the deprecated 'google-generativeai' package).",
                ) from exc

            self.validate_config()
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _build_contents_and_system(
        self, messages: Sequence[Message]
    ) -> tuple[list[Any], Optional[str]]:
        """Split framework messages into Gemini ``contents`` + system instruction.

        Gemini has no ``system`` role in its content list; system prompts
        are passed separately via ``GenerateContentConfig.system_instruction``.
        Assistant turns map to Gemini's ``"model"`` role. Assistant
        messages that requested tool calls are encoded as
        ``function_call`` parts, and tool-result messages are encoded as
        ``function_response`` parts on a ``"user"``-role turn, per the
        Gemini function-calling protocol. When a :class:`ToolCall` carries
        a ``thought_signature`` in ``provider_data`` (set by this same
        provider on a prior turn), it is echoed back onto the
        reconstructed ``function_call`` part -- Gemini requires this for
        multi-turn tool-calling conversations and rejects the request
        with ``400 INVALID_ARGUMENT`` otherwise.
        """
        from google.genai import types

        system_parts: list[str] = []
        contents: list[Any] = []
        for message in messages:
            if message.role == Role.SYSTEM:
                system_parts.append(message.content)
                continue

            if message.role == Role.ASSISTANT and message.tool_calls:
                parts = []
                if message.content:
                    parts.append(types.Part.from_text(text=message.content))
                for call in message.tool_calls:
                    function_call_part = types.Part.from_function_call(
                        name=call.name, args=call.arguments
                    )
                    if isinstance(call.provider_data, bytes):
                        function_call_part.thought_signature = call.provider_data
                    parts.append(function_call_part)
                contents.append(types.Content(role="model", parts=parts))
                continue

            if message.role == Role.TOOL:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=message.name or "tool",
                                response={"result": message.content},
                            )
                        ],
                    )
                )
                continue

            gemini_role = "model" if message.role == Role.ASSISTANT else "user"
            contents.append(
                types.Content(role=gemini_role, parts=[types.Part.from_text(text=message.content)])
            )

        system_instruction = "\n".join(system_parts) if system_parts else None
        return contents, system_instruction

    def _build_config(
        self,
        *,
        temperature: Optional[float],
        system_instruction: Optional[str],
        tools: Optional[Sequence[Tool]],
        response_model: Optional[type[BaseModel]],
        kwargs: dict[str, Any],
    ) -> Any:
        from google.genai import types

        config_kwargs: dict[str, Any] = dict(kwargs)

        if response_model is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_model
        elif tools:
            function_declarations = [
                types.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    # parameters_json_schema accepts a raw JSON Schema dict directly
                    # (typed Any upstream) -- unlike `parameters`, which expects a
                    # fully-built `types.Schema` object. Using the JSON-schema field
                    # lets us pass Tool.parameters_schema as-is.
                    parameters_json_schema=t.parameters_schema
                    or {"type": "object", "properties": {}},
                )
                for t in tools
            ]
            config_kwargs["tools"] = [types.Tool(function_declarations=function_declarations)]

        return types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            **config_kwargs,
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
        contents, system_instruction = self._build_contents_and_system(messages)
        config = self._build_config(
            temperature=temperature,
            system_instruction=system_instruction,
            tools=tools,
            response_model=response_model,
            kwargs=kwargs,
        )
        try:
            response = client.models.generate_content(
                model=resolved_model,
                contents=contents,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderException(
                f"Gemini generate_content failed: {exc}",
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
        client = self._get_client()
        resolved_model = model or self._model
        contents, system_instruction = self._build_contents_and_system(messages)
        config = self._build_config(
            temperature=temperature,
            system_instruction=system_instruction,
            tools=tools,
            response_model=response_model,
            kwargs=kwargs,
        )
        try:
            response = await client.aio.models.generate_content(
                model=resolved_model,
                contents=contents,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderException(
                f"Gemini async generate_content failed: {exc}",
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
        response_model: Optional[type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        client = self._get_client()
        contents, system_instruction = self._build_contents_and_system(messages)
        config = self._build_config(
            temperature=temperature,
            system_instruction=system_instruction,
            tools=tools,
            response_model=response_model,
            kwargs=kwargs,
        )
        tool_calls: list[ToolCall] = []
        try:
            stream = client.models.generate_content_stream(
                model=model or self._model,
                contents=contents,
                config=config,
            )
            for chunk in stream:
                text = chunk.text or ""
                candidates = getattr(chunk, "candidates", None)
                if candidates and getattr(candidates[0], "content", None) is not None:
                    parts = getattr(candidates[0].content, "parts", None) or []
                    tool_calls.extend(_extract_tool_calls_from_parts(parts, len(tool_calls)))
                yield StreamChunk(delta=text, is_final=False, raw=chunk)
            yield StreamChunk(delta="", is_final=True, tool_calls=tool_calls)
        except Exception as exc:  # noqa: BLE001
            raise ProviderException(
                f"Gemini streaming failed: {exc}",
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
        client = self._get_client()
        contents, system_instruction = self._build_contents_and_system(messages)
        config = self._build_config(
            temperature=temperature,
            system_instruction=system_instruction,
            tools=tools,
            response_model=response_model,
            kwargs=kwargs,
        )
        tool_calls: list[ToolCall] = []
        try:
            stream = await client.aio.models.generate_content_stream(
                model=model or self._model,
                contents=contents,
                config=config,
            )
            async for chunk in stream:
                text = chunk.text or ""
                candidates = getattr(chunk, "candidates", None)
                if candidates and getattr(candidates[0], "content", None) is not None:
                    parts = getattr(candidates[0].content, "parts", None) or []
                    tool_calls.extend(_extract_tool_calls_from_parts(parts, len(tool_calls)))
                yield StreamChunk(delta=text, is_final=False, raw=chunk)
            yield StreamChunk(delta="", is_final=True, tool_calls=tool_calls)
        except Exception as exc:  # noqa: BLE001
            raise ProviderException(
                f"Gemini async streaming failed: {exc}",
                provider=self.name,
                original_error=exc,
            ) from exc

    def _to_chat_response(
        self, response: Any, model: str, *, response_model: Optional[type[BaseModel]] = None
    ) -> ChatResponse:
        """Convert a ``google-genai`` response object into a :class:`ChatResponse`.

        Walks ``candidates[0].content.parts`` directly rather than using the
        ``response.text`` / ``response.function_calls`` convenience
        properties: both discard the per-part ``thought_signature`` field
        that newer Gemini API versions require echoed back verbatim on
        function-call parts in later turns, and ``response.text`` also logs
        a noisy warning whenever the response contains a function call.
        """
        usage_meta = getattr(response, "usage_metadata", None)
        usage = Usage(
            prompt_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
            total_tokens=getattr(usage_meta, "total_token_count", 0) or 0,
        )
        finish_reason = None
        candidates = getattr(response, "candidates", None)
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
            finish_reason = str(finish_reason) if finish_reason is not None else None

        parts: list[Any] = []
        if candidates and getattr(candidates[0], "content", None) is not None:
            parts = getattr(candidates[0].content, "parts", None) or []

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for index, part in enumerate(parts):
            part_text = getattr(part, "text", None)
            if part_text:
                text_parts.append(part_text)

            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                args = dict(getattr(function_call, "args", None) or {})
                call_name = getattr(function_call, "name", "call")
                tool_calls.append(
                    ToolCall(
                        id=f"{call_name}-{index}",
                        name=call_name,
                        arguments=args,
                        provider_data=getattr(part, "thought_signature", None),
                    )
                )

        parsed = None
        if response_model is not None:
            parsed = getattr(response, "parsed", None)

        return ChatResponse(
            content="".join(text_parts),
            model=model,
            provider=self.name,
            usage=usage,
            raw=response,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            parsed=parsed,
        )
