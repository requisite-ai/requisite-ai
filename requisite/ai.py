"""
The ``AI`` facade.

This is the class most developers will use directly. It hides provider
selection, configuration loading, and message-list bookkeeping behind a
small, Pythonic API, while remaining fully driven by dependency
injection underneath -- nothing here reaches for global state.

Examples
--------
>>> from requisite import AI
>>> ai = AI()  # doctest: +SKIP
>>> ai.chat("Explain LangGraph in one sentence.")  # doctest: +SKIP

Switching providers is a configuration change, not a code change:

>>> ai = AI(provider="gemini", model="gemini-2.5-flash")  # doctest: +SKIP

Injecting an isolated registry / settings (useful in tests):

>>> from requisite.config.settings import Settings
>>> from requisite.providers.factory import ProviderRegistry
>>> ai = AI(settings=Settings(default_provider="openai"), registry=ProviderRegistry())  # doctest: +SKIP
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Optional, TypeVar, Union

from pydantic import BaseModel

from requisite.config.settings import Settings
from requisite.core.exceptions import ConfigurationException
from requisite.core.interfaces import ChatResponse, Message, StreamChunk
from requisite.core.rate_limiter import RateLimiter
from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry, default_registry
from requisite.telemetry.otel import get_meter, get_tracer
from requisite.tools.registry import ToolLike, resolve_tool_like

_tracer = get_tracer("requisite.ai")
_meter = get_meter("requisite.ai")
_request_counter = _meter.create_counter(
    "requisite.ai.requests", description="AI facade calls made to a provider", unit="1"
)
_request_duration = _meter.create_histogram(
    "requisite.ai.request.duration", description="AI facade call duration", unit="s"
)
_token_counter = _meter.create_counter(
    "requisite.ai.tokens", description="Tokens used per AI facade call", unit="1"
)

logger = logging.getLogger("requisite")

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


class AI:
    """Simple, provider-agnostic entry point for chat completions.

    Parameters
    ----------
    provider:
        Either a provider name (e.g. ``"openai"``, ``"gemini"``) resolved
        through ``registry``, or an already-constructed
        :class:`~requisite.providers.base.BaseProvider` instance
        (useful for tests or custom providers). Defaults to
        ``settings.default_provider``.
    model:
        Overrides the provider's default model. Defaults to ``settings.model``.
    settings:
        Configuration source. Defaults to a freshly loaded
        :class:`~requisite.config.settings.Settings` (reads environment
        variables / ``.env``). Inject an explicit instance in tests to
        avoid depending on the process environment.
    registry:
        The :class:`~requisite.providers.factory.ProviderRegistry` used
        to resolve ``provider`` when it's given as a string. Defaults to
        the framework's built-in registry, which already knows about
        ``"openai"`` and ``"gemini"``.
    system_prompt:
        Optional system message applied to every call made through this
        instance, unless overridden per-call.
    rate_limiter:
        A :class:`~requisite.core.rate_limiter.RateLimiter` that every
        call made through this instance waits on before reaching the
        provider. Pass the *same* instance to several ``AI``/
        :class:`~requisite.agents.agent.Agent` objects to share one
        budget across them -- this is the usual case when they draw on
        the same underlying API key. Defaults to ``None``; if unset and
        ``settings.rate_limit_rpm`` is configured, a private limiter is
        built from that instead (see :class:`~requisite.config.settings.Settings`).

    Examples
    --------
    Basic usage:

    >>> ai = AI(provider="openai")  # doctest: +SKIP
    >>> ai.chat("Hello!")  # doctest: +SKIP

    Streaming:

    >>> for chunk in ai.stream("Hello!"):  # doctest: +SKIP
    ...     print(chunk, end="")

    Full response object (with usage, model, provider metadata):

    >>> response = ai.chat_response("Hello!")  # doctest: +SKIP
    >>> response.usage.total_tokens  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        provider: Optional[Union[str, BaseProvider]] = None,
        model: Optional[str] = None,
        settings: Optional[Settings] = None,
        registry: Optional[ProviderRegistry] = None,
        system_prompt: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        self._settings = settings or Settings()
        self._registry = registry or default_registry
        self._system_prompt = system_prompt
        self._provider = self._resolve_provider(provider, model)
        self._rate_limiter = rate_limiter or self._build_default_rate_limiter()

    def _build_default_rate_limiter(self) -> Optional[RateLimiter]:
        if self._settings.rate_limit_rpm is None:
            return None
        return RateLimiter(
            requests_per_minute=self._settings.rate_limit_rpm,
            max_wait_seconds=self._settings.rate_limit_max_wait_seconds,
        )

    def _resolve_provider(
        self, provider: Optional[Union[str, BaseProvider]], model: Optional[str]
    ) -> BaseProvider:
        if isinstance(provider, BaseProvider):
            return provider

        provider_name = provider or self._settings.default_provider
        if not provider_name:
            raise ConfigurationException(
                "No provider specified and no default_provider configured.",
            )

        resolved_model = model or self._settings.model
        api_key = self._settings.api_key_for(provider_name)
        extra_kwargs = self._settings.provider_kwargs(provider_name)

        return self._registry.create(
            provider_name,
            api_key=api_key,
            model=resolved_model,
            timeout=self._settings.request_timeout,
            max_retries=self._settings.max_retries,
            **extra_kwargs,
        )

    @property
    def provider(self) -> BaseProvider:
        """The underlying, fully-resolved provider instance backing this ``AI``."""
        return self._provider

    @property
    def rate_limiter(self) -> Optional[RateLimiter]:
        """The :class:`~requisite.core.rate_limiter.RateLimiter` every call waits
        on before reaching the provider, if one is configured."""
        return self._rate_limiter

    def _build_messages(
        self, prompt: Union[str, Sequence[Message]], system_prompt: Optional[str]
    ) -> list[Message]:
        messages: list[Message] = []
        effective_system = system_prompt if system_prompt is not None else self._system_prompt
        if effective_system:
            messages.append(Message.system(effective_system))

        if isinstance(prompt, str):
            messages.append(Message.user(prompt))
        else:
            messages.extend(prompt)
        return messages

    def chat(
        self,
        prompt: Union[str, Sequence[Message]],
        *,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence[ToolLike]] = None,
        response_model: Optional[type[ResponseModelT]] = None,
        **kwargs: Any,
    ) -> Any:
        """Send a prompt and return the response.

        Parameters
        ----------
        prompt:
            Either a plain string (treated as a single user message) or
            a full conversation as a sequence of :class:`Message` objects.
        system_prompt:
            Overrides the instance-level ``system_prompt`` for this call.
        model:
            Overrides the configured model for this call.
        temperature:
            Overrides the configured temperature for this call.
        tools:
            Tools the model may call -- :class:`~requisite.tools.base.Tool`
            instances, plain functions, or ``@tool``-decorated functions
            (all accepted directly; a decorated function's attached
            ``.tool`` is resolved automatically, the same as
            :class:`~requisite.tools.registry.ToolRegistry` does). When
            the model requests one, the tool is *not* executed
            automatically here -- inspect ``chat_response(...).tool_calls``
            (or use an :class:`~requisite.agents.agent.Agent`, which runs
            the full tool-calling loop for you).
        response_model:
            A Pydantic model class. When given, the response is
            constrained to match its schema and this method returns a
            validated instance of it instead of a string.
        **kwargs:
            Passed through to the underlying provider.

        Returns
        -------
        str | BaseModel
            The generated text, or a validated ``response_model``
            instance when one was requested. Use :meth:`chat_response`
            if you need usage stats, tool calls, or the raw provider
            payload alongside the content.
        """
        response = self.chat_response(
            prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            tools=tools,
            response_model=response_model,
            **kwargs,
        )
        if response_model is not None:
            return response.parsed
        return response.content

    def chat_response(
        self,
        prompt: Union[str, Sequence[Message]],
        *,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence[ToolLike]] = None,
        response_model: Optional[type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a prompt and return the full, normalized :class:`ChatResponse`."""
        messages = self._build_messages(prompt, system_prompt)
        effective_temperature = (
            temperature if temperature is not None else self._settings.temperature
        )
        resolved_tools = [resolve_tool_like(t) for t in tools] if tools else None
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        attributes = {
            "requisite.provider": self._provider.name,
            "requisite.model": model or self._settings.model,
        }
        with _tracer.start_as_current_span("requisite.ai.chat_response", attributes=attributes):
            start = time.monotonic()
            try:
                response = self._provider.chat(
                    messages,
                    model=model,
                    temperature=effective_temperature,
                    tools=resolved_tools,
                    response_model=response_model,
                    **kwargs,
                )
            except Exception:
                # start_as_current_span already records the exception and sets
                # an error status on the span automatically as it propagates
                # out of this `with` block -- only the metric needs recording here.
                _request_counter.add(1, {**attributes, "requisite.status": "error"})
                raise
            _request_duration.record(time.monotonic() - start, attributes)
            _request_counter.add(1, {**attributes, "requisite.status": "success"})
            _token_counter.add(
                response.usage.prompt_tokens, {**attributes, "requisite.token_type": "prompt"}
            )
            _token_counter.add(
                response.usage.completion_tokens,
                {**attributes, "requisite.token_type": "completion"},
            )
            return response

    async def achat(
        self,
        prompt: Union[str, Sequence[Message]],
        *,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence[ToolLike]] = None,
        response_model: Optional[type[ResponseModelT]] = None,
        **kwargs: Any,
    ) -> Any:
        """Async counterpart to :meth:`chat`."""
        response = await self.achat_response(
            prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            tools=tools,
            response_model=response_model,
            **kwargs,
        )
        if response_model is not None:
            return response.parsed
        return response.content

    async def achat_response(
        self,
        prompt: Union[str, Sequence[Message]],
        *,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence[ToolLike]] = None,
        response_model: Optional[type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Async counterpart to :meth:`chat_response`."""
        messages = self._build_messages(prompt, system_prompt)
        effective_temperature = (
            temperature if temperature is not None else self._settings.temperature
        )
        resolved_tools = [resolve_tool_like(t) for t in tools] if tools else None
        if self._rate_limiter is not None:
            await self._rate_limiter.aacquire()
        attributes = {
            "requisite.provider": self._provider.name,
            "requisite.model": model or self._settings.model,
        }
        with _tracer.start_as_current_span("requisite.ai.achat_response", attributes=attributes):
            start = time.monotonic()
            try:
                response = await self._provider.achat(
                    messages,
                    model=model,
                    temperature=effective_temperature,
                    tools=resolved_tools,
                    response_model=response_model,
                    **kwargs,
                )
            except Exception:
                # start_as_current_span already records the exception and sets
                # an error status on the span automatically as it propagates
                # out of this `with` block -- only the metric needs recording here.
                _request_counter.add(1, {**attributes, "requisite.status": "error"})
                raise
            _request_duration.record(time.monotonic() - start, attributes)
            _request_counter.add(1, {**attributes, "requisite.status": "success"})
            _token_counter.add(
                response.usage.prompt_tokens, {**attributes, "requisite.token_type": "prompt"}
            )
            _token_counter.add(
                response.usage.completion_tokens,
                {**attributes, "requisite.token_type": "completion"},
            )
            return response

    def stream(
        self,
        prompt: Union[str, Sequence[Message]],
        *,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence[ToolLike]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream the response as plain text chunks.

        Parameters
        ----------
        tools:
            Tools the model may call, same as :meth:`chat`. When the
            model requests one, it is *not* executed or reported here --
            this method only ever yields text deltas. Use
            :meth:`stream_response` if you need the tool call(s)
            alongside the streamed text.

        Yields
        ------
        str
            Incremental text deltas, in order.
        """
        messages = self._build_messages(prompt, system_prompt)
        effective_temperature = (
            temperature if temperature is not None else self._settings.temperature
        )
        resolved_tools = [resolve_tool_like(t) for t in tools] if tools else None
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        attributes = {
            "requisite.provider": self._provider.name,
            "requisite.model": model or self._settings.model,
        }
        with _tracer.start_as_current_span("requisite.ai.stream", attributes=attributes):
            start = time.monotonic()
            try:
                for chunk in self._provider.stream(
                    messages,
                    model=model,
                    temperature=effective_temperature,
                    tools=resolved_tools,
                    **kwargs,
                ):
                    if chunk.delta:
                        yield chunk.delta
            except Exception:
                # start_as_current_span already records the exception and sets
                # an error status on the span automatically as it propagates
                # out of this `with` block -- only the metric needs recording here.
                _request_counter.add(1, {**attributes, "requisite.status": "error"})
                raise
            _request_duration.record(time.monotonic() - start, attributes)
            _request_counter.add(1, {**attributes, "requisite.status": "success"})

    async def astream(
        self,
        prompt: Union[str, Sequence[Message]],
        *,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence[ToolLike]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async counterpart to :meth:`stream`."""
        messages = self._build_messages(prompt, system_prompt)
        effective_temperature = (
            temperature if temperature is not None else self._settings.temperature
        )
        resolved_tools = [resolve_tool_like(t) for t in tools] if tools else None
        if self._rate_limiter is not None:
            await self._rate_limiter.aacquire()
        attributes = {
            "requisite.provider": self._provider.name,
            "requisite.model": model or self._settings.model,
        }
        with _tracer.start_as_current_span("requisite.ai.astream", attributes=attributes):
            start = time.monotonic()
            try:
                async for chunk in self._provider.astream(
                    messages,
                    model=model,
                    temperature=effective_temperature,
                    tools=resolved_tools,
                    **kwargs,
                ):
                    if chunk.delta:
                        yield chunk.delta
            except Exception:
                # start_as_current_span already records the exception and sets
                # an error status on the span automatically as it propagates
                # out of this `with` block -- only the metric needs recording here.
                _request_counter.add(1, {**attributes, "requisite.status": "error"})
                raise
            _request_duration.record(time.monotonic() - start, attributes)
            _request_counter.add(1, {**attributes, "requisite.status": "success"})

    def stream_response(
        self,
        prompt: Union[str, Sequence[Message]],
        *,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence[ToolLike]] = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Stream the full, normalized :class:`StreamChunk` sequence.

        Unlike :meth:`stream`, each chunk is returned in full -- including
        ``tool_calls`` once the model completes one or more, typically on
        the final chunk (``is_final=True``). See
        :class:`~requisite.core.interfaces.StreamChunk`'s docstring for
        the exact completion contract; check ``chunk.has_tool_calls``
        rather than assuming any particular chunk carries them.
        """
        messages = self._build_messages(prompt, system_prompt)
        effective_temperature = (
            temperature if temperature is not None else self._settings.temperature
        )
        resolved_tools = [resolve_tool_like(t) for t in tools] if tools else None
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
        attributes = {
            "requisite.provider": self._provider.name,
            "requisite.model": model or self._settings.model,
        }
        with _tracer.start_as_current_span("requisite.ai.stream_response", attributes=attributes):
            start = time.monotonic()
            try:
                yield from self._provider.stream(
                    messages,
                    model=model,
                    temperature=effective_temperature,
                    tools=resolved_tools,
                    **kwargs,
                )
            except Exception:
                # start_as_current_span already records the exception and sets
                # an error status on the span automatically as it propagates
                # out of this `with` block -- only the metric needs recording here.
                _request_counter.add(1, {**attributes, "requisite.status": "error"})
                raise
            _request_duration.record(time.monotonic() - start, attributes)
            _request_counter.add(1, {**attributes, "requisite.status": "success"})

    async def astream_response(
        self,
        prompt: Union[str, Sequence[Message]],
        *,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence[ToolLike]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Async counterpart to :meth:`stream_response`."""
        messages = self._build_messages(prompt, system_prompt)
        effective_temperature = (
            temperature if temperature is not None else self._settings.temperature
        )
        resolved_tools = [resolve_tool_like(t) for t in tools] if tools else None
        if self._rate_limiter is not None:
            await self._rate_limiter.aacquire()
        attributes = {
            "requisite.provider": self._provider.name,
            "requisite.model": model or self._settings.model,
        }
        with _tracer.start_as_current_span("requisite.ai.astream_response", attributes=attributes):
            start = time.monotonic()
            try:
                async for chunk in self._provider.astream(
                    messages,
                    model=model,
                    temperature=effective_temperature,
                    tools=resolved_tools,
                    **kwargs,
                ):
                    yield chunk
            except Exception:
                # start_as_current_span already records the exception and sets
                # an error status on the span automatically as it propagates
                # out of this `with` block -- only the metric needs recording here.
                _request_counter.add(1, {**attributes, "requisite.status": "error"})
                raise
            _request_duration.record(time.monotonic() - start, attributes)
            _request_counter.add(1, {**attributes, "requisite.status": "success"})

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"AI(provider={self._provider.name!r}, model={self._provider.model!r})"
