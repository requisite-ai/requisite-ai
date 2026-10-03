"""
Provider interface.

Every AI provider (OpenAI, Gemini, Anthropic, ...) implements
:class:`BaseProvider`. Business logic and the public :class:`~requisite.ai.AI`
facade depend only on this abstract interface -- never on a concrete
provider class -- which is what allows switching providers via
configuration alone.

Adding a new provider requires implementing this interface only; no
changes to any other part of the framework are needed.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Optional, TypeVar

from pydantic import BaseModel

from requisite.core.interfaces import ChatResponse, Message, StreamChunk

if TYPE_CHECKING:
    from requisite.tools.base import Tool

logger = logging.getLogger("requisite.providers")

_T = TypeVar("_T")


def network_errors() -> tuple[type[BaseException], ...]:
    """Connection-level exceptions worth retrying: dropped/aborted/reset
    connections and timeouts, for providers whose SDK is httpx-based.

    ``httpx.TransportError`` covers ``ReadError``, ``ConnectError``,
    ``RemoteProtocolError`` and the timeout family -- notably wider than
    ``google-genai``'s own transient list, which only has
    ``TimeoutException``/``ConnectError`` and so misses a mid-request
    ``ReadError`` (e.g. Windows ``WinError 10053``).
    """
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx ships with every httpx-based SDK
        return (ConnectionError, TimeoutError)
    return (httpx.TransportError, ConnectionError, TimeoutError)


class BaseProvider(ABC):
    """Abstract base class for all LLM providers.

    Parameters
    ----------
    api_key:
        Provider API key. Concrete implementations validate this is
        present before making any network call.
    model:
        Default model identifier to use for requests that don't
        override it explicitly.
    timeout:
        Default request timeout, in seconds.
    max_retries:
        Number of retries for transient failures (rate limits,
        connection errors). Concrete implementations should delegate
        this to the underlying SDK's own retry mechanism when possible;
        when the SDK has none (or its own misses real connection
        errors), override :meth:`_transient_errors` and wrap the raw SDK
        call in :meth:`_call_with_retries`/:meth:`_acall_with_retries`.
    **kwargs:
        Additional provider-specific options (e.g. ``base_url`` for
        self-hosted or proxy endpoints).

    Notes
    -----
    Implementations must support both synchronous and asynchronous
    execution, plus streaming, without duplicating logic where
    avoidable (e.g. by having the sync path call into shared helpers).
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._extra_options = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, unique, lowercase identifier for this provider (e.g. ``"openai"``)."""

    @property
    def model(self) -> str:
        """The default model identifier configured for this provider instance."""
        return self._model

    @abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence["Tool"]] = None,
        response_model: Optional[type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Synchronously generate a chat completion.

        Parameters
        ----------
        messages:
            Ordered conversation history, oldest first.
        model:
            Overrides the provider's default model for this call.
        temperature:
            Overrides the default sampling temperature for this call.
        tools:
            Tools the model may call. When the model requests one or more,
            they're reported on the returned ``ChatResponse.tool_calls``
            rather than executed automatically -- execution is the
            caller's (or an ``Agent``'s) responsibility.
        response_model:
            When given, the provider constrains generation to match this
            Pydantic model's schema and populates
            ``ChatResponse.parsed`` with a validated instance.
        **kwargs:
            Provider-specific passthrough options (e.g. ``top_p``).

        Returns
        -------
        ChatResponse
            Normalized response.

        Raises
        ------
        requisite.core.exceptions.ProviderException
            If the underlying provider call fails.
        """

    @abstractmethod
    async def achat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence["Tool"]] = None,
        response_model: Optional[type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Asynchronous counterpart to :meth:`chat`. Same parameters and behavior."""

    @abstractmethod
    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence["Tool"]] = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Synchronously stream a chat completion, chunk by chunk.

        Parameters
        ----------
        messages:
            Ordered conversation history, oldest first.
        model:
            Overrides the provider's default model for this call.
        temperature:
            Overrides the default sampling temperature for this call.
        tools:
            Tools the model may call. Implementations accumulate any
            incremental/fragmented tool-call data internally (SDKs differ
            here) and only attach fully-assembled tool calls to a
            :class:`StreamChunk` once complete -- see
            :class:`StreamChunk`'s docstring for the exact contract.
        **kwargs:
            Provider-specific passthrough options (e.g. ``top_p``).

        Yields
        ------
        StreamChunk
            Incremental pieces of the response, in order.
        """

    @abstractmethod
    def astream(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence["Tool"]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Asynchronous counterpart to :meth:`stream`. Same parameters and behavior."""

    def _transient_errors(self) -> tuple[type[BaseException], ...]:
        """Exception types :meth:`_call_with_retries` should retry.

        Empty by default, i.e. no retry: providers whose SDK already
        retries internally (OpenAI, Anthropic and everything wire-compatible
        with them) must keep it that way, or retries would multiply.
        """
        return ()

    def _retry_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter: ~0.5s, 1s, 2s, ... capped at 8s."""
        return float(min(0.5 * 2**attempt, 8.0) * (0.5 + random.random() / 2))

    def _call_with_retries(self, call: Callable[[], _T]) -> _T:
        """Run ``call``, retrying up to ``max_retries`` times on
        :meth:`_transient_errors` with exponential backoff. Anything else,
        and the final failure once retries are exhausted, propagates
        unchanged. Wrap only the raw SDK call -- not response conversion.
        """
        attempt = 0
        while True:
            try:
                return call()
            except self._transient_errors() as exc:
                if attempt >= self._max_retries:
                    raise
                delay = self._retry_delay(attempt)
                attempt += 1
                logger.warning(
                    "%s: transient error (%s: %s); retry %d/%d in %.1fs",
                    self.name,
                    type(exc).__name__,
                    exc,
                    attempt,
                    self._max_retries,
                    delay,
                )
                time.sleep(delay)

    async def _acall_with_retries(self, call: Callable[[], Awaitable[_T]]) -> _T:
        """Asynchronous counterpart to :meth:`_call_with_retries`."""
        attempt = 0
        while True:
            try:
                return await call()
            except self._transient_errors() as exc:
                if attempt >= self._max_retries:
                    raise
                delay = self._retry_delay(attempt)
                attempt += 1
                logger.warning(
                    "%s: transient error (%s: %s); retry %d/%d in %.1fs",
                    self.name,
                    type(exc).__name__,
                    exc,
                    attempt,
                    self._max_retries,
                    delay,
                )
                await asyncio.sleep(delay)

    def validate_config(self) -> None:
        """Raise :class:`~requisite.core.exceptions.ConfigurationException`
        if this provider instance is missing required configuration.

        Concrete providers may override this for provider-specific checks;
        the default implementation only verifies an API key is present.
        """
        from requisite.core.exceptions import ConfigurationException

        if not self._api_key:
            raise ConfigurationException(
                f"Missing API key for provider '{self.name}'. "
                f"Set it via configuration or the appropriate environment variable.",
            )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}(model={self._model!r})"
