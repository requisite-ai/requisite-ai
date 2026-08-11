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

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel

from requisite.core.interfaces import ChatResponse, Message, StreamChunk

if TYPE_CHECKING:
    from requisite.tools.base import Tool

logger = logging.getLogger("requisite.providers")


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
        this to the underlying SDK's own retry mechanism when possible.
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
