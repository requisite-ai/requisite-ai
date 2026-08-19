"""
OpenAI embedding provider.

Uses the same ``openai`` package as :class:`~requisite.providers.openai_provider.OpenAIProvider`
(client-based SDK, ``client.embeddings.create``). Default model
``text-embedding-3-small`` -- current, cost-effective, and not
deprecated; override ``model=`` for ``text-embedding-3-large`` or a
newer model as needed.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Optional

from requisite.core.exceptions import ConfigurationException, ProviderException
from requisite.rag.base import BaseEmbeddingProvider


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider backed by OpenAI's embeddings API.

    Parameters
    ----------
    api_key:
        OpenAI API key. Falls back to the ``OPENAI_API_KEY`` environment
        variable if not provided.
    model:
        Embedding model, e.g. ``"text-embedding-3-small"`` (default),
        ``"text-embedding-3-large"``.
    timeout:
        Per-request timeout, in seconds.
    max_retries:
        Number of retries the SDK performs on transient errors.

    Examples
    --------
    >>> provider = OpenAIEmbeddingProvider(api_key="sk-...")
    >>> vector = provider.embed_one("hello world")  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        timeout: float = 60.0,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: Any = None

    @property
    def name(self) -> str:
        return "openai"

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - exercised only without the dep
                raise ConfigurationException(
                    "The 'openai' package is required for OpenAIEmbeddingProvider. "
                    "Install it with: pip install openai",
                ) from exc
            if not self._api_key:
                raise ConfigurationException(
                    "Missing API key for OpenAIEmbeddingProvider. Set it via configuration "
                    "or the OPENAI_API_KEY environment variable.",
                )
            self._client = OpenAI(
                api_key=self._api_key, timeout=self._timeout, max_retries=self._max_retries
            )
        return self._client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        client = self._get_client()
        try:
            response = client.embeddings.create(input=list(texts), model=self._model)
        except Exception as exc:  # noqa: BLE001
            raise ProviderException(
                f"OpenAI embeddings.create failed: {exc}",
                provider=self.name,
                original_error=exc,
            ) from exc
        return [item.embedding for item in response.data]
