"""
Gemini embedding provider.

Uses the current unified ``google-genai`` SDK (``client.models.embed_content``),
same package as :class:`~requisite.providers.gemini_provider.GeminiProvider`.
Default model ``gemini-embedding-001`` -- the current GA embedding model.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Optional

from requisite.core.exceptions import ConfigurationException, ProviderException
from requisite.rag.base import BaseEmbeddingProvider


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider backed by Gemini's embed_content API.

    Parameters
    ----------
    api_key:
        Gemini API key. Falls back to the ``GEMINI_API_KEY`` /
        ``GOOGLE_API_KEY`` environment variables if not provided.
    model:
        Embedding model. Defaults to ``"gemini-embedding-001"`` (current GA).

    Examples
    --------
    >>> provider = GeminiEmbeddingProvider(api_key="...")
    >>> vector = provider.embed_one("hello world")  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gemini-embedding-001",
        **kwargs: Any,
    ) -> None:
        self._api_key = (
            api_key
            if api_key is not None
            else os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        self._model = model
        self._client: Any = None

    @property
    def name(self) -> str:
        return "gemini"

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - exercised only without the dep
                raise ConfigurationException(
                    "The 'google-genai' package is required for GeminiEmbeddingProvider. "
                    "Install it with: pip install google-genai",
                ) from exc
            if not self._api_key:
                raise ConfigurationException(
                    "Missing API key for GeminiEmbeddingProvider. Set it via configuration "
                    "or the GEMINI_API_KEY environment variable.",
                )
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        client = self._get_client()
        try:
            response = client.models.embed_content(model=self._model, contents=list(texts))
        except Exception as exc:  # noqa: BLE001
            raise ProviderException(
                f"Gemini embed_content failed: {exc}",
                provider=self.name,
                original_error=exc,
            ) from exc
        return [list(embedding.values) for embedding in response.embeddings]
