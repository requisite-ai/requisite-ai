"""
Groq provider.

Groq's API is intentionally OpenAI-wire-compatible (confirmed against
Groq's own docs: same request/response shapes, ``client.chat.completions.create``).
Rather than duplicate :class:`~requisite.providers.openai_provider.OpenAIProvider`'s
translation logic, this provider *is* one, configured with Groq's base
URL and a Groq-appropriate default model. This is the intended pattern
for any other OpenAI-wire-compatible provider (OpenRouter, Together AI,
...) -- see ``CONTRIBUTING.md``.

Uses the ``openai`` package (no separate ``groq`` package dependency
required), so no new optional dependency group is needed beyond ``openai``.
"""

from __future__ import annotations

from typing import Any

from requisite.providers.openai_provider import OpenAIProvider

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(OpenAIProvider):
    """Provider implementation backed by Groq's OpenAI-compatible endpoint.

    Parameters
    ----------
    api_key:
        Groq API key. Falls back to the ``GROQ_API_KEY`` environment
        variable if not provided.
    model:
        Default model, e.g. ``"llama-3.3-70b-versatile"``, ``"openai/gpt-oss-20b"``.
    timeout:
        Per-request timeout, in seconds.
    max_retries:
        Number of retries the SDK performs on transient errors.

    Examples
    --------
    >>> provider = GroqProvider(api_key="gsk_...", model="llama-3.3-70b-versatile")
    >>> response = provider.chat([Message.user("Say hi in one word.")])  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
        timeout: float = 60.0,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            base_url=_GROQ_BASE_URL,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "groq"
