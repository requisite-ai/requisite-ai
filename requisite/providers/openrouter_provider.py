"""
OpenRouter provider.

OpenRouter's API is intentionally OpenAI-wire-compatible (confirmed
against OpenRouter's own docs: same request/response shapes, works as a
drop-in with the ``openai`` Python client pointed at
``https://openrouter.ai/api/v1``). Rather than duplicate
:class:`~requisite.providers.openai_provider.OpenAIProvider`'s
translation logic, this provider *is* one, configured with OpenRouter's
base URL and a sensible default model -- the same pattern already used
for :class:`~requisite.providers.groq_provider.GroqProvider` and
:class:`~requisite.providers.azure_openai_provider.AzureOpenAIProvider`.

Uses the ``openai`` package (no separate ``openrouter`` package
required), so no new optional dependency group is needed beyond ``openai``.
"""

from __future__ import annotations

from typing import Any, Optional

from requisite.providers.openai_provider import OpenAIProvider

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAIProvider):
    """Provider implementation backed by OpenRouter's OpenAI-compatible endpoint.

    OpenRouter routes to many underlying model providers through one API,
    so ``model`` uses OpenRouter's ``vendor/model`` naming convention
    (e.g. ``"openai/gpt-4o-mini"``, ``"anthropic/claude-sonnet-4-6"``) --
    see https://openrouter.ai/models for the full catalog.

    Parameters
    ----------
    api_key:
        OpenRouter API key. Falls back to the ``OPENROUTER_API_KEY``
        environment variable if not provided.
    model:
        Default model, in OpenRouter's ``vendor/model`` form.
    timeout:
        Per-request timeout, in seconds.
    max_retries:
        Number of retries the SDK performs on transient errors.

    Examples
    --------
    >>> provider = OpenRouterProvider(api_key="sk-or-...", model="openai/gpt-4o-mini")
    >>> response = provider.chat([Message.user("Say hi in one word.")])  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
        timeout: float = 60.0,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            base_url=_OPENROUTER_BASE_URL,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "openrouter"
