"""
Together AI provider.

Together AI's API is intentionally OpenAI-wire-compatible (confirmed
against Together's own docs: ``chat.completions.create`` with the plain
``openai`` client works as a drop-in against
``https://api.together.ai/v1``, including ``tools=`` and
``response_format=`` for structured output). Rather than duplicate
:class:`~requisite.providers.openai_provider.OpenAIProvider`'s
translation logic, this provider *is* one, configured with Together's
base URL and a sensible default model -- the same pattern already used
for :class:`~requisite.providers.groq_provider.GroqProvider` and
:class:`~requisite.providers.azure_openai_provider.AzureOpenAIProvider`.

Uses the ``openai`` package (no separate ``together`` package required),
so no new optional dependency group is needed beyond ``openai``.
"""

from __future__ import annotations

from typing import Any, Optional

from requisite.providers.openai_provider import OpenAIProvider

_TOGETHER_BASE_URL = "https://api.together.ai/v1"


class TogetherProvider(OpenAIProvider):
    """Provider implementation backed by Together AI's OpenAI-compatible endpoint.

    Together hosts open-source models under namespaced model IDs (e.g.
    ``"meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"``) -- see
    https://www.together.ai/models for the full catalog.

    Parameters
    ----------
    api_key:
        Together API key. Falls back to the ``TOGETHER_API_KEY``
        environment variable if not provided.
    model:
        Default model, in Together's namespaced ``org/Model-Name`` form.
    timeout:
        Per-request timeout, in seconds.
    max_retries:
        Number of retries the SDK performs on transient errors.

    Examples
    --------
    >>> provider = TogetherProvider(
    ...     api_key="...", model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    ... )
    >>> response = provider.chat([Message.user("Say hi in one word.")])  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        timeout: float = 60.0,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            base_url=_TOGETHER_BASE_URL,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "together"
