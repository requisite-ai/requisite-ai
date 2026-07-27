"""
Azure OpenAI provider.

Uses Azure OpenAI's **v1 GA API** (generally available since August 2025),
which lets you use the plain ``OpenAI`` client pointed at
``{azure_endpoint}/openai/v1/`` -- no ``AzureOpenAI``-specific client class
and no dated ``api_version`` string required. This deliberately avoids the
older ``AzureOpenAI`` / ``AsyncAzureOpenAI`` client + monthly-dated
``api_version`` pattern, which Microsoft's own current docs describe as
superseded by the v1 API for new code (confirmed live against Microsoft
Learn and the ``openai`` package's own Azure examples).

Because the v1 API's request/response shape is otherwise identical to
OpenAI's, this provider -- like :class:`~requisite.providers.groq_provider.GroqProvider`
-- is implemented as a thin :class:`~requisite.providers.openai_provider.OpenAIProvider`
subclass rather than duplicating its translation logic.

No separate SDK: this uses the ``openai`` package.
"""

from __future__ import annotations

from typing import Any

from requisite.core.exceptions import ConfigurationException
from requisite.providers.openai_provider import OpenAIProvider


class AzureOpenAIProvider(OpenAIProvider):
    """Provider implementation backed by Azure OpenAI's v1 GA API.

    Parameters
    ----------
    api_key:
        Azure OpenAI API key. Falls back to the ``AZURE_OPENAI_API_KEY``
        environment variable if not provided.
    azure_endpoint:
        Your Azure OpenAI resource endpoint, e.g.
        ``"https://your-resource-name.openai.azure.com"``. Required --
        there is no sensible framework-wide default.
    model:
        Your **deployment name** in Azure OpenAI Studio/Foundry (e.g.
        ``"gpt-4.1-nano"``), not necessarily the underlying model's name.
    timeout:
        Per-request timeout, in seconds.
    max_retries:
        Number of retries the SDK performs on transient errors.

    Examples
    --------
    >>> provider = AzureOpenAIProvider(
    ...     api_key="...",
    ...     azure_endpoint="https://your-resource.openai.azure.com",
    ...     model="gpt-4.1-nano",  # your deployment name
    ... )
    >>> response = provider.chat([Message.user("Say hi in one word.")])  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> None:
        if not azure_endpoint:
            raise ConfigurationException(
                "AzureOpenAIProvider requires 'azure_endpoint' (your Azure OpenAI "
                "resource URL, e.g. https://your-resource.openai.azure.com). Set it "
                "via Settings.azure_openai_endpoint / AZURE_OPENAI_ENDPOINT, or pass "
                "it directly.",
            )
        base_url = f"{azure_endpoint.rstrip('/')}/openai/v1/"
        super().__init__(
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            base_url=base_url,
            **kwargs,
        )
        self._azure_endpoint = azure_endpoint

    @property
    def name(self) -> str:
        return "azure_openai"
