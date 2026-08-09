"""
Provider registry and factory.

The registry lets new providers be added -- by the framework itself or
by third-party plugins -- without ever modifying framework code: just
call :meth:`ProviderRegistry.register` with a name and a callable that
builds a :class:`~requisite.providers.base.BaseProvider`.

Design notes
------------
``ProviderRegistry`` is a plain, instantiable class rather than a
classic Singleton -- you can create as many independent registries as
you like (handy for tests, or for isolating plugin sets per
application). For convenience, a single shared ``default_registry``
instance is created at import time and pre-populated with the built-in
providers; most applications will just use that one via
:func:`create_provider`, but nothing prevents constructing a fresh
:class:`ProviderRegistry` and injecting it into :class:`requisite.ai.AI`
explicitly.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from requisite.core.exceptions import ConfigurationException
from requisite.providers.base import BaseProvider

logger = logging.getLogger("requisite.providers.factory")

ProviderBuilder = Callable[..., BaseProvider]


class ProviderRegistry:
    """A registry mapping provider names to constructors.

    Examples
    --------
    Registering a custom or third-party provider:

    >>> from requisite.providers.factory import ProviderRegistry
    >>> from requisite.providers.base import BaseProvider
    >>> registry = ProviderRegistry()
    >>> class EchoProvider(BaseProvider):
    ...     name = "echo"  # doctest: +SKIP
    >>> registry.register("echo", EchoProvider)  # doctest: +SKIP

    Creating a provider instance by name:

    >>> provider = registry.create("openai", api_key="sk-...", model="gpt-4o-mini")  # doctest: +SKIP
    """

    def __init__(self) -> None:
        self._builders: dict[str, ProviderBuilder] = {}

    def register(self, name: str, builder: ProviderBuilder) -> None:
        """Register a provider constructor under ``name``.

        Parameters
        ----------
        name:
            Unique, lowercase identifier (e.g. ``"openai"``). Re-registering
            an existing name overwrites it, which allows plugins or tests
            to substitute alternative implementations deliberately.
        builder:
            A callable (typically the provider class itself) that accepts
            the same keyword arguments as
            :class:`~requisite.providers.base.BaseProvider` and returns
            an instance of it.
        """
        key = name.lower().strip()
        if not key:
            raise ConfigurationException("Provider name must be a non-empty string.")
        self._builders[key] = builder
        logger.debug("Registered provider '%s'", key, extra={"provider": key})

    def unregister(self, name: str) -> None:
        """Remove a provider registration, if present. No-op if absent."""
        self._builders.pop(name.lower().strip(), None)

    def is_registered(self, name: str) -> bool:
        """Return whether a provider is registered under ``name``."""
        return name.lower().strip() in self._builders

    def available(self) -> list[str]:
        """Return the sorted list of currently registered provider names."""
        return sorted(self._builders)

    def create(self, name: str, **kwargs: Any) -> BaseProvider:
        """Instantiate the provider registered under ``name``.

        Parameters
        ----------
        name:
            Provider identifier, e.g. ``"openai"`` or ``"gemini"``.
        **kwargs:
            Forwarded to the provider's constructor (``api_key``, ``model``,
            ``timeout``, ``max_retries``, provider-specific options, ...).

        Returns
        -------
        BaseProvider

        Raises
        ------
        ConfigurationException
            If no provider is registered under ``name``.
        """
        key = name.lower().strip()
        builder = self._builders.get(key)
        if builder is None:
            raise ConfigurationException(
                f"Unknown provider '{name}'. Available providers: {self.available()}",
            )
        return builder(**kwargs)


def _register_builtin_providers(registry: ProviderRegistry) -> None:
    """Register the providers that ship with the framework.

    Imports are performed lazily, inside small factory functions, so
    that importing ``requisite`` never requires every optional
    provider SDK (``openai``, ``google-genai``, ...) to be installed --
    only the SDK for the provider you actually use.
    """

    def _build_openai(**kwargs: Any) -> BaseProvider:
        from requisite.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(**kwargs)

    def _build_gemini(**kwargs: Any) -> BaseProvider:
        from requisite.providers.gemini_provider import GeminiProvider

        return GeminiProvider(**kwargs)

    def _build_anthropic(**kwargs: Any) -> BaseProvider:
        from requisite.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(**kwargs)

    def _build_groq(**kwargs: Any) -> BaseProvider:
        from requisite.providers.groq_provider import GroqProvider

        return GroqProvider(**kwargs)

    def _build_azure_openai(**kwargs: Any) -> BaseProvider:
        from requisite.providers.azure_openai_provider import AzureOpenAIProvider

        return AzureOpenAIProvider(**kwargs)

    def _build_openrouter(**kwargs: Any) -> BaseProvider:
        from requisite.providers.openrouter_provider import OpenRouterProvider

        return OpenRouterProvider(**kwargs)

    def _build_together(**kwargs: Any) -> BaseProvider:
        from requisite.providers.together_provider import TogetherProvider

        return TogetherProvider(**kwargs)

    def _build_ollama(**kwargs: Any) -> BaseProvider:
        from requisite.providers.ollama_provider import OllamaProvider

        return OllamaProvider(**kwargs)

    registry.register("openai", _build_openai)
    registry.register("gemini", _build_gemini)
    registry.register("anthropic", _build_anthropic)
    registry.register("groq", _build_groq)
    registry.register("azure_openai", _build_azure_openai)
    registry.register("openrouter", _build_openrouter)
    registry.register("together", _build_together)
    registry.register("ollama", _build_ollama)


default_registry = ProviderRegistry()
_register_builtin_providers(default_registry)


def create_provider(name: str, **kwargs: Any) -> BaseProvider:
    """Convenience wrapper around ``default_registry.create``.

    Equivalent to ``default_registry.create(name, **kwargs)``. Prefer
    working with an explicit :class:`ProviderRegistry` instance in
    library code and tests; this function is a shortcut for scripts and
    simple applications.
    """
    return default_registry.create(name, **kwargs)
