"""
Built-in AI providers and the extensible provider registry.

Provider SDKs (``openai``, ``google-genai``, ...) are imported lazily by
each provider module, so simply importing ``requisite.providers``
never requires every optional dependency to be installed.
"""

from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry, create_provider, default_registry

__all__ = ["BaseProvider", "ProviderRegistry", "default_registry", "create_provider"]
