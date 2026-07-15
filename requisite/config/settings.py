"""
Application configuration.

Uses ``pydantic-settings`` (the current, supported way to load typed
settings in Pydantic v2 -- the old ``pydantic.BaseSettings`` was removed
from pydantic core and now lives in this separate package) so that
developers never have to manually call ``os.environ.get`` or parse a
``.env`` file by hand.

Examples
--------
>>> from requisite.config.settings import Settings
>>> settings = Settings()  # reads OPENAI_API_KEY, GEMINI_API_KEY, etc.
>>> settings.default_provider
'openai'

Override anything via keyword arguments (useful in tests):

>>> settings = Settings(default_provider="gemini", model="gemini-2.0-flash")
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application configuration.

    All fields can be supplied via environment variables, a ``.env``
    file in the current working directory, or explicit keyword
    arguments (which always take precedence). See ``.env.example`` in
    the project root for the full list of supported variables.

    Attributes
    ----------
    openai_api_key:
        API key for OpenAI. Required only if the OpenAI provider is used.
    gemini_api_key:
        API key for Google Gemini. Required only if the Gemini provider
        is used.
    default_provider:
        Name of the provider to use when none is explicitly passed to
        :class:`requisite.ai.AI`. Defaults to ``"openai"``.
    model:
        Default model identifier used when none is explicitly supplied.
    temperature:
        Default sampling temperature, in ``[0.0, 2.0]``.
    request_timeout:
        Default per-request timeout, in seconds, applied to provider
        HTTP clients.
    max_retries:
        Default number of retries for transient provider failures.
    log_level:
        Standard library logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
    log_format:
        ``"plain"`` (default) or ``"json"``. Passed to
        ``requisite.telemetry.configure_logging(json_format=...)`` if the
        application opts into it -- Settings only stores the preference,
        it never configures logging itself (see that function's
        docstring for why).

    Notes
    -----
    API keys are stored as :class:`pydantic.SecretStr` so that they are
    never accidentally logged or printed -- ``repr(settings)`` masks
    them automatically.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openai_api_key: Optional[SecretStr] = Field(default=None)
    gemini_api_key: Optional[SecretStr] = Field(default=None)
    anthropic_api_key: Optional[SecretStr] = Field(default=None)
    groq_api_key: Optional[SecretStr] = Field(default=None)
    azure_openai_api_key: Optional[SecretStr] = Field(default=None)
    azure_openai_endpoint: Optional[str] = Field(
        default=None,
        description=(
            "Your Azure OpenAI resource endpoint, e.g. "
            "'https://your-resource.openai.azure.com'. Required only if "
            "provider='azure_openai' is used."
        ),
    )

    default_provider: str = Field(default="openai")
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    request_timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)

    log_level: str = Field(default="INFO")
    log_format: str = Field(default="plain")

    def api_key_for(self, provider: str) -> Optional[str]:
        """Return the plaintext API key for the given provider name.

        Parameters
        ----------
        provider:
            Provider identifier, e.g. ``"openai"``, ``"gemini"``,
            ``"anthropic"``, ``"groq"``, or ``"azure_openai"``.

        Returns
        -------
        Optional[str]
            The plaintext key, or ``None`` if not configured.
        """
        mapping: dict[str, Optional[SecretStr]] = {
            "openai": self.openai_api_key,
            "gemini": self.gemini_api_key,
            "anthropic": self.anthropic_api_key,
            "groq": self.groq_api_key,
            "azure_openai": self.azure_openai_api_key,
        }
        secret = mapping.get(provider.lower())
        return secret.get_secret_value() if secret else None

    def provider_kwargs(self, provider: str) -> dict[str, Any]:
        """Return extra, provider-specific constructor kwargs beyond the
        common ``api_key``/``model``/``timeout``/``max_retries`` set.

        Most providers need nothing extra here. This exists specifically
        for providers whose constructor requires configuration that has
        no equivalent on other providers -- today, only
        ``azure_openai``'s ``azure_endpoint``. See ADR-0002 for the
        reasoning behind keeping this generic rather than special-casing
        Azure inside :class:`~requisite.ai.AI`.

        Parameters
        ----------
        provider:
            Provider identifier.

        Returns
        -------
        dict
            Extra keyword arguments to merge into the provider's
            constructor call. Empty for providers with no extra config.
        """
        if provider.lower() == "azure_openai":
            return {"azure_endpoint": self.azure_openai_endpoint}
        return {}


@lru_cache(maxsize=1)
def get_cached_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance.

    This is an opt-in convenience for callers that want to avoid
    re-parsing the environment on every call. It is *not* used
    internally by :class:`requisite.ai.AI` by default -- that class
    accepts an explicit ``Settings`` instance via constructor injection
    so tests can freely construct isolated configurations.
    """
    return Settings()
