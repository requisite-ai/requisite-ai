"""Unit tests for :class:`requisite.config.settings.Settings`."""

from __future__ import annotations

from requisite.config.settings import Settings


def test_defaults_when_env_is_empty(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.default_provider == "openai"
    assert settings.temperature == 0.2
    assert settings.openai_api_key is None


def test_reads_api_keys_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abc")
    monkeypatch.setenv("GEMINI_API_KEY", "g-abc")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.api_key_for("openai") == "sk-abc"
    assert settings.api_key_for("gemini") == "g-abc"


def test_reads_new_provider_api_keys_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abc")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-abc")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-abc")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.api_key_for("anthropic") == "sk-ant-abc"
    assert settings.api_key_for("groq") == "gsk-abc"
    assert settings.api_key_for("azure_openai") == "az-abc"


def test_provider_kwargs_returns_azure_endpoint_for_azure_openai() -> None:
    settings = Settings(
        azure_openai_endpoint="https://my-resource.openai.azure.com", _env_file=None
    )  # type: ignore[call-arg]
    assert settings.provider_kwargs("azure_openai") == {
        "azure_endpoint": "https://my-resource.openai.azure.com"
    }


def test_provider_kwargs_empty_for_providers_without_extra_config() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.provider_kwargs("openai") == {}
    assert settings.provider_kwargs("anthropic") == {}
    assert settings.provider_kwargs("groq") == {}


def test_api_key_for_unknown_provider_returns_none() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.api_key_for("does-not-exist") is None


def test_secret_values_are_masked_in_repr() -> None:
    settings = Settings(openai_api_key="super-secret-key", _env_file=None)  # type: ignore[call-arg]
    assert "super-secret-key" not in repr(settings)


def test_explicit_kwargs_override_environment(monkeypatch) -> None:
    monkeypatch.setenv("DEFAULT_PROVIDER", "openai")
    settings = Settings(default_provider="gemini", _env_file=None)  # type: ignore[call-arg]
    assert settings.default_provider == "gemini"
