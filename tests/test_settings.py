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
