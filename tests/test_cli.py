"""
Unit tests for the ``requisite`` CLI (``requisite.cli``).

Calls ``requisite.cli.main.main(argv=[...])`` in-process and asserts on
``capsys`` output, following the "no test may require network access or
a real API key" convention -- the same DI shape ``AI``/``Agent`` already
use (``registry=``, ``settings=``) is threaded through ``main()``
specifically so a fake provider can be injected here instead of touching
the shared ``default_registry``, or reading the repo's real ``.env``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Any, Optional

import pytest

from requisite.cli.main import main
from requisite.config.settings import Settings
from requisite.core.interfaces import ChatResponse, Message, StreamChunk, Usage
from requisite.providers.base import BaseProvider
from requisite.providers.factory import ProviderRegistry


class FakeProvider(BaseProvider):
    """Deterministic in-memory provider -- mirrors tests/test_ai.py's fixture."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(api_key="fake-key", model=kwargs.get("model", "fake-model"))

    @property
    def name(self) -> str:
        return "fake"

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence[Any]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        last = messages[-1].content if messages else ""
        return ChatResponse(
            content=f"fake reply to: {last}",
            model=model or self._model,
            provider=self.name,
            usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )

    async def achat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[Sequence[Any]] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        return self.chat(messages, model=model, temperature=temperature, tools=tools, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Iterator[StreamChunk]:
        raise NotImplementedError  # pragma: no cover - unused by these tests

    async def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError  # pragma: no cover - unused by these tests
        yield  # pragma: no cover


@pytest.fixture
def fake_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("fake", FakeProvider)
    return registry


@pytest.fixture
def fake_settings() -> Settings:
    return Settings(default_provider="fake", model="fake-model", _env_file=None)  # type: ignore[call-arg]


# --- init -------------------------------------------------------------------


def test_init_scaffolds_a_project(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dest = tmp_path / "myproject"
    exit_code = main(["init", str(dest), "--provider", "gemini"])

    assert exit_code == 0
    for filename in (
        ".env.example",
        ".gitignore",
        "requirements.txt",
        "agents.py",
        "main.py",
        "README.md",
    ):
        assert (dest / filename).is_file()

    assert "GEMINI_API_KEY" in (dest / ".env.example").read_text()
    assert "gemini" in (dest / "requirements.txt").read_text()
    assert "agent_registry" in (dest / "agents.py").read_text()

    out = capsys.readouterr().out
    assert "Created" in out
    assert "Next steps" in out


def test_init_refuses_nonempty_directory_without_force(tmp_path: Path) -> None:
    dest = tmp_path / "myproject"
    dest.mkdir()
    (dest / "existing.txt").write_text("hi")

    exit_code = main(["init", str(dest)])
    assert exit_code == 1


def test_init_force_overwrites(tmp_path: Path) -> None:
    dest = tmp_path / "myproject"
    dest.mkdir()
    (dest / "existing.txt").write_text("hi")

    exit_code = main(["init", str(dest), "--force"])
    assert exit_code == 0
    assert (dest / "agents.py").is_file()


# --- providers ----------------------------------------------------------------


def test_providers_lists_registered_providers(
    fake_registry: ProviderRegistry, fake_settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["providers"], registry=fake_registry, settings=fake_settings)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "fake" in out
    assert "PROVIDER" in out


# --- capabilities ---------------------------------------------------------------


def test_capabilities_lists_builtin_capabilities(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["capabilities"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "weather" in out
    assert "filesystem" in out
    assert "internet_search" in out
    assert "priority=" in out


# --- plugins ------------------------------------------------------------------


def test_plugins_reports_no_plugins_found(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("requisite.plugins.entry_points", lambda *, group: [])

    exit_code = main(["plugins"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "No plugins found" in out


def test_plugins_lists_loaded_plugins(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from importlib.metadata import EntryPoint

    from requisite.plugins import DEFAULT_GROUP

    ep = EntryPoint(name="fixture", value="tests.fixture_plugin", group=DEFAULT_GROUP)
    monkeypatch.setattr("requisite.plugins.entry_points", lambda *, group: [ep])

    exit_code = main(["plugins"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Loaded:" in out
    assert "fixture" in out


def test_plugins_reports_failures_and_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from importlib.metadata import EntryPoint

    from requisite.plugins import DEFAULT_GROUP

    ep = EntryPoint(name="broken", value="tests.fixture_plugin:does_not_exist", group=DEFAULT_GROUP)
    monkeypatch.setattr("requisite.plugins.entry_points", lambda *, group: [ep])

    exit_code = main(["plugins"])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "Failed:" in out
    assert "broken" in out


def test_plugins_custom_group(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib.metadata import EntryPoint

    captured: dict[str, str] = {}

    def fake_entry_points(*, group: str) -> list[EntryPoint]:
        captured["group"] = group
        return []

    monkeypatch.setattr("requisite.plugins.entry_points", fake_entry_points)

    main(["plugins", "--group", "custom.group"])
    assert captured["group"] == "custom.group"


# --- agents -----------------------------------------------------------------


def test_agents_without_agents_py_prints_helpful_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    exit_code = main(["agents"])
    assert exit_code == 1


def test_agents_lists_scaffolded_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["init", str(tmp_path), "--force"])
    monkeypatch.chdir(tmp_path)

    exit_code = main(["agents"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "assistant" in out


# --- chat ---------------------------------------------------------------------


def test_chat_one_shot(
    fake_registry: ProviderRegistry, fake_settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["chat", "hello there"], registry=fake_registry, settings=fake_settings)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "fake reply to: hello there" in out


def test_chat_interactive_repl(
    fake_registry: ProviderRegistry,
    fake_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses = iter(["hi", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(responses))

    exit_code = main(["chat"], registry=fake_registry, settings=fake_settings)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "fake reply to: hi" in out


def test_chat_with_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _StubResult:
        content = "stub agent reply"

    class _StubAgent:
        def run(self, prompt: str) -> "_StubResult":
            return _StubResult()

    class _StubRegistry:
        def get(self, name: str) -> "_StubAgent":
            assert name == "assistant"
            return _StubAgent()

    monkeypatch.setattr(
        "requisite.cli.commands.load_agent_registry", lambda module_path: _StubRegistry()
    )
    monkeypatch.chdir(tmp_path)

    exit_code = main(["chat", "hello", "--agent", "assistant"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "stub agent reply" in out


def test_chat_with_agent_and_provider_reports_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression test: --agent together with --provider/--model
    previously silently discarded --provider/--model with no warning
    (a named agent already has its own configured provider). See
    docs/adr/0031-code-review-fixes.md."""

    class _StubRegistry:
        def get(self, name: str) -> None:  # pragma: no cover - must not be reached
            raise AssertionError("agent registry should not be consulted")

    monkeypatch.setattr(
        "requisite.cli.commands.load_agent_registry", lambda module_path: _StubRegistry()
    )
    monkeypatch.chdir(tmp_path)

    exit_code = main(["chat", "hello", "--agent", "assistant", "--provider", "anthropic"])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "--agent" in err


def test_chat_with_unknown_agent_reports_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main(["init", str(tmp_path), "--force"])
    monkeypatch.chdir(tmp_path)

    exit_code = main(["chat", "hello", "--agent", "nonexistent"])
    assert exit_code == 1


# --- misc ---------------------------------------------------------------------


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
