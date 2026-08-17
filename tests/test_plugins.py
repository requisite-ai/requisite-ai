"""Unit tests for requisite.plugins.

Constructs real `importlib.metadata.EntryPoint` objects pointing at
tests/fixture_plugin.py (a genuinely importable module, not a mock) and
monkeypatches `requisite.plugins.entry_points` -- the name imported into
that module -- to return them, so `discover()`'s actual `.load()` /
call-if-callable logic runs for real. Only the *source* of entry points
(normally read from installed package metadata) is faked.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint

import pytest

from requisite import plugins as plugins_module
from tests import fixture_plugin


def _entry_point(name: str, value: str) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=plugins_module.DEFAULT_GROUP)


def test_discover_loads_module_only_target(monkeypatch: pytest.MonkeyPatch) -> None:
    ep = _entry_point("module_target", "tests.fixture_plugin")
    monkeypatch.setattr(plugins_module, "entry_points", lambda *, group: [ep])

    result = plugins_module.discover()

    assert result.loaded == ["module_target"]
    assert result.failed == {}


def test_discover_calls_callable_target(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_plugin.register_call_count = 0
    ep = _entry_point("callable_target", "tests.fixture_plugin:register")
    monkeypatch.setattr(plugins_module, "entry_points", lambda *, group: [ep])

    result = plugins_module.discover()

    assert result.loaded == ["callable_target"]
    assert fixture_plugin.register_call_count == 1


def test_discover_records_failure_without_blocking_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken = _entry_point("broken", "tests.fixture_plugin:does_not_exist")
    working = _entry_point("working", "tests.fixture_plugin")
    monkeypatch.setattr(plugins_module, "entry_points", lambda *, group: [broken, working])

    result = plugins_module.discover()

    assert result.loaded == ["working"]
    assert "broken" in result.failed
    assert "does_not_exist" in result.failed["broken"]


def test_discover_returns_empty_result_when_no_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plugins_module, "entry_points", lambda *, group: [])

    result = plugins_module.discover()

    assert result.loaded == []
    assert result.failed == {}


def test_discover_passes_group_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_entry_points(*, group: str) -> list[EntryPoint]:
        captured["group"] = group
        return []

    monkeypatch.setattr(plugins_module, "entry_points", fake_entry_points)

    plugins_module.discover(group="custom.group")

    assert captured["group"] == "custom.group"
