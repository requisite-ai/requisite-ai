"""Unit tests for requisite.telemetry (structured/JSON logging)."""

from __future__ import annotations

import json
import logging

from requisite.telemetry.logging import JSONFormatter, configure_logging


def _make_record(
    message: str = "hello",
    *,
    level: int = logging.INFO,
    extra: dict | None = None,
    exc_info=None,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name="requisite.example",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=exc_info,
    )
    for key, value in (extra or {}).items():
        setattr(record, key, value)
    return record


def test_json_formatter_includes_standard_fields() -> None:
    formatter = JSONFormatter()
    record = _make_record("hello world")
    payload = json.loads(formatter.format(record))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "requisite.example"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload


def test_json_formatter_merges_extra_fields() -> None:
    formatter = JSONFormatter()
    record = _make_record("resolved", extra={"capability": "weather", "priority": 5})
    payload = json.loads(formatter.format(record))
    assert payload["capability"] == "weather"
    assert payload["priority"] == 5


def test_json_formatter_includes_exception_info() -> None:
    formatter = JSONFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _make_record("failed", exc_info=sys.exc_info())
    payload = json.loads(formatter.format(record))
    assert "boom" in payload["exception"]


def test_json_formatter_output_is_single_line_valid_json() -> None:
    formatter = JSONFormatter()
    record = _make_record("multi\nline\nmessage")
    line = formatter.format(record)
    assert "\n" not in line
    json.loads(line)  # does not raise


def test_configure_logging_attaches_json_formatter(capsys) -> None:
    logger = configure_logging(level="DEBUG", json_format=True, logger_name="requisite.test_config")
    logger.debug("test message", extra={"foo": "bar"})
    captured = capsys.readouterr()
    payload = json.loads(captured.err.strip())
    assert payload["message"] == "test message"
    assert payload["foo"] == "bar"

    # cleanup so this test doesn't leak a handler into other tests
    logger.handlers.clear()


def test_configure_logging_attaches_plain_formatter(capsys) -> None:
    logger = configure_logging(
        level="INFO", json_format=False, logger_name="requisite.test_config_plain"
    )
    logger.info("plain message")
    captured = capsys.readouterr()
    assert "plain message" in captured.err
    try:
        json.loads(captured.err.strip())
        raise AssertionError("plain formatter output should not be valid JSON")
    except json.JSONDecodeError:
        pass

    logger.handlers.clear()


def test_configure_logging_respects_level(capsys) -> None:
    logger = configure_logging(level="WARNING", logger_name="requisite.test_config_level")
    logger.info("should not appear")
    logger.warning("should appear")
    captured = capsys.readouterr()
    assert "should not appear" not in captured.err
    assert "should appear" in captured.err

    logger.handlers.clear()
