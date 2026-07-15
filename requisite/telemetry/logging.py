"""
Structured logging.

Every module in the framework already logs through the standard library
(``logging.getLogger("requisite.<subpackage>")``) with plain string
messages -- see ``DEVELOPMENT.md``'s logging conventions. This module adds
an opt-in :class:`JSONFormatter` that renders those same log records as
single-line JSON, plus a :func:`configure_logging` convenience for
wiring it up.

Deliberately **not** called automatically anywhere in the framework:
importing ``requisite`` must never mutate global logging configuration
as a side effect. Applications call :func:`configure_logging` themselves
(typically once, at startup) if they want it -- consistent with the
framework never reading environment/global state implicitly outside of
:class:`~requisite.config.settings.Settings`.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import IO, Any, Optional

_STANDARD_LOG_RECORD_ATTRS = frozenset(vars(logging.makeLogRecord({})).keys())


class JSONFormatter(logging.Formatter):
    """Renders each log record as a single line of JSON.

    Always includes ``timestamp``, ``level``, ``logger``, and ``message``.
    Includes ``exception`` (a formatted traceback) when the record carries
    exception info. Any additional fields passed via the standard
    library's ``extra=`` keyword on a log call are merged in as-is, which
    is how this formatter supports structured, queryable fields rather
    than just a flat message string.

    Examples
    --------
    >>> import logging
    >>> logger = logging.getLogger("requisite.example")
    >>> logger.info("Resolved capability", extra={"capability": "weather", "provider_name": "open-meteo"})  # doctest: +SKIP
    # -> {"timestamp": "...", "level": "INFO", "logger": "requisite.example",
    #     "message": "Resolved capability", "capability": "weather", "provider_name": "open-meteo"}
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt or "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_RECORD_ATTRS and key not in payload:
                payload[key] = value

        return json.dumps(payload, default=str)


def configure_logging(
    *,
    level: str = "INFO",
    json_format: bool = False,
    stream: Optional[IO[str]] = None,
    logger_name: str = "requisite",
) -> logging.Logger:
    """Attach a single handler + formatter to the ``requisite`` logger tree.

    A convenience for applications that want reasonable logging output
    with one line of setup -- entirely optional, and safe to skip if your
    application configures logging its own way (every framework logger
    is a normal, undecorated ``logging.Logger`` that works fine with any
    external configuration too).

    Parameters
    ----------
    level:
        Standard library level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``, ``"CRITICAL"``). Typically sourced from
        ``Settings.log_level``.
    json_format:
        If ``True``, use :class:`JSONFormatter`. Otherwise, a plain
        human-readable line format. Typically sourced from
        ``Settings.log_format == "json"``.
    stream:
        Where to write log output. Defaults to ``sys.stderr``.
    logger_name:
        Which logger to attach the handler to. Defaults to the root of
        the framework's own logger tree (``"requisite"``), so it applies
        to every framework subsystem without touching the application's
        own loggers.

    Returns
    -------
    logging.Logger
        The configured logger, in case the caller wants to adjust it further.

    Examples
    --------
    >>> from requisite.telemetry import configure_logging
    >>> configure_logging(level="DEBUG", json_format=True)  # doctest: +SKIP
    """
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(
        JSONFormatter()
        if json_format
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger
