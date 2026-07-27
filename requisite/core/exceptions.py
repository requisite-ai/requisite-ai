"""
Exception hierarchy for requisite.

Every exception raised by the framework inherits from :class:`AIException`,
which makes it easy for calling code to catch "anything from this
framework" with a single ``except AIException`` clause, while still
allowing fine-grained handling of specific failure modes.

Design notes
------------
* Exceptions are never swallowed internally -- provider SDK errors are
  always wrapped and re-raised with additional context, never discarded.
* Each exception carries an optional ``provider`` / ``details`` payload
  so calling code (and logs) can introspect *why* something failed
  without parsing strings.
"""

from __future__ import annotations

from typing import Any


class AIException(Exception):
    """Base class for all exceptions raised by requisite.

    Parameters
    ----------
    message:
        Human readable description of the failure.
    details:
        Optional structured context (e.g. the offending payload, a
        provider error code). Never put secrets (API keys, tokens) here.

    Examples
    --------
    >>> try:
    ...     raise AIException("something went wrong")
    ... except AIException as exc:
    ...     print(str(exc))
    something went wrong
    """

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.details:
            return f"{self.message} | details={self.details}"
        return self.message


class ConfigurationException(AIException):
    """Raised when configuration is missing, invalid, or inconsistent.

    Examples
    --------
    Raised when an API key required by the selected provider is not set,
    or when an unknown provider name is requested from the registry.
    """


class ProviderException(AIException):
    """Raised when a provider (OpenAI, Gemini, ...) fails to fulfil a request.

    Parameters
    ----------
    message:
        Human readable description of the failure.
    provider:
        Name of the provider that raised the error (e.g. ``"openai"``).
    original_error:
        The underlying exception raised by the provider SDK, preserved
        for debugging via ``raise ... from original_error`` semantics.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        original_error: BaseException | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.provider = provider
        self.original_error = original_error


class ToolException(AIException):
    """Raised when a tool registration or tool execution fails."""


class SkillException(AIException):
    """Raised when a skill registration or skill execution fails."""


class AgentException(AIException):
    """Raised when agent creation or agent execution fails."""


class MCPException(AIException):
    """Raised when an MCP client/server interaction fails."""


class CapabilityException(AIException):
    """Raised when a declared capability cannot be resolved to any available
    implementation, or when a capability name is registered incorrectly.

    Examples
    --------
    Raised by ``Agent.requires("weather")`` when no provider is registered
    for the ``"weather"`` capability, or when every registered provider
    for it reports itself unavailable (e.g. missing an API key).
    """


class PromptException(AIException):
    """Raised when a prompt template is misused: rendered without a
    required variable, or a named template isn't found in a registry.
    """
