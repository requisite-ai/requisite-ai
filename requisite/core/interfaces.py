"""
Core data models and interfaces.

These types are provider-agnostic. Every provider implementation
translates its own SDK's request/response shapes into these models so
that the rest of the framework -- and end-user code -- never has to
deal with provider-specific formats.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class Role(str, Enum):
    """Conversation participant role, mirrored across all providers."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A single tool invocation requested by the model.

    Parameters
    ----------
    id:
        Provider-assigned identifier for this call. Used to correlate a
        tool's result with the call that requested it when reporting
        results back to the model.
    name:
        Name of the tool the model wants to invoke.
    arguments:
        Parsed keyword arguments for the call. Providers report
        arguments as a JSON string on the wire; this is already decoded
        into a dict.
    """

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A single message in a conversation.

    Parameters
    ----------
    role:
        Who sent the message (system, user, assistant, or tool).
    content:
        The text content of the message.
    name:
        Optional identifier, primarily used for tool messages to
        indicate which tool produced the content.
    tool_call_id:
        For a ``TOOL``-role message, the id of the :class:`ToolCall` this
        message is reporting the result of.
    tool_calls:
        For an ``ASSISTANT``-role message that requested tool calls,
        the calls it requested. Empty otherwise.

    Examples
    --------
    >>> Message(role=Role.USER, content="Hello!")
    Message(role=<Role.USER: 'user'>, content='Hello!', name=None, tool_call_id=None, tool_calls=[])
    """

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Convenience constructor for a user message."""
        return cls(role=Role.USER, content=content)

    @classmethod
    def system(cls, content: str) -> "Message":
        """Convenience constructor for a system message."""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Convenience constructor for an assistant message."""
        return cls(role=Role.ASSISTANT, content=content)

    @classmethod
    def assistant_tool_calls(cls, tool_calls: list[ToolCall], *, content: str = "") -> "Message":
        """Convenience constructor for an assistant message requesting tool calls."""
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool_result(
        cls, content: str, *, tool_call_id: str, name: Optional[str] = None
    ) -> "Message":
        """Convenience constructor for a message reporting a tool's result back to the model."""
        return cls(role=Role.TOOL, content=content, name=name, tool_call_id=tool_call_id)


class Usage(BaseModel):
    """Token usage reported by a provider, when available.

    All fields default to ``0`` because not every provider reports
    every figure; missing data should never raise an error.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """Normalized response returned by every provider's ``chat`` call.

    Parameters
    ----------
    content:
        The generated text. Empty when the model's entire response is a
        tool call with no accompanying text.
    model:
        The concrete model identifier that produced the response
        (e.g. ``"gpt-4o"``, ``"gemini-2.0-flash"``).
    provider:
        Name of the provider that produced the response (e.g. ``"openai"``).
    usage:
        Token usage information, when the provider reports it.
    raw:
        The unprocessed, provider-native response object, kept for
        advanced use cases. Treat this as an escape hatch, not part of
        the stable public API.
    finish_reason:
        Provider-reported reason generation stopped (e.g. ``"stop"``,
        ``"length"``), normalized to a string when possible.
    tool_calls:
        Tools the model wants to invoke, if any. Empty when the model
        did not request any tool calls.
    parsed:
        The validated Pydantic model instance produced when a
        ``response_model`` was requested for structured output. ``None``
        otherwise.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: str
    model: str
    provider: str
    usage: Usage = Field(default_factory=Usage)
    raw: Any = None
    finish_reason: Optional[str] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    parsed: Optional[Any] = None

    @property
    def has_tool_calls(self) -> bool:
        """Whether the model requested one or more tool calls."""
        return len(self.tool_calls) > 0

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.content


class StreamChunk(BaseModel):
    """A single chunk yielded while streaming a response.

    Parameters
    ----------
    delta:
        The incremental piece of text produced since the last chunk.
    is_final:
        Whether this chunk marks the end of the stream.
    raw:
        The unprocessed, provider-native chunk object.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    delta: str
    is_final: bool = False
    raw: Any = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.delta
