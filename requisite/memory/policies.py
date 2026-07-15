"""
Conversation management policies.

"Memory" (``requisite.memory``) answers *where* conversation history is
stored. This module answers a different question: *how much* of that
history should actually be sent to the model on any given turn, once it
grows past what's useful (or affordable, or fits the context window).

A :class:`BaseConversationPolicy` takes the full message history and
returns a possibly-shorter one. It's applied by
:class:`~requisite.agents.agent.Agent` once, to the initial message list,
before the tool-calling loop starts -- never mid-loop, and never to what
gets persisted back to memory (each future load re-applies the policy
fresh over the then-current stored history). See
``docs/adr/0003-prompt-templates-structured-logging-conversation-policy.md``
for the reasoning.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from requisite.core.exceptions import ConfigurationException
from requisite.core.interfaces import Message, Role

if TYPE_CHECKING:
    from requisite.ai import AI

logger = logging.getLogger("requisite.memory.policies")


class BaseConversationPolicy(ABC):
    """Abstract interface for conversation retention policies."""

    @abstractmethod
    def apply(self, messages: list[Message]) -> list[Message]:
        """Return a possibly-shorter version of ``messages``.

        Implementations should be safe to call on a history that's
        already within policy (typically a no-op / passthrough in that
        case), since callers apply this unconditionally.
        """

    async def aapply(self, messages: list[Message]) -> list[Message]:
        """Async counterpart to :meth:`apply`.

        Default implementation runs :meth:`apply` in a worker thread so a
        synchronous policy (including one that calls a synchronous LLM
        method internally) remains usable from async code without
        blocking the event loop. :class:`SummarizingPolicy` overrides
        this with a true async implementation.
        """
        return await asyncio.to_thread(self.apply, messages)


class MessageCountPolicy(BaseConversationPolicy):
    """Keep only the most recent ``max_messages`` messages.

    Parameters
    ----------
    max_messages:
        Maximum number of messages to keep.
    keep_system:
        If ``True`` (default), system messages are always kept in
        addition to the most recent ``max_messages`` non-system messages
        -- trimming conversational back-and-forth without silently
        dropping the agent's own instructions.

    Examples
    --------
    >>> policy = MessageCountPolicy(max_messages=2)
    >>> history = [Message.user("1"), Message.assistant("2"), Message.user("3")]
    >>> [m.content for m in policy.apply(history)]
    ['2', '3']
    """

    def __init__(self, max_messages: int, *, keep_system: bool = True) -> None:
        if max_messages < 1:
            raise ConfigurationException("MessageCountPolicy requires max_messages >= 1.")
        self.max_messages = max_messages
        self.keep_system = keep_system

    def apply(self, messages: list[Message]) -> list[Message]:
        if len(messages) <= self.max_messages:
            return list(messages)

        if not self.keep_system:
            return messages[-self.max_messages :]

        system_messages = [m for m in messages if m.role == Role.SYSTEM]
        other_messages = [m for m in messages if m.role != Role.SYSTEM]
        keep_count = max(self.max_messages - len(system_messages), 0)
        return [*system_messages, *other_messages[-keep_count:]]


class SummarizingPolicy(BaseConversationPolicy):
    """When history exceeds ``max_messages``, collapse the oldest messages
    into a single LLM-generated summary, keeping the most recent
    ``keep_recent`` messages verbatim.

    Parameters
    ----------
    ai:
        The :class:`~requisite.ai.AI` instance used to generate the
        summary. A separate, cheaper/faster model than the agent's own is
        a reasonable choice here (e.g. ``AI(model="gpt-4o-mini")``), since
        summarization quality requirements are usually lower than the
        agent's main task.
    max_messages:
        History longer than this triggers summarization.
    keep_recent:
        How many of the most recent messages to keep verbatim,
        un-summarized. Must be less than ``max_messages``.
    summarization_prompt:
        Overrides the default summarization instruction. Must contain a
        ``{conversation}`` placeholder.

    Examples
    --------
    >>> from requisite import AI
    >>> policy = SummarizingPolicy(AI(), max_messages=20, keep_recent=4)  # doctest: +SKIP
    >>> agent = Agent(name="A", provider="openai", conversation_policy=policy)  # doctest: +SKIP
    """

    DEFAULT_PROMPT = (
        "Summarize the conversation below concisely, preserving names, "
        "facts, decisions, and any context a continuation of this "
        "conversation would need. Write the summary as plain prose, not "
        "a transcript.\n\n{conversation}"
    )

    def __init__(
        self,
        ai: "AI",
        *,
        max_messages: int,
        keep_recent: int = 4,
        summarization_prompt: Optional[str] = None,
    ) -> None:
        if keep_recent >= max_messages:
            raise ConfigurationException(
                f"SummarizingPolicy requires keep_recent ({keep_recent}) < "
                f"max_messages ({max_messages}).",
            )
        prompt = summarization_prompt or self.DEFAULT_PROMPT
        if "{conversation}" not in prompt:
            raise ConfigurationException(
                "summarization_prompt must contain a '{conversation}' placeholder.",
            )
        self.ai = ai
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self.summarization_prompt = prompt

    def _split(self, messages: list[Message]) -> Optional[tuple[list[Message], list[Message]]]:
        if len(messages) <= self.max_messages:
            return None
        split_index = len(messages) - self.keep_recent
        return messages[:split_index], messages[split_index:]

    def _render_conversation(self, messages: list[Message]) -> str:
        return "\n".join(f"{m.role.value}: {m.content}" for m in messages if m.content)

    def _summary_message(self, summary_text: str) -> Message:
        return Message.system(f"Summary of earlier conversation:\n{summary_text}")

    def apply(self, messages: list[Message]) -> list[Message]:
        split = self._split(messages)
        if split is None:
            return list(messages)
        to_summarize, recent = split

        conversation_text = self._render_conversation(to_summarize)
        summary_text = self.ai.chat(
            self.summarization_prompt.format(conversation=conversation_text)
        )
        logger.debug(
            "Summarized %d messages into one summary message",
            len(to_summarize),
            extra={"messages_summarized": len(to_summarize)},
        )
        return [self._summary_message(summary_text), *recent]

    async def aapply(self, messages: list[Message]) -> list[Message]:
        split = self._split(messages)
        if split is None:
            return list(messages)
        to_summarize, recent = split

        conversation_text = self._render_conversation(to_summarize)
        summary_text = await self.ai.achat(
            self.summarization_prompt.format(conversation=conversation_text)
        )
        logger.debug(
            "Summarized %d messages into one summary message",
            len(to_summarize),
            extra={"messages_summarized": len(to_summarize)},
        )
        return [self._summary_message(summary_text), *recent]
