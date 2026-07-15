"""Unit tests for prompt templates."""

from __future__ import annotations

import pytest

from requisite.core.exceptions import PromptException
from requisite.core.interfaces import Role
from requisite.prompts.registry import PromptTemplateRegistry
from requisite.prompts.template import ChatPromptTemplate, PromptTemplate


def test_from_template_derives_input_variables() -> None:
    template = PromptTemplate.from_template("Translate to {language}: {text}")
    assert template.input_variables == ["language", "text"]


def test_from_template_ignores_positional_fields() -> None:
    template = PromptTemplate.from_template("Braces example: {} and {name}")
    assert template.input_variables == ["name"]


def test_format_substitutes_variables() -> None:
    template = PromptTemplate.from_template("Translate to {language}: {text}")
    assert template.format(language="French", text="Hello") == "Translate to French: Hello"


def test_format_raises_on_missing_variable() -> None:
    template = PromptTemplate.from_template("Translate to {language}: {text}")
    with pytest.raises(PromptException, match="text"):
        template.format(language="French")


def test_partial_prefills_and_leaves_remaining_variables() -> None:
    template = PromptTemplate.from_template("Translate to {language}: {text}")
    partially_filled = template.partial(language="French")
    assert partially_filled.input_variables == ["text"]
    assert partially_filled.format(text="Good morning") == "Translate to French: Good morning"


def test_partial_can_be_chained() -> None:
    template = PromptTemplate.from_template("{a}-{b}-{c}")
    result = template.partial(a="1").partial(b="2").format(c="3")
    assert result == "1-2-3"


def test_prompt_template_is_frozen() -> None:
    template = PromptTemplate.from_template("hi {name}")
    with pytest.raises(Exception):  # noqa: B017 - pydantic frozen model raises its own error type
        template.template = "changed"  # type: ignore[misc]


def test_chat_prompt_template_from_messages() -> None:
    chat_template = ChatPromptTemplate.from_messages(
        [("system", "You are a {persona}."), ("user", "{question}")]
    )
    assert chat_template.input_variables == ["persona", "question"]
    messages = chat_template.format_messages(persona="pirate", question="Where's the treasure?")
    assert [m.role for m in messages] == [Role.SYSTEM, Role.USER]
    assert messages[0].content == "You are a pirate."
    assert messages[1].content == "Where's the treasure?"


def test_chat_prompt_template_accepts_role_enum() -> None:
    chat_template = ChatPromptTemplate.from_messages([(Role.USER, "{question}")])
    messages = chat_template.format_messages(question="hi")
    assert messages[0].role == Role.USER


def test_chat_prompt_template_partial() -> None:
    chat_template = ChatPromptTemplate.from_messages(
        [("system", "You are a {persona}."), ("user", "{question}")]
    )
    partially_filled = chat_template.partial(persona="pirate")
    assert partially_filled.input_variables == ["question"]
    messages = partially_filled.format_messages(question="Ahoy?")
    assert messages[0].content == "You are a pirate."


def test_chat_prompt_template_missing_variable_raises() -> None:
    chat_template = ChatPromptTemplate.from_messages([("user", "{question}")])
    with pytest.raises(PromptException):
        chat_template.format_messages()


# ---------------------------------------------------------------------------
# PromptTemplateRegistry
# ---------------------------------------------------------------------------


def test_registry_register_and_get() -> None:
    registry = PromptTemplateRegistry()
    template = PromptTemplate.from_template("hi {name}")
    registry.register("greeting", template)
    assert registry.get("greeting") is template
    assert "greeting" in registry
    assert len(registry) == 1


def test_registry_get_missing_raises() -> None:
    registry = PromptTemplateRegistry()
    with pytest.raises(PromptException):
        registry.get("nope")


def test_registry_register_empty_name_raises() -> None:
    registry = PromptTemplateRegistry()
    with pytest.raises(PromptException):
        registry.register("", PromptTemplate.from_template("hi"))


def test_registry_unregister_is_idempotent() -> None:
    registry = PromptTemplateRegistry()
    registry.register("greeting", PromptTemplate.from_template("hi {name}"))
    registry.unregister("greeting")
    registry.unregister("greeting")  # no error
    assert "greeting" not in registry


def test_registry_can_hold_chat_prompt_templates_too() -> None:
    registry = PromptTemplateRegistry()
    chat_template = ChatPromptTemplate.from_messages([("user", "{question}")])
    registry.register("qa", chat_template)
    assert registry.get("qa") is chat_template
