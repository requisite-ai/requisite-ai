"""
Prompt templates: reusable, parameterized prompts.

Two shapes, matching the two things applications actually template:

- :class:`PromptTemplate` -- a single string with ``{named}`` placeholders,
  for a plain-text prompt (e.g. what you'd pass to ``ai.chat(...)``).
- :class:`ChatPromptTemplate` -- an ordered sequence of role/template pairs,
  for a full conversation (e.g. a system prompt + a user turn), rendering
  directly to a ``list[Message]`` you can pass to ``ai.chat(...)`` or
  ``agent.run(...)``.

Deliberately dependency-free: variable substitution is Python's own
``str.format`` (named fields only -- positional/anonymous ``{}`` fields
are not treated as template variables). No new templating language to
learn, no Jinja2 dependency for what is, in practice, almost always a
flat set of named substitutions.
"""

from __future__ import annotations

import string
from typing import Any, Union

from pydantic import BaseModel, ConfigDict, Field

from requisite.core.exceptions import PromptException
from requisite.core.interfaces import Message, Role


class _PartialSafeDict(dict[str, Any]):
    """A dict that echoes back ``{key}`` for any key it doesn't have.

    Used to implement :meth:`PromptTemplate.partial` -- substituting some
    variables now while leaving any others' placeholders intact in the
    resulting template string, ready to be filled in later.
    """

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _extract_variables(template: str) -> list[str]:
    """Return the named ``{variable}`` fields in ``template``, in order of
    first appearance. Positional (``{}``) and numeric (``{0}``) fields are
    skipped -- named variables only are treated as template inputs.

    For a dotted/indexed field (``{cfg.api_key}``, ``{items[0]}``), the
    *root* name (``cfg``, ``items``) is what :meth:`PromptTemplate.format`
    actually needs supplied as a kwarg -- ``str.format`` resolves the
    attribute/index access against whatever that kwarg is. Capturing only
    the root (not the full dotted expression, and not skipping it
    entirely) is what makes :meth:`PromptTemplate.format`'s missing-
    variable check able to catch a genuinely missing ``cfg`` before ever
    calling ``str.format`` -- previously such a field was silently
    excluded from ``input_variables`` altogether, so a missing root
    surfaced as a raw ``KeyError``/``AttributeError`` instead of
    :class:`~requisite.core.exceptions.PromptException`. See
    ``docs/adr/0031-code-review-fixes.md``.
    """
    seen: list[str] = []
    for _literal, field_name, _format_spec, _conversion in string.Formatter().parse(template):
        if not field_name:
            continue
        root = field_name.split(".")[0].split("[")[0]
        if root.isidentifier() and root not in seen:
            seen.append(root)
    return seen


class PromptTemplate(BaseModel):
    """A single prompt string with named ``{variable}`` placeholders.

    Parameters
    ----------
    template:
        The template string, e.g. ``"Translate to {language}: {text}"``.
    input_variables:
        The variable names this template requires. Usually left to be
        auto-derived by :meth:`from_template` rather than set directly.

    Examples
    --------
    >>> template = PromptTemplate.from_template("Translate to {language}: {text}")
    >>> template.input_variables
    ['language', 'text']
    >>> template.format(language="French", text="Hello")
    'Translate to French: Hello'

    Pre-filling some variables now, leaving the rest for later:

    >>> french_translator = template.partial(language="French")
    >>> french_translator.format(text="Good morning")
    'Translate to French: Good morning'
    """

    model_config = ConfigDict(frozen=True)

    template: str
    input_variables: list[str] = Field(default_factory=list)

    @classmethod
    def from_template(cls, template: str) -> "PromptTemplate":
        """Build a :class:`PromptTemplate`, auto-deriving ``input_variables``
        from the ``{named}`` placeholders found in ``template``.
        """
        return cls(template=template, input_variables=_extract_variables(template))

    def format(self, **kwargs: Any) -> str:
        """Render the template, substituting each variable.

        Raises
        ------
        requisite.core.exceptions.PromptException
            If any variable in :attr:`input_variables` is missing from
            ``kwargs``.
        """
        missing = [v for v in self.input_variables if v not in kwargs]
        if missing:
            raise PromptException(
                f"Missing template variable(s): {missing}. Required: {self.input_variables}",
            )
        return self.template.format(**kwargs)

    def partial(self, **kwargs: Any) -> "PromptTemplate":
        """Return a new :class:`PromptTemplate` with some variables pre-filled.

        The given variables are substituted into the template text now;
        any remaining ``input_variables`` are left as placeholders in the
        returned template, to be filled in by a later :meth:`format` call.

        A string value containing literal ``{``/``}`` characters is
        escaped (doubled, ``str.format``'s own literal-brace convention)
        before substitution, so it can't introduce a *new* placeholder
        into the returned template's text for a later :meth:`format` call
        to unexpectedly fill -- e.g. ``template.partial(name="{secret}")``
        must not make a later ``.format(secret=...)`` interpolate that
        value into ``name``'s position too. Non-string values are passed
        through unescaped, so a value with its own ``__format__`` (e.g. a
        number used with a ``{x:.2f}``-style spec still in the template)
        keeps working exactly as before. See
        ``docs/adr/0031-code-review-fixes.md``.
        """
        escaped_kwargs = {
            key: value.replace("{", "{{").replace("}", "}}") if isinstance(value, str) else value
            for key, value in kwargs.items()
        }
        rendered = self.template.format_map(_PartialSafeDict(escaped_kwargs))
        remaining = [v for v in self.input_variables if v not in kwargs]
        return PromptTemplate(template=rendered, input_variables=remaining)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"PromptTemplate(input_variables={self.input_variables!r})"


class ChatPromptTemplate(BaseModel):
    """An ordered sequence of role/template pairs, rendering to a full
    ``list[Message]`` conversation.

    Parameters
    ----------
    messages:
        Ordered ``(role, PromptTemplate)`` pairs.

    Examples
    --------
    >>> chat_template = ChatPromptTemplate.from_messages([
    ...     ("system", "You are a {persona}."),
    ...     ("user", "{question}"),
    ... ])
    >>> chat_template.input_variables
    ['persona', 'question']
    >>> messages = chat_template.format_messages(persona="pirate", question="Where's the treasure?")
    >>> [m.content for m in messages]
    ['You are a pirate.', "Where's the treasure?"]

    The result is a plain ``list[Message]`` -- pass it straight to ``ai.chat(...)``
    or ``agent.run(...)``:

    >>> from requisite import AI
    >>> ai = AI()  # doctest: +SKIP
    >>> ai.chat(messages)  # doctest: +SKIP
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    messages: list[tuple[Role, PromptTemplate]]

    @classmethod
    def from_messages(cls, messages: list[tuple[Union[str, Role], str]]) -> "ChatPromptTemplate":
        """Build a :class:`ChatPromptTemplate` from ``(role, template_text)`` pairs.

        Parameters
        ----------
        messages:
            E.g. ``[("system", "You are a {persona}."), ("user", "{question}")]``.
            ``role`` accepts either a :class:`~requisite.core.interfaces.Role`
            or its string value (``"system"``, ``"user"``, ``"assistant"``).
        """
        built: list[tuple[Role, PromptTemplate]] = []
        for role, text in messages:
            resolved_role = role if isinstance(role, Role) else Role(role)
            built.append((resolved_role, PromptTemplate.from_template(text)))
        return cls(messages=built)

    @property
    def input_variables(self) -> list[str]:
        """The union of all variables required across every message template, in order."""
        seen: list[str] = []
        for _role, template in self.messages:
            for variable in template.input_variables:
                if variable not in seen:
                    seen.append(variable)
        return seen

    def format_messages(self, **kwargs: Any) -> list[Message]:
        """Render every message template, returning a ``list[Message]``.

        Raises
        ------
        requisite.core.exceptions.PromptException
            If any message's required variables are missing from ``kwargs``.
        """
        return [
            Message(role=role, content=template.format(**kwargs))
            for role, template in self.messages
        ]

    def partial(self, **kwargs: Any) -> "ChatPromptTemplate":
        """Return a new :class:`ChatPromptTemplate` with some variables pre-filled
        across every message template. See :meth:`PromptTemplate.partial`.
        """
        return ChatPromptTemplate(
            messages=[(role, template.partial(**kwargs)) for role, template in self.messages]
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"ChatPromptTemplate(roles={[r.value for r, _ in self.messages]!r})"
