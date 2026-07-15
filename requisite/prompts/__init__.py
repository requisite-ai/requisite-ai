"""
Prompt templates: reusable, parameterized prompts.

``PromptTemplate`` for a single string; ``ChatPromptTemplate`` for a
full role-tagged conversation, rendering directly to ``list[Message]``.
``PromptTemplateRegistry`` for naming and reusing templates across an
application.
"""

from requisite.prompts.registry import PromptTemplateRegistry
from requisite.prompts.registry import default_registry as default_prompt_registry
from requisite.prompts.template import ChatPromptTemplate, PromptTemplate

__all__ = [
    "PromptTemplate",
    "ChatPromptTemplate",
    "PromptTemplateRegistry",
    "default_prompt_registry",
]
