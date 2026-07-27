"""
Text chunking.

One deliberately simple strategy for v1: fixed-size character windows
with overlap, breaking on whitespace near the boundary where possible so
words aren't split mid-token. No dependency on a tokenizer library --
``chunk_size``/``chunk_overlap`` are in characters, not model tokens,
which is an approximation but a dependency-free one. See
``docs/adr/0005-rag-integration.md`` for why this was chosen over a
token-aware chunker for the first version.
"""

from __future__ import annotations

from requisite.core.exceptions import ConfigurationException

_WHITESPACE = " \t\n\r"


def chunk_text(text: str, *, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Split ``text`` into overlapping chunks of roughly ``chunk_size`` characters.

    Parameters
    ----------
    text:
        The text to split.
    chunk_size:
        Target maximum characters per chunk.
    chunk_overlap:
        How many characters of overlap between consecutive chunks, so
        context near a chunk boundary isn't lost entirely to one side.
        Must be smaller than ``chunk_size``.

    Returns
    -------
    list[str]
        The chunks, in order. Empty input returns an empty list.

    Examples
    --------
    >>> chunks = chunk_text("a " * 600, chunk_size=1000, chunk_overlap=100)
    >>> len(chunks) > 1
    True
    """
    if chunk_size < 1:
        raise ConfigurationException("chunk_text requires chunk_size >= 1.")
    if chunk_overlap < 0:
        raise ConfigurationException("chunk_text requires chunk_overlap >= 0.")
    if chunk_overlap >= chunk_size:
        raise ConfigurationException("chunk_text requires chunk_overlap < chunk_size.")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    text_length = len(text)
    step = chunk_size - chunk_overlap

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Prefer breaking on whitespace near the boundary, so words aren't
        # split mid-token -- search backward from `end` for the nearest
        # whitespace, but don't shrink the chunk by more than ~20% doing so.
        if end < text_length:
            search_floor = max(start, end - int(chunk_size * 0.2))
            break_at = -1
            for i in range(end, search_floor, -1):
                if text[i - 1] in _WHITESPACE:
                    break_at = i
                    break
            if break_at != -1:
                end = break_at

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break
        start += step if step > 0 else chunk_size

    return chunks
