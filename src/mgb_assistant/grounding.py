"""Grounding rules shared by the composer, which enforces them, and the eval checks, which grade them.

CLAUDE.md rule 5: never output a number without a source. A number counts as sourced when
its sentence carries a [n] marker that points at a real citation.
"""

from __future__ import annotations

import re
from collections.abc import Collection

from mgb_assistant.answer import Answer

MARKER = re.compile(r"\[(\d+)\]")
# A standalone number, kept whole across separators: 0.015, 1-3-4-2, 32/36. Units may follow
# ("0.015in", "12V"), but a letter or digit may not come before it, so "HS4" is not a number.
NUMBER = re.compile(r"(?<![A-Za-z0-9.])\d+(?:[-–./:]\d+)*")
LIST_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _pieces(line: str) -> list[str]:
    """A line's sentences. A piece that is only markers ("... 0.015 in. [1]") joins its sentence."""
    pieces: list[str] = []
    for piece in SENTENCE_END.split(line):
        if not piece.strip():
            continue
        if pieces and not MARKER.sub("", piece).strip():
            pieces[-1] += " " + piece
        else:
            pieces.append(piece)
    return pieces


def sentences(text: str) -> list[str]:
    """Every sentence in the text, with list numbering ("1. ") stripped so it isn't read as a number."""
    return [piece for line in text.splitlines() for piece in _pieces(LIST_PREFIX.sub("", line))]


def is_sourced(sentence: str, valid: Collection[int]) -> bool:
    """True if the sentence has no numbers, or carries a marker pointing at a valid citation."""
    return not NUMBER.findall(MARKER.sub("", sentence)) or any(int(m) in valid for m in MARKER.findall(sentence))


def _split_numbers(full: str, valid: Collection[int]) -> tuple[set[str], list[str]]:
    """The full answer's cited numbers, and its uncited ones."""
    cited: set[str] = set()
    uncited: list[str] = []
    for sentence in sentences(full):
        numbers = NUMBER.findall(MARKER.sub("", sentence))
        if is_sourced(sentence, valid):
            cited.update(numbers)
        else:
            uncited.extend(numbers)
    return cited, uncited


def uncited_numbers(answer: Answer) -> list[str]:
    """Numbers stated without a source.

    A number in the full answer counts as cited when its sentence carries a valid marker.
    A number in the spoken answer counts as cited when the full answer cites it.
    """
    cited, uncited = _split_numbers(answer.full, range(1, len(answer.citations) + 1))
    uncited.extend(n for n in NUMBER.findall(answer.spoken) if n not in cited and n not in uncited)
    return uncited


def spoken_is_sourced(full: str, spoken: str, citation_count: int) -> bool:
    """True if every number in the spoken form is one the full answer cites."""
    cited, _ = _split_numbers(full, range(1, citation_count + 1))
    return all(n in cited for n in NUMBER.findall(spoken))


def drop_unsourced(text: str, valid: Collection[int]) -> tuple[str, int]:
    """The text without sentences that state a number with no valid marker, and how many went."""
    kept_lines: list[str] = []
    dropped = 0
    for line in text.splitlines():
        prefix_match = LIST_PREFIX.match(line)
        prefix = prefix_match.group(0) if prefix_match else ""
        pieces = _pieces(line[len(prefix):])
        kept = [p for p in pieces if is_sourced(p, valid)]
        dropped += len(pieces) - len(kept)
        if kept:
            kept_lines.append(prefix + " ".join(kept))
        elif not pieces:
            kept_lines.append(line)  # blank lines keep the layout
    return "\n".join(kept_lines).strip(), dropped
