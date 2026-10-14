"""The car's systems, and matching questions to them."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

_SYSTEM_ID = re.compile(r"^[a-z_]+(\.[a-z_]+)*$")
_WORD = re.compile(r"[a-z0-9]+")


def covers(scope: str, system: str) -> bool:
    """True if `system` is `scope` or sits under it: fuel.carburetor covers fuel.carburetor.float."""
    return system == scope or system.startswith(scope + ".")


def overlaps(a: str, b: str) -> bool:
    """True if either system sits under the other."""
    return covers(a, b) or covers(b, a)


def _raw_words(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("’", "'").replace("'s", ""))


def words(text: str) -> tuple[str, ...]:
    """Lowercase words with a trailing plural "s" dropped, so "carbs" and "carb" match."""
    return tuple(w[:-1] if len(w) > 2 and w.endswith("s") and not w.endswith("ss") else w for w in _raw_words(text))


@dataclass(frozen=True)
class System:
    id: str
    terms: tuple[str, ...]
    safety_critical: bool = False


@dataclass(frozen=True)
class Match:
    term: str  # the words as they appeared in the question
    systems: tuple[str, ...]


class Taxonomy:
    def __init__(self, systems: Iterable[System]):
        self.systems = {s.id: s for s in systems}
        self._terms: dict[tuple[str, ...], list[str]] = {}
        for system in self.systems.values():
            for term in system.terms:
                self._terms.setdefault(words(term), []).append(system.id)
        self._longest = max((len(t) for t in self._terms), default=0)

    def match(self, question: str) -> list[Match]:
        """Terms in the question, longest first at each position, left to right."""
        asked, normalized = _raw_words(question), words(question)
        matches: list[Match] = []
        i = 0
        while i < len(normalized):
            for length in range(min(self._longest, len(normalized) - i), 0, -1):
                systems = self._terms.get(normalized[i : i + length])
                if systems:
                    matches.append(Match(" ".join(asked[i : i + length]), tuple(systems)))
                    i += length
                    break
            else:
                i += 1
        return matches

    def is_safety_critical(self, system: str) -> bool:
        return any(s.safety_critical and covers(s.id, system) for s in self.systems.values())

    def unknown_ids(self, ids: Iterable[str]) -> list[str]:
        """The ids that aren't systems in this taxonomy."""
        return [i for i in ids if i not in self.systems]


def load_taxonomy(path: Path) -> Taxonomy:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    systems = []
    for system_id, entry in (data.get("systems") or {}).items():
        if not _SYSTEM_ID.match(system_id):
            raise ValueError(f"{path}: system id {system_id!r} should be lowercase words joined by dots")
        terms = tuple((entry or {}).get("terms") or ())
        if not terms:
            raise ValueError(f"{path}: system {system_id} has no terms")
        systems.append(System(system_id, terms, bool(entry.get("safety_critical", False))))
    return Taxonomy(systems)
