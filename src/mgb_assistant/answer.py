"""The answer contract between the assistant and everything that consumes it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Citation:
    """One source behind an answer.

    tier: 1 = car record, 2 = conversion docs, 3 = workshop manual, 4 = community.
    source: "profile", "log", or the document's name.
    page: the page number printed in the document (a string, since labels like "D.4" exist).
    entry_id: the log entry or profile deviation id, for tier 1.
    system: the system the cited passage covers, e.g. "fuel.carburetor".
    """

    tier: int
    source: str
    page: str | None = None
    entry_id: str | None = None
    system: str | None = None


@dataclass(frozen=True)
class Answer:
    """Both forms of an answer, plus the structure the behavior rules depend on.

    full: the complete answer. Each sentence that states a fact carries [n] markers,
        placed before its closing punctuation, that point into `citations` (1-based).
    spoken: at most two sentences, for text-to-speech.
    conflict: the one-line statement of a tier conflict, when there was one.
    safety_flag: set on safety-critical jobs (brakes, fuel lines).
    abstained: set when no source covers the question.
    """

    full: str
    spoken: str
    citations: tuple[Citation, ...] = ()
    conflict: str | None = None
    safety_flag: bool = False
    abstained: bool = False


class Assistant(Protocol):
    """Anything the eval harness can question."""

    model_id: str  # recorded with every eval run

    def answer(self, question: str, *, use_profile: bool) -> Answer: ...
