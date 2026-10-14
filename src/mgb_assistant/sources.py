"""The source index: page-cited passages from documents, tagged by tier and system.

The index is a JSONL file, one passage per line. It's built from manuals/, so it holds
manual text and is gitignored like them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

INDEX_TIERS = (2, 3, 4)  # tier 1 is the profile and the log, which aren't indexed documents


@dataclass(frozen=True)
class Passage:
    id: str
    source_id: str  # e.g. "factory-manual"
    source: str  # the name a citation shows, e.g. "workshop manual"
    tier: int
    page: str | None  # the page number printed in the document
    system: str
    text: str
    pdf_page: int | None = None  # where it sits in the PDF, for tracing a bad page mapping


def _problem(p: Passage) -> str | None:
    if p.tier not in INDEX_TIERS:
        return f"tier {p.tier}: the index holds tiers 2-4; tier 1 is the profile and the log"
    if p.tier in (2, 3) and not p.page:
        return f"tier {p.tier} passages need the printed page"
    if not p.text.strip():
        return "the passage has no text"
    return None


def load_index(path: Path) -> list[Passage]:
    passages: list[Passage] = []
    seen: set[str] = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            passage = Passage(**json.loads(line))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"{path}:{n}: not a passage ({exc})") from exc
        problem = _problem(passage)
        if problem:
            raise ValueError(f"{path}:{n}: {passage.id}: {problem}")
        if passage.id in seen:
            raise ValueError(f"{path}:{n}: passage id {passage.id} appears twice")
        seen.add(passage.id)
        passages.append(passage)
    return passages


def write_index(passages: list[Passage], path: Path) -> None:
    for passage in passages:
        problem = _problem(passage)
        if problem:
            raise ValueError(f"{passage.id}: {problem}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(p), ensure_ascii=False) for p in passages]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
