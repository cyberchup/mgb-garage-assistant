"""Loading and validating the eval file."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from mgb_assistant.evals.checks import CASE_CHECKS

MODALITIES = ("text", "voice")


class EvalFileError(ValueError):
    pass


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    modality: str
    expected_behavior: str
    pass_when: str
    question: str | None = None
    utterance: str | None = None
    audio: tuple[str, ...] = ()
    expected_answer: str | list[str] | None = None
    expected_source: dict | None = None
    checks: dict[str, object] = field(default_factory=dict)
    review: str | None = None  # the part of pass_when a person has to judge


@dataclass(frozen=True)
class EvalSet:
    path: Path
    root: Path  # paths inside the eval file are relative to this
    version: int
    sha256: str
    car_profile: Path
    metrics: tuple[str, ...]
    cases: tuple[Case, ...]


def find_root(start: Path) -> Path:
    """The repository root: the nearest folder holding pyproject.toml."""
    for folder in [start, *start.parents]:
        if (folder / "pyproject.toml").is_file():
            return folder
    raise EvalFileError(f"no pyproject.toml above {start}; can't resolve paths in the eval file")


def load_eval_set(path: Path) -> EvalSet:
    path = path.resolve()
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise EvalFileError(f"{path}: expected a mapping at the top level")
    for key in ("version", "car_profile", "cases"):
        if key not in data:
            raise EvalFileError(f"{path}: missing {key!r}")

    cases = tuple(_load_case(path, entry) for entry in data["cases"])
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise EvalFileError(f"{path}: case id {case.id} appears twice")
        seen.add(case.id)

    root = find_root(path.parent)
    return EvalSet(
        path=path,
        root=root,
        version=data["version"],
        # Hash with normalized line endings so a checkout with CRLF gets the same hash.
        sha256=hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
        car_profile=root / data["car_profile"],
        metrics=tuple(data.get("metrics") or ()),
        cases=cases,
    )


def _load_case(path: Path, entry: dict) -> Case:
    case_id = entry.get("id")
    where = f"{path}: case {case_id}"
    for key in ("id", "category", "modality", "expected_behavior", "pass_when"):
        if not entry.get(key):
            raise EvalFileError(f"{where}: missing {key!r}")
    modality = entry["modality"]
    if modality not in MODALITIES:
        raise EvalFileError(f"{where}: modality {modality!r} is not one of {', '.join(MODALITIES)}")
    if modality == "text" and not entry.get("question"):
        raise EvalFileError(f"{where}: text cases need a question")
    if modality == "voice" and not entry.get("utterance"):
        raise EvalFileError(f"{where}: voice cases need an utterance")

    checks = dict(entry.get("checks") or {})
    review = checks.pop("review", None)
    if review is not None and not isinstance(review, str):
        raise EvalFileError(f"{where}: review must be text")
    for name, value in checks.items():
        if name not in CASE_CHECKS:
            known = ", ".join(sorted([*CASE_CHECKS, "review"]))
            raise EvalFileError(f"{where}: unknown check {name!r} (known: {known})")
        _, expected_type, item_type = CASE_CHECKS[name]
        if not isinstance(value, expected_type):
            raise EvalFileError(f"{where}: check {name} takes a {expected_type.__name__}")
        if item_type and not all(type(item) is item_type for item in value):
            raise EvalFileError(f"{where}: check {name} takes a list of {item_type.__name__}")

    return Case(
        id=case_id,
        category=entry["category"],
        modality=modality,
        expected_behavior=" ".join(entry["expected_behavior"].split()),
        pass_when=entry["pass_when"],
        question=entry.get("question"),
        utterance=entry.get("utterance"),
        audio=tuple(entry.get("audio") or ()),
        expected_answer=entry.get("expected_answer"),
        expected_source=entry.get("expected_source"),
        checks=checks,
        review=review,
    )
