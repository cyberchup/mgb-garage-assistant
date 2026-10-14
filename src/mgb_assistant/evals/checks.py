"""Machine checks for graded answers.

Each check returns pass, fail or not_ready (the eval file is missing ground truth).
Anything that needs judgment stays out of here and goes in a case's `review` note.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mgb_assistant.answer import Answer
from mgb_assistant.profile import Profile

if TYPE_CHECKING:
    from mgb_assistant.evals.cases import Case


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # pass | fail | not_ready
    detail: str = ""


_MARKER = re.compile(r"\[(\d+)\]")
# A standalone number, kept whole across separators: 0.015, 1-3-4-2, 32/36. Units may follow
# ("0.015in", "12V"), but a letter or digit may not come before it, so "HS4" is not a number.
_NUMBER = re.compile(r"(?<![A-Za-z0-9.])\d+(?:[-–./:]\d+)*")
_LIST_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for line in text.splitlines():
        for piece in _SENTENCE_END.split(_LIST_PREFIX.sub("", line)):
            if not piece.strip():
                continue
            if sentences and not _MARKER.sub("", piece).strip():
                sentences[-1] += " " + piece  # "... 0.015 in. [1]": the marker belongs to its sentence
            else:
                sentences.append(piece)
    return sentences


def uncited_numbers(answer: Answer) -> list[str]:
    """Numbers stated without a source.

    A number in the full answer counts as cited when its sentence carries a valid marker.
    A number in the spoken answer counts as cited when the full answer cites it.
    """
    valid = set(range(1, len(answer.citations) + 1))
    cited: set[str] = set()
    uncited: list[str] = []
    for sentence in _sentences(answer.full):
        numbers = _NUMBER.findall(_MARKER.sub("", sentence))
        if valid & {int(m) for m in _MARKER.findall(sentence)}:
            cited.update(numbers)
        else:
            uncited.extend(numbers)
    uncited.extend(n for n in _NUMBER.findall(answer.spoken) if n not in cited and n not in uncited)
    return uncited


def _normalize(text: str) -> str:
    text = text.lower().replace("–", "-").replace("—", "-")
    text = re.sub(r"\s*-\s*", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def _result(name: str, problems: list[str], passed_detail: str = "") -> CheckResult:
    if problems:
        return CheckResult(name, "fail", "; ".join(problems))
    return CheckResult(name, "pass", passed_detail)


# --- Global checks: CLAUDE.md rule 5, applied to every text answer ---


def citations_valid(answer: Answer, case: Case, profile: Profile, param: object) -> CheckResult:
    used = {int(m) for m in _MARKER.findall(answer.full)}
    problems = [f"marker [{m}] has no citation" for m in sorted(used) if not 1 <= m <= len(answer.citations)]
    for i, c in enumerate(answer.citations, 1):
        if c.tier not in (1, 2, 3, 4):
            problems.append(f"[{i}] tier {c.tier} is not 1-4")
        if c.tier in (2, 3) and not c.page:
            problems.append(f"[{i}] tier {c.tier} citation has no page")
        if c.tier >= 2 and not c.system:
            problems.append(f"[{i}] has no system tag")
        if c.source == "log" and not c.entry_id:
            problems.append(f"[{i}] log citation has no entry id")
        if i not in used:
            problems.append(f"[{i}] is never referenced in the answer")
    return _result("citations_valid", problems)


def grounded(answer: Answer, case: Case, profile: Profile, param: object) -> CheckResult:
    if answer.abstained:
        return CheckResult("grounded", "pass", "abstained")
    if answer.citations and _MARKER.search(answer.full):
        return CheckResult("grounded", "pass")
    return CheckResult("grounded", "fail", "no citations and no abstention")


def no_uncited_numbers(answer: Answer, case: Case, profile: Profile, param: object) -> CheckResult:
    numbers = uncited_numbers(answer)
    return _result("no_uncited_numbers", [f"uncited: {', '.join(numbers)}"] if numbers else [])


# --- Per-case checks, switched on in the eval file ---


def contains_expected(answer: Answer, case: Case, profile: Profile, param: object) -> CheckResult:
    if case.expected_answer is None:
        return CheckResult("contains_expected", "not_ready", "expected_answer is not filled in")
    expected = [case.expected_answer] if isinstance(case.expected_answer, str) else case.expected_answer
    text = _normalize(_MARKER.sub("", answer.full))
    missing = [e for e in expected if _normalize(e) not in text]
    return _result("contains_expected", [f"missing {m!r}" for m in missing])


def cites_expected_source(answer: Answer, case: Case, profile: Profile, param: object) -> CheckResult:
    source = case.expected_source or {}
    if source.get("page") is None:
        return CheckResult("cites_expected_source", "not_ready", "expected_source.page is not filled in")
    tier, page = source.get("tier"), str(source["page"]).strip().lower()
    for c in answer.citations:
        if c.tier == tier and c.page is not None and c.page.strip().lower() == page:
            return CheckResult("cites_expected_source", "pass")
    return CheckResult("cites_expected_source", "fail", f"no tier {tier} citation of page {source['page']}")


def no_demoted_citations(answer: Answer, case: Case, profile: Profile, param: object) -> CheckResult:
    """No citation from a tier the profile demotes for that system (rules 1 and 2)."""
    problems = []
    for i, c in enumerate(answer.citations, 1):
        deviation = profile.overriding(c.tier, c.system) if c.system else None
        if deviation:
            problems.append(f"[{i}] tier {c.tier} on {c.system} is demoted by {deviation.id}")
    return _result("no_demoted_citations", problems)


def conflict_stated(answer: Answer, case: Case, profile: Profile, param: object) -> CheckResult:
    if not answer.conflict or not answer.conflict.strip():
        return CheckResult("conflict_stated", "fail", "no conflict stated")
    if "\n" in answer.conflict.strip():
        return CheckResult("conflict_stated", "fail", "conflict runs past one line")
    return CheckResult("conflict_stated", "pass")


def safety_flag(answer: Answer, case: Case, profile: Profile, param: object) -> CheckResult:
    return _result("safety_flag", [] if answer.safety_flag else ["safety flag not set"])


def not_tier4_alone(answer: Answer, case: Case, profile: Profile, param: object) -> CheckResult:
    tiers = {c.tier for c in answer.citations}
    if 4 in tiers and not tiers & {1, 2, 3}:
        return CheckResult("not_tier4_alone", "fail", "only community sources cited")
    return CheckResult("not_tier4_alone", "pass")


def log_only(answer: Answer, case: Case, profile: Profile, param: object) -> CheckResult:
    """Answered from the log, or "no record". Notes from the car's record may ride along; the manual may not."""
    outside = [f"[{i}] {c.source}" for i, c in enumerate(answer.citations, 1) if c.tier != 1]
    if outside:
        return CheckResult("log_only", "fail", f"cites sources outside the car's record: {', '.join(outside)}")
    if any(c.source == "log" for c in answer.citations):
        return CheckResult("log_only", "pass")
    if answer.abstained:
        return CheckResult("log_only", "pass", "no record")
    return CheckResult("log_only", "fail", "no log entry cited and no 'no record' answer")


def cites_tier(answer: Answer, case: Case, profile: Profile, param: object) -> CheckResult:
    """At least one citation from the listed tiers: the answer drew on the source that should win."""
    tiers = param if isinstance(param, list) else []
    if any(c.tier in tiers for c in answer.citations):
        return CheckResult("cites_tier", "pass")
    return CheckResult("cites_tier", "fail", f"no citation from tier {' or '.join(map(str, tiers))}")


def must_mention(answer: Answer, case: Case, profile: Profile, param: object) -> CheckResult:
    terms = param if isinstance(param, list) else []
    missing = [t for t in terms if not re.search(rf"(?<!\w){re.escape(t)}(?!\w)", answer.full, re.IGNORECASE)]
    return _result("must_mention", [f"missing {', '.join(missing)}"] if missing else [])


Check = Callable[[Answer, "Case", Profile, object], CheckResult]

GLOBAL_CHECKS: dict[str, Check] = {
    "citations_valid": citations_valid,
    "grounded": grounded,
    "no_uncited_numbers": no_uncited_numbers,
}

# name -> (check, type of its value in the eval file, type of the list items for list values)
CASE_CHECKS: dict[str, tuple[Check, type, type | None]] = {
    "contains_expected": (contains_expected, bool, None),
    "cites_expected_source": (cites_expected_source, bool, None),
    "no_demoted_citations": (no_demoted_citations, bool, None),
    "conflict_stated": (conflict_stated, bool, None),
    "safety_flag": (safety_flag, bool, None),
    "not_tier4_alone": (not_tier4_alone, bool, None),
    "log_only": (log_only, bool, None),
    "cites_tier": (cites_tier, list, int),
    "must_mention": (must_mention, list, str),
}


def grade(answer: Answer, case: Case, profile: Profile) -> list[CheckResult]:
    results = [check(answer, case, profile, None) for check in GLOBAL_CHECKS.values()]
    for name, value in case.checks.items():
        if value is False:
            continue
        check = CASE_CHECKS[name][0]
        results.append(check(answer, case, profile, value))
    return results
