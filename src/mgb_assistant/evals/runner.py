"""Running an eval set against an assistant, and recording the run."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from mgb_assistant.answer import Answer, Assistant
from mgb_assistant.evals.cases import Case, EvalSet
from mgb_assistant.evals.checks import CheckResult, grade, uncited_numbers
from mgb_assistant.profile import Profile

HARNESS_VERSION = "0.1.0"

# Cases that count toward the auto-pass rate. not_ready and skipped cases can't be scored yet.
GRADED = ("pass", "needs_review", "fail", "error")
AUTO_PASSED = ("pass", "needs_review")


@dataclass
class CaseResult:
    case_id: str
    category: str
    status: str  # pass | needs_review | fail | error | not_ready | skipped
    checks: list[CheckResult] = field(default_factory=list)
    review: str | None = None
    answer: Answer | None = None
    uncited_numbers: list[str] = field(default_factory=list)
    note: str | None = None  # why it was skipped, or the error


@dataclass
class ProfilePass:
    """Every case run once, with the profile on or off."""

    profile: str  # on | off
    results: list[CaseResult]

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts

    def auto_pass_rate(self) -> float | None:
        graded = [r.status in AUTO_PASSED for r in self.results if r.status in GRADED]
        return _rate(graded)

    def metrics(self) -> dict[str, float | None]:
        spec = [r for r in self.results if r.category == "spec_lookup"]
        traps = [r.status in AUTO_PASSED for r in self.results if r.category == "weber_trap" and r.status in GRADED]
        answered = [r for r in self.results if r.answer is not None]
        uncited = sum(len(r.uncited_numbers) for r in answered)
        return {
            "spec_accuracy": _check_rate(spec, "contains_expected"),
            # Until the source index exists, this only checks the cited page against expected_source.
            "citation_accuracy": _check_rate(spec, "cites_expected_source"),
            "override_accuracy": _rate(traps),
            "uncited_numbers_per_100": round(100 * uncited / len(answered), 1) if answered else None,
            # Voice metrics wait for the voice path (phase 3).
            "voice_term_accuracy": None,
            "voice_number_accuracy": None,
            "readback_catch_rate": None,
        }


@dataclass
class Run:
    eval_set: EvalSet
    assistant_name: str
    model_id: str
    date: datetime
    passes: list[ProfilePass]

    def ablation_gap(self) -> float | None:
        by_profile = {p.profile: p.auto_pass_rate() for p in self.passes}
        on, off = by_profile.get("on"), by_profile.get("off")
        if on is None or off is None:
            return None
        return round(on - off, 3)

    def to_record(self) -> dict:
        es = self.eval_set
        return {
            "harness_version": HARNESS_VERSION,
            "date": self.date.isoformat(timespec="seconds"),
            "eval_file": es.path.relative_to(es.root).as_posix(),
            "eval_version": es.version,
            "eval_sha256": es.sha256,
            "car_profile": es.car_profile.relative_to(es.root).as_posix(),
            "git": _git_state(es.root),
            "assistant": self.assistant_name,
            "model": self.model_id,
            "profile_ablation_gap": self.ablation_gap(),
            "passes": [_pass_record(p, es.metrics) for p in self.passes],
        }


def run_case(case: Case, assistant: Assistant, profile: Profile, *, use_profile: bool) -> CaseResult:
    if case.modality != "text":
        return CaseResult(case.id, case.category, "skipped", note="the voice path isn't built yet (phase 3)")
    try:
        answer = assistant.answer(case.question, use_profile=use_profile)
    except Exception as exc:  # one broken case shouldn't sink the whole run
        return CaseResult(case.id, case.category, "error", note=f"{type(exc).__name__}: {exc}")

    checks = grade(answer, case, profile)
    statuses = {c.status for c in checks}
    if "fail" in statuses:
        status = "fail"
    elif "not_ready" in statuses:
        status = "not_ready"
    elif case.review:
        status = "needs_review"
    else:
        status = "pass"
    return CaseResult(
        case.id,
        case.category,
        status,
        checks=checks,
        review=case.review,
        answer=answer,
        uncited_numbers=uncited_numbers(answer),
    )


def run_evals(
    eval_set: EvalSet,
    profile: Profile,
    assistant: Assistant,
    assistant_name: str,
    *,
    profile_settings: tuple[str, ...] = ("on", "off"),
    case_ids: set[str] | None = None,
) -> Run:
    """Run the cases once per profile setting. The profile is always used for grading;
    the setting only controls whether the assistant sees it."""
    cases = [c for c in eval_set.cases if case_ids is None or c.id in case_ids]
    passes = [
        ProfilePass(setting, [run_case(c, assistant, profile, use_profile=setting == "on") for c in cases])
        for setting in profile_settings
    ]
    return Run(eval_set, assistant_name, assistant.model_id, datetime.now(), passes)


def write_run(run: Run, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = "-".join(p.profile for p in run.passes)
    path = out_dir / f"{run.date:%Y%m%dT%H%M%S}-{run.assistant_name}-profile-{settings}.json"
    path.write_text(json.dumps(run.to_record(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _pass_record(p: ProfilePass, metric_names: tuple[str, ...]) -> dict:
    metrics = p.metrics()
    return {
        "profile": p.profile,
        "auto_pass_rate": p.auto_pass_rate(),
        "counts": p.counts(),
        # The eval file's metric list decides what's reported; the gap is reported once, per run.
        "metrics": {name: metrics.get(name) for name in metric_names if name != "profile_ablation_gap"},
        "results": [asdict(r) for r in p.results],
    }


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _check_rate(results: list[CaseResult], name: str) -> float | None:
    return _rate([c.status == "pass" for r in results for c in r.checks if c.name == name and c.status != "not_ready"])


def _git_state(root: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
