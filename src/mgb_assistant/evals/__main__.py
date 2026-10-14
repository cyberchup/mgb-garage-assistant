"""Command line: python -m mgb_assistant.evals [EVAL_FILE] [options]"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from mgb_assistant.answer import Assistant
from mgb_assistant.composer import Composer, make_llm
from mgb_assistant.evals.baselines import NullAssistant, ProfileOnlyAssistant
from mgb_assistant.evals.cases import EvalSet, load_eval_set
from mgb_assistant.evals.runner import Run, run_evals, write_run
from mgb_assistant.paths import DEFAULT_INDEX
from mgb_assistant.profile import load_profile
from mgb_assistant.resolver import load_resolver
from mgb_assistant.retriever import load_retriever


def _composer(llm_name: str) -> Callable[[EvalSet], Assistant]:
    def build(eval_set: EvalSet) -> Assistant:
        return Composer(load_retriever(eval_set.root / DEFAULT_INDEX, eval_set.car_profile), make_llm(llm_name))

    return build


ASSISTANTS: dict[str, Callable[[EvalSet], Assistant]] = {
    "null": lambda eval_set: NullAssistant(),
    "profile-only": lambda eval_set: ProfileOnlyAssistant(load_resolver(eval_set.car_profile)),
    # The real pipeline: resolver, retriever and an LLM. These call a model and, for Claude, cost money.
    "claude": _composer("claude"),
    "ollama": _composer("ollama"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mgb_assistant.evals",
        description="Run the eval set against an assistant and record the run.",
    )
    parser.add_argument("eval_file", nargs="?", default="evals/v1.yaml", type=Path)
    parser.add_argument("--assistant", choices=sorted(ASSISTANTS), default="null")
    parser.add_argument(
        "--profile",
        choices=("on", "off", "both"),
        default="both",
        help="both (the default) runs the profile ablation",
    )
    parser.add_argument("--case", action="append", dest="cases", metavar="ID", help="run only this case; repeatable")
    parser.add_argument("--out", type=Path, help="where to write the run record (default: runs/ at the repo root)")
    args = parser.parse_args(argv)

    try:
        eval_set = load_eval_set(args.eval_file)
        profile = load_profile(eval_set.car_profile)
        assistant = ASSISTANTS[args.assistant](eval_set)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    unknown = set(args.cases or ()) - {c.id for c in eval_set.cases}
    if unknown:
        print(f"error: no case with id {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    run = run_evals(
        eval_set,
        profile,
        assistant,
        args.assistant,
        profile_settings=("on", "off") if args.profile == "both" else (args.profile,),
        case_ids=set(args.cases) if args.cases else None,
    )
    path = write_run(run, args.out or eval_set.root / "runs")
    print(format_summary(run))
    print(f"\nRun written to {path}")
    return 0


def format_summary(run: Run) -> str:
    record = run.to_record()
    git = record["git"]
    git_note = f"git {git['commit'][:7]}{', uncommitted changes' if git['dirty'] else ''}" if git["commit"] else "not in git"
    lines = [
        f"Eval set v{record['eval_version']} (sha256 {record['eval_sha256'][:8]}) | "
        f"assistant {record['assistant']}, model {record['model']} | {git_note}",
        "",
    ]

    headers = [f"profile {p.profile}" for p in run.passes]
    lines.append(f"{'case':<6}{'category':<18}" + "".join(f"{h:<14}" for h in headers))
    for i, first in enumerate(run.passes[0].results):
        statuses = [p.results[i].status for p in run.passes]
        lines.append(f"{first.case_id:<6}{first.category:<18}" + "".join(f"{s:<14}" for s in statuses))

    lines.append("")
    lines.append(f"{'':<24}" + "".join(f"{h:<14}" for h in headers))
    rows = [("auto-pass rate", [p.auto_pass_rate() for p in run.passes])]
    metrics = [p.metrics() for p in run.passes]
    rows += [(name, [m.get(name) for m in metrics]) for name in run.eval_set.metrics if name != "profile_ablation_gap"]
    for name, values in rows:
        lines.append(f"{name:<24}" + "".join(f"{_fmt(v):<14}" for v in values))
    gap = run.ablation_gap()
    if gap is not None:
        lines.append(f"{'profile_ablation_gap':<24}{gap:+.2f}  (auto-pass rate, on minus off)")

    first = run.passes[0]
    problems = [
        f"  {r.case_id:<4}{c.name}: {c.detail}"
        for r in first.results
        for c in r.checks
        if c.status != "pass"
    ]
    problems += [f"  {r.case_id:<4}error: {r.note}" for r in first.results if r.status == "error"]
    if problems:
        lines += ["", f"Checks not passing, profile {first.profile}:", *problems]
    reviews = [f"  {r.case_id:<4}{r.review}" for r in first.results if r.status == "needs_review"]
    if reviews:
        lines += ["", f"Needs review, profile {first.profile} (machine checks passed):", *reviews]
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    sys.exit(main())
