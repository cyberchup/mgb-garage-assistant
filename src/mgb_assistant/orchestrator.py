"""The orchestrator: sends each question to the part that should answer it.

"When did I last..." questions are answered from the maintenance log alone (no model call).
Everything else goes to the composer. Log commands ("Log it: ...") arrive with voice logging.

Command line: python -m mgb_assistant "When did I last set the points gap?" [--llm ollama]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from mgb_assistant.answer import Answer, Assistant
from mgb_assistant.composer import Composer, format_answer, make_llm
from mgb_assistant.llm import LLMError
from mgb_assistant.logbook import LogBook, load_logbook
from mgb_assistant.paths import DEFAULT_INDEX, DEFAULT_LOG, DEFAULT_PROFILE, find_root
from mgb_assistant.retriever import load_retriever

_HISTORY = re.compile(
    r"\b(when did (i|we)\b|when was\b.*\blast\b|last time (i|we)\b|how long since (i|we)\b|(have|did) (i|we) ever\b)",
    re.IGNORECASE,
)


def is_history_question(question: str) -> bool:
    return bool(_HISTORY.search(question))


class Orchestrator:
    """An Assistant (see answer.py) that routes between the log and the composer."""

    def __init__(self, composer: Assistant, logbook: LogBook):
        self.composer = composer
        self.logbook = logbook
        self.model_id = composer.model_id

    def answer(self, question: str, *, use_profile: bool) -> Answer:
        # The ablation switches off the profile only; the log is the car's record either way.
        if is_history_question(question):
            return self.logbook.history_answer(question)
        return self.composer.answer(question, use_profile=use_profile)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mgb_assistant",
        description="Ask about the car: history questions go to the maintenance log, the rest to the composer.",
    )
    parser.add_argument("question")
    parser.add_argument("--llm", choices=("claude", "ollama"), default=os.environ.get("MGB_LLM", "claude"))
    parser.add_argument("--model", help="model name for the chosen LLM")
    parser.add_argument("--no-profile", action="store_true", help="switch the profile off, as in the eval ablation")
    parser.add_argument("--index", type=Path, help="passage index (default: index/passages.jsonl in the repo)")
    parser.add_argument("--profile", type=Path, help="profile YAML (default: profile/mgb-1973.yaml in the repo)")
    parser.add_argument("--log", type=Path, help="maintenance log (default: log/maintenance.yaml in the repo)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        root = find_root(Path.cwd()) if not (args.index and args.profile and args.log) else None
        profile_path = args.profile or root / DEFAULT_PROFILE
        logbook = load_logbook(args.log or root / DEFAULT_LOG, profile_path)
        if is_history_question(args.question):
            answer = logbook.history_answer(args.question)  # no index or model needed
        else:
            retriever = load_retriever(args.index or root / DEFAULT_INDEX, profile_path)
            orchestrator = Orchestrator(Composer(retriever, make_llm(args.llm, args.model)), logbook)
            answer = orchestrator.answer(args.question, use_profile=not args.no_profile)
    except (OSError, ValueError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_answer(answer))
    return 0


if __name__ == "__main__":
    sys.exit(main())
