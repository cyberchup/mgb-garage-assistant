from pathlib import Path

import pytest

from mgb_assistant.answer import Answer
from mgb_assistant.evals.cases import load_eval_set
from mgb_assistant.evals.runner import run_evals
from mgb_assistant.logbook import load_logbook
from mgb_assistant.orchestrator import Orchestrator, is_history_question, main
from mgb_assistant.profile import load_profile

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "profile" / "mgb-1973.yaml"
FIXTURE = REPO / "evals" / "fixtures" / "maintenance-log.yaml"


class FakeComposer:
    model_id = "fake"

    def __init__(self):
        self.calls: list[tuple[str, bool]] = []

    def answer(self, question, *, use_profile):
        self.calls.append((question, use_profile))
        return Answer(full="From the composer.", spoken="From the composer.", abstained=True)


@pytest.mark.parametrize(
    "question",
    [
        "When did I last set the points gap?",
        "When was the oil last changed?",
        "Last time I bled the brakes?",
        "How long since we changed the plugs?",
        "Have I ever replaced the condenser?",
    ],
)
def test_history_questions_are_recognised(question):
    assert is_history_question(question)


@pytest.mark.parametrize(
    "question",
    ["What's the points gap?", "How do I set the idle mixture?", "What should I check before the last drive?"],
)
def test_other_questions_are_not(question):
    assert not is_history_question(question)


def test_history_goes_to_the_log_and_everything_else_to_the_composer():
    composer = FakeComposer()
    orchestrator = Orchestrator(composer, load_logbook(FIXTURE, PROFILE))
    history = orchestrator.answer("When did I last set the points gap?", use_profile=True)
    assert history.citations[0].source == "log" and composer.calls == []
    orchestrator.answer("What's the points gap?", use_profile=False)
    assert composer.calls == [("What's the points gap?", False)]


def test_the_real_pipeline_passes_l1_on_the_repo_evals():
    eval_set = load_eval_set(REPO / "evals" / "v1.yaml")
    orchestrator = Orchestrator(FakeComposer(), load_logbook(eval_set.maintenance_log, eval_set.car_profile))
    run = run_evals(eval_set, load_profile(eval_set.car_profile), orchestrator, "fake", case_ids={"L1"})
    assert [r.status for p in run.passes for r in p.results] == ["pass", "pass"]


def test_cli_answers_history_without_an_index_or_a_model(tmp_path, capsys):
    missing_index = tmp_path / "none.jsonl"
    args = ["When did I last set the points gap?", "--log", str(FIXTURE), "--profile", str(PROFILE),
            "--index", str(missing_index)]
    assert main(args) == 0
    assert "Last logged on 2026-10-11" in capsys.readouterr().out
