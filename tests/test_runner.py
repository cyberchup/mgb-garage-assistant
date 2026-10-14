import json
from pathlib import Path

import pytest

from mgb_assistant.answer import Answer, Citation
from mgb_assistant.evals.__main__ import main
from mgb_assistant.evals.baselines import ProfileOnlyAssistant
from mgb_assistant.evals.cases import load_eval_set
from mgb_assistant.evals.runner import run_evals, write_run
from mgb_assistant.profile import load_profile
from mgb_assistant.resolver import load_resolver

REPO = Path(__file__).resolve().parents[1]

PROFILE_YAML = """\
vehicle: {id: test-car}
deviations:
  - id: carb-weber
    system: fuel.carburetor
    confidence: believed
    affects: [fuel.carburetor]
    demotes_tiers: [3]
"""

EVAL_YAML = """\
version: 1
car_profile: car.yaml
metrics: [override_accuracy, profile_ablation_gap, uncited_numbers_per_100]
cases:
  - id: T1
    category: weber_trap
    modality: text
    question: "How do I balance the carbs?"
    expected_behavior: Single Weber, nothing to balance.
    pass_when: No SU procedure; conflict stated.
    checks: {no_demoted_citations: true, conflict_stated: true}
  - id: C1
    category: confidence
    modality: text
    question: "What jets are in my carb?"
    expected_behavior: Says unknown.
    pass_when: No invented jet sizes.
    checks: {review: Says the jetting is unknown.}
  - id: S2
    category: spec_lookup
    modality: text
    question: "What's the points gap?"
    expected_answer: null
    expected_source: {tier: 3, page: null}
    expected_behavior: Manual value.
    pass_when: Matches the manual.
    checks: {contains_expected: true}
  - id: V1
    category: voice
    modality: voice
    utterance: "Log it: points gap fifteen thou"
    expected_behavior: Reads it back.
    pass_when: Confirmation step.
"""

WEBER = Citation(tier=2, source="weber install guide", page="3", system="fuel.carburetor")
SU_PAGE = Citation(tier=3, source="workshop manual", page="D.7", system="fuel.carburetor")


class ScriptedAssistant:
    """Answers T1 the right way only when it can see the profile."""

    model_id = "scripted-1"

    def answer(self, question: str, *, use_profile: bool) -> Answer:
        if "balance" in question:
            if use_profile:
                return Answer(
                    full="This car has one Weber, so there's nothing to balance; set the idle instead [1].",
                    spoken="There's only one carb, so there's nothing to balance.",
                    citations=(WEBER,),
                    conflict="The manual describes twin SUs; this car runs a single Weber.",
                )
            return Answer(full="Balance the SUs with a flow meter [1].", spoken="Use a flow meter.", citations=(SU_PAGE,))
        if "jets" in question:
            return Answer(full="The jetting isn't recorded yet.", spoken="I don't know yet.", abstained=True)
        raise RuntimeError("no script for this question")


@pytest.fixture
def eval_set(tmp_path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / "car.yaml").write_text(PROFILE_YAML, encoding="utf-8")
    (tmp_path / "evals.yaml").write_text(EVAL_YAML, encoding="utf-8")
    return load_eval_set(tmp_path / "evals.yaml")


def run(eval_set, **kwargs):
    return run_evals(eval_set, load_profile(eval_set.car_profile), ScriptedAssistant(), "scripted", **kwargs)


def statuses(profile_pass):
    return {r.case_id: r.status for r in profile_pass.results}


def test_ablation_shows_the_profile_making_the_difference(eval_set):
    on, off = run(eval_set).passes
    assert statuses(on) == {"T1": "pass", "C1": "needs_review", "S2": "error", "V1": "skipped"}
    assert statuses(off)["T1"] == "fail"
    assert on.metrics()["override_accuracy"] == 1.0
    assert off.metrics()["override_accuracy"] == 0.0
    # Graded cases: T1, C1 and S2 (an error counts against the assistant).
    assert on.auto_pass_rate() == pytest.approx(2 / 3, abs=0.001)
    assert run(eval_set).ablation_gap() == pytest.approx(1 / 3, abs=0.001)


def test_missing_ground_truth_is_not_ready_not_failed(eval_set):
    class Abstains:
        model_id = "abstains"

        def answer(self, question, *, use_profile):
            return Answer(full="No source covers this.", spoken="No source.", abstained=True)

    result = run_evals(eval_set, load_profile(eval_set.car_profile), Abstains(), "abstains", case_ids={"S2"})
    assert statuses(result.passes[0]) == {"S2": "not_ready"}


def test_run_record_holds_what_reproducing_the_run_needs(eval_set, tmp_path):
    path = write_run(run(eval_set), tmp_path / "runs")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert "scripted-profile-on-off" in path.name
    assert record["eval_file"] == "evals.yaml"
    assert record["eval_version"] == 1
    assert record["eval_sha256"] == eval_set.sha256
    assert record["model"] == "scripted-1"
    assert set(record["git"]) == {"commit", "dirty"}
    assert [p["profile"] for p in record["passes"]] == ["on", "off"]
    assert set(record["passes"][0]["metrics"]) == {"override_accuracy", "uncited_numbers_per_100"}
    t1 = record["passes"][0]["results"][0]
    assert t1["answer"]["citations"][0]["tier"] == 2


def test_profile_only_baseline_opens_an_ablation_gap_on_the_repo_evals():
    eval_set = load_eval_set(REPO / "evals" / "v1.yaml")
    assistant = ProfileOnlyAssistant(load_resolver(eval_set.car_profile))
    result = run_evals(eval_set, load_profile(eval_set.car_profile), assistant, "profile-only")
    on, off = result.passes
    assert statuses(on)["T1"] == "needs_review"  # machine checks pass; the Weber idle advice is for review
    assert statuses(off)["T1"] == "fail"
    assert on.metrics()["override_accuracy"] == 1.0
    assert off.metrics()["override_accuracy"] == 0.0
    assert on.metrics()["uncited_numbers_per_100"] == 0.0
    assert result.ablation_gap() > 0


def test_cli_runs_the_repo_eval_set(tmp_path, capsys):
    assert main([str(REPO / "evals" / "v1.yaml"), "--out", str(tmp_path)]) == 0
    assert "profile_ablation_gap" in capsys.readouterr().out
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_cli_rejects_an_unknown_case_id(tmp_path, capsys):
    assert main([str(REPO / "evals" / "v1.yaml"), "--case", "Z9", "--out", str(tmp_path)]) == 2
    assert "Z9" in capsys.readouterr().err
