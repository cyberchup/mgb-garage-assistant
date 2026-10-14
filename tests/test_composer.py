# Passage text here is invented for the tests. Never paste manual text into tests (CLAUDE.md).

from pathlib import Path

import pytest

from mgb_assistant import composer as composer_module
from mgb_assistant.answer import Answer
from mgb_assistant.composer import DROPPED, NO_SOURCE, SPOKEN_FALLBACK, Composer, main
from mgb_assistant.evals.cases import Case
from mgb_assistant.evals.checks import grade
from mgb_assistant.grounding import uncited_numbers
from mgb_assistant.llm import LLMError
from mgb_assistant.profile import load_profile
from mgb_assistant.resolver import load_resolver
from mgb_assistant.retriever import Retriever
from mgb_assistant.sources import Passage, write_index

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "profile" / "mgb-1973.yaml"

WEBER_IDLE = Passage("w-idle", "weber-guide", "Weber install guide", 2, "4", "tuning.idle",
                     "Adjusting the carb idle: turn the idle mixture screw in until the idle drops, then back out.")
SU_DAMPER = Passage("m-damper", "factory-manual", "workshop manual", 3, "D.7", "fuel.carburetor",
                    "Topping up the carb dampers: unscrew each damper cap and add oil to the dashpot.")
BRAKE_FORUM = Passage("f-bleed", "forum", "forum thread", 4, None, "brakes", "Bleeding the brakes, forum tips.")


class FakeLLM:
    model_id = "fake"

    def __init__(self, answer: str = "", spoken: str = "", covered: bool = True, reply: dict | None = None):
        self.reply = reply if reply is not None else {"answer": answer, "spoken": spoken, "covered": covered}
        self.calls: list[tuple[str, str, dict]] = []

    def complete_json(self, system, user, schema):
        self.calls.append((system, user, schema))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.fixture(scope="module")
def retriever():
    return Retriever([WEBER_IDLE, SU_DAMPER, BRAKE_FORUM], load_resolver(PROFILE))


IDLE_QUESTION = "How do I adjust the carb idle mixture?"


def test_the_answer_leads_with_the_conflict_and_ends_with_the_profile_notes(retriever):
    # Sources: [1] carb-weber, [2] weber-variant (both profile), [3] the Weber idle passage.
    llm = FakeLLM("Turn the idle mixture screw in until the idle drops, then back out [3].", "Turn the mixture screw in.")
    answer = Composer(retriever, llm).answer(IDLE_QUESTION, use_profile=True)
    assert answer.full == (
        "The workshop manual covers the factory twin SU HS4; this car has Weber 32/36, variant unknown [1].\n\n"
        "Turn the idle mixture screw in until the idle drops, then back out [2].\n\n"
        "That's believed, not confirmed. To confirm: read the stamping on the carb body [1]. "
        "Not known yet: Weber variant and jetting [3]."
    )
    assert [c.entry_id or c.page for c in answer.citations] == ["carb-weber", "4", "weber-variant"]
    assert answer.conflict.startswith("The workshop manual covers the factory twin SU HS4")
    assert answer.spoken == "Turn the mixture screw in."
    assert not answer.abstained and not answer.safety_flag


def test_composed_answers_pass_the_harness_checks(retriever):
    llm = FakeLLM("Turn the idle mixture screw in until the idle drops [3].", "Turn the screw in.")
    answer = Composer(retriever, llm).answer(IDLE_QUESTION, use_profile=True)
    case = Case("T", "weber_trap", "text", "-", "-", question=IDLE_QUESTION,
                checks={"no_demoted_citations": True, "cites_tier": [1, 2], "conflict_stated": True})
    results = {r.name: r.status for r in grade(answer, case, load_profile(PROFILE))}
    assert set(results.values()) == {"pass"}, results


def test_sentences_with_unsourced_numbers_are_dropped_and_the_answer_says_so(retriever):
    llm = FakeLLM("Set the screw 2 turns out. Turn the idle mixture screw in until the idle drops [3].", "Turn it in.")
    answer = Composer(retriever, llm).answer(IDLE_QUESTION, use_profile=True)
    assert "2 turns" not in answer.full
    assert DROPPED in answer.full
    assert uncited_numbers(answer) == []


def test_markers_past_the_last_source_are_removed_and_lists_of_markers_split(retriever):
    llm = FakeLLM("It needs 3.5 psi [9]. Turn the screw in [1, 3].", "Turn the screw in.")
    answer = Composer(retriever, llm).answer(IDLE_QUESTION, use_profile=True)
    assert "3.5" not in answer.full
    assert "Turn the screw in [1][2]." in answer.full


def test_the_spoken_form_never_carries_an_uncited_number(retriever):
    llm = FakeLLM("Turn the idle mixture screw in until the idle drops [3].", "Set it to 42 turns.")
    answer = Composer(retriever, llm).answer(IDLE_QUESTION, use_profile=True)
    assert answer.spoken == "Turn the idle mixture screw in until the idle drops."


def test_the_spoken_form_is_at_most_two_sentences(retriever):
    llm = FakeLLM("Turn the screw in [3].", "One. Two. Three.")
    assert Composer(retriever, llm).answer(IDLE_QUESTION, use_profile=True).spoken == "One. Two."


def test_with_nothing_to_ground_on_the_model_isnt_called(retriever):
    llm = FakeLLM("never used", "never used")
    answer = Composer(retriever, llm).answer("It's sat a year. What do I do first?", use_profile=True)
    assert llm.calls == []
    assert answer.full == NO_SOURCE and answer.abstained


def test_a_lone_community_source_on_brakes_is_withheld_and_the_safety_prompt_added(retriever):
    llm = FakeLLM("never used", "never used")
    answer = Composer(retriever, llm).answer("Walk me through bleeding the brakes.", use_profile=True)
    assert llm.calls == []
    assert answer.safety_flag and answer.abstained
    assert answer.full.endswith("Safety-critical job: check a second source before relying on this.")


def test_an_uncovered_question_is_marked_abstained(retriever):
    llm = FakeLLM("The sources don't give the jet sizes.", "I don't know the jet sizes.", covered=False)
    answer = Composer(retriever, llm).answer("What jets are in my carb?", use_profile=True)
    assert answer.abstained


def test_the_prompt_numbers_sources_and_lists_the_added_lines(retriever):
    llm = FakeLLM("Turn the screw in [3].", "Turn it in.")
    Composer(retriever, llm).answer(IDLE_QUESTION, use_profile=True)
    system, user, schema = llm.calls[0]
    assert user.startswith(f"Question: {IDLE_QUESTION}\n")
    assert "- The workshop manual covers the factory twin SU HS4" in user
    assert "[1] tier 1, car record, profile entry carb-weber" in user
    assert "[3] tier 2, Weber install guide p.4, tuning.idle" in user
    assert "Topping up the carb dampers" not in user  # demoted, so the model never sees it
    assert schema["required"] == ["answer", "spoken", "covered"]


def test_with_the_profile_off_the_model_sees_the_manual_page_and_no_car_record(retriever):
    llm = FakeLLM("Top up the dampers [1].", "Top up the dampers.")
    answer = Composer(retriever, llm).answer("How do I top up the carb dampers?", use_profile=False)
    user = llm.calls[0][1]
    assert "car record" not in user
    assert "Topping up the carb dampers" in user
    assert answer.conflict is None


def test_weak_matches_are_not_offered(retriever):
    llm = FakeLLM("Turn the screw in [1].", "Turn it in.")
    Composer(retriever, llm).answer("Where is the idle mixture screw?", use_profile=False)
    assert len(llm.calls) == 1  # the Weber idle passage clears the default floor...
    Composer(retriever, llm, min_score=1000).answer("Where is the idle mixture screw?", use_profile=False)
    assert len(llm.calls) == 1  # ...but not a raised one, leaving nothing to send


def test_malformed_model_json_is_an_error(retriever):
    with pytest.raises(LLMError, match="missing answer, spoken or covered"):
        Composer(retriever, FakeLLM(reply={"answer": "x"})).answer(IDLE_QUESTION, use_profile=True)


def test_cli_prints_the_answer_sources_and_flags(tmp_path, capsys, monkeypatch):
    index = tmp_path / "passages.jsonl"
    write_index([WEBER_IDLE], index)
    monkeypatch.setattr(composer_module, "make_llm", lambda name, model=None: FakeLLM("Turn the screw in [3].", "Turn it."))
    assert main([IDLE_QUESTION, "--index", str(index), "--profile", str(PROFILE)]) == 0
    out = capsys.readouterr().out
    assert "Spoken: Turn it." in out
    assert "[2] tier 2 Weber install guide p.4, tuning.idle" in out


def test_cli_reports_model_errors(tmp_path, capsys, monkeypatch):
    index = tmp_path / "passages.jsonl"
    write_index([WEBER_IDLE], index)
    failing = FakeLLM(reply=LLMError("no Anthropic credentials"))
    monkeypatch.setattr(composer_module, "make_llm", lambda name, model=None: failing)
    assert main([IDLE_QUESTION, "--index", str(index), "--profile", str(PROFILE)]) == 2
    assert "no Anthropic credentials" in capsys.readouterr().err


def test_the_fixed_lines_have_no_numbers():
    for line in (NO_SOURCE, DROPPED, SPOKEN_FALLBACK):
        assert uncited_numbers(Answer(full="", spoken=line)) == [], line
