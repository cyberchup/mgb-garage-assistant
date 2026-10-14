from mgb_assistant.answer import Answer, Citation
from mgb_assistant.evals.cases import Case
from mgb_assistant.evals.checks import GLOBAL_CHECKS, grade, uncited_numbers
from mgb_assistant.profile import Deviation, Profile

PROFILE = Profile(
    "mgb",
    (Deviation("carb-weber", "fuel.carburetor", "believed", ("fuel.carburetor", "tuning.idle"), (3,)),),
)
MANUAL = Citation(tier=3, source="workshop manual", page="B.4", system="ignition.firing_order")
WEBER_DOC = Citation(tier=2, source="weber install guide", page="3", system="fuel.carburetor")
SU_PAGE = Citation(tier=3, source="workshop manual", page="D.7", system="fuel.carburetor.dampers")
FORUM = Citation(tier=4, source="forum thread", system="brakes.hydraulics")
LOG = Citation(tier=1, source="log", entry_id="2026-10-20-01", system="ignition.points")


def case(**fields) -> Case:
    base = dict(id="X1", category="spec_lookup", modality="text", expected_behavior="-", pass_when="-", question="?")
    return Case(**{**base, **fields})


def answer(full: str, *citations: Citation, spoken: str = "", **flags) -> Answer:
    return Answer(full=full, spoken=spoken, citations=citations, **flags)


def check(name: str, a: Answer, c: Case | None = None):
    c = c or case(checks={} if name in GLOBAL_CHECKS else {name: True})
    return next(r for r in grade(a, c, PROFILE) if r.name == name)


# --- uncited numbers ---


def test_number_in_a_cited_sentence_is_cited():
    assert uncited_numbers(answer("The firing order is 1-3-4-2 [1].", MANUAL)) == []


def test_number_without_a_marker_is_uncited():
    assert uncited_numbers(answer("The gap is 0.015 in.")) == ["0.015"]


def test_marker_after_the_period_still_cites_its_sentence():
    assert uncited_numbers(answer("The gap is 0.015 in. [1]", MANUAL)) == []


def test_marker_must_point_at_a_real_citation():
    assert uncited_numbers(answer("The gap is 0.015 in [2].", MANUAL)) == ["0.015"]


def test_list_numbering_and_part_names_are_not_numbers():
    assert uncited_numbers(answer("1. Drain the float bowl.\n2. Ignore the HS4 steps.")) == []


def test_ranges_and_fractions_stay_whole():
    assert uncited_numbers(answer("It runs a Weber 32/36 and wants 55-60 lb ft.")) == ["32/36", "55-60"]


def test_spoken_numbers_need_a_cited_twin_in_the_full_answer():
    a = answer("The gap is 0.015 in [1].", MANUAL, spoken="Set it to 0.015, or maybe 0.016.")
    assert uncited_numbers(a) == ["0.016"]


# --- global checks ---


def test_citations_valid_catches_each_kind_of_broken_citation():
    no_page = Citation(tier=3, source="workshop manual", system="ignition")
    log_without_entry = Citation(tier=1, source="log")
    result = check("citations_valid", answer("A [1]. B [2]. C [4].", no_page, log_without_entry, MANUAL))
    assert result.status == "fail"
    for problem in ("[4] has no citation", "[1] tier 3 citation has no page", "[2] log citation has no entry id",
                    "[3] is never referenced"):
        assert problem in result.detail


def test_grounded_accepts_a_citation_or_an_abstention():
    assert check("grounded", answer("1-3-4-2 [1].", MANUAL)).status == "pass"
    assert check("grounded", answer("Nothing covers that.", abstained=True)).status == "pass"
    assert check("grounded", answer("It's 1-3-4-2.")).status == "fail"


# --- per-case checks ---


def test_contains_expected_ignores_spacing_and_dash_style():
    c = case(expected_answer="1-3-4-2", checks={"contains_expected": True})
    assert check("contains_expected", answer("It's 1 – 3 – 4 – 2 [1].", MANUAL), c).status == "pass"


def test_contains_expected_needs_every_part_of_a_list():
    c = case(expected_answer=["0.015 in", "cold"], checks={"contains_expected": True})
    assert check("contains_expected", answer("Set it to 0.015 in [1].", MANUAL), c).status == "fail"


def test_contains_expected_is_not_ready_without_ground_truth():
    assert check("contains_expected", answer("Anything.", abstained=True)).status == "not_ready"


def test_cites_expected_source_matches_tier_and_printed_page():
    c = case(expected_source={"tier": 3, "page": "b.4"}, checks={"cites_expected_source": True})
    assert check("cites_expected_source", answer("1-3-4-2 [1].", MANUAL), c).status == "pass"
    c = case(expected_source={"tier": 3, "page": "B.5"}, checks={"cites_expected_source": True})
    assert check("cites_expected_source", answer("1-3-4-2 [1].", MANUAL), c).status == "fail"


def test_cites_expected_source_is_not_ready_without_a_page():
    c = case(expected_source={"tier": 3, "page": None}, checks={"cites_expected_source": True})
    assert check("cites_expected_source", answer("1-3-4-2 [1].", MANUAL), c).status == "not_ready"


def test_no_demoted_citations_rejects_the_manual_on_overridden_systems():
    assert check("no_demoted_citations", answer("Top up the dampers [1].", SU_PAGE)).status == "fail"
    assert check("no_demoted_citations", answer("Adjust the idle screw [1].", WEBER_DOC)).status == "pass"
    assert check("no_demoted_citations", answer("Firing order is 1-3-4-2 [1].", MANUAL)).status == "pass"


def test_cites_tier_needs_a_citation_from_a_listed_tier():
    c = case(checks={"cites_tier": [1, 2]})
    assert check("cites_tier", answer("Adjust the idle screw [1].", WEBER_DOC), c).status == "pass"
    assert check("cites_tier", answer("Top up the dampers [1].", SU_PAGE), c).status == "fail"
    assert check("cites_tier", answer("No source covers this.", abstained=True), c).status == "fail"


def test_conflict_must_be_stated_in_one_line():
    assert check("conflict_stated", answer("x", abstained=True)).status == "fail"
    assert check("conflict_stated", answer("x", abstained=True, conflict="a\nb")).status == "fail"
    one_line = answer("x", abstained=True, conflict="The manual covers twin SUs; this car has one Weber.")
    assert check("conflict_stated", one_line).status == "pass"


def test_safety_flag_must_be_set():
    assert check("safety_flag", answer("Bleed [1].", MANUAL)).status == "fail"
    assert check("safety_flag", answer("Bleed [1].", MANUAL, safety_flag=True)).status == "pass"


def test_community_sources_cannot_stand_alone():
    assert check("not_tier4_alone", answer("Bleed [1].", FORUM)).status == "fail"
    assert check("not_tier4_alone", answer("Bleed [1] [2].", FORUM, MANUAL)).status == "pass"


def test_log_only_accepts_log_citations_or_no_record():
    assert check("log_only", answer("Set on 2026-10-20 [1].", LOG)).status == "pass"
    assert check("log_only", answer("No record of that.", abstained=True)).status == "pass"
    assert check("log_only", answer("The manual says [1].", MANUAL)).status == "fail"
    assert check("log_only", answer("Set on 2026-10-20 [1] [2].", LOG, MANUAL)).status == "fail"


def test_log_only_lets_notes_from_the_profile_ride_along():
    profile_note = Citation(tier=1, source="profile", entry_id="ignition-type")
    no_record = answer("No record. Ignition type unknown [1].", profile_note, abstained=True)
    assert check("log_only", no_record).status == "pass"
    assert check("log_only", answer("Ignition type unknown [1].", profile_note)).status == "fail"


def test_must_mention_matches_whole_words_in_any_case():
    c = case(checks={"must_mention": ["Weber"]})
    assert check("must_mention", answer("Drain the weber float bowl.", abstained=True), c).status == "pass"
    assert check("must_mention", answer("Drain the carbs.", abstained=True), c).status == "fail"


def test_checks_set_to_false_are_skipped():
    names = [r.name for r in grade(answer("x", abstained=True), case(checks={"safety_flag": False}), PROFILE)]
    assert "safety_flag" not in names
