# Passage text here is invented for the tests. Never paste manual text into tests (CLAUDE.md).

from pathlib import Path

import pytest

from mgb_assistant.answer import Citation
from mgb_assistant.resolver import load_resolver
from mgb_assistant.retriever import Bm25, Hit, Retriever, _rank, main
from mgb_assistant.sources import Passage, write_index

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "profile" / "mgb-1973.yaml"


def passage(pid: str, tier: int, source: str, page: str | None, system: str, text: str) -> Passage:
    return Passage(pid, source.replace(" ", "-"), source, tier, page, system, text)


SU_DAMPER = passage("m-damper", 3, "workshop manual", "D.7", "fuel.carburetor",
                    "Topping up the carb dampers: unscrew each damper cap and add oil to the dashpot.")
WEBER_FLOAT = passage("w-float", 2, "Weber install guide", "4", "fuel.carburetor",
                      "Setting the float level on the Weber: hold the float bowl cover upright and measure.")
FORUM_FLOAT = passage("f-float", 4, "forum thread", None, "fuel.carburetor",
                      "Weber float level, float level, float level: what worked on my float.")
MANUAL_BLEED = passage("m-bleed", 3, "workshop manual", "M.4", "brakes",
                       "Bleeding the brakes: start at the wheel furthest from the master cylinder.")
FORUM_BLEED = passage("f-bleed", 4, "forum thread", None, "brakes",
                      "Bleeding the brakes with a one-person kit, bleeding tips from the forum.")


@pytest.fixture(scope="module")
def resolver():
    return load_resolver(PROFILE)


def retriever(resolver, *passages: Passage) -> Retriever:
    return Retriever(passages, resolver)


def ids(hits) -> list[str]:
    return [h.passage.id for h in hits]


def test_demoted_passages_are_set_aside_not_returned(resolver):
    r = retriever(resolver, SU_DAMPER, WEBER_FLOAT).search("How do I top up the carb dampers?")
    assert "m-damper" not in ids(r.hits)
    assert [(s.hit.passage.id, s.reason) for s in r.set_aside] == [
        ("m-damper", "demoted by profile entry carb-weber")
    ]


def test_without_the_profile_nothing_is_demoted(resolver):
    r = retriever(resolver, SU_DAMPER).without_profile().search("How do I top up the carb dampers?")
    assert ids(r.hits) == ["m-damper"]
    assert r.set_aside == ()


def test_higher_tier_wins_within_a_system_even_when_it_scores_lower(resolver):
    r = retriever(resolver, WEBER_FLOAT, FORUM_FLOAT).search("What's the float level on the Weber?")
    scores = {h.passage.id: h.score for h in r.hits}
    assert scores["f-float"] > scores["w-float"]
    assert ids(r.hits) == ["w-float", "f-float"]


def test_community_source_cannot_stand_alone_on_a_safety_critical_job(resolver):
    alone = retriever(resolver, FORUM_BLEED).search("Walk me through bleeding the brakes.")
    assert alone.hits == ()
    assert [s.reason for s in alone.set_aside] == ["community source can't stand alone on a safety-critical job"]

    backed = retriever(resolver, FORUM_BLEED, MANUAL_BLEED).search("Walk me through bleeding the brakes.")
    assert ids(backed.hits) == ["m-bleed", "f-bleed"]


def test_passages_on_the_questions_system_get_a_boost(resolver):
    head = passage("m-head", 3, "workshop manual", "A.9", "engine.head", "Nut torque and order.")
    damper = passage("m-shock", 3, "workshop manual", "K.3", "suspension", "Nut torque and order.")
    r = retriever(resolver, damper, head).search("What's the head nut torque and order?")
    assert ids(r.hits) == ["m-head", "m-shock"]


def test_most_relevant_system_comes_first_with_tier_order_inside_it():
    hits = [
        Hit(MANUAL_BLEED, 5.0),
        Hit(FORUM_FLOAT, 9.0),
        Hit(WEBER_FLOAT, 2.0),
    ]
    assert ids(_rank(hits)) == ["w-float", "f-float", "m-bleed"]


def test_questions_with_no_shared_terms_find_nothing(resolver):
    r = retriever(resolver, SU_DAMPER, MANUAL_BLEED).search("What is it?")
    assert r.hits == () and r.set_aside == ()


def test_hits_cite_their_page_and_system(resolver):
    hit = retriever(resolver, WEBER_FLOAT).search("Weber float level").hits[0]
    assert hit.citation() == Citation(tier=2, source="Weber install guide", page="4", system="fuel.carburetor")


def test_rejects_passages_on_systems_the_taxonomy_lacks(resolver):
    typo = passage("x", 3, "workshop manual", "1", "fuel.carberetor", "Anything.")
    with pytest.raises(ValueError, match="fuel.carberetor"):
        Retriever([typo], resolver)


def test_bm25_weighs_rare_terms_above_common_ones():
    bm25 = Bm25([["float", "level"], ["float", "jet"], ["float", "choke"]])
    float_only, jet_doc, _ = bm25.scores(["float", "jet"])
    assert jet_doc > float_only > 0
    assert bm25.scores(["missing"]) == [0.0, 0.0, 0.0]


def test_cli_shows_hits_and_what_was_set_aside(tmp_path, capsys):
    index = tmp_path / "passages.jsonl"
    write_index([SU_DAMPER, WEBER_FLOAT], index)
    assert main(["How do I top up the carb dampers?", "--index", str(index), "--profile", str(PROFILE)]) == 0
    out = capsys.readouterr().out
    assert "Set aside:" in out and "demoted by profile entry carb-weber" in out


def test_cli_explains_a_missing_index(tmp_path, capsys):
    missing = tmp_path / "none.jsonl"
    assert main(["anything", "--index", str(missing), "--profile", str(PROFILE)]) == 2
    assert "ingest a source first" in capsys.readouterr().err
