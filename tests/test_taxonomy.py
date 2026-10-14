from pathlib import Path

import pytest

from mgb_assistant.taxonomy import System, Taxonomy, load_taxonomy, overlaps

REPO = Path(__file__).resolve().parents[1]

TAXONOMY = Taxonomy(
    [
        System("fuel.carburetor", ("carb", "SU", "carb damper")),
        System("fuel.pump", ("SU pump",)),
        System("suspension", ("damper",)),
        System("brakes", ("brake", "master cylinder"), safety_critical=True),
        System("clutch", ("clutch", "clutch master cylinder")),
        System("engine", ("engine", "firing order")),
        System("engine.head", ("head",)),
    ]
)


def systems(question: str) -> list[str]:
    return [s for m in TAXONOMY.match(question) for s in m.systems]


def test_longest_term_wins_over_the_words_inside_it():
    assert systems("How do I top up the carb dampers?") == ["fuel.carburetor"]
    assert systems("How do I top up the dampers?") == ["suspension"]
    assert systems("Is the SU pump ticking?") == ["fuel.pump"]
    assert systems("Bleed the clutch master cylinder") == ["clutch"]
    assert systems("Bleed the master cylinder") == ["brakes"]


def test_plurals_and_case_match():
    assert systems("Balance the CARBS") == ["fuel.carburetor"]
    assert systems("Tune the SUs") == ["fuel.carburetor"]


def test_curly_apostrophes_match():
    assert systems("What’s the firing order?") == ["engine"]


def test_match_reports_the_words_as_asked():
    assert [m.term for m in TAXONOMY.match("top up the Carb Dampers")] == ["carb dampers"]


def test_a_question_with_no_terms_matches_nothing():
    assert TAXONOMY.match("It's sat a year.") == []


def test_safety_critical_covers_everything_under_it():
    assert TAXONOMY.is_safety_critical("brakes")
    assert TAXONOMY.is_safety_critical("brakes.hydraulics")
    assert not TAXONOMY.is_safety_critical("fuel.carburetor")


def test_overlaps_runs_both_ways():
    assert overlaps("fuel", "fuel.carburetor")
    assert overlaps("fuel.carburetor", "fuel")
    assert not overlaps("fuel.carburetor", "fuel.choke")


def test_synonyms_fold_into_the_first_word_of_their_group():
    taxonomy = Taxonomy([System("fuel.carburetor", ("carb",))], synonyms=[["carb", "carburetor", "carburetter"]])
    assert taxonomy.normalize("Carburetters and carburetors") == ("carb", "and", "carb")
    assert [m.systems for m in taxonomy.match("Remove the carburetter")] == [("fuel.carburetor",)]


def test_rejects_a_synonym_group_with_phrases(tmp_path):
    path = tmp_path / "systems.yaml"
    path.write_text("systems:\n  fuel:\n    terms: [fuel]\nsynonyms:\n  - [petrol, motor spirit]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="single words"):
        load_taxonomy(path)


def test_the_repo_taxonomy_loads():
    taxonomy = load_taxonomy(REPO / "profile" / "systems.yaml")
    assert taxonomy.is_safety_critical("fuel.lines")
    assert not taxonomy.is_safety_critical("fuel.carburetor")


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ("systems:\n  Fuel:\n    terms: [fuel]\n", "lowercase"),
        ("systems:\n  fuel:\n    terms: []\n", "no terms"),
    ],
)
def test_rejects_a_malformed_system(tmp_path, body, error):
    path = tmp_path / "systems.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_taxonomy(path)
