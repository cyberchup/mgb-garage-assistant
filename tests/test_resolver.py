from pathlib import Path

import pytest

from mgb_assistant.profile import Deviation, Profile, Unknown
from mgb_assistant.resolver import SECOND_SOURCE_PROMPT, Resolver, load_resolver, main
from mgb_assistant.taxonomy import System, Taxonomy

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def resolver() -> Resolver:
    return load_resolver(REPO / "profile" / "mgb-1973.yaml")


def texts(resolution) -> list[str]:
    return [n.text for n in resolution.notes()]


@pytest.mark.parametrize(
    "question",
    ["How do I balance the carbs?", "What needle should I run?", "How do I top up the carb dampers?"],
)
def test_weber_traps_override_the_manual(resolver, question):
    r = resolver.resolve(question)
    assert [o.deviation.id for o in r.overrides] == ["carb-weber"]
    conflict = r.overrides[0].conflict
    assert "twin SU HS4" in conflict and "Weber 32/36" in conflict and "\n" not in conflict
    assert r.overrides[0].verify == "That's believed, not confirmed. To confirm: read the stamping on the carb body."


def test_each_how_to_check_is_given_once(resolver):
    notes = texts(resolver.resolve("What jets are in my carb?"))
    assert sum("read the stamping on the carb body" in n for n in notes) == 1
    assert "Not known yet: Weber variant and jetting." in notes


def test_notes_carry_the_profile_entry_to_cite(resolver):
    notes = resolver.resolve("What jets are in my carb?").notes()
    assert [n.entry_id for n in notes] == ["carb-weber", "carb-weber", "weber-variant"]


def test_choke_question_reaches_the_weber_and_its_unknown_variant(resolver):
    r = resolver.resolve("Is my choke manual or electric?")
    assert r.systems == ("fuel.choke",)
    assert [o.deviation.id for o in r.overrides] == ["carb-weber"]
    assert [q.id for q in r.open_questions] == ["weber-variant"]


def test_points_question_flags_the_unknown_ignition_type(resolver):
    r = resolver.resolve("What's the points gap?")
    assert r.overrides == ()
    assert texts(r) == [
        "Not known yet: ignition type (points or electronic). "
        "To check: lift the distributor cap and look for points or an electronic module."
    ]


def test_brake_jobs_are_safety_critical(resolver):
    r = resolver.resolve("Walk me through bleeding the brakes.")
    assert r.safety_systems == ("brakes",)
    assert texts(r)[-1] == SECOND_SOURCE_PROMPT


def test_without_the_profile_only_the_systems_and_safety_remain(resolver):
    r = resolver.without_profile().resolve("How do I balance the carbs?")
    assert r.systems == ("fuel.carburetor",)
    assert r.overrides == () and r.open_questions == ()
    assert resolver.without_profile().resolve("Bleed the brakes").safety_critical


def test_demotion_holds_whatever_the_question(resolver):
    assert resolver.demoted(3, "fuel.carburetor.float").id == "carb-weber"
    assert resolver.demoted(2, "fuel.carburetor") is None
    assert resolver.demoted(3, "brakes") is None
    assert resolver.without_profile().demoted(3, "fuel.carburetor") is None


TAXONOMY = Taxonomy([System("ignition", ("ignition",)), System("fuel", ("fuel",))])


def test_an_unknown_deviation_overrides_nothing_and_becomes_an_open_question():
    electronic = Deviation(
        "ign-electronic", "ignition", "unknown", ("ignition",), (3,),
        as_found="electronic ignition, maybe", verify_by="look under the distributor cap",
    )
    r = Resolver(TAXONOMY, Profile("car", (electronic,))).resolve("Check the ignition")
    assert r.overrides == ()
    assert texts(r) == ["Not known yet: electronic ignition, maybe. To check: look under the distributor cap."]


def test_rejects_a_profile_using_systems_the_taxonomy_lacks():
    profile = Profile("car", (), (Unknown("u", "something", ("fuel.carberetor",)),))
    with pytest.raises(ValueError, match="fuel.carberetor"):
        Resolver(TAXONOMY, profile)


def test_cli_prints_the_resolution(capsys):
    assert main(["How do I balance the carbs?", "--profile", str(REPO / "profile" / "mgb-1973.yaml")]) == 0
    out = capsys.readouterr().out
    assert "carbs -> fuel.carburetor" in out
    assert "Override: carb-weber (believed)" in out


def test_cli_can_switch_the_profile_off(capsys):
    assert main(["How do I balance the carbs?", "--no-profile", "--profile", str(REPO / "profile" / "mgb-1973.yaml")]) == 0
    assert "Override" not in capsys.readouterr().out
