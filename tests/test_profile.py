from pathlib import Path

import pytest

from mgb_assistant.profile import Deviation, Profile, covers, load_profile

REPO = Path(__file__).resolve().parents[1]


def weber(confidence: str) -> Profile:
    return Profile("mgb", (Deviation("carb-weber", "fuel.carburetor", confidence, ("fuel.carburetor",), (3,)),))


def test_covers_matches_the_system_and_what_sits_under_it():
    assert covers("fuel.carburetor", "fuel.carburetor")
    assert covers("fuel.carburetor", "fuel.carburetor.float")
    assert not covers("fuel.carburetor", "fuel.carburetors")
    assert not covers("fuel.carburetor", "fuel")


def test_believed_deviation_demotes_the_manual_for_affected_systems():
    profile = weber("believed")
    assert profile.overriding(3, "fuel.carburetor.float").id == "carb-weber"
    assert profile.overriding(2, "fuel.carburetor") is None
    assert profile.overriding(3, "brakes.hydraulics") is None


def test_unknown_deviation_overrides_nothing():
    assert weber("unknown").overriding(3, "fuel.carburetor") is None


def test_loads_the_repo_profile():
    profile = load_profile(REPO / "profile" / "mgb-1973.yaml")
    assert profile.vehicle_id == "mgb-1973"
    assert profile.overriding(3, "fuel.choke").id == "carb-weber"
    assert profile.deviations[0].verify_by == "read the stamping on the carb body"
    assert [u.id for u in profile.unknowns] == ["engine-number", "weber-variant", "ignition-type"]
    assert profile.systems_file == REPO / "profile" / "systems.yaml"
    assert "ignition" in profile.system_ids()


def test_rejects_an_unknown_without_its_systems(tmp_path):
    path = tmp_path / "car.yaml"
    path.write_text("vehicle: {id: x}\nunknowns:\n  - engine number\n", encoding="utf-8")
    with pytest.raises(ValueError, match="each unknown needs"):
        load_profile(path)


def test_rejects_an_unrecognized_confidence(tmp_path):
    path = tmp_path / "car.yaml"
    path.write_text(
        "vehicle: {id: x}\n"
        "deviations:\n"
        "  - {id: d, system: fuel, confidence: probably, affects: [fuel], demotes_tiers: [3]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="probably"):
        load_profile(path)
