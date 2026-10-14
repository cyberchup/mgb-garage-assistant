from datetime import date
from pathlib import Path

import pytest

from mgb_assistant.evals.cases import Case
from mgb_assistant.evals.checks import grade
from mgb_assistant.logbook import (
    NO_RECORD,
    LogBook,
    LogEntry,
    LogError,
    Measurement,
    ProfileUpdate,
    load_logbook,
    main,
    parse_measurement,
    readback,
)
from mgb_assistant.profile import load_profile

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "profile" / "mgb-1973.yaml"
FIXTURE = REPO / "evals" / "fixtures" / "maintenance-log.yaml"


def entry(entry_id: str = "2026-10-11-01", **fields) -> LogEntry:
    base = dict(id=entry_id, date=date.fromisoformat(entry_id[:10]), system="ignition.points", action="set points gap",
                measurements=(Measurement("points_gap", 0.015, "in"),))
    return LogEntry(**{**base, **fields})


@pytest.fixture
def logbook(tmp_path) -> LogBook:
    return load_logbook(tmp_path / "log" / "maintenance.yaml", PROFILE)


# --- reading ---


def test_the_schema_example_is_a_valid_log():
    entries = load_logbook(REPO / "schemas" / "log-entry.example.yaml", PROFILE).entries()
    assert [(e.id, e.source, e.measurements[0].value) for e in entries] == [("2026-10-20-01", "voice", 0.015)]


def test_unconfirmed_entries_never_feed_answers():
    logbook = load_logbook(FIXTURE, PROFILE)
    assert "2026-10-12-01" in [e.id for e in logbook.entries()]
    assert "2026-10-12-01" not in [e.id for e in logbook.confirmed()]
    assert logbook.latest(["ignition.points"]).id == "2026-10-11-01"


def test_a_missing_file_or_no_file_is_an_empty_log(tmp_path):
    assert load_logbook(tmp_path / "none.yaml", PROFILE).entries() == []
    assert load_logbook(None, PROFILE).entries() == []


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ("- {id: 2026-10-11, date: 2026-10-11, system: ignition, action: x}", "id should look like"),
        ("- {id: 2026-10-11-01, date: 2026-10-12, system: ignition, action: x}", "date should be a date matching"),
        ("- {id: 2026-10-11-01, date: 2026-10-11, system: ignition.pionts, action: x}", "isn't in the taxonomy"),
        ("- {id: 2026-10-11-01, date: 2026-10-11, system: ignition, action: x, measurements: [{name: gap, value: 1}]}",
         "needs a unit"),
        ("- {id: 2026-10-11-01, date: 2026-10-11, system: ignition, action: x, source: phone}", "source should be"),
        ("- {id: 2026-10-11-01, date: 2026-10-11, system: ignition, action: x,"
         " proposes_profile_update: {target: nonsense, value: DGV}}", "isn't a profile unknown"),
    ],
)
def test_rejects_a_malformed_entry(tmp_path, body, error):
    path = tmp_path / "log.yaml"
    path.write_text(body + "\n", encoding="utf-8")
    with pytest.raises(LogError, match=error):
        load_logbook(path, PROFILE).entries()


# --- writing ---


def test_only_confirmed_entries_are_written(logbook):
    with pytest.raises(LogError, match="only confirmed entries are written"):
        logbook.write(entry(confirmed=False, source="voice"))
    assert logbook.entries() == []


def test_confirming_a_draft_writes_it(logbook):
    written = logbook.confirm(entry(confirmed=False, source="voice"))
    assert written.confirmed
    assert logbook.confirmed() == [written]


def test_the_log_is_append_only_and_keeps_hand_written_comments(logbook):
    logbook.write(entry("2026-10-11-01"))
    with logbook.path.open("a", encoding="utf-8") as f:
        f.write("# a note the owner typed in by hand\n")
    logbook.write(entry("2026-10-11-02", action="checked dwell"))
    text = logbook.path.read_text(encoding="utf-8")
    assert "# a note the owner typed in by hand" in text
    assert [e.id for e in logbook.entries()] == ["2026-10-11-01", "2026-10-11-02"]


def test_ids_number_each_days_jobs(logbook):
    assert logbook.next_id(date(2026, 10, 11)) == "2026-10-11-01"
    logbook.write(entry("2026-10-11-01"))
    assert logbook.next_id(date(2026, 10, 11)) == "2026-10-11-02"
    assert logbook.next_id(date(2026, 10, 12)) == "2026-10-12-01"


def test_duplicate_ids_are_refused(logbook):
    logbook.write(entry())
    with pytest.raises(LogError, match="already in the log"):
        logbook.write(entry())


def test_entries_round_trip_through_the_file(logbook):
    original = entry(odometer_mi=61240, parts=("points set",), observations="Pitted contacts.",
                     proposes_profile_update=ProfileUpdate("ignition-type", "points"))
    logbook.write(original)
    assert logbook.entries() == [original]


# --- measurements and read-back ---


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("points_gap=0.015in", Measurement("points_gap", 0.015, "in")),
        ("points_gap=15thou", Measurement("points_gap", 0.015, "in")),
        ("idle_speed=850rpm", Measurement("idle_speed", 850.0, "rpm")),
    ],
)
def test_parses_measurements(text, expected):
    assert parse_measurement(text) == expected


def test_a_measurement_without_a_unit_is_refused():
    with pytest.raises(LogError, match="should look like"):
        parse_measurement("points_gap=0.015")


def test_readback_states_every_number():
    text = readback(entry(odometer_mi=61240, proposes_profile_update=ProfileUpdate("ignition-type", "points")))
    assert "points gap 0.015 in" in text
    assert "Odometer 61240 mi" in text
    assert "ignition-type = points" in text


# --- answering ---


def test_history_answers_pass_the_l1_checks():
    answer = load_logbook(FIXTURE, PROFILE).history_answer("When did I last set the points gap?")
    case = Case("L1", "log", "text", "-", "-", question="?", expected_answer=["2026-10-11", "0.015"],
                checks={"log_only": True, "contains_expected": True})
    results = {r.name: r.status for r in grade(answer, case, load_profile(PROFILE))}
    assert set(results.values()) == {"pass"}, results
    assert "0.018" not in answer.full


def test_no_matching_entry_is_no_record():
    answer = load_logbook(FIXTURE, PROFILE).history_answer("When did I last bleed the brakes?")
    assert answer.full == NO_RECORD and answer.abstained and answer.citations == ()


def test_proposals_wait_for_approval(logbook):
    logbook.write(entry("2026-10-11-01"))
    logbook.write(entry("2026-10-11-02", system="fuel.carburetor", action="read the carb stamping", measurements=(),
                        proposes_profile_update=ProfileUpdate("weber-variant", "32/36 DGV")))
    assert [e.id for e in logbook.proposals()] == ["2026-10-11-02"]


# --- command line ---


def cli(tmp_path, *args):
    return main(["--log", str(tmp_path / "log.yaml"), "--profile", str(PROFILE), *args])


def test_cli_add_reads_back_then_writes_on_yes(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert cli(tmp_path, "add", "--system", "ignition.points", "--action", "set points gap",
               "--measure", "points_gap=15thou", "--date", "2026-10-11") == 0
    out = capsys.readouterr().out
    assert "points gap 0.015 in." in out and "Written as 2026-10-11-01" in out
    assert load_logbook(tmp_path / "log.yaml", PROFILE).confirmed()[0].measurements[0].value == 0.015


def test_cli_add_writes_nothing_without_a_yes(tmp_path, capsys, monkeypatch):
    def no_input(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_input)
    assert cli(tmp_path, "add", "--system", "ignition.points", "--action", "set points gap") == 1
    assert "Not written." in capsys.readouterr().out
    assert not (tmp_path / "log.yaml").exists()


def test_cli_rejects_an_unknown_system(tmp_path, capsys):
    assert cli(tmp_path, "add", "--system", "ignition.pionts", "--action", "x", "--yes") == 2
    assert "isn't in the taxonomy" in capsys.readouterr().err


def test_cli_lists_entries(tmp_path, capsys):
    assert main(["--log", str(FIXTURE), "--profile", str(PROFILE), "list"]) == 0
    out = capsys.readouterr().out
    assert "2026-10-11-01" in out and "2026-10-12-01" not in out
