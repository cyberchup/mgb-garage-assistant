# Passage text here is invented for the tests. Never paste manual text into tests (CLAUDE.md).

import json

import pytest

from mgb_assistant.sources import Passage, load_index, write_index

MANUAL = Passage("m-1", "factory-manual", "workshop manual", 3, "B.4", "ignition", "Invented passage text.")
FORUM = Passage("f-1", "forum", "forum thread", 4, None, "brakes", "Invented forum text.")


def test_round_trip(tmp_path):
    path = tmp_path / "index" / "passages.jsonl"
    write_index([MANUAL, FORUM], path)
    assert load_index(path) == [MANUAL, FORUM]


@pytest.mark.parametrize(
    ("passage", "error"),
    [
        (Passage("p", "log", "log", 1, None, "ignition", "x"), "tier 1 is the profile and the log"),
        (Passage("p", "factory-manual", "workshop manual", 3, None, "ignition", "x"), "need the printed page"),
        (Passage("p", "factory-manual", "workshop manual", 3, "B.4", "ignition", "  "), "no text"),
    ],
)
def test_rejects_passages_that_cannot_be_cited(tmp_path, passage, error):
    with pytest.raises(ValueError, match=error):
        write_index([passage], tmp_path / "passages.jsonl")
    path = tmp_path / "hand-edited.jsonl"
    path.write_text(json.dumps(passage.__dict__) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_index(path)


def test_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "passages.jsonl"
    line = json.dumps(MANUAL.__dict__)
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="appears twice"):
        load_index(path)


def test_rejects_lines_that_are_not_passages(tmp_path):
    path = tmp_path / "passages.jsonl"
    path.write_text('{"id": "p", "colour": "red"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="passages.jsonl:1: not a passage"):
        load_index(path)
