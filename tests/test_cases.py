from pathlib import Path

import pytest

from mgb_assistant.evals.cases import EvalFileError, load_eval_set

REPO = Path(__file__).resolve().parents[1]

TEXT_CASE = """\
  - id: {id}
    category: abstention
    modality: text
    question: "What's the torque?"
    expected_behavior: Says so.
    pass_when: No uncited number.
"""


def write_eval(tmp_path: Path, cases: str) -> Path:
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    path = tmp_path / "evals.yaml"
    path.write_text(f"version: 1\ncar_profile: car.yaml\ncases:\n{cases}", encoding="utf-8")
    return path


def test_the_repo_eval_file_loads():
    eval_set = load_eval_set(REPO / "evals" / "v1.yaml")
    assert eval_set.version == 1
    assert eval_set.car_profile.is_file()
    t1 = next(c for c in eval_set.cases if c.id == "T1")
    assert t1.checks == {"no_demoted_citations": True, "cites_tier": [1, 2], "conflict_stated": True}
    assert t1.review
    assert {c.modality for c in eval_set.cases} == {"text", "voice"}


def test_paths_resolve_from_the_repo_root(tmp_path):
    eval_set = load_eval_set(write_eval(tmp_path, TEXT_CASE.format(id="A1")))
    assert eval_set.root == tmp_path.resolve()
    assert eval_set.car_profile == tmp_path.resolve() / "car.yaml"


def test_rejects_duplicate_ids(tmp_path):
    path = write_eval(tmp_path, TEXT_CASE.format(id="A1") + TEXT_CASE.format(id="A1"))
    with pytest.raises(EvalFileError, match="A1 appears twice"):
        load_eval_set(path)


def test_rejects_a_misspelled_check(tmp_path):
    path = write_eval(tmp_path, TEXT_CASE.format(id="A1") + "    checks: {safety_flags: true}\n")
    with pytest.raises(EvalFileError, match="unknown check 'safety_flags'"):
        load_eval_set(path)


def test_rejects_a_check_value_of_the_wrong_type(tmp_path):
    path = write_eval(tmp_path, TEXT_CASE.format(id="A1") + "    checks: {must_mention: Weber}\n")
    with pytest.raises(EvalFileError, match="must_mention takes a list"):
        load_eval_set(path)


def test_rejects_list_items_of_the_wrong_type(tmp_path):
    path = write_eval(tmp_path, TEXT_CASE.format(id="A1") + "    checks: {cites_tier: [one, 2]}\n")
    with pytest.raises(EvalFileError, match="cites_tier takes a list of int"):
        load_eval_set(path)


def test_text_cases_need_a_question(tmp_path):
    path = write_eval(tmp_path, TEXT_CASE.format(id="A1").replace('    question: "What\'s the torque?"\n', ""))
    with pytest.raises(EvalFileError, match="need a question"):
        load_eval_set(path)


def test_hash_ignores_line_endings(tmp_path):
    path = write_eval(tmp_path, TEXT_CASE.format(id="A1"))
    lf_bytes = path.read_bytes().replace(b"\r\n", b"\n")  # write_text uses CRLF on Windows
    path.write_bytes(lf_bytes)
    lf = load_eval_set(path).sha256
    path.write_bytes(lf_bytes.replace(b"\n", b"\r\n"))
    assert load_eval_set(path).sha256 == lf
