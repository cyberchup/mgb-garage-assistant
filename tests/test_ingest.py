# Page text here is invented for the tests. Never paste manual text into tests (CLAUDE.md).

import re
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from mgb_assistant.ingest import (
    LabelRange,
    SourceSpec,
    SystemRange,
    clean_pages,
    ingest_source,
    load_source_spec,
    main,
    split_passages,
    tag_system,
)
from mgb_assistant.sources import load_index
from mgb_assistant.taxonomy import System, Taxonomy, load_taxonomy

REPO = Path(__file__).resolve().parents[1]

TAXONOMY = Taxonomy(
    [
        System("fuel", ("fuel",)),
        System("fuel.carburetor", ("carb", "jet", "float")),
        System("fuel.choke", ("choke",)),
        System("tuning.idle", ("idle", "idle speed")),
        System("brakes", ("brake",), safety_critical=True),
    ],
    synonyms=[["carb", "carburetor", "carburetter"]],
)
CARB_PAGES = SystemRange(1, 99, "fuel.carburetor", ("fuel.choke", "tuning.idle"))


def spec(**fields) -> SourceSpec:
    base = dict(id="guide", source="test guide", tier=2, file=Path("missing.pdf"), systems=(CARB_PAGES,))
    return SourceSpec(**{**base, **fields})


# --- cleaning ---


def test_running_headers_are_dropped_even_with_changing_page_numbers():
    pages = ["TEST GUIDE Page 1\nFirst body line.", "TEST GUIDE Page 2\nSecond body line.", "TEST GUIDE Page 3\nThird."]
    assert clean_pages(pages) == ["First body line.", "Second body line.", "Third."]


def test_a_line_on_too_few_pages_stays_unless_a_pattern_drops_it():
    pages = ["Note to reader.\nOne.", "Note to reader.\nTwo.", "Three.", "Four.", "Five."]
    assert clean_pages(pages)[0] == "Note to reader.\nOne."
    assert clean_pages(pages, [re.compile("^Note to")])[0] == "One."


def test_symbol_glyphs_and_curly_typography_are_cleaned():
    assert clean_pages([" The engine’s “idle” – slow"]) == ['The engine\'s "idle" - slow']


# --- splitting ---


def test_headings_start_and_prefix_passages():
    text = "FITTING THE CARB\n\nBolt it down evenly and check it.\n\nADJUSTING\n\nTurn the idle screw slowly in."
    assert split_passages(text) == [
        "FITTING THE CARB: Bolt it down evenly and check it.",
        "ADJUSTING: Turn the idle screw slowly in.",
    ]


def test_long_runs_split_at_paragraphs():
    paragraph = " ".join(["word"] * 50)
    assert len(split_passages("\n\n".join([paragraph] * 3), max_words=120)) == 2


def test_short_paragraphs_join_their_neighbours_and_lone_fragments_are_dropped():
    assert split_passages("Too short.\n\nThis one has enough words.") == ["Too short. This one has enough words."]
    assert split_passages("Too short.") == []
    assert split_passages("Real text for the first section.\n\nLONE HEADING\n\nTwo words") == [
        "Real text for the first section."
    ]


@pytest.mark.parametrize("line", ["PART # AB1, AB2, AB3", "DO NOT OVERTIGHTEN THE JETS."])
def test_lists_and_shouted_sentences_are_not_headings(line):
    assert split_passages(f"{line}\n\nThe body text of this passage.")[0].startswith(line)


# --- tagging ---


def test_tagging_falls_back_to_the_default():
    assert tag_system("Nothing relevant in these words.", CARB_PAGES, TAXONOMY) == "fuel.carburetor"


def test_an_allowed_system_wins_when_its_words_dominate():
    assert tag_system("Set the idle speed, then the idle again.", CARB_PAGES, TAXONOMY) == "tuning.idle"


def test_ties_go_to_the_default():
    assert tag_system("The carb and the choke.", CARB_PAGES, TAXONOMY) == "fuel.carburetor"


def test_the_most_specific_candidate_gets_the_credit():
    broad = SystemRange(1, 9, "fuel", ("fuel.carburetor",))
    assert tag_system("Check the jet and the float.", broad, TAXONOMY) == "fuel.carburetor"


def test_synonyms_reach_the_tagger():
    assert tag_system("Remove the carburetter.", SystemRange(1, 9, "fuel", ("fuel.carburetor",)), TAXONOMY) == (
        "fuel.carburetor"
    )


# --- page labels ---


def test_label_ranges_win_over_the_pdf_label():
    s = spec(labels=(LabelRange(10, 20, start=1, prefix="D."),))
    assert s.label(13, "13") == "D.4"
    assert s.label(9, "ix") == "ix"


# --- source files ---


def write_spec(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "guide.yaml"
    path.write_text(body, encoding="utf-8")
    return path


GOOD_SPEC = """\
id: guide
source: test guide
tier: 3
file: manuals/guide.pdf
labels:
  - {pages: 3-5, start: 1, prefix: "A."}
drop_lines: ["^Notice"]
systems:
  - {pages: 1-2, default: fuel}
  - {pages: 3-9, default: fuel.carburetor, allow: [fuel.choke]}
"""


def test_loads_a_source_file(tmp_path):
    s = load_source_spec(write_spec(tmp_path, GOOD_SPEC), tmp_path)
    assert s.file == tmp_path / "manuals" / "guide.pdf"
    assert s.system_range(4) == SystemRange(3, 9, "fuel.carburetor", ("fuel.choke",))
    assert s.label(3, "3") == "A.1"


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (("tier: 3", "tier: 1"), "tier 1 can't be indexed"),
        (("pages: 1-2", "pages: 2-1"), "backwards"),
        (("pages: 3-9", "pages: 2-9"), "overlap"),
        (("source: test guide\n", ""), "missing 'source'"),
        (('"^Notice"', '"^(Notice"'), "doesn't compile"),
    ],
)
def test_rejects_a_broken_source_file(tmp_path, change, error):
    with pytest.raises(ValueError, match=error):
        load_source_spec(write_spec(tmp_path, GOOD_SPEC.replace(*change)), tmp_path)


def test_the_repo_source_files_load_and_use_known_systems():
    taxonomy = load_taxonomy(REPO / "profile" / "systems.yaml")
    for path in sorted((REPO / "sources").glob("*.yaml")):
        s = load_source_spec(path, REPO)
        assert taxonomy.unknown_ids(sorted(s.system_ids())) == [], path.name


# --- ingesting ---


def test_ingest_builds_cited_tagged_passages():
    pages = [("GUIDE\nFit the carb onto the studs evenly.", "1"), ("GUIDE\nSet the idle speed and idle mixture.", "2")]
    passages, report = ingest_source(spec(), TAXONOMY, pages)
    assert [(p.id, p.page, p.system, p.pdf_page) for p in passages] == [
        ("guide:p1:1", "1", "fuel.carburetor", 1),
        ("guide:p2:1", "2", "tuning.idle", 2),
    ]
    assert passages[0].text == "Fit the carb onto the studs evenly."
    assert report.systems == {"fuel.carburetor": 1, "tuning.idle": 1}


def test_pages_without_text_are_reported_as_needing_ocr():
    _, report = ingest_source(spec(), TAXONOMY, [("Some real text on this page.", "1"), ("  ", "2")])
    assert report.no_text == [2]


def test_every_page_needs_a_systems_entry():
    with pytest.raises(ValueError, match="no systems entry covers PDF pages 3-4"):
        ingest_source(spec(systems=(SystemRange(1, 2, "fuel"),)), TAXONOMY, [("Text here.", str(n)) for n in range(1, 5)])


def test_rejects_systems_the_taxonomy_lacks():
    with pytest.raises(ValueError, match="fuel.carberetor"):
        ingest_source(spec(systems=(SystemRange(1, 9, "fuel.carberetor"),)), TAXONOMY, [])


def test_a_missing_pdf_says_where_it_belongs():
    with pytest.raises(FileNotFoundError, match="put your copy of the test guide in manuals/"):
        ingest_source(spec(), TAXONOMY)


def make_pdf(path: Path, pages: list[list[str]]) -> None:
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    for lines in pages:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        stream = DecodedStreamObject()
        stream.set_data(" ".join(["BT /F1 12 Tf 72 720 Td", *(f"({line}) Tj 0 -14 Td" for line in lines), "ET"]).encode())
        page.replace_contents(stream)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer.write(path)


def test_cli_ingests_a_pdf_end_to_end(tmp_path, capsys, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    make_pdf(tmp_path / "manuals" / "guide.pdf", [
        ["TEST GUIDE", "Fit the carburetor onto the four studs."],
        ["TEST GUIDE", "Set the idle speed with the engine warm."],
    ])
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "guide.yaml").write_text(
        "id: guide\nsource: test guide\ntier: 2\nfile: manuals/guide.pdf\n"
        "systems:\n  - {pages: 1-2, default: fuel.carburetor, allow: [tuning.idle]}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert main(["--profile", str(REPO / "profile" / "mgb-1973.yaml")]) == 0
    assert "2 PDF pages -> 2 passages" in capsys.readouterr().out
    passages = load_index(tmp_path / "index" / "passages.jsonl")
    assert [(p.page, p.system, p.text) for p in passages] == [
        ("1", "fuel.carburetor", "Fit the carburetor onto the four studs."),
        ("2", "tuning.idle", "Set the idle speed with the engine warm."),
    ]
