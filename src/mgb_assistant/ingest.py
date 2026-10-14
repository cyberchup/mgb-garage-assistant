"""Ingesting source documents into the passage index.

Each source has a metadata file in sources/ (committed: page references and systems, no
document text) pointing at a PDF in manuals/ (gitignored). Ingestion reads every source
and rebuilds index/passages.jsonl from scratch.

Command line: python -m mgb_assistant.ingest
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pypdf import PdfReader

from mgb_assistant.paths import DEFAULT_INDEX, DEFAULT_PROFILE, find_root
from mgb_assistant.profile import load_profile
from mgb_assistant.sources import INDEX_TIERS, Passage, write_index
from mgb_assistant.taxonomy import Taxonomy, covers, load_taxonomy

MAX_WORDS = 120  # a passage's length cap, so a hit points at one job rather than a whole page
MIN_WORDS = 4  # shorter fragments, like a stray caption, aren't worth indexing
REPEATED_SHARE = 0.5  # a line on at least this share of pages is a running header or footer

_PRIVATE_USE = re.compile("[-]")  # symbol-font glyphs, usually bullets
_TYPOGRAPHY = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


@dataclass(frozen=True)
class SystemRange:
    """PDF pages whose passages get `default`, or an `allow`ed system that fits their words better."""

    first: int
    last: int
    default: str
    allow: tuple[str, ...] = ()


@dataclass(frozen=True)
class LabelRange:
    """PDF pages whose printed numbers run prefix+start, prefix+start+1, and so on."""

    first: int
    last: int
    start: int = 1
    prefix: str = ""


@dataclass(frozen=True)
class SourceSpec:
    id: str
    source: str  # the name citations show
    tier: int
    file: Path
    systems: tuple[SystemRange, ...]
    labels: tuple[LabelRange, ...] = ()
    drop_lines: tuple[re.Pattern[str], ...] = ()
    spec_path: Path | None = None

    def system_range(self, pdf_page: int) -> SystemRange | None:
        return next((r for r in self.systems if r.first <= pdf_page <= r.last), None)

    def label(self, pdf_page: int, pdf_label: str) -> str:
        """The printed page number: an explicit label range wins, then the PDF's own label."""
        for r in self.labels:
            if r.first <= pdf_page <= r.last:
                return f"{r.prefix}{r.start + pdf_page - r.first}"
        return pdf_label

    def system_ids(self) -> set[str]:
        return {s for r in self.systems for s in (r.default, *r.allow)}


def _page_range(value: object, where: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", str(value))
    if not m:
        raise ValueError(f"{where}: pages should look like 4 or 4-12, got {value!r}")
    first, last = int(m[1]), int(m[2] or m[1])
    if first < 1 or last < first:
        raise ValueError(f"{where}: page range {value!r} is backwards or starts below 1")
    return first, last


def _check_no_overlap(ranges: Sequence[SystemRange | LabelRange], where: str) -> None:
    ordered = sorted(ranges, key=lambda r: r.first)
    for a, b in zip(ordered, ordered[1:]):
        if b.first <= a.last:
            raise ValueError(f"{where}: pages {a.first}-{a.last} and {b.first}-{b.last} overlap")


def load_source_spec(path: Path, root: Path) -> SourceSpec:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for key in ("id", "source", "tier", "file", "systems"):
        if key not in data:
            raise ValueError(f"{path}: missing {key!r}")
    if data["tier"] not in INDEX_TIERS:
        raise ValueError(f"{path}: tier {data['tier']} can't be indexed; documents are tiers 2-4")

    systems = []
    for i, entry in enumerate(data["systems"], 1):
        where = f"{path}: systems entry {i}"
        first, last = _page_range(entry.get("pages"), where)
        if not entry.get("default"):
            raise ValueError(f"{where}: missing a default system")
        systems.append(SystemRange(first, last, entry["default"], tuple(entry.get("allow") or ())))
    _check_no_overlap(systems, f"{path}: systems")

    labels = []
    for i, entry in enumerate(data.get("labels") or [], 1):
        first, last = _page_range(entry.get("pages"), f"{path}: labels entry {i}")
        labels.append(LabelRange(first, last, int(entry.get("start", 1)), str(entry.get("prefix", ""))))
    _check_no_overlap(labels, f"{path}: labels")

    try:
        drop_lines = tuple(re.compile(p) for p in data.get("drop_lines") or ())
    except re.error as exc:
        raise ValueError(f"{path}: drop_lines pattern doesn't compile: {exc}") from exc

    return SourceSpec(
        id=data["id"],
        source=data["source"],
        tier=data["tier"],
        file=root / data["file"],
        systems=tuple(systems),
        labels=tuple(labels),
        drop_lines=drop_lines,
        spec_path=path,
    )


def _clean_line(line: str) -> str:
    return " ".join(_PRIVATE_USE.sub("", line).translate(_TYPOGRAPHY).split())


def _line_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.lower())  # "Page 2" and "Page 3" are the same header


def clean_pages(pages: Sequence[str], drop_lines: Sequence[re.Pattern[str]] = ()) -> list[str]:
    """Strip symbol glyphs, plain the typography, and drop running headers and footers."""
    split = [[_clean_line(line) for line in page.splitlines()] for page in pages]
    pages_with = Counter(key for lines in split for key in {_line_key(line) for line in lines if line})
    threshold = max(2, math.ceil(REPEATED_SHARE * len(pages)))
    repeated = {key for key, n in pages_with.items() if n >= threshold}

    def keep(line: str) -> bool:
        return not line or (_line_key(line) not in repeated and not any(p.search(line) for p in drop_lines))

    return ["\n".join(line for line in lines if keep(line)) for lines in split]


def _is_heading(paragraph: str) -> bool:
    """A short run of capitals that doesn't end a sentence, like "CARBURETOR INSTALLATION".

    Commas rule a line out: "PART # WK050, WK054, WK055" is a list, not a heading.
    """
    letters = [c for c in paragraph if c.isalpha()]
    if not letters or len(paragraph.split()) > 10 or paragraph.endswith((".", "!", "?")) or "," in paragraph:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= 0.7


def split_passages(text: str, *, max_words: int = MAX_WORDS, min_words: int = MIN_WORDS) -> list[str]:
    """A page's paragraphs grouped into passages. A heading starts a new passage and prefixes it."""
    passages: list[str] = []
    heading: str | None = None
    current: list[str] = []

    def flush() -> None:
        body = " ".join(current)
        if len(body.split()) >= min_words:
            passages.append(f"{heading}: {body}" if heading else body)
        current.clear()

    for paragraph in (" ".join(block.split()) for block in re.split(r"\n\s*\n", text)):
        if not paragraph:
            continue
        if _is_heading(paragraph):
            flush()
            heading = paragraph.rstrip(":")
            continue
        if current and len(" ".join(current).split()) + len(paragraph.split()) > max_words:
            flush()
        current.append(paragraph)
    flush()
    return passages


def tag_system(text: str, system_range: SystemRange, taxonomy: Taxonomy) -> str:
    """The range's default system, unless an allowed one matches more of the passage's words."""
    candidates = (system_range.default, *system_range.allow)
    counts: Counter[str] = Counter()
    for match in taxonomy.match(text):
        for system in match.systems:
            covering = [c for c in candidates if covers(c, system)]
            if covering:
                counts[max(covering, key=len)] += 1  # credit the most specific candidate
    return max(candidates, key=lambda c: (counts[c], c == system_range.default))


def read_pdf(path: Path) -> list[tuple[str, str]]:
    """Each page's text and the PDF's own page label (its page number when it has none)."""
    reader = PdfReader(path)
    labels = reader.page_labels
    return [(page.extract_text() or "", labels[i]) for i, page in enumerate(reader.pages)]


@dataclass
class SourceReport:
    spec: SourceSpec
    pages: int = 0
    passages: int = 0
    no_text: list[int] = field(default_factory=list)  # PDF pages without a text layer
    systems: Counter[str] = field(default_factory=Counter)


def ingest_source(
    spec: SourceSpec, taxonomy: Taxonomy, pages: Sequence[tuple[str, str]] | None = None
) -> tuple[list[Passage], SourceReport]:
    """Passages from one source. `pages` (text, PDF label) stands in for reading the PDF."""
    unknown = taxonomy.unknown_ids(sorted(spec.system_ids()))
    if unknown:
        raise ValueError(f"{spec.spec_path}: systems the taxonomy lacks: {', '.join(unknown)}")
    if pages is None:
        if not spec.file.is_file():
            raise FileNotFoundError(f"{spec.file} not found; put your copy of the {spec.source} in manuals/")
        pages = read_pdf(spec.file)
    uncovered = [n for n in range(1, len(pages) + 1) if spec.system_range(n) is None]
    if uncovered:
        raise ValueError(f"{spec.spec_path}: no systems entry covers PDF pages {_ranges(uncovered)}")

    report = SourceReport(spec, pages=len(pages))
    passages: list[Passage] = []
    texts = clean_pages([text for text, _ in pages], spec.drop_lines)
    for n, ((raw, pdf_label), text) in enumerate(zip(pages, texts), 1):
        if not raw.strip():
            report.no_text.append(n)
            continue
        system_range = spec.system_range(n)
        for i, body in enumerate(split_passages(text), 1):
            system = tag_system(body, system_range, taxonomy)
            passages.append(
                Passage(
                    id=f"{spec.id}:p{n}:{i}",
                    source_id=spec.id,
                    source=spec.source,
                    tier=spec.tier,
                    page=spec.label(n, pdf_label),
                    system=system,
                    text=body,
                    pdf_page=n,
                )
            )
            report.systems[system] += 1
    report.passages = len(passages)
    return passages, report


def ingest(spec_paths: Sequence[Path], taxonomy: Taxonomy, root: Path) -> tuple[list[Passage], list[SourceReport]]:
    specs = [load_source_spec(p, root) for p in spec_paths]
    ids = Counter(s.id for s in specs)
    duplicated = sorted(i for i, n in ids.items() if n > 1)
    if duplicated:
        raise ValueError(f"source ids used twice: {', '.join(duplicated)}")
    passages: list[Passage] = []
    reports = []
    for spec in specs:
        source_passages, report = ingest_source(spec, taxonomy)
        passages += source_passages
        reports.append(report)
    return passages, reports


def _ranges(pages: Sequence[int]) -> str:
    """[3, 5, 6, 7] -> "3, 5-7"."""
    spans: list[list[int]] = []
    for n in pages:
        if spans and n == spans[-1][1] + 1:
            spans[-1][1] = n
        else:
            spans.append([n, n])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in spans)


def format_report(reports: Sequence[SourceReport], index_path: Path, total: int) -> str:
    lines = []
    for r in reports:
        spec_name = r.spec.spec_path.name if r.spec.spec_path else r.spec.id
        lines.append(f"{r.spec.source} (tier {r.spec.tier}, {spec_name})")
        lines.append(f"  {r.pages} PDF pages -> {r.passages} passages")
        lines.append("  systems: " + ", ".join(f"{s} {n}" for s, n in r.systems.most_common()))
        if r.no_text:
            lines.append(f"  no text layer on PDF pages {_ranges(r.no_text)}: a scan needs OCR before ingesting")
    lines.append(f"Index written to {index_path} ({total} passages)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mgb_assistant.ingest",
        description="Rebuild the passage index from every source file in sources/.",
    )
    parser.add_argument("--sources", type=Path, help="folder of source files (default: sources/ in the repo)")
    parser.add_argument("--out", type=Path, help="index to write (default: index/passages.jsonl in the repo)")
    parser.add_argument("--profile", type=Path, help="profile whose systems file tags passages")
    args = parser.parse_args(argv)
    try:
        root = find_root(Path.cwd())
        spec_paths = sorted((args.sources or root / "sources").glob("*.yaml"))
        if not spec_paths:
            raise FileNotFoundError(f"no source files in {args.sources or root / 'sources'}")
        profile = load_profile(args.profile or root / DEFAULT_PROFILE)
        if profile.systems_file is None:
            raise ValueError("the profile doesn't name its systems file")
        passages, reports = ingest(spec_paths, load_taxonomy(profile.systems_file), root)
        out = args.out or root / DEFAULT_INDEX
        write_index(passages, out)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_report(reports, out, len(passages)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
