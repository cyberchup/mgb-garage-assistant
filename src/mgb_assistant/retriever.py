"""The tiered retriever: page-cited passages for a question, ranked by tier within each system.

It runs after the profile resolver. Passages the profile demotes are set aside, never
returned as hits, and so are community passages that would stand alone on a
safety-critical job.

Command line: python -m mgb_assistant.retriever "What's the float level on the Weber?"
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mgb_assistant.answer import Citation
from mgb_assistant.paths import DEFAULT_INDEX, DEFAULT_PROFILE, find_root
from mgb_assistant.resolver import Resolution, Resolver, load_resolver
from mgb_assistant.sources import Passage, load_index
from mgb_assistant.taxonomy import Taxonomy, overlaps, words

STOPWORDS = frozenset(
    words(
        "a an and are as at be by can do does for from has have how i if in into is it its me my "
        "of on or should that the their there this to was what when where which will with you your"
    )
)
SYSTEM_BOOST = 1.5  # passages on a system the question touches


def terms(text: str, taxonomy: Taxonomy) -> list[str]:
    """Search terms: the taxonomy's folded words, minus stopwords."""
    return [w for w in taxonomy.normalize(text) if w not in STOPWORDS]


class Bm25:
    """Okapi BM25 keyword relevance over pre-split documents."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.average = max(sum(self.lengths) / len(docs), 1.0) if docs else 1.0
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (len(docs) - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: Sequence[str]) -> list[float]:
        query_terms = [t for t in dict.fromkeys(query) if t in self.idf]
        scores = []
        for tf, length in zip(self.tf, self.lengths):
            score = 0.0
            for t in query_terms:
                f = tf.get(t, 0)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * length / self.average)
                    score += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            scores.append(score)
        return scores


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float

    def citation(self) -> Citation:
        p = self.passage
        return Citation(tier=p.tier, source=p.source, page=p.page, system=p.system)


@dataclass(frozen=True)
class SetAside:
    """A relevant passage the answer must not rest on. It can still be named in a conflict line."""

    hit: Hit
    reason: str


@dataclass(frozen=True)
class Retrieval:
    resolution: Resolution
    hits: tuple[Hit, ...]
    set_aside: tuple[SetAside, ...]


class Retriever:
    def __init__(self, passages: Sequence[Passage], resolver: Resolver):
        unknown = resolver.taxonomy.unknown_ids(sorted({p.system for p in passages}))
        if unknown:
            raise ValueError(f"the index uses systems the taxonomy lacks: {', '.join(unknown)}")
        self.passages = tuple(passages)
        self.resolver = resolver
        self._bm25 = Bm25([terms(p.text, resolver.taxonomy) for p in self.passages])

    def without_profile(self) -> Retriever:
        """The same retriever with the profile switched off, for the ablation."""
        clone = copy.copy(self)
        clone.resolver = self.resolver.without_profile()
        return clone

    def search(self, question: str, *, k: int = 5) -> Retrieval:
        resolution = self.resolver.resolve(question)
        hits: list[Hit] = []
        set_aside: list[SetAside] = []
        for passage, score in zip(self.passages, self._bm25.scores(terms(question, self.resolver.taxonomy))):
            if score <= 0:
                continue
            if any(overlaps(passage.system, s) for s in resolution.systems):
                score *= SYSTEM_BOOST
            hit = Hit(passage, score)
            deviation = self.resolver.demoted(passage.tier, passage.system)
            if deviation:
                set_aside.append(SetAside(hit, f"demoted by profile entry {deviation.id}"))
            else:
                hits.append(hit)

        # Rule 6: on a safety-critical system, community sources never stand alone.
        safety = [h for h in hits if self.resolver.taxonomy.is_safety_critical(h.passage.system)]
        if safety and all(h.passage.tier == 4 for h in safety):
            set_aside += [SetAside(h, "community source can't stand alone on a safety-critical job") for h in safety]
            hits = [h for h in hits if h not in safety]

        set_aside.sort(key=lambda s: -s.hit.score)
        return Retrieval(resolution, tuple(_rank(hits)[:k]), tuple(set_aside[:k]))


def _rank(hits: list[Hit]) -> list[Hit]:
    """The most relevant system first. Within a system, the higher tier wins (rule 1), then relevance."""
    groups: dict[str, list[Hit]] = {}
    for hit in hits:
        groups.setdefault(hit.passage.system, []).append(hit)
    ordered = sorted(groups.values(), key=lambda group: -max(h.score for h in group))
    return [h for group in ordered for h in sorted(group, key=lambda h: (h.passage.tier, -h.score))]


def load_retriever(index_path: Path, profile_path: Path) -> Retriever:
    if not index_path.is_file():
        raise FileNotFoundError(f"no index at {index_path} yet; ingest a source first")
    return Retriever(load_index(index_path), load_resolver(profile_path))


def _describe(p: Passage) -> str:
    page = f" p.{p.page}" if p.page else ""
    return f"tier {p.tier}  {p.source}{page}  {p.system}"


def format_retrieval(r: Retrieval) -> str:
    lines = [f"Question: {r.resolution.question}", f"Systems:  {', '.join(r.resolution.systems) or 'none'}"]
    lines.append("Hits:" if r.hits else "Hits: none")
    for i, hit in enumerate(r.hits, 1):
        snippet = " ".join(hit.passage.text.split())
        lines.append(f"  {i}. {_describe(hit.passage)}  score {hit.score:.2f}")
        lines.append(f"     {snippet[:100]}{'...' if len(snippet) > 100 else ''}")
    if r.set_aside:
        lines.append("Set aside:")
        lines += [f"  - {_describe(s.hit.passage)}: {s.reason}" for s in r.set_aside]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mgb_assistant.retriever",
        description="Show the passages the retriever finds for a question, and what it sets aside.",
    )
    parser.add_argument("question")
    parser.add_argument("--index", type=Path, help="passage index (default: index/passages.jsonl in the repo)")
    parser.add_argument("--profile", type=Path, help="profile YAML (default: profile/mgb-1973.yaml in the repo)")
    parser.add_argument("--no-profile", action="store_true", help="switch the profile off, as in the eval ablation")
    parser.add_argument("-k", type=int, default=5, help="how many hits to show (default 5)")
    args = parser.parse_args(argv)
    try:
        root = find_root(Path.cwd()) if not (args.index and args.profile) else None
        retriever = load_retriever(args.index or root / DEFAULT_INDEX, args.profile or root / DEFAULT_PROFILE)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.no_profile:
        retriever = retriever.without_profile()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # a Windows console can't show every character in a PDF
    print(format_retrieval(retriever.search(args.question, k=args.k)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
