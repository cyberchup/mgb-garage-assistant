"""The profile resolver: what the car's own record says about a question, before retrieval runs.

Command line: python -m mgb_assistant.resolver "How do I balance the carbs?" [--no-profile]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from mgb_assistant.paths import DEFAULT_PROFILE, find_root
from mgb_assistant.profile import Deviation, Profile, load_profile
from mgb_assistant.taxonomy import Match, Taxonomy, load_taxonomy, overlaps

TIER_NAMES = {1: "car's record", 2: "conversion docs", 3: "workshop manual", 4: "community sources"}
SECOND_SOURCE_PROMPT = "Safety-critical job: check a second source before relying on this."


@dataclass(frozen=True)
class Override:
    """A deviation that outranks lower tiers for the systems the question touches."""

    deviation: Deviation

    @property
    def conflict(self) -> str:
        """The conflict, in one line (rule 3)."""
        d = self.deviation
        if d.factory_spec and d.as_found:
            demoted = " and ".join(TIER_NAMES[t] for t in d.demotes_tiers if t in TIER_NAMES) or "factory spec"
            return f"The {demoted} covers the factory {d.factory_spec}; this car has {d.as_found}."
        return f"This car differs from factory spec on {d.system} (profile entry {d.id})."

    @property
    def verify(self) -> str | None:
        """For a believed fact, the prompt to verify it (rule 4)."""
        d = self.deviation
        if d.confidence != "believed":
            return None
        if d.verify_by:
            return f"That's believed, not confirmed. To confirm: {d.verify_by}."
        return "That's believed, not confirmed."


@dataclass(frozen=True)
class OpenQuestion:
    """An unknown the answer depends on. The answer says so, and how to check (rule 4)."""

    id: str
    what: str
    check_by: str | None

    def prompt(self, *, with_check: bool = True) -> str:
        if self.check_by and with_check:
            return f"Not known yet: {self.what}. To check: {self.check_by}."
        return f"Not known yet: {self.what}."


@dataclass(frozen=True)
class Note:
    """A line the answer has to include."""

    text: str
    entry_id: str | None  # the profile entry to cite; None for the safety prompt


@dataclass(frozen=True)
class Resolution:
    question: str
    matches: tuple[Match, ...]
    systems: tuple[str, ...]
    overrides: tuple[Override, ...]
    open_questions: tuple[OpenQuestion, ...]
    safety_systems: tuple[str, ...]

    @property
    def safety_critical(self) -> bool:
        return bool(self.safety_systems)

    def notes(self) -> list[Note]:
        """What the answer has to say on top of its content, in order."""
        notes = []
        for override in self.overrides:
            notes.append(Note(override.conflict, override.deviation.id))
            if override.verify:
                notes.append(Note(override.verify, override.deviation.id))
        # Give each "how to check" once: reading the carb's stamping settles both a believed
        # deviation and the unknown variant.
        given = {o.deviation.verify_by for o in self.overrides if o.verify}
        notes += [Note(q.prompt(with_check=q.check_by not in given), q.id) for q in self.open_questions]
        if self.safety_critical:
            notes.append(Note(SECOND_SOURCE_PROMPT, None))
        return notes


class Resolver:
    def __init__(self, taxonomy: Taxonomy, profile: Profile | None):
        if profile is not None:
            unknown = taxonomy.unknown_ids(sorted(profile.system_ids()))
            if unknown:
                raise ValueError(f"profile {profile.vehicle_id} uses systems its taxonomy lacks: {', '.join(unknown)}")
        self.taxonomy = taxonomy
        self.profile = profile

    def without_profile(self) -> Resolver:
        """The same resolver with the profile switched off, for the ablation."""
        return Resolver(self.taxonomy, None)

    def demoted(self, tier: int, system: str) -> Deviation | None:
        """The deviation that demotes a passage of this tier and system, if any.

        This holds whatever the question was: a passage on an overridden system is wrong
        for this car even when the question's wording didn't match that system.
        """
        return self.profile.overriding(tier, system) if self.profile else None

    def resolve(self, question: str) -> Resolution:
        matches = tuple(self.taxonomy.match(question))
        systems = tuple(dict.fromkeys(s for m in matches for s in m.systems))

        def touched(affects: tuple[str, ...]) -> bool:
            return any(overlaps(a, s) for a in affects for s in systems)

        overrides: list[Override] = []
        open_questions: list[OpenQuestion] = []
        if self.profile is not None:
            for d in self.profile.deviations:
                if not touched(d.affects):
                    continue
                if d.confidence == "unknown":
                    open_questions.append(OpenQuestion(d.id, d.as_found or d.system, d.verify_by))
                else:
                    overrides.append(Override(d))
            open_questions += [
                OpenQuestion(u.id, u.what, u.check_by) for u in self.profile.unknowns if touched(u.affects)
            ]
        safety = tuple(s for s in systems if self.taxonomy.is_safety_critical(s))
        return Resolution(question, matches, systems, tuple(overrides), tuple(open_questions), safety)


def load_resolver(profile_path: Path) -> Resolver:
    profile = load_profile(profile_path)
    if profile.systems_file is None:
        raise ValueError(f"{profile_path}: the profile doesn't name its systems file")
    return Resolver(load_taxonomy(profile.systems_file), profile)


def format_resolution(r: Resolution) -> str:
    matched = ", ".join(f"{m.term} -> {'/'.join(m.systems)}" for m in r.matches) or "none"
    lines = [f"Question: {r.question}", f"Systems:  {matched}"]
    for o in r.overrides:
        d = o.deviation
        tiers = "/".join(map(str, d.demotes_tiers))
        lines.append(f"Override: {d.id} ({d.confidence}) demotes tier {tiers} on {', '.join(d.affects)}")
    if r.safety_critical:
        lines.append(f"Safety:   {', '.join(r.safety_systems)}")
    notes = r.notes()
    if notes:
        lines.append("The answer must say:")
        lines += [f"  - {n.text}" + (f"  [profile: {n.entry_id}]" if n.entry_id else "") for n in notes]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m mgb_assistant.resolver",
        description="Show what the car profile says about a question, before retrieval.",
    )
    parser.add_argument("question")
    parser.add_argument("--profile", type=Path, help="profile YAML (default: profile/mgb-1973.yaml in the repo)")
    parser.add_argument("--no-profile", action="store_true", help="switch the profile off, as in the eval ablation")
    args = parser.parse_args(argv)
    try:
        resolver = load_resolver(args.profile or find_root(Path.cwd()) / DEFAULT_PROFILE)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.no_profile:
        resolver = resolver.without_profile()
    print(format_resolution(resolver.resolve(args.question)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
