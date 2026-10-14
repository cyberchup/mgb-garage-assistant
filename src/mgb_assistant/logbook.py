"""The maintenance log: what was done to the car, when, and what was measured. Tier 1.

Entries live in log/maintenance.yaml (gitignored: it's personal), one YAML list item per job,
in the format of schemas/log-entry.example.yaml. The file is append-only.

Rule 7: a spoken entry is a draft until the owner accepts its read-back, and only then is it
written. An unconfirmed entry in the file, added by hand say, never feeds an answer.

Command line: python -m mgb_assistant.logbook add|list|proposals
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import yaml

from mgb_assistant.answer import Answer, Citation
from mgb_assistant.paths import DEFAULT_LOG, DEFAULT_PROFILE, find_root
from mgb_assistant.profile import load_profile
from mgb_assistant.taxonomy import Taxonomy, load_taxonomy, overlaps

SOURCES = ("text", "voice")
NO_RECORD = "No record of that in the maintenance log."
NO_RECORD_SPOKEN = "There's no record of that in the log."
LOG_HEADER = (
    "# Maintenance log: one entry per job. Tier 1, alongside the car profile.\n"
    "# Written by `python -m mgb_assistant.logbook add`; append-only. Format: schemas/log-entry.example.yaml.\n"
)
_ID = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}$")
_MEASURE = re.compile(r"^([a-z][a-z0-9_]*)=(\d*\.?\d+)\s*([A-Za-z°%]\S*)$")  # a unit starts with a letter, ° or %
_UNIT_ALIASES = {"thou": ("in", 0.001)}  # "15 thou" is logged as 0.015 in


class LogError(ValueError):
    pass


def _fmt(value: float) -> str:
    return f"{value:g}"


@dataclass(frozen=True)
class SpecRef:
    """The spec a measurement was checked against: a value's source and page, never its text."""

    tier: int
    source: str
    page: str | None = None


@dataclass(frozen=True)
class Measurement:
    name: str
    value: float
    unit: str
    spec_ref: SpecRef | None = None

    def describe(self) -> str:
        return f"{self.name.replace('_', ' ')} {_fmt(self.value)} {self.unit}"


@dataclass(frozen=True)
class ProfileUpdate:
    """A profile unknown or deviation this entry settles. The owner approves it before the profile changes."""

    target: str
    value: str


@dataclass(frozen=True)
class LogEntry:
    id: str
    date: date
    system: str
    action: str
    source: str = "text"
    confirmed: bool = True
    odometer_mi: int | None = None
    parts: tuple[str, ...] = ()
    measurements: tuple[Measurement, ...] = ()
    observations: str | None = None
    proposes_profile_update: ProfileUpdate | None = None

    def to_dict(self) -> dict:
        """Field order follows schemas/log-entry.example.yaml."""
        measurements = []
        for m in self.measurements:
            item: dict = {"name": m.name, "value": m.value, "unit": m.unit}
            if m.spec_ref:
                item["spec_ref"] = {"tier": m.spec_ref.tier, "source": m.spec_ref.source, "page": m.spec_ref.page}
            measurements.append(item)
        update = self.proposes_profile_update
        return {
            "id": self.id,
            "date": self.date,
            "odometer_mi": self.odometer_mi,
            "system": self.system,
            "action": self.action,
            "parts": list(self.parts),
            "measurements": measurements,
            "observations": self.observations,
            "source": self.source,
            "confirmed": self.confirmed,
            "proposes_profile_update": {"target": update.target, "value": update.value} if update else None,
        }


def parse_entry(raw: object, where: str) -> LogEntry:
    if not isinstance(raw, dict):
        raise LogError(f"{where}: an entry should be a mapping")
    entry_id, day = raw.get("id"), raw.get("date")
    where = f"{where} ({entry_id})"
    if not isinstance(entry_id, str) or not _ID.match(entry_id):
        raise LogError(f"{where}: id should look like 2026-10-20-01")
    if not isinstance(day, date) or entry_id[:10] != day.isoformat():
        raise LogError(f"{where}: date should be a date matching the id")
    for key in ("system", "action"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise LogError(f"{where}: missing {key}")
    if raw.get("source", "text") not in SOURCES:
        raise LogError(f"{where}: source should be one of {', '.join(SOURCES)}")
    if not isinstance(raw.get("confirmed", True), bool):
        raise LogError(f"{where}: confirmed should be true or false")
    odometer = raw.get("odometer_mi")
    if odometer is not None and (not isinstance(odometer, int) or isinstance(odometer, bool) or odometer < 0):
        raise LogError(f"{where}: odometer_mi should be a whole number of miles")

    measurements = []
    for m in raw.get("measurements") or []:
        value = m.get("value") if isinstance(m, dict) else None
        if not isinstance(m, dict) or not m.get("name") or not isinstance(value, (int, float)) or isinstance(value, bool):
            raise LogError(f"{where}: each measurement needs a name and a numeric value")
        if not m.get("unit"):
            raise LogError(f"{where}: measurement {m['name']} needs a unit")
        ref = m.get("spec_ref")
        spec_ref = None
        if ref:
            if ref.get("tier") not in (1, 2, 3, 4) or not ref.get("source"):
                raise LogError(f"{where}: spec_ref for {m['name']} needs a tier (1-4) and a source")
            spec_ref = SpecRef(ref["tier"], ref["source"], None if ref.get("page") is None else str(ref["page"]))
        measurements.append(Measurement(m["name"], float(value), str(m["unit"]), spec_ref))

    update = raw.get("proposes_profile_update")
    if update is not None and (not isinstance(update, dict) or not update.get("target") or update.get("value") is None):
        raise LogError(f"{where}: proposes_profile_update needs a target and a value")
    return LogEntry(
        id=entry_id,
        date=day,
        system=raw["system"],
        action=raw["action"].strip(),
        source=raw.get("source", "text"),
        confirmed=raw.get("confirmed", True),
        odometer_mi=odometer,
        parts=tuple(str(p) for p in raw.get("parts") or ()),
        measurements=tuple(measurements),
        observations=raw.get("observations"),
        proposes_profile_update=ProfileUpdate(update["target"], str(update["value"])) if update else None,
    )


def parse_measurement(text: str) -> Measurement:
    """'points_gap=0.015in' or 'points_gap=15thou' (logged as 0.015 in)."""
    m = _MEASURE.match(text.strip())
    if not m:
        raise LogError(f"measurement {text!r} should look like points_gap=0.015in")
    value, unit = float(m[2]), m[3]
    if unit in _UNIT_ALIASES:
        unit, factor = _UNIT_ALIASES[unit]
        value = round(value * factor, 6)
    return Measurement(m[1], value, unit)


def readback(entry: LogEntry) -> str:
    """What the owner hears or sees before saying yes (rule 7): every number, spelled out in full."""
    lines = [f"Log for {entry.date.isoformat()}: {entry.action} ({entry.system})."]
    lines += [f"{m.describe()}." for m in entry.measurements]
    if entry.odometer_mi is not None:
        lines.append(f"Odometer {entry.odometer_mi} mi.")
    if entry.parts:
        lines.append(f"Parts: {', '.join(entry.parts)}.")
    if entry.observations:
        lines.append(f"Notes: {entry.observations}")
    if entry.proposes_profile_update:
        u = entry.proposes_profile_update
        lines.append(f"Proposes a profile update: {u.target} = {u.value} (you approve it separately).")
    return "\n".join(lines)


class LogBook:
    def __init__(self, path: Path | None, taxonomy: Taxonomy, profile_ids: Collection[str] = ()):
        self.path = path  # None is an empty, read-only log
        self.taxonomy = taxonomy
        self.profile_ids = set(profile_ids)

    def entries(self) -> list[LogEntry]:
        """Every entry in the file, confirmed or not. A missing file is an empty log."""
        if self.path is None or not self.path.is_file():
            return []
        data = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if data is None:
            return []
        if not isinstance(data, list):
            raise LogError(f"{self.path}: the log should be a list of entries")
        entries = [parse_entry(raw, f"{self.path}: entry {i}") for i, raw in enumerate(data, 1)]
        seen: set[str] = set()
        for entry in entries:
            self._check(entry)
            if entry.id in seen:
                raise LogError(f"{self.path}: entry id {entry.id} appears twice")
            seen.add(entry.id)
        return entries

    def _check(self, entry: LogEntry) -> None:
        if self.taxonomy.unknown_ids([entry.system]):
            raise LogError(f"{entry.id}: system {entry.system!r} isn't in the taxonomy (profile/systems.yaml)")
        update = entry.proposes_profile_update
        if update and self.profile_ids and update.target not in self.profile_ids:
            raise LogError(f"{entry.id}: {update.target!r} isn't a profile unknown or deviation")

    def confirmed(self) -> list[LogEntry]:
        """The entries that may feed answers (rule 7)."""
        return [e for e in self.entries() if e.confirmed]

    def next_id(self, day: date) -> str:
        taken = [e.id for e in self.entries() if e.date == day]
        return f"{day.isoformat()}-{len(taken) + 1:02d}"

    def write(self, entry: LogEntry) -> LogEntry:
        """Append a confirmed entry. Drafts are never written (rule 7): confirm them first."""
        if self.path is None:
            raise LogError("this log has no file to write to")
        if not entry.confirmed:
            raise LogError(f"{entry.id}: only confirmed entries are written; read it back and confirm first")
        self._check(entry)
        if any(e.id == entry.id for e in self.entries()):
            raise LogError(f"entry id {entry.id} is already in the log")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.is_file()
        ends_cleanly = new or self.path.read_bytes().endswith(b"\n")
        block = yaml.safe_dump([entry.to_dict()], sort_keys=False, allow_unicode=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(LOG_HEADER if new else ("" if ends_cleanly else "\n"))
            f.write("\n" + block)
        return entry

    def confirm(self, draft: LogEntry) -> LogEntry:
        """The owner said yes to the read-back: mark the draft confirmed and write it."""
        return self.write(replace(draft, confirmed=True))

    def latest(self, systems: Sequence[str]) -> LogEntry | None:
        matching = [e for e in self.confirmed() if any(overlaps(e.system, s) for s in systems)]
        return max(matching, key=lambda e: e.id, default=None)

    def history_answer(self, question: str) -> Answer:
        """'When did I last...' from the log alone, citing the entry (never the manual)."""
        systems = [s for m in self.taxonomy.match(question) for s in m.systems]
        entry = self.latest(systems) if systems else None
        if entry is None:
            return Answer(full=NO_RECORD, spoken=NO_RECORD_SPOKEN, abstained=True)
        day, action = entry.date.isoformat(), entry.action.rstrip(".")
        sentences = [f"Last logged on {day}: {action} [1]."]
        if entry.measurements:
            sentences.append(f"Recorded {', '.join(m.describe() for m in entry.measurements)} [1].")
        if entry.odometer_mi is not None:
            sentences.append(f"Odometer {entry.odometer_mi} mi [1].")
        if entry.observations:
            sentences.append(f"Notes: {entry.observations.rstrip('.')} [1].")
        return Answer(
            full=" ".join(sentences),
            spoken=f"You last logged that on {day}: {action}.",
            citations=(Citation(tier=1, source="log", entry_id=entry.id, system=entry.system),),
        )

    def proposals(self) -> list[LogEntry]:
        """Confirmed entries proposing a profile update, waiting for the owner's approval."""
        return [e for e in self.confirmed() if e.proposes_profile_update]


def load_logbook(log_path: Path | None, profile_path: Path) -> LogBook:
    profile = load_profile(profile_path)
    if profile.systems_file is None:
        raise LogError(f"{profile_path}: the profile doesn't name its systems file")
    ids = {d.id for d in profile.deviations} | {u.id for u in profile.unknowns}
    return LogBook(log_path, load_taxonomy(profile.systems_file), ids)


def _ask(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mgb_assistant.logbook", description="Read and write the maintenance log.")
    parser.add_argument("--log", type=Path, help="log file (default: log/maintenance.yaml in the repo)")
    parser.add_argument("--profile", type=Path, help="profile YAML (default: profile/mgb-1973.yaml in the repo)")
    commands = parser.add_subparsers(dest="command", required=True)

    add = commands.add_parser("add", help="add an entry; it's read back before anything is written")
    add.add_argument("--system", required=True, help="a system id from profile/systems.yaml, e.g. ignition.points")
    add.add_argument("--action", required=True, help='what was done, e.g. "set points gap"')
    add.add_argument("--measure", action="append", default=[], metavar="NAME=VALUEUNIT", help="e.g. points_gap=0.015in")
    add.add_argument("--date", type=date.fromisoformat, default=None, help="YYYY-MM-DD (default: today)")
    add.add_argument("--odometer", type=int, help="odometer reading in miles")
    add.add_argument("--part", action="append", default=[], help="a part fitted; repeatable")
    add.add_argument("--note", help="observations")
    add.add_argument("--proposes", metavar="TARGET=VALUE", help="settle a profile unknown, e.g. weber-variant=DGV")
    add.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    listing = commands.add_parser("list", help="list confirmed entries")
    listing.add_argument("--all", action="store_true", help="include unconfirmed entries")
    commands.add_parser("proposals", help="list profile updates waiting for approval")
    args = parser.parse_args(argv)

    try:
        root = find_root(Path.cwd()) if not (args.log and args.profile) else None
        logbook = load_logbook(args.log or root / DEFAULT_LOG, args.profile or root / DEFAULT_PROFILE)
        if args.command == "add":
            day = args.date or date.today()
            update = None
            if args.proposes:
                target, _, value = args.proposes.partition("=")
                if not value:
                    raise LogError("--proposes should look like weber-variant=DGV")
                update = ProfileUpdate(target.strip(), value.strip())
            draft = LogEntry(
                id=logbook.next_id(day),
                date=day,
                system=args.system,
                action=args.action,
                source="text",
                confirmed=False,
                odometer_mi=args.odometer,
                parts=tuple(args.part),
                measurements=tuple(parse_measurement(m) for m in args.measure),
                observations=args.note,
                proposes_profile_update=update,
            )
            logbook._check(draft)
            print(readback(draft))
            if not (args.yes or _ask("Write it? [y/N] ")):
                print("Not written.")
                return 1
            entry = logbook.confirm(draft)
            print(f"Written as {entry.id} in {logbook.path}")
        elif args.command == "list":
            entries = logbook.entries() if args.all else logbook.confirmed()
            for e in entries:
                flag = "" if e.confirmed else "  [unconfirmed]"
                measured = "; ".join(m.describe() for m in e.measurements)
                print(f"{e.id}  {e.system:<20} {e.action}" + (f" ({measured})" if measured else "") + flag)
            if not entries:
                print("The log is empty.")
        else:
            for e in logbook.proposals():
                u = e.proposes_profile_update
                print(f"{e.id}  {u.target} = {u.value}")
            if not logbook.proposals():
                print("No profile updates waiting.")
    except (OSError, ValueError) as exc:  # LogError is a ValueError
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
