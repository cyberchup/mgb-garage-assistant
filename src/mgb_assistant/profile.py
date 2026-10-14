"""The car profile: the car as it actually sits. Tier 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from mgb_assistant.taxonomy import covers

CONFIDENCE_LEVELS = ("confirmed", "believed", "unknown")


@dataclass(frozen=True)
class Deviation:
    """A difference from factory spec, and the systems where it outranks other tiers."""

    id: str
    system: str
    confidence: str
    affects: tuple[str, ...]
    demotes_tiers: tuple[int, ...]
    factory_spec: str | None = None
    as_found: str | None = None
    verify_by: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class Unknown:
    """A fact answers depend on that nobody has checked yet. It overrides nothing (rule 4)."""

    id: str
    what: str
    affects: tuple[str, ...]
    check_by: str | None = None


@dataclass(frozen=True)
class Profile:
    vehicle_id: str
    deviations: tuple[Deviation, ...]
    unknowns: tuple[Unknown, ...] = ()
    systems_file: Path | None = None  # the taxonomy this profile's system ids come from

    def overriding(self, tier: int, system: str) -> Deviation | None:
        """The deviation that demotes `tier` for `system`, if any.

        A deviation marked `unknown` overrides nothing (rule 4).
        """
        for deviation in self.deviations:
            if deviation.confidence == "unknown" or tier not in deviation.demotes_tiers:
                continue
            if any(covers(scope, system) for scope in deviation.affects):
                return deviation
        return None

    def system_ids(self) -> set[str]:
        """Every system id the profile refers to."""
        ids = {s for d in self.deviations for s in (d.system, *d.affects)}
        return ids | {s for u in self.unknowns for s in u.affects}


def load_profile(path: Path) -> Profile:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    deviations = []
    for raw in data.get("deviations") or []:
        confidence = raw.get("confidence")
        if confidence not in CONFIDENCE_LEVELS:
            raise ValueError(
                f"{path}: deviation {raw.get('id')!r} has confidence {confidence!r}; "
                f"expected one of {', '.join(CONFIDENCE_LEVELS)}"
            )
        deviations.append(
            Deviation(
                id=raw["id"],
                system=raw["system"],
                confidence=confidence,
                affects=tuple(raw.get("affects") or ()),
                demotes_tiers=tuple(raw.get("demotes_tiers") or ()),
                factory_spec=raw.get("factory_spec"),
                as_found=raw.get("as_found"),
                verify_by=raw.get("verify_by"),
                notes=raw.get("notes"),
            )
        )
    unknowns = []
    for raw in data.get("unknowns") or []:
        if not isinstance(raw, dict) or not raw.get("id") or not raw.get("what") or not raw.get("affects"):
            raise ValueError(f"{path}: each unknown needs an id, what, and the systems it affects; got {raw!r}")
        unknowns.append(Unknown(raw["id"], raw["what"], tuple(raw["affects"]), raw.get("check_by")))
    systems = data.get("systems")
    return Profile(
        vehicle_id=data["vehicle"]["id"],
        deviations=tuple(deviations),
        unknowns=tuple(unknowns),
        systems_file=path.parent / systems if systems else None,
    )
