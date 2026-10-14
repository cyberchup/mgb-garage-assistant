"""The car profile: the car as it actually sits. Tier 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIDENCE_LEVELS = ("confirmed", "believed", "unknown")


@dataclass(frozen=True)
class Deviation:
    """A difference from factory spec, and the systems where it outranks other tiers."""

    id: str
    system: str
    confidence: str
    affects: tuple[str, ...]
    demotes_tiers: tuple[int, ...]


@dataclass(frozen=True)
class Profile:
    vehicle_id: str
    deviations: tuple[Deviation, ...]

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


def covers(scope: str, system: str) -> bool:
    """True if `system` is `scope` or sits under it: fuel.carburetor covers fuel.carburetor.float."""
    return system == scope or system.startswith(scope + ".")


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
            )
        )
    return Profile(vehicle_id=data["vehicle"]["id"], deviations=tuple(deviations))
