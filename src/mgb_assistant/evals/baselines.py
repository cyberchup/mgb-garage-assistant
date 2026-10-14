"""Stand-in assistants for exercising the harness before the real pipeline exists."""

from __future__ import annotations

from mgb_assistant.answer import Answer, Citation
from mgb_assistant.resolver import Resolver


class NullAssistant:
    """Abstains on every question: the floor any real assistant has to beat."""

    model_id = "none"

    def answer(self, question: str, *, use_profile: bool) -> Answer:
        return Answer(
            full="No source covers this yet.",
            spoken="I don't have a source for that yet.",
            abstained=True,
        )


class ProfileOnlyAssistant:
    """Says only what the profile resolver knows, with no retrieval.

    It measures what the profile alone is worth: with the profile switched off it knows
    nothing, so the ablation gap is all profile.
    """

    model_id = "none"

    def __init__(self, resolver: Resolver):
        self.resolver = resolver

    def answer(self, question: str, *, use_profile: bool) -> Answer:
        resolver = self.resolver if use_profile else self.resolver.without_profile()
        resolution = resolver.resolve(question)
        notes = resolution.notes()
        entries: list[str] = []
        sentences = [] if resolution.overrides else ["No source covers this yet."]
        for note in notes:
            if note.entry_id is None:
                sentences.append(note.text)
                continue
            if note.entry_id not in entries:
                entries.append(note.entry_id)
            sentences.append(f"{note.text.rstrip('.')} [{entries.index(note.entry_id) + 1}].")
        return Answer(
            full=" ".join(sentences),
            spoken=notes[0].text if resolution.overrides else "I don't have a source for that yet.",
            citations=tuple(Citation(tier=1, source="profile", entry_id=e) for e in entries),
            conflict=" ".join(o.conflict for o in resolution.overrides) or None,
            safety_flag=resolution.safety_critical,
            abstained=not resolution.overrides,
        )
