"""Stand-in assistants for exercising the harness before the real pipeline exists."""

from __future__ import annotations

from mgb_assistant.answer import Answer


class NullAssistant:
    """Abstains on every question: the floor any real assistant has to beat."""

    model_id = "none"

    def answer(self, question: str, *, use_profile: bool) -> Answer:
        return Answer(
            full="No source covers this yet.",
            spoken="I don't have a source for that yet.",
            abstained=True,
        )
