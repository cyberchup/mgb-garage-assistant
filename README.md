# MGB Garage Assistant

A local voice-and-text assistant for a 1973 MGB Roadster that trusts the car's own record over the factory manual, and says so when the two disagree.

> Status: early. Phase 0 (groundwork) → Phase 1 (text Q&A).

## Why this isn't just a manual chatbot

This car runs a Weber carburetor instead of the factory twin SUs, so every fuel-system procedure in the workshop manual is confidently wrong for it. The assistant is built around that problem:

- **Source precedence:** a car profile records deviations from factory spec and outranks the manual for the systems they affect.
- **Confidence:** every fact is `confirmed`, `believed` or `unknown`, and answers hedge to match.
- **Grounding:** every answer cites a manual page or a maintenance-log entry. No source, no answer.
- **Voice in a loud garage:** push-to-talk, a custom vocabulary for technical terms, and read-back before any number is logged.

Each of these is measured by an eval set (`evals/v1.yaml`), including a profile ablation that runs the same questions with the profile switched off.

## Running the evals

Requires [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run python -m mgb_assistant.evals            # profile on and off (the ablation)
uv run python -m mgb_assistant.evals --case T1  # one case
uv run python -m mgb_assistant.evals --assistant profile-only
uv run python -m mgb_assistant.resolver "How do I balance the carbs?"
uv run python -m mgb_assistant.retriever "What's the float level on the Weber?"
uv run pytest
```

Each run writes a record to `runs/` (gitignored, since model answers can paraphrase the manual) with the model, the profile setting, the eval file's version and hash, the git commit and the date.

There are two baseline assistants so far. `null` abstains on everything and sets the floor. `profile-only` says only what the profile resolver knows (overrides, unknowns, safety flags) with no retrieval, so its ablation gap is the profile's contribution on its own.

The resolver command shows which systems a question touches, which profile entries override the manual, and what the answer has to say. Systems and the words that point at them live in `profile/systems.yaml`.

The retriever command searches the source index (`index/passages.jsonl`, gitignored because it holds manual text) and shows the hits plus what it set aside: passages the profile demotes, and community sources that would stand alone on a brakes or fuel-line job. Within a system, the higher tier ranks first. Relevance is keyword-based (BM25) for now.

## Bring your own manual

Workshop manuals are copyrighted and aren't included. Put your own PDFs in `manuals/` (gitignored). Scanned paper needs OCR first: the ingester reads the PDF's text layer and reports any page without one.

Each document has a source file in `sources/` (committed). It holds only metadata: the tier, the PDF's path, which systems each page range covers, how PDF pages map to printed page numbers, and patterns for boilerplate lines. `sources/weber-install-guide.yaml` is a worked example for the free Weber conversion kit guide. Then rebuild the index:

```
uv run python -m mgb_assistant.ingest
```

Every PDF page must fall inside a `systems` range. That keeps tagging deliberate, which matters most for the workshop manual: an SU page tagged with the wrong system would escape the Weber override.

## Roadmap

0. Groundwork: confirm the as-built config and ingest the manual
1. Text Q&A with citations and profile overrides
2. Maintenance log and recommissioning workflow
3. Voice logging with read-back
4. Voice Q&A and step mode
5. More vehicles, a wake word, and photo-based part ID
