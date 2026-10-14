# MGB Garage Assistant

A local voice-and-text assistant for one car: a 1973 MGB Roadster. It answers from the car's own record first and the workshop manual second, and says when they disagree. This is a portfolio project. The hard parts are source precedence, confidence, grounding and voice accuracy, and each one is measured by the eval set in `evals/`.

## The car

The source of truth is `profile/mgb-1973.yaml`. In summary:

- 1973 MGB Roadster, chrome bumper, 1798cc B-series.
- It runs a **Weber carb (believed to be a 32/36, unconfirmed)** instead of the factory twin SU HS4s. Factory fuel-system procedures do NOT apply to this car.
- The goal is a reliable driver that keeps improving, not a concours restoration.
- It last ran around October 2025. The first real workflow is recommissioning it.

## Core behavior rules

These are the product's requirements. Every feature and every test must hold to them.

1. **Source tiers.** In order: 1 = the car's record (profile and maintenance log), 2 = conversion docs (Weber), 3 = workshop manual, 4 = community (forums, club articles). A higher tier wins only within the systems it covers.
2. **Overrides.** Each profile deviation lists the systems it `affects`. A question touching one of those systems demotes tier 3 passages for that system.
3. **Conflicts.** When tiers disagree, the answer comes from the higher tier and states the conflict in one line.
4. **Confidence.** A fact marked `believed` still overrides, but the answer adds a prompt to verify it. A fact marked `unknown` overrides nothing; the assistant asks, or says how to check.
5. **Grounding.** Every answer cites a manual page or a log entry. If no source covers the question, the assistant says so. Never output a number without a source.
6. **Safety-critical jobs** (brakes, fuel lines) get a flag and a prompt to check a second source. Tier 4 never stands alone on these jobs.
7. **Voice logging.** Every spoken number is read back, and nothing is written until the user says yes. Unconfirmed log entries never feed answers.

## Architecture

- **Clients:** text chat (phone or laptop), voice (push-to-talk, local speech-to-text and text-to-speech), and the eval harness. The harness replays recorded audio through the voice path.
- **Backend** (local, on an HP EliteDesk or an Unraid server):
  - The orchestrator sorts questions from log commands and returns two forms of every answer: spoken (two sentences at most) and full, with citations.
  - The profile resolver runs before retrieval.
  - The tiered retriever returns page-cited passages.
  - The log writer reads numbers back before writing.
- **Stores:** the car profile, the source index (page-cited, tagged by tier and system), and the maintenance log.
- **Voice:** push-to-talk first. Add a wake word later, kept off while the engine runs. Prime speech-to-text with a vocabulary list built from the profile: Weber, Lucas, SU, thou, float bowl, jets, choke, distributor, points.

## Repo layout

- `profile/`: car profile YAML, and `systems.yaml`: the system ids and the words that point at them. Deviations, unknowns, source passages and questions all use these ids.
- `schemas/`: example maintenance log entry.
- `evals/v1.yaml`: eval questions and their machine checks. The harness reads this file.
- `src/mgb_assistant/`: the Python package. `answer.py` holds the answer contract (citations with tier, printed page and system; `[n]` markers in the text). `resolver.py` is the profile resolver. `evals/` holds the harness and the baseline assistants.
- `tests/`: pytest suite.
- `runs/`: eval run records. Gitignored, since model answers can paraphrase the manual.
- `manuals/`: private source PDFs. Gitignored; **never commit manual content**.

Commands (Python 3.12 via uv): `uv sync`, `uv run pytest`, `uv run python -m mgb_assistant.evals` (profile on and off; `--case ID` for one case; `--assistant profile-only`), `uv run python -m mgb_assistant.resolver "QUESTION"` (`--no-profile` for the ablation view).

## Build phases

Each phase ends on an eval gate, not on a feature list.

0. **Groundwork.** Confirm the Weber variant, the engine number and the ignition type, ingest the manual, and fill in the profile. Gate: no `unknown` values left in the fuel system.
1. **Text Q&A.** Ingest the manual with page numbers, then build tiered retrieval, the profile resolver, citations and the eval harness. Gate: the S, T, C and A rows pass, and the profile ablation (evals with the profile switched off) shows a clear gap.
2. **Maintenance log (text).** Add log writes, log-grounded answers and the recommissioning workflow. Gate: L1 and R1 pass.
3. **Voice logging.** Add push-to-talk, speech-to-text with the custom vocabulary, and read-back. Gate: V1 passes on both the quiet and the noisy recordings.
4. **Voice Q&A and step mode.** Add two-form answers, text-to-speech and step navigation (next, back, repeat, repeat the spec, stop). Gate: V2 passes.
5. **Stretch.** Add profiles for the F-150 and the outboard, a wake word, and photo-based part ID.

**Current phase: 0 → 1.**

## Conventions

- Put the LLM behind a provider interface. Whether to use a local model or a hosted API is still undecided.
- Everything except an optional hosted model must run offline.
- Eval runs must be reproducible. Record the model, whether the profile was on or off, the eval file version and the date for each run.
- Never copy manual text into the repo, eval answers, tests, fixtures or commit messages. Store values and page references only.
- Ask before choosing major dependencies (LLM provider, vector store, speech models).

## Decisions

- **Manual:** the factory workshop manual reprint (MGB and MGB GT, 1962–1980, ISBN 9781855201743). It's a paper copy, to be scanned into a searchable PDF in `manuals/`. Citations use the page numbers printed in the book, not the PDF's own page numbers.

## Open questions (ask; don't assume)

- The Weber variant (DGV, DGEV or DGAV) and its jetting; the engine number; points or electronic ignition.
- Local LLM or hosted API.
- The host: an EliteDesk, the Unraid server, or both.
- Which device sits at the garage bench, and Wi-Fi coverage there.
- The project name.
