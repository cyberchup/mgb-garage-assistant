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

## Bring your own manual

Workshop manuals are copyrighted and aren't included. Put your own PDFs in `manuals/` (gitignored).

## Roadmap

0. Groundwork: confirm the as-built config and ingest the manual
1. Text Q&A with citations and profile overrides
2. Maintenance log and recommissioning workflow
3. Voice logging with read-back
4. Voice Q&A and step mode
5. More vehicles, a wake word, and photo-based part ID
