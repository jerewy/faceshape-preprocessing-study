# CLAUDE.md

Entry point for Claude Code in this repo. Read **`AGENTS.md`** for the full
working instructions, then **`README.md`** for the experiment design.

Quick facts:
- Purpose: face-shape classification **preprocessing-impact comparison**
  (4 models × 4 preprocessing versions = 16 runs) on the Niten Lama dataset.
- Companion to Paper 1 (SLR). App is a separate showcase — not in this repo.
- Use `py -3.12` for the venv on Windows (bare `python` is an old 3.6).
- Train + test on Niten Lama only. No KOMNET. No Indonesian-generalization claim.
- First action after setup: a 1-epoch smoke test, then the full matrix.
