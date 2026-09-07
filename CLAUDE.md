# CLAUDE.md

Entry point for Claude Code in this repo. Read **`AGENTS.md`** for the full
working instructions, then **`README.md`** for the experiment design.

Quick facts:
- Purpose: face-shape classification **preprocessing-impact comparison** on the
  Niten Lama dataset. Main matrix is 4 models × 4 preprocessing versions = 16 runs
  in `runs/`; the follow-up adds an alignment ablation and two seed replications
  for 56 runs total.
- Companion to a systematic literature review. The app is a separate showcase,
  not in this repo.
- On Windows use `py -3.12` for the venv; verify what bare `python` resolves to
  before relying on it.
- Three image caches are required before training: `resize` (D1), `crop`
  (D2/D3/D3n), `align` (D4/D4n). See step 3 in `AGENTS.md`.
- Train + test on Niten Lama only. No KOMNET. No Indonesian-generalization claim.
- First action after setup: a 1-epoch smoke test, then the full matrix.
- Never write into `runs/` from follow-up work — it is the provenance for the
  originally reported results.
