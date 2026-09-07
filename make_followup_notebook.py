"""Generate run_followup.ipynb — the reviewer-response runs (ICORIS 2026 resubmission).

Separate from make_notebook.py/run_matrix.ipynb on purpose: that notebook still holds
the executed outputs of the original 16 runs, which are the provenance for Table II.
Nothing here writes into runs/.
"""
import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": s}


cells = []

cells.append(md(r'''# Paper 2 — Follow-up Runs (reviewer response)

Answers the two reviewer objections that need compute. **Nothing here touches
`runs/`, `run_matrix.ipynb`, or `splits/split.csv`** — the original 16 runs stay
exactly as published.

| Tier | What | Runs | Time | Answers |
|---|---|---|---|---|
| **1** | Alignment ablation `D3n`/`D4n` (augmentation without rotation) | 8 | ~3.5 h | R2: "D4 confounds rotation augmentation with alignment" |
| **2a** | Full D1–D4 grid at training seed **123** | 16 | ~7 h | R1: "why only one random seed" · R2: "multi-seed or k-fold" |
| **2b** | Full D1–D4 grid at training seed **2025** | 16 | ~7 h | (same — gives 3 seeds total with the original) |

**The data split never changes.** `src/train.py` pins it to `SPLIT_SEED = 42`; `--seed`
now controls training randomness only (weight init, shuffling, augmentation sampling).
The near-duplicate audit is tied to that partition, so re-splitting would silently
invalidate the deduplication.

**How to run:** select the `.venv` kernel, then run tiers **one at a time** (each is
resumable — any run with a `results.json` is skipped).'''))

# ---------------------------------------------------------------- 0. guard rails
cells.append(md('## 0. Setup and guard rails'))
cells.append(code(r'''import os, sys, json
from pathlib import Path

REPO_DIR = Path.cwd()   # <-- set to the repo root if this notebook is opened elsewhere
os.chdir(REPO_DIR)
print("cwd   :", os.getcwd())
print("python:", sys.executable)

import torch
print("torch :", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

for sub in ["data/preprocessed/resize", "data/preprocessed/crop",
            "data/preprocessed/align", "splits/split.csv"]:
    print(("OK   " if (REPO_DIR / sub).exists() else "MISSING ") + sub)'''))

cells.append(md('''### Protect the original results
Records the state of `runs/` now, so the last cell can prove it was never modified.
Also asserts that no follow-up output directory is `runs/`.'''))
cells.append(code(r'''ORIGINAL_RUNS = sorted(p.parent.name for p in Path("runs").glob("*/results.json"))
ORIGINAL_FINGERPRINT = {
    d: round(json.loads(Path(f"runs/{d}/results.json").read_text())["test_accuracy"], 6)
    for d in ORIGINAL_RUNS
}
print(f"original runs found: {len(ORIGINAL_RUNS)} (expected 16)")
assert len(ORIGINAL_RUNS) == 16, "original matrix incomplete — stop and investigate"

ALIGN_DIR = "results_align_ablation"
SEED_DIRS = {123: "results_seed123", 2025: "results_seed2025"}

for d in [ALIGN_DIR, *SEED_DIRS.values()]:
    assert Path(d).resolve() != Path("runs").resolve(), f"{d} would overwrite runs/"
print("output dirs OK:", [ALIGN_DIR, *SEED_DIRS.values()])'''))

# ---------------------------------------------------------------- 1. verify arms
cells.append(md('''## 1. Verify the new preprocessing arms

`D3n` / `D4n` reuse the **same precomputed images** as `D3` / `D4` — only the
augmentation differs (no `RandomRotation`). So `D4n vs D3n` isolates in-plane
alignment, which `D4 vs D3` cannot. No new preprocessing pass is needed.'''))
cells.append(code(r'''import importlib
from src import preprocessing
importlib.reload(preprocessing)

def ops(v, split="train"):
    return [type(t).__name__ for t in preprocessing.build_transform(v, split).transforms]

for v in ["D3", "D3n", "D4", "D4n"]:
    print(f"{v:4s} <- {str(preprocessing.source_root_for(v, 'data/raw', 'data/preprocessed')):32s} {ops(v)}")

# regression guard: the original D3/D4 regime must be untouched
assert ops("D3") == ["RandomHorizontalFlip", "RandomRotation", "ColorJitter",
                     "RandomResizedCrop", "ToTensor", "Normalize"]
assert ops("D3n") == ["RandomHorizontalFlip", "ColorJitter",
                      "RandomResizedCrop", "ToTensor", "Normalize"]
assert preprocessing.source_root_for("D3n", "data/raw", "data/preprocessed") == \
       preprocessing.source_root_for("D3", "data/raw", "data/preprocessed")
assert preprocessing.source_root_for("D4n", "data/raw", "data/preprocessed") == \
       preprocessing.source_root_for("D4", "data/raw", "data/preprocessed")
print("\nOK — D3/D4 unchanged, D3n/D4n = same images, augmentation minus rotation")'''))

# ---------------------------------------------------------------- 2. run everything
cells.append(md('''## 2. Run everything (40 runs, ~18 h) — start it and walk away

One cell, three tiers, in order of importance: the alignment ablation first (the
reject-level objection), then the two extra seeds. It streams progress and is fully
**resumable** — any run that already has a `results.json` is skipped, so if the machine
reboots you just re-run this cell and it picks up where it stopped.

Measured from the original matrix: **~28 min/run** (17 min MobileNetV3-Small to 37 min
Swin-T), so 40 runs is roughly 18 hours. Only the GPU is busy; you can keep using the PC.

*If you would rather go tier by tier, the cell below this one has them separated.*'''))
cells.append(code(r'''import subprocess, sys, time

PLAN = [
    ("Tier 1  alignment ablation", ALIGN_DIR, ["D3n", "D4n"], 42),
    ("Tier 2a seed 123",           SEED_DIRS[123], ["D1", "D2", "D3", "D4"], 123),
    ("Tier 2b seed 2025",          SEED_DIRS[2025], ["D1", "D2", "D3", "D4"], 2025),
]

t0 = time.time()
for name, out_dir, versions, seed in PLAN:
    print(f"\n{'=' * 70}\n{name}  ->  {out_dir}/  (seed {seed}, {len(versions) * 4} runs)\n{'=' * 70}",
          flush=True)
    subprocess.run([sys.executable, "-m", "src.run_matrix",
                    "--output-dir", out_dir, "--versions", *versions,
                    "--epochs", "30",
                    "--extra", "--seed", str(seed), "--num-workers", "6"], check=True)
    print(f"[{name}] done — {(time.time() - t0) / 3600:.1f} h elapsed", flush=True)

print(f"\nALL TIERS COMPLETE in {(time.time() - t0) / 3600:.1f} h")'''))

cells.append(md('''<details><summary>Alternative: run one tier at a time</summary>

Skip this if you ran the cell above.
</details>'''))
cells.append(code(r'''# Tier 1 only — alignment ablation, 8 runs, ~3.5 h
# !"{sys.executable}" -m src.run_matrix --output-dir {ALIGN_DIR} --versions D3n D4n --epochs 30 --extra --num-workers 6

# Tier 2a only — seed 123, 16 runs, ~7 h
# !"{sys.executable}" -m src.run_matrix --output-dir {SEED_DIRS[123]} --epochs 30 --extra --seed 123 --num-workers 6

# Tier 2b only — seed 2025, 16 runs, ~7 h
# !"{sys.executable}" -m src.run_matrix --output-dir {SEED_DIRS[2025]} --epochs 30 --extra --seed 2025 --num-workers 6'''))

# ---------------------------------------------------------------- 5. collect
cells.append(md('''## 5. Collect every run into one table
Reads `runs/` (seed 42, original) plus the follow-up directories. Original files are
only ever **read**.'''))
cells.append(code(r'''import pandas as pd

SOURCES = [("runs", 42), (ALIGN_DIR, 42), (SEED_DIRS[123], 123), (SEED_DIRS[2025], 2025)]

rows = []
for d, default_seed in SOURCES:
    for res in sorted(Path(d).glob("*/results.json")):
        r = json.loads(res.read_text(encoding="utf-8"))
        rows.append({
            "model": r["model"], "preprocess": r["preprocess"],
            "seed": r.get("seed", default_seed),
            "test_accuracy": r["test_accuracy"], "macro_f1": r["macro_f1"],
            "epochs_run": r.get("epochs_run"),
            "train_minutes": round(r["train_seconds"] / 60, 1) if r.get("train_seconds") else None,
            "gpu": (r.get("environment") or {}).get("gpu"),
            "source": d,
        })

allruns = pd.DataFrame(rows).sort_values(["preprocess", "model", "seed"])
allruns.to_csv("followup_all_runs.csv", index=False)
print(f"{len(allruns)} runs across {allruns.source.nunique()} directories")
display(allruns.head(12))'''))

# ---------------------------------------------------------------- 6. seed table
cells.append(md('''## 6. Seed variance — the replacement for Table II
`mean ± sd` over the three training seeds. Report this instead of single-seed numbers;
it answers R1 directly and lets you say whether the model spread exceeds seed noise.'''))
cells.append(code(r'''grid = allruns[allruns.preprocess.isin(["D1", "D2", "D3", "D4"])]
n_seeds = grid.groupby(["model", "preprocess"]).seed.nunique()
print("seeds per cell (want 3 everywhere):")
display(n_seeds.unstack())

stat = (grid.groupby(["model", "preprocess"]).test_accuracy
            .agg(["mean", "std", "count"]) * [100, 100, 1])
stat["cell"] = stat.apply(lambda r: f"{r['mean']:.1f} ± {r['std']:.1f}", axis=1)
table2 = stat["cell"].unstack()
print("\nTest accuracy %, mean ± sd over seeds:")
display(table2)

sd = grid.groupby(["model", "preprocess"]).test_accuracy.std().mean() * 100
spread_d3 = grid[grid.preprocess == "D3"].groupby("model").test_accuracy.mean()
print(f"\nmean within-cell seed sd : {sd:.2f} pts")
print(f"model spread at D3       : {(spread_d3.max() - spread_d3.min()) * 100:.2f} pts")
print("-> architecture effect is real if the spread clearly exceeds the seed sd")'''))

# ---------------------------------------------------------------- 7. ablation
cells.append(md('''## 7. Alignment ablation — the clean test
`D4 vs D3` = alignment **with** the rotation confound (what the paper reported).
`D4n vs D3n` = alignment **without** it. McNemar on paired test predictions, same
exact binomial test used in the paper.'''))
cells.append(code(r'''import numpy as np
from scipy.stats import binomtest

def preds(d, model, ver):
    z = np.load(Path(d) / f"{model}_{ver}" / "test_predictions.npz")
    return z["y_pred"], z["y_true"]

def mcnemar(pa, pb, y):
    ca, cb = (pa == y), (pb == y)
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue
    return b, c, p

# original D4 vs D3 predictions live in the archived npz
orig = np.load("runs/test_predictions.npz")
MODELS = ["resnet50", "efficientnetv2s", "mobilenetv3small", "swint"]

out = []
for m in MODELS:
    y = orig["targets"]
    b, c, p = mcnemar(orig[f"{m}_D4"], orig[f"{m}_D3"], y)
    row = {"model": m, "contrast": "D4 vs D3 (confounded)",
           "delta_pts": ((orig[f"{m}_D4"] == y).mean() - (orig[f"{m}_D3"] == y).mean()) * 100,
           "b": b, "c": c, "p": p}
    out.append(row)
    try:
        p3, y3 = preds(ALIGN_DIR, m, "D3n")
        p4, _ = preds(ALIGN_DIR, m, "D4n")
        b, c, p = mcnemar(p4, p3, y3)
        out.append({"model": m, "contrast": "D4n vs D3n (clean)",
                    "delta_pts": ((p4 == y3).mean() - (p3 == y3).mean()) * 100,
                    "b": b, "c": c, "p": p})
    except FileNotFoundError:
        print(f"[pending] {m} ablation runs not finished yet")

ab = pd.DataFrame(out).sort_values(["model", "contrast"])
ab["delta_pts"] = ab.delta_pts.round(2)
ab["p"] = ab.p.round(4)
display(ab)
ab.to_csv("followup_alignment_ablation.csv", index=False)
print("\nHolm-correct these p-values alongside the rest before quoting them in the paper.")'''))

# ---------------------------------------------------------------- 8. compute env
cells.append(md('''## 8. Compute environment (for §II.D)
R1 asked for GPU model, training time, CUDA version, and memory. Paste this into the
Training subsection.'''))
cells.append(code(r'''envs = [json.loads(p.read_text(encoding="utf-8")).get("environment")
         for d, _ in SOURCES for p in Path(d).glob("*/results.json")]
envs = [e for e in envs if e]
if envs:
    e = envs[0]
    print(f"GPU        : {e.get('gpu')} ({e.get('gpu_memory_total_gb')} GB)")
    print(f"CUDA/cuDNN : {e.get('cuda')} / {e.get('cudnn')}")
    print(f"PyTorch    : {e.get('torch')}  ·  Python {e.get('python')}")
    print(f"OS         : {e.get('platform')}")

t = allruns.dropna(subset=["train_minutes"])
if len(t):
    print(f"\ntraining time: {t.train_minutes.min():.0f}-{t.train_minutes.max():.0f} min per run "
          f"(median {t.train_minutes.median():.0f}), {t.train_minutes.sum() / 60:.1f} GPU-hours over {len(t)} logged runs")
    display(t.groupby("model").train_minutes.agg(["median", "min", "max"]).round(1))
print("\nNote: the original 16 runs predate timing capture. Their wall-clock is recoverable "
      "from results.json mtimes (sequential execution): ~26 min/run, 09:28-15:53 on 2026-06-23.")'''))

# ---------------------------------------------------------------- 9. verify
cells.append(md('## 9. Verify the original results were never modified'))
cells.append(code(r'''now = {
    d: round(json.loads(Path(f"runs/{d}/results.json").read_text())["test_accuracy"], 6)
    for d in sorted(p.parent.name for p in Path("runs").glob("*/results.json"))
}
if now == ORIGINAL_FINGERPRINT:
    print(f"OK — all {len(now)} original runs unchanged")
else:
    changed = {k for k in set(now) | set(ORIGINAL_FINGERPRINT)
               if now.get(k) != ORIGINAL_FINGERPRINT.get(k)}
    print("CHANGED:", changed)'''))

cells.append(md('''## Done

Outputs: `followup_all_runs.csv`, `followup_alignment_ablation.csv`, and per-run
folders under `results_align_ablation/`, `results_seed123/`, `results_seed2025/`.

Next, for the manuscript:
1. Replace Table II with the `mean ± sd` table from §6.
2. Turn the D3/D4 confound from a limitation (§II.B, §III.F) into a result using §7.
3. Fill in the compute environment in §II.D from §8.'''))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python (.venv)", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
Path("run_followup.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote run_followup.ipynb with", len(cells), "cells")
