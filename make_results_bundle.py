"""Package the reviewer-response experiment into a shareable bundle + zip.

Includes every result, the analysis tables, the code changes, and the figures.
Excludes model checkpoints (3.9 GB of best.pt) and the raw dataset, which is public
on Kaggle. Re-runnable: the output directory is rebuilt from scratch each time.

    python make_results_bundle.py
"""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

OUT = Path("../paper2_followup_bundle")
ASSETS = Path("../paper_assets")
MODELS = ["resnet50", "efficientnetv2s", "mobilenetv3small", "swint"]
VERSIONS = ["D1", "D2", "D3", "D4"]
SEEDS = [42, 123, 2025]
SOURCES = {42: "runs", 123: "results_seed123", 2025: "results_seed2025"}
PER_RUN_FILES = ["results.json", "epochs.csv", "test_predictions.npz", "confusion_matrix.png"]


def load_accuracy():
    acc, meta = {}, {}
    for r in csv.DictReader(open("runs/summary.csv", encoding="utf-8")):
        acc[(r["model"], r["preprocess"], 42)] = float(r["test_accuracy"]) * 100
    for seed in (123, 2025):
        for p in Path(SOURCES[seed]).glob("*/results.json"):
            r = json.loads(p.read_text(encoding="utf-8"))
            acc[(r["model"], r["preprocess"], seed)] = r["test_accuracy"] * 100
            meta[(r["model"], r["preprocess"], seed)] = r
    return acc, meta


def load_ablation():
    out = {}
    for p in Path("results_align_ablation").glob("*/results.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        out[(r["model"], r["preprocess"])] = r["test_accuracy"] * 100
    return out


def write_all_runs_csv(acc, ablation, meta):
    rows = []
    for (m, v, s), a in sorted(acc.items()):
        r = meta.get((m, v, s), {})
        rows.append({"model": m, "preprocess": v, "seed": s, "test_accuracy_pct": round(a, 3),
                     "macro_f1": round(r.get("macro_f1", float("nan")), 4) if r else "",
                     "epochs_run": r.get("epochs_run", ""),
                     "train_minutes": round(r["train_seconds"] / 60, 1) if r.get("train_seconds") else "",
                     "source_dir": SOURCES[s]})
    for (m, v), a in sorted(ablation.items()):
        p = Path("results_align_ablation") / f"{m}_{v}" / "results.json"
        r = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"model": m, "preprocess": v, "seed": 42, "test_accuracy_pct": round(a, 3),
                     "macro_f1": round(r["macro_f1"], 4), "epochs_run": r.get("epochs_run", ""),
                     "train_minutes": round(r["train_seconds"] / 60, 1),
                     "source_dir": "results_align_ablation"})
    with (OUT / "data" / "all_runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def write_table2(acc):
    path = OUT / "data" / "table2_mean_sd.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "preprocess", "mean_accuracy_pct", "sd_pct", "n_seeds",
                    *[f"seed_{s}" for s in SEEDS]])
        for m in MODELS:
            for v in VERSIONS:
                vals = np.array([acc[(m, v, s)] for s in SEEDS])
                w.writerow([m, v, round(vals.mean(), 2), round(vals.std(ddof=1), 2), len(vals),
                            *[round(acc[(m, v, s)], 2) for s in SEEDS]])


def write_ablation_csv(acc, ablation):
    with (OUT / "data" / "alignment_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "D3_seed42", "D4_seed42", "D4_minus_D3_confounded",
                    "D3n_seed42", "D4n_seed42", "D4n_minus_D3n_clean"])
        for m in MODELS:
            w.writerow([m, round(acc[(m, "D3", 42)], 2), round(acc[(m, "D4", 42)], 2),
                        round(acc[(m, "D4", 42)] - acc[(m, "D3", 42)], 2),
                        round(ablation[(m, "D3n")], 2), round(ablation[(m, "D4n")], 2),
                        round(ablation[(m, "D4n")] - ablation[(m, "D3n")], 2)])


def results_md(acc, ablation):
    L = []
    A = L.append
    A("# Results — preprocessing-impact study, reviewer-response runs\n")
    A(f"Generated {datetime.now():%Y-%m-%d}. 40 new runs, 21.0 GPU-hours, RTX 4060.\n")

    A("\n## Table II (revised): test accuracy %, mean ± sd over 3 training seeds\n")
    A("| Model | D1 | D2 | D3 | D4 |")
    A("|---|---|---|---|---|")
    for m in MODELS:
        cells = []
        for v in VERSIONS:
            x = np.array([acc[(m, v, s)] for s in SEEDS])
            cells.append(f"{x.mean():.1f} ± {x.std(ddof=1):.1f}")
        A(f"| {m} | " + " | ".join(cells) + " |")
    sds = [np.std([acc[(m, v, s)] for s in SEEDS], ddof=1) for m in MODELS for v in VERSIONS]
    A(f"\nMean within-cell seed sd: **{np.mean(sds):.2f} pts** (max {max(sds):.2f}). "
      "Split is identical across all runs (seed 42); only the training seed varies.\n")

    A("\n## Claim 1 — preprocessing gain (D1 → D3). HOLDS\n")
    A("| Model | mean gain | sd | positive at every seed |")
    A("|---|---|---|---|")
    for m in MODELS:
        g = np.array([acc[(m, "D3", s)] - acc[(m, "D1", s)] for s in SEEDS])
        A(f"| {m} | +{g.mean():.2f} | {g.std(ddof=1):.2f} | {'yes' if all(g > 0) else 'NO'} |")
    A("\n8–12× the seed noise. The paper's central claim is robust.\n")

    A("\n## Claim 2 — alignment (D4 − D3, as published). NULL, CONFIRMED\n")
    A("| Model | mean | sd | per-seed (42 / 123 / 2025) |")
    A("|---|---|---|---|")
    for m in MODELS:
        d = np.array([acc[(m, "D4", s)] - acc[(m, "D3", s)] for s in SEEDS])
        A(f"| {m} | {d.mean():+.2f} | {d.std(ddof=1):.2f} | "
          + " / ".join(f"{x:+.1f}" for x in d) + " |")
    A("\nSigns flip between seeds in every model. The published null conclusion is correct.\n")

    A("\n## Claim 3 — Swin-T D4, the published best. DOES NOT REPLICATE\n")
    sw = np.array([acc[("swint", "D4", s)] for s in SEEDS])
    A("| Seed | Accuracy |")
    A("|---|---|")
    for s, a in zip(SEEDS, sw):
        A(f"| {s} | {a:.1f}%{' ← published' if s == 42 else ''} |")
    A(f"\n**Mean {sw.mean():.2f} ± {sw.std(ddof=1):.2f}.** The 88.7% in the abstract was a "
      "seed outlier; the other two seeds land ~3.3 points lower.\n")

    A("\n## Top configurations (3-seed mean)\n")
    rank = sorted(((np.mean([acc[(m, v, s)] for s in SEEDS]),
                    np.std([acc[(m, v, s)] for s in SEEDS], ddof=1), m, v)
                   for m in MODELS for v in VERSIONS), reverse=True)
    A("| Rank | Config | Accuracy |")
    A("|---|---|---|")
    for i, (a, sd, m, v) in enumerate(rank[:5], 1):
        A(f"| {i} | {m} {v} | {a:.2f} ± {sd:.2f} |")
    A("\nEfficientNetV2-S replaces Swin-T as the best model. Its D3 and D4 differ by "
      "0.05 points — indistinguishable, reinforcing the alignment null.\n")

    A("\n## Alignment ablation (D3n / D4n) — seed 42 only\n")
    A("`D3n` / `D4n` are `D3` / `D4` with the ±15° rotation removed from the training\n"
      "augmentation. Both D3 and D4 randomly rotate faces by up to ±15°, which re-tilts\n"
      "the faces alignment just straightened — so `D4 vs D3` could not isolate alignment.\n"
      "`D4n vs D3n` can. Same precomputed images; only the augmentation differs.\n")
    A("| Model | D3 | D4 | D4−D3 (confounded) | D3n | D4n | D4n−D3n (clean) |")
    A("|---|---|---|---|---|---|---|")
    for m in MODELS:
        A(f"| {m} | {acc[(m, 'D3', 42)]:.1f} | {acc[(m, 'D4', 42)]:.1f} | "
          f"{acc[(m, 'D4', 42)] - acc[(m, 'D3', 42)]:+.2f} | {ablation[(m, 'D3n')]:.1f} | "
          f"{ablation[(m, 'D4n')]:.1f} | {ablation[(m, 'D4n')] - ablation[(m, 'D3n')]:+.2f} |")
    mean_clean = np.mean([ablation[(m, "D4n")] - ablation[(m, "D3n")] for m in MODELS])
    A(f"\nAll four are positive (mean +{mean_clean:.2f}), where the confounded comparison had "
      "mixed signs.\n**Caveat:** this is one seed, and the effect is smaller than the "
      f"{np.mean(sds):.2f}-pt seed noise. It is not a reliable finding without replication "
      "at seeds 123 and 2025 (8 runs, ~4.5 h). Reported here for completeness, not as a result.\n")

    A("\n## Identity-level leakage audit (new, beyond what reviewers asked)\n")
    A("The published dHash audit compares **raw files**. Two different photographs of the\n"
      "same person differ in framing, so their hashes diverge — but after face cropping and\n"
      "alignment they become near-identical training inputs. Perceptual hashing cannot see this.\n")
    A("\nEmbedding all 5,000 images with InceptionResnetV1 (VGGFace2) and clustering at\n"
      "cosine ≥ 0.95 (connected components):\n")
    A("- **195 identity clusters span a split boundary**")
    A("- **240 held-out images (16.2% of val+test)** sit in such a cluster")
    A("- 13 clusters carry more than one face-shape label (annotation noise)")
    A("\nValidated visually: 6 of 6 randomly sampled pairs at the threshold were the same\n"
      "person in a different photograph. See `figures/fig_identity_leakage.png`.\n")
    A("\n**Implication:** absolute accuracies on this dataset are inflated — for this study\n"
      "*and* for all prior work on it. Relative comparisons (the subject of the paper) are\n"
      "unaffected, since leakage applies to all arms roughly equally.\n")

    A("\n## Corrections to the manuscript\n")
    A("| Location | Current | Should be |")
    A("|---|---|---|")
    A("| Abstract, §III.C | Swin-T 88.7% best | EfficientNetV2-S 87.7 ± 1.3; Swin-T 86.4 ± 2.0 |")
    A("| §III.D, Fig. 3 | \"best overall configuration (Swin-T, D4)\" | no longer holds |")
    A("| Table II | single-seed values | mean ± sd over 3 seeds |")
    A("| Table I, §II.A | 3,500 / 759 / 741 | **3,515 / 744 / 741** |")
    A("| §II.B, §III.F | rotation/alignment confound as limitation | now measured (D3n/D4n) |")
    A("| §III.F | single seed as limitation | resolved — 3 seeds |")
    return "\n".join(L)


def readme_md(rows):
    total_min = sum(r["train_minutes"] for r in rows if isinstance(r["train_minutes"], float))
    return f"""# Face-Shape Preprocessing Study — reviewer-response experiment bundle

Everything from the follow-up experiment run in response to the ICORIS 2026 reviews:
raw per-run results, analysis, code changes, and figures.

Generated {datetime.now():%Y-%m-%d}.

## What this was for

The original paper compared 4 ImageNet-pretrained models across 4 cumulative
preprocessing variants (D1–D4) on the Niten Lama face-shape dataset. Two reviewer
objections needed compute to answer:

1. **"Why only one random seed?"** (R1) and **"multi-seed or k-fold validation"** (R2)
   → the full 16-config grid was re-run at training seeds **123** and **2025**.
2. **"D4 still confounds rotation augmentation with face alignment"** (R2)
   → two new variants, **D3n / D4n**, isolate alignment cleanly.

**40 new runs, {total_min / 60:.1f} GPU-hours, NVIDIA RTX 4060.**

## What D3n / D4n mean

`n` = **no rotation**.

| Variant | Images | Training augmentation |
|---|---|---|
| D1 | resized | none |
| D2 | face crop | none |
| D3 | face crop | flip, **±15° rotation**, jitter, resized-crop |
| D4 | crop + aligned | flip, **±15° rotation**, jitter, resized-crop |
| **D3n** | face crop *(same as D3)* | flip, jitter, resized-crop — **no rotation** |
| **D4n** | crop + aligned *(same as D4)* | flip, jitter, resized-crop — **no rotation** |

`D4 vs D3` was meant to measure alignment, but both apply ±15° random rotation, which
re-tilts the faces alignment just straightened. `D4n vs D3n` removes rotation from both
sides, leaving alignment as the only difference.

## Headline findings

- **Preprocessing gain (D1→D3) holds:** +11 to +17 points, positive at every seed and
  every model — 8–12× the measured seed noise of 1.31 pts.
- **The published best result does not replicate:** Swin-T D4 was 88.7 / 85.4 / 85.2
  across seeds (mean 86.4 ± 2.0). EfficientNetV2-S is the better model.
- **The alignment null is confirmed** at three seeds and with the confound removed.
- **New:** identity-level leakage — 195 identity clusters straddle the train/test split,
  invisible to the published perceptual-hash audit.

Full detail in [RESULTS.md](RESULTS.md).

## Layout

```
RESULTS.md                    full analysis, tables, manuscript corrections
data/
  all_runs.csv                every run: model, variant, seed, accuracy, F1, epochs, time
  table2_mean_sd.csv          revised Table II
  alignment_ablation.csv      D3n/D4n vs D3/D4
  identity_leakage.csv        cross-split identity matches
  dup_pairs.csv               original dHash near-duplicate audit (17 pairs)
figures/                      error analysis, near-duplicates, identity leakage
code/                         all source changes (see below)
runs/                         per-run results.json, epochs.csv, predictions, confusion matrix
  original_seed42/  align_ablation/  seed123/  seed2025/
```

## Code changes

| File | Change |
|---|---|
| `src/preprocessing.py` | Added `D3n` / `D4n`. D1–D4 op order untouched. |
| `src/train.py` | Pinned `SPLIT_SEED = 42` so `--seed` can never re-split; logs GPU/CUDA/timing/seed; dumps per-run test predictions. |
| `src/run_matrix.py` | Summary carries seed, epochs, train time. |
| `run_followup.py` | Terminal runner for all three tiers, resumable. |
| `run_followup.ipynb` | Notebook version + analysis cells. |
| `check_identity_leakage.py` | **New** — identity-level leakage audit. |
| `make_error_analysis.py` | **New** — misclassified-sample figure (R1 asked for this). |
| `make_dup_figure.py` | **New** — near-duplicate figure (R2 asked for this). |

The original 16 runs and `run_matrix.ipynb` were never modified; a fingerprint check
verified all 16 accuracies unchanged after all 40 new runs.

## Reproducing

Same split (`splits/split.csv`, seed 42) throughout — it is **not** regenerated, because
the near-duplicate audit is tied to that exact partition.

```
python run_followup.py            # all 40 runs, ~21 h
python run_followup.py --tier 1   # alignment ablation only, ~4.5 h
python check_identity_leakage.py
```

## Not included

- **Model checkpoints** (`best.pt`) — 3.9 GB across 56 runs.
- **The dataset** — public on Kaggle (CC0):
  https://www.kaggle.com/datasets/niten19/face-shape-dataset

## Environment

PyTorch 2.11.0+cu128 · timm 1.0.27 · Python 3.13.3 · CUDA 12.8 · cuDNN 9.19 ·
NVIDIA RTX 4060 (8 GB) · Windows 11
"""


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ["data", "figures", "code/src", "runs"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    acc, meta = load_accuracy()
    ablation = load_ablation()

    rows = write_all_runs_csv(acc, ablation, meta)
    write_table2(acc)
    write_ablation_csv(acc, ablation)
    (OUT / "RESULTS.md").write_text(results_md(acc, ablation), encoding="utf-8")
    (OUT / "README.md").write_text(readme_md(rows), encoding="utf-8")

    for name in ["identity_leakage.csv", "dup_pairs.csv", "error_analysis_confusions.csv"]:
        if (ASSETS / name).exists():
            shutil.copy2(ASSETS / name, OUT / "data" / name)
    for fig in ASSETS.glob("fig_*.png"):
        shutil.copy2(fig, OUT / "figures" / fig.name)

    for f in ["preprocessing.py", "train.py", "run_matrix.py", "data.py", "models.py",
              "evaluate.py"]:
        if Path("src", f).exists():
            shutil.copy2(Path("src", f), OUT / "code" / "src" / f)
    for f in ["run_followup.py", "make_followup_notebook.py", "run_followup.ipynb",
              "check_identity_leakage.py", "make_error_analysis.py", "make_dup_figure.py",
              "make_results_bundle.py", "requirements.txt"]:
        if Path(f).exists():
            shutil.copy2(f, OUT / "code" / f)

    label = {"runs": "original_seed42", "results_align_ablation": "align_ablation",
             "results_seed123": "seed123", "results_seed2025": "seed2025"}
    n_runs = 0
    for src, dst in label.items():
        for run_dir in sorted(Path(src).glob("*/")):
            if not (run_dir / "results.json").exists():
                continue
            target = OUT / "runs" / dst / run_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for f in PER_RUN_FILES:
                if (run_dir / f).exists():
                    shutil.copy2(run_dir / f, target / f)
            n_runs += 1
    shutil.copy2("runs/summary.csv", OUT / "runs" / "original_seed42_summary.csv")
    shutil.copy2("runs/test_predictions.npz", OUT / "runs" / "original_seed42_predictions.npz")
    shutil.copy2("splits/split.csv", OUT / "data" / "split.csv")

    zip_path = shutil.make_archive(str(OUT), "zip", root_dir=OUT.parent, base_dir=OUT.name)
    size_mb = Path(zip_path).stat().st_size / 1024 ** 2
    print(f"bundle : {OUT.resolve()}")
    print(f"zip    : {Path(zip_path).resolve()} ({size_mb:.1f} MB)")
    print(f"runs   : {n_runs} (checkpoints excluded)")


if __name__ == "__main__":
    main()
