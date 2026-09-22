# Face-Shape Classification — Preprocessing-Impact Study

Code, fixed data splits, and complete per-run results for an empirical study of how
**data preparation affects face-shape classification accuracy** across CNN and Vision
Transformer architectures.

> **Research question.** How much does data preparation (resize / face-crop / alignment /
> augmentation) change face-shape classification accuracy, and is the effect consistent
> across architectures?

This repository is the empirical companion to a systematic literature review,
*Data Preparation in Face-Shape Classification: A Systematic Literature Review*.
It contains every number in the paper that can be recomputed without the dataset:
56 training runs with their per-epoch logs, per-sample test predictions, and
confusion matrices. Model checkpoints (3.9 GB) and the dataset itself are not
committed, so the Grad-CAM, error-analysis and audit figures must be regenerated
locally — see [Dataset](#dataset).

## Contents at a glance

| Directory | Runs | Seed | What it is |
|---|---|---|---|
| `runs/` | 16 | 42 | Main matrix: 4 models × 4 preprocessing versions (D1–D4) |
| `results_seed123/` | 16 | 123 | Seed replication of the main matrix |
| `results_seed2025/` | 16 | 2025 | Seed replication of the main matrix |
| `results_align_ablation/` | 8 | 42 | Alignment ablation: 4 models × {D3n, D4n} |
| **Total** | **56** | | |

Each run directory holds `results.json` (final metrics), `epochs.csv` (the per-epoch
train/val curve), and `confusion_matrix.png`. Each results directory also has a
`summary.csv` aggregating its runs.

Per-sample test predictions are committed too, so significance tests can be recomputed
without retraining — but they are stored differently in the two generations of runs:

- `runs/` — one bundled `runs/test_predictions.npz` holding every model's predictions
  plus the shared `targets` array. This is what `correct_stats.py` reads.
- `results_seed123/`, `results_seed2025/`, `results_align_ablation/` — one
  `test_predictions.npz` per run directory.

`runs/summary.csv` carries test accuracy and macro precision / recall / F1 plus the best
validation macro-F1. The three follow-up `summary.csv` files add `seed`, `epochs_run`,
and `train_seconds` columns.

## Experimental design

**Models** — four ImageNet-pretrained backbones via `timm`, all at 224×224 input so that
preprocessing is the only variable changing across the matrix:

| Key | `timm` id | Role |
|---|---|---|
| `resnet50` | `resnet50` | Standard CNN baseline |
| `efficientnetv2s` | `tf_efficientnetv2_s` | Efficient CNN |
| `mobilenetv3small` | `mobilenetv3_small_100` | Mobile deployment target |
| `swint` | `swin_tiny_patch4_window7_224` | Vision Transformer |

**Preprocessing versions** — the independent variable. Augmentation is applied to the
**train split only**; validation and test are always a plain resize.

| Version | Image source | Train-split augmentation |
|---|---|---|
| D1 | Resized raw (`data/preprocessed/resize`) | none |
| D2 | Face crop (`data/preprocessed/crop`) | none |
| D3 | Face crop | flip, ±15° rotation, brightness/contrast jitter, resized crop |
| D4 | Crop + in-plane alignment (`data/preprocessed/align`) | flip, ±15° rotation, jitter, resized crop |
| D3n | Face crop *(same source as D3)* | as D3, **without rotation** |
| D4n | Crop + alignment *(same source as D4)* | as D4, **without rotation** |

**Why D3n / D4n exist.** In the main matrix, D4 adds alignment but keeps the ±15°
rotation augmentation, which re-introduces exactly the tilt that alignment removes.
D4 vs D3 therefore cannot isolate the effect of alignment. The `n` variants drop the
rotation from both arms, so D4n vs D3n measures alignment alone. See
`SOURCE_BY_VERSION` and `NO_ROTATION_VERSIONS` in [`src/preprocessing.py`](src/preprocessing.py).

**Split** — one fixed stratified 70 / 15 / 15 split (`seed=42`), created once and reused
by all 56 runs, so no run gets a partition advantage. See
[Split provenance](#split-provenance).

## Dataset

**Niten Lama Face Shape Dataset** — 5 balanced classes (`heart, oblong, oval, round,
square`), ~5,000 images, CC0, available at
<https://www.kaggle.com/datasets/niten19/face-shape-dataset>.
Images are **not** committed here. See [`data/README.md`](data/README.md) for the
required folder layout.

Training and testing are on this dataset only.

## Reproducing

Steps 1–4 are one-time setup; step 5 is the main matrix, roughly 2–3 h on a single
modern GPU.

```bash
# 1. Environment  (Windows: scripts\setup_windows.ps1, which uses py -3.12)
bash scripts/setup.sh

# 2. Dataset at data/raw/<class>/*.jpg — see data/README.md

# 3. Build the three image caches. All three are required:
#    resize is what D1 reads, crop is D2/D3/D3n, align is D4/D4n.
python scripts/make_resize_cache.py
python -m src.preprocess_faces --mode crop  --data-root data/raw --out data/preprocessed/crop
python -m src.preprocess_faces --mode align --data-root data/raw --out data/preprocessed/align

# 4. Smoke test — 1 epoch on the smallest model, confirms the pipeline runs
python -m src.train --model mobilenetv3small --preprocess D1 --epochs 1

# 5. Main matrix (16 runs -> runs/). Resumable: it skips any run that already
#    has a results.json, so an interruption costs only the run in flight.
python -m src.run_matrix --output-dir runs

# 6. Follow-up matrix (40 runs: alignment ablation + 2 seed replications, ~18 h).
#    Streams to console and followup_log.txt. Never writes into runs/.
python run_followup.py
```

`splits/split.csv` is committed, so step 5 reproduces the exact partitions used in the
paper without regenerating the split. **Do not run `python -m src.data`** unless you
mean to: it overwrites `splits/split.csv` with a freshly generated split, which is the
*pre*-deduplication partition (see [Split provenance](#split-provenance)) and would
silently invalidate comparisons against the committed results.

Device is auto-detected: CUDA → Intel XPU → CPU. A CUDA GPU is strongly recommended;
the committed runs were produced on one. `run_matrix.py` lowers the batch size to 16
for `swint` and `efficientnetv2s`.

`run_matrix.ipynb` and `run_followup.ipynb` are notebook equivalents of steps 3–6.
`run_matrix.ipynb` retains its executed outputs as provenance for the main matrix. Both
notebooks are generated by `make_notebook.py` and `make_followup_notebook.py`.

## Analysis scripts

These regenerate the paper's tables and figures from the committed results. Most read
`runs/`, `results_*/`, and `splits/` and write to `../paper_assets/` — a sibling
directory outside this repository, which you must create first:

```bash
mkdir -p ../paper_assets
```

Three write somewhere else, and two of those overwrite committed provenance — check the
**Writes** column before running them.

| Script | Needs | Produces | Writes |
|---|---|---|---|
| `correct_stats.py` | `runs/test_predictions.npz` (committed) | McNemar tests and CIs for the D-version contrasts; runs on a fresh clone | `../paper_assets/` |
| `check_leakage_impact.py` | `identity_leakage.csv` + committed predictions | Whether leakage shifts the preprocessing contrasts; no dataset needed | `../paper_assets/` |
| `journal_stats.py` | dataset + checkpoints | Per-sample predictions, significance, efficiency, per-class breakdown | `../paper_assets/` **and overwrites `runs/test_predictions.npz`** |
| `make_training_curves.py` | `epochs.csv` files | Training-curve figure | `../paper_assets/` |
| `make_results_bundle.py` | all results directories, plus the audit CSVs in `../paper_assets/` | Shareable bundle + zip of results, tables, and figures | `../paper2_followup_bundle/` |
| `review_dups.py` | dataset | Near-duplicate dHash audit → `dup_pairs.csv`, contact sheet | `../paper_assets/` |
| `make_dup_figure.py` | `dup_pairs.csv` from `review_dups.py` | Publication figure for the duplicate audit | `../paper_assets/` |
| `reassign_dups.py` | dataset + `splits/split.csv` | Applies the dedup split reassignment | **overwrites `splits/split.csv` and `splits/split_original.csv`** |
| `check_identity_leakage.py` | dataset (+ `facenet-pytorch`) | Identity-level cross-split leakage audit | `../paper_assets/` |
| `check_preprocessing.py` | dataset + caches | Sanity check that the crop/align caches match raw | nothing |
| `make_error_analysis.py` | dataset + checkpoints | Misclassified-sample figure and table | `../paper_assets/` |
| `make_gradcam.py` | dataset + checkpoints (+ `pytorch-grad-cam`) | Grad-CAM overlays | `../paper_assets/` |

Scripts listed as needing checkpoints require `best.pt` files, which are git-ignored
(~70 MB average, 6 MB to 110 MB depending on the backbone) and must be regenerated by
re-running training.

Two of these destroy committed provenance if run on a fresh clone, because the clone
already contains their output:

- **`journal_stats.py`** rebuilds `runs/test_predictions.npz` from whatever `best.pt`
  files exist. With a partial set of checkpoints it silently replaces the committed
  16-model bundle — the one `correct_stats.py` reads — with a smaller one.
- **`reassign_dups.py`** copies the current `splits/split.csv` over
  `splits/split_original.csv` and then rewrites `split.csv`. On a fresh clone
  `split.csv` is *already* the deduplicated split, so re-running it destroys the
  pre-deduplication partition. The reassignment has already been applied; you do not
  need to run this script to reproduce the committed results.

`followup_log.txt` is the raw console log of the 40-run follow-up job, kept as
provenance for the wall-clock timings and early-stopping epochs.

## Split provenance

Two split files are committed, and the difference between them matters:

- **`splits/split_original.csv`** — the first stratified split: 3,500 train / 750 val /
  750 test.
- **`splits/split.csv`** — the split actually used by all 56 committed runs, after the
  near-duplicate audit: **3,515 train / 744 val / 741 test.**

The audit (`review_dups.py`, dHash over raw images) found duplicate and near-duplicate
image pairs straddling split boundaries. `reassign_dups.py` groups each near-duplicate
cluster and assigns the whole cluster to its majority split; here that moved 9 images out
of test and 6 out of val into train, 15 in total. No class label was changed. The result
is that **no held-out image has a dHash near-duplicate (Hamming ≤ 5) in training.**

### Identity-level leakage is *not* resolved

Image-level dHash compares raw files. It cannot catch the same person photographed on a
different day: two such photographs have very different framing, so their raw hashes
diverge — yet after face cropping and alignment they become near-identical model inputs.
That is the leakage mode that matters for a celebrity-heavy dataset.

`check_identity_leakage.py` measures it, embedding every image with a VGGFace2
face-recognition model and clustering at cosine ≥ 0.95, calibrated against a random-pair
null drawn from this same dataset. On this split it reports:

- **195 identity clusters span a split boundary**
- **240 held-out images — 16.2% of val + test — sit in such a cluster**
- 13 clusters carry more than one face-shape label (annotation noise, independent of
  leakage)

**Absolute accuracies on this dataset are therefore inflated** — for this study and for
any other work using this dataset without an identity-aware split. Treat every accuracy
in `runs/` and `results_*/` as an upper bound.

Whether leakage also distorts the *relative* preprocessing comparison — the actual
subject of this study — is a separate and weaker question, because cropping and
alignment may normalise identity cues differently across D1–D4. Do not assume it away:
`check_leakage_impact.py` tests it directly, re-scoring every committed run on the
leakage-free subset of the test set and comparing each contrast against a random-subset
null. It needs only the audit's CSV and the committed predictions, no dataset and no
retraining.

Neither audit's output CSV is committed, because regenerating either requires the
dataset. Reproduce them with:

```bash
python review_dups.py                # -> ../paper_assets/dup_pairs.csv
python check_identity_leakage.py     # -> ../paper_assets/identity_leakage.csv
python check_leakage_impact.py       # -> ../paper_assets/leakage_impact.md
```

## Scope of claims

- Training and testing are on the Niten Lama dataset **only**. The comparison across
  preprocessing versions is internally valid; the absolute accuracies are not
  transferable to other datasets.
- **Absolute accuracies are inflated by identity leakage.** 16.2% of held-out images
  share an identity cluster with a training image (see [Split provenance](#split-provenance)).
  Every number here is an upper bound on performance for an unseen person. Whether the
  relative D1–D4 comparison is affected is testable with `check_leakage_impact.py` and
  should not be assumed either way.
- The dataset is female-celebrity-dominant and non-Indonesian. Face morphology varies
  with sex and ancestry, so these results **do not** support claims about Indonesian or
  cross-population performance. Cross-population evaluation is future work.
- Single-seed differences between adjacent configurations can be smaller than seed
  variance. Use the 3-seed mean ± sd (`runs/`, `results_seed123/`, `results_seed2025/`)
  rather than a single run when comparing configurations.
- **The alignment ablation has only one seed.** `results_align_ablation/` is seed 42
  only, so D4n − D3n is exactly the kind of single-seed difference the bullet above warns
  about: it is +0.7 to +2.0 points in all four models, but no model reaches significance
  (exact McNemar on the committed predictions gives raw p ≥ 0.15, Holm-adjusted p ≥ 0.58;
  a sign test on 4/4 positive gives p = 0.125). Read it as suggestive, not as a result.
  Replicating it at seeds 123 and 2025 is 8 runs, roughly 5 GPU-hours.
- KOMNET, IMSFD, and other Indonesian datasets are out of scope for this study.

## Layout

```
src/                        training code
  data.py                   fixed split + dataset
  preprocessing.py          D1-D4 / D3n / D4n transforms and image sources
  preprocess_faces.py       MediaPipe face-crop and alignment caches (run once)
  models.py                 timm model factory
  train.py                  train one (model, version) experiment
  evaluate.py               metrics + confusion matrix
  run_matrix.py             loop a set of experiments + write summary.csv
scripts/
  setup.sh, setup_windows.ps1   environment setup
  make_resize_cache.py      builds the D1 resize cache
  analyze_dataset.py        image sizes + face-detection rate per class
configs/experiments.yaml    matrix and hyperparameters (reference)
splits/                     fixed train/val/test assignments (committed)
runs/, results_*/           committed per-run results
data/README.md              dataset placement instructions
```

## License

Code is released under the MIT License — see [`LICENSE`](LICENSE).

The committed result files and split CSVs are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see
[`LICENSE-RESULTS`](LICENSE-RESULTS).

The Niten Lama dataset is CC0 and is distributed by its authors; no dataset images are
redistributed here.

## Citing

Please cite the tagged release rather than `main`, so that the artifact cited is the one
you read. See [`CITATION.cff`](CITATION.cff) and the repository's Releases page.
