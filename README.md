# Face-Shape Classification — Preprocessing-Impact Study

Training pipeline for **Paper 2** of the pre-thesis project. This repo is the
empirical companion to Paper 1 (the SLR *Data Preparation in Image-Based
Face-Shape Classification*). It answers one question:

> **How much does data preparation (resize / face-crop / alignment / augmentation)
> change face-shape classification accuracy across CNN and Vision Transformer
> architectures?**

It is a self-contained, clone-and-run repo. No context from any chat is needed —
everything an agent or human needs is in `README.md` and `AGENTS.md`.

## Experiment design

**Dataset:** Niten Lama Face Shape Dataset — 5 balanced classes
(`heart, oblong, oval, round, square`), ~5,000 images, CC0. **Train + test on this
dataset only.** See `data/README.md` for how to obtain and place it.

**Fixed split (seeded, reused across all runs):** 70% train / 15% val / 15% test,
stratified per class, `seed=42`. Created once into `splits/split.csv`.

**Four preprocessing versions (the independent variable):**

| Version | Pipeline | Notes |
|---|---|---|
| D1 | Resize only (224×224) | Baseline |
| D2 | Face-crop (MediaPipe) → resize | Tests cropping |
| D3 | Face-crop → augmentation → resize | Tests augmentation |
| D4 | Face-crop → alignment → augmentation → resize | Full pipeline (heaviest / optional) |

Augmentation (D3/D4, **train split only**): horizontal flip, ±15° rotation,
brightness/contrast jitter.

**Four models (the architecture axis), all ImageNet-pretrained via `timm`:**

| Key | `timm` id | Role |
|---|---|---|
| `resnet50` | `resnet50` | Standard CNN baseline |
| `efficientnetv2s` | `tf_efficientnetv2_s` | Efficient CNN |
| `mobilenetv3small` | `mobilenetv3_small_100` | **App deployment target** |
| `swint` | `swin_tiny_patch4_window7_224` | Vision Transformer |

**Matrix:** 4 models × 4 versions = **16 runs**. Metrics per run: accuracy,
macro precision/recall/F1, 5×5 confusion matrix.

## Quickstart

```bash
# 1. Environment (Windows: scripts\setup_windows.ps1 — uses Python 3.12)
bash scripts/setup.sh           # or the .ps1 on Windows

# 2. Put the dataset at data/raw/<class>/*.jpg  (see data/README.md)

# 3. Precompute face-crop and aligned variants (needed for D2/D3/D4)
python -m src.preprocess_faces --mode crop  --data-root data/raw --out data/preprocessed/crop
python -m src.preprocess_faces --mode align --data-root data/raw --out data/preprocessed/align

# 4. SMOKE TEST first (1 epoch, smallest model) — confirm the pipeline runs
python -m src.train --model mobilenetv3small --preprocess D1 --epochs 1

# 5. Full 16-run matrix
python -m src.run_matrix --output-dir runs

# 6. Results: runs/summary.csv  + per-run runs/<model>_<version>/{results.json,confusion_matrix.png,epochs.csv}
```

Training auto-detects the device: **CUDA** (NVIDIA) → Intel **XPU** → **CPU**.
A CUDA GPU is strongly recommended. With no NVIDIA GPU, run on a free
**Kaggle T4** notebook instead (clone this repo into the notebook).

## Scope of claims (read before writing the paper)

- Train + test on Niten Lama **only**. The comparison is internally valid.
- **Do NOT claim Indonesian / cross-population generalization.** Niten Lama is
  female-celebrity-dominant and non-Indonesian; face morphology varies by sex and
  ancestry. State this as a limitation; cross-population evaluation is future work.
- KOMNET and other Indonesian datasets are **not** part of this study.

## Layout

```
src/                 training code
  data.py            fixed split + dataset
  preprocessing.py   D1–D4 transform builders
  preprocess_faces.py  MediaPipe face-crop / alignment (run once)
  models.py          timm model factory
  train.py           train one (model, version) experiment
  evaluate.py        metrics + confusion matrix
  run_matrix.py      loop all 16 experiments + summary.csv
configs/experiments.yaml   matrix + hyperparameters (reference)
scripts/             environment setup (Windows + bash)
data/README.md       dataset placement instructions
```
