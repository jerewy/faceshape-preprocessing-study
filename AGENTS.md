# AGENTS.md — instructions for agents running this repo

You are running the training pipeline for a face-shape classification
**preprocessing-impact comparison study**. This file is self-contained; you do
not need any external chat context. Read `README.md` for the full design.

## Your job, in order

1. **Set up the environment.**
   - Windows: `scripts\setup_windows.ps1` (uses `py -3.12`; the bare `python` on
     this user's machine is an old 3.6 — do not use it).
   - Linux/Kaggle: `scripts/setup.sh`.
   - Install the **CUDA** build of torch if the machine has an NVIDIA GPU
     (check `nvidia-smi`); otherwise CPU build, or move to a Kaggle T4 notebook.
2. **Confirm the dataset** is at `data/raw/<class>/*.jpg` for all 5 classes
   (`heart, oblong, oval, round, square`). See `data/README.md`. Do not proceed
   without it.
3. **Precompute** face-crop and aligned variants (one-time, needed for D2/D3/D4):
   ```
   python -m src.preprocess_faces --mode crop  --data-root data/raw --out data/preprocessed/crop
   python -m src.preprocess_faces --mode align --data-root data/raw --out data/preprocessed/align
   ```
4. **Smoke test before the full run:**
   `python -m src.train --model mobilenetv3small --preprocess D1 --epochs 1`
   Confirm it trains, evaluates, and writes `runs/mobilenetv3small_D1/results.json`.
5. **Run the full matrix:** `python -m src.run_matrix --output-dir runs`
   (it skips runs that already have `results.json`, so it is resumable).
6. **Report** `runs/summary.csv` plus the best model per version and best version
   per model. Highlight MobileNetV3Small (the deployment target).

## Acceptance criteria

- All 16 runs produce `results.json` + `confusion_matrix.png`.
- `runs/summary.csv` has 16 rows with accuracy and macro-F1.
- The fixed split (`splits/split.csv`, `seed=42`) is identical across all runs.
- No augmentation leaks into val/test (only train, only D3/D4).

## Hardware notes

- Device auto-detected: CUDA → Intel XPU → CPU.
- Memory-hungry nets are `swint` and `efficientnetv2s`; `run_matrix.py` already
  lowers their batch size to 16. If you hit CUDA OOM, lower `--batch-size`
  further or add gradient accumulation.
- 16 runs on a single T4 ≈ a few hours total.

## Do / don't

- **Do** keep input size at 224×224 for every model so preprocessing is the only
  variable that changes across the matrix.
- **Do** fix the implementation, not the metric, if results look wrong.
- **Don't** train on KOMNET or any Indonesian dataset — out of scope.
- **Don't** claim Indonesian/cross-population performance. Niten Lama bias is a
  disclosed limitation (female-celebrity, non-Indonesian).
- **Don't** commit the dataset, checkpoints, or `.venv` (see `.gitignore`).
