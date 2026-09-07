"""Generate run_matrix.ipynb — the local VS Code runner for the 16-run matrix."""
import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": s}


cells = []

cells.append(md(r'''# Paper 2 — Run the 16-Run Matrix (local · VS Code)

Runs all **16 experiments** (4 preprocessing versions × 4 models) on your RTX 4060
and collects the results. Everything else is already on disk (data, crop/align, the
de-duplicated split), so this just builds the fast **D1 resize-cache** and trains.

**How to run:** select the **`.venv` (Python 3.13)** kernel (top-right), then **Run All**.

**Timeline:** resize cache ~2 min · smoke ~30 s · **full matrix ~2–3 h (leave running).**'''))

cells.append(md('## 0. Setup & sanity checks'))
cells.append(code(r'''import os, sys
from pathlib import Path

REPO_DIR = Path.cwd()   # <-- set to the repo root if this notebook is opened elsewhere
os.chdir(REPO_DIR)
print("cwd   :", os.getcwd())
print("python:", sys.executable)

import torch
print("torch :", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

for sub in ["data/raw", "data/preprocessed/resize", "data/preprocessed/crop",
            "data/preprocessed/align", "splits/split.csv"]:
    print(("OK   " if (REPO_DIR / sub).exists() else "MISSING ") + sub)'''))

cells.append(md('''## 1. Build the D1 resize-cache (the speed fix)
Downscales raw images once to `data/preprocessed/resize/` so D1 stops re-decoding
full-size JPEGs every epoch. Idempotent — skips files that already exist.'''))
cells.append(code(r'''from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASSES = ["heart", "oblong", "oval", "round", "square"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SRC, DST, SIZE = Path("data/raw"), Path("data/preprocessed/resize"), 256

made = skipped = 0
for c in CLASSES:
    (DST / c).mkdir(parents=True, exist_ok=True)
    for p in (SRC / c).rglob("*"):
        if p.suffix.lower() not in EXTS:
            continue
        out = DST / c / p.name
        if out.exists():
            skipped += 1; continue
        try:
            Image.open(p).convert("RGB").resize((SIZE, SIZE)).save(out, quality=90)
            made += 1
        except Exception as e:
            print("skip", p.name, e)
print(f"resize cache -> made {made}, existed {skipped}, total {sum(1 for c in CLASSES for _ in (DST/c).glob('*'))}")'''))

cells.append(md('## 2. Verify the D1 fix is active'))
cells.append(code(r'''import importlib
from src import preprocessing
importlib.reload(preprocessing)
assert preprocessing.SOURCE_BY_VERSION["D1"] == "resize", "D1 fix NOT applied — check src/preprocessing.py"
for v in ["D1", "D2", "D3", "D4"]:
    print(v, "->", preprocessing.source_root_for(v, "data/raw", "data/preprocessed"))'''))

cells.append(md('''## 3. Fresh start (optional)
`True` clears previous run outputs for a clean matrix; set `False` to resume after an
interruption (run_matrix skips any cell that already has a `results.json`).'''))
cells.append(code(r'''import shutil
FRESH_START = True
if FRESH_START and Path("runs").exists():
    shutil.rmtree("runs"); print("cleared runs/")
Path("runs").mkdir(exist_ok=True)'''))

cells.append(md('## 4. Smoke test — confirm D1 now loads fast (~30 s)'))
cells.append(code(r'''!"{sys.executable}" -m src.train --model mobilenetv3small --preprocess D1 --epochs 1'''))

cells.append(md('''## 5. Run the full 16-run matrix  (~2–3 h — leave it running)
Streams progress per run. You can keep using your PC; it only uses the GPU. Resumable.'''))
cells.append(code(r'''!"{sys.executable}" -m src.run_matrix --epochs 30 --extra --num-workers 6'''))

cells.append(md('## 6. Results — run after the matrix finishes'))
cells.append(code(r'''import pandas as pd
df = pd.read_csv("runs/summary.csv")
acc = df.pivot(index="model", columns="preprocess", values="test_accuracy").round(4)
f1  = df.pivot(index="model", columns="preprocess", values="macro_f1").round(4)
print("Test accuracy (model x version)"); display(acc)
print("Macro-F1 (model x version)"); display(f1)
best = df.loc[df.test_accuracy.idxmax(), ["model", "preprocess", "test_accuracy"]]
print("Best overall:", dict(best))
print("Best version per model:", df.loc[df.groupby('model').test_accuracy.idxmax(),
      ['model','preprocess','test_accuracy']].to_dict('records'))'''))

cells.append(code(r'''import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(acc.values, cmap="viridis")
ax.set_xticks(range(len(acc.columns))); ax.set_xticklabels(acc.columns)
ax.set_yticks(range(len(acc.index)));   ax.set_yticklabels(acc.index)
for i in range(acc.shape[0]):
    for j in range(acc.shape[1]):
        ax.text(j, i, f"{acc.values[i, j]:.3f}", ha="center", va="center", color="w", fontsize=9)
ax.set_title("Test accuracy — model x preprocessing"); fig.colorbar(im); plt.tight_layout()
out = Path("../paper_assets/fig_results_heatmap.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight"); fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
print("saved", out)'''))

cells.append(md('''## Done
`runs/summary.csv` has the 16 rows; per-run confusion matrices are in
`runs/<model>_<version>/`. The results figure is saved to `../paper_assets/`.'''))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python (.venv)", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
Path("run_matrix.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote run_matrix.ipynb with", len(cells), "cells")
