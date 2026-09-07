# Dataset placement

This study trains and tests on the **Niten Lama Face Shape Dataset** only
(5 classes, ~5,000 images, CC0):
<https://www.kaggle.com/datasets/niten19/face-shape-dataset>

Images are **not** committed to git.

## Required layout

Place images so that each class is a subfolder of `data/raw/`:

```
data/raw/
  heart/   *.jpg
  oblong/  *.jpg
  oval/    *.jpg
  round/   *.jpg
  square/  *.jpg
```

If the download ships pre-split (e.g. `training_set/` + `testing_set/`), **merge** the
matching class folders together under `data/raw/<class>/`. This repository uses its own
fixed 70/15/15 split so that all runs see identical partitions — do not rely on the
dataset's own split.

Remove any non-image files before proceeding. The Kaggle download has been observed to
contain a Windows `desktop.ini` inside `testing_set/Round`, which makes the class count
read as 1001 instead of 1000.

## After placing images

Build all three preprocessing caches. Every one of them is required by some
preprocessing version: D1 reads `resize`, D2/D3/D3n read `crop`, D4/D4n read `align`.

```bash
python scripts/make_resize_cache.py
python -m src.preprocess_faces --mode crop  --data-root data/raw --out data/preprocessed/crop
python -m src.preprocess_faces --mode align --data-root data/raw --out data/preprocessed/align
```

These mirror the class/filename structure under `data/preprocessed/`. The split CSV
remaps each raw filename onto the matching preprocessed file, so the caches must keep
the original filenames.

## What is and is not tracked

- `data/` is git-ignored in full — raw images and all generated caches
  (`data/preprocessed/`, `data/cache/`).
- `splits/` **is** tracked. `splits/split.csv` is the exact partition used by every
  committed run, so results reproduce without regenerating the split. See "Split
  provenance" in the top-level `README.md` for how it differs from
  `splits/split_original.csv`.
- Model checkpoints (`best.pt`, ~110 MB each) are git-ignored.

## Out of scope

KOMNET, IMSFD, and any other Indonesian dataset are out of scope for this repository.
