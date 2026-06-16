# Dataset placement

This study trains and tests on the **Niten Lama Face Shape Dataset** only
(5 classes, ~5,000 images, CC0). Images are **not** committed to git.

## Required layout

Place images so each class is a subfolder of `data/raw/`:

```
data/raw/
  heart/   *.jpg
  oblong/  *.jpg
  oval/    *.jpg
  round/   *.jpg
  square/  *.jpg
```

If the download ships pre-split (e.g. `training_set/` + `testing_set/`), **merge**
the matching class folders together under `data/raw/<class>/`. This repo creates
its own fixed 70/15/15 split (`splits/split.csv`, `seed=42`) so all 16 runs use
identical partitions — do not rely on the dataset's own split.

## After placing images

```
# face-crop + aligned variants (needed for D2/D3/D4)
python -m src.preprocess_faces --mode crop  --data-root data/raw --out data/preprocessed/crop
python -m src.preprocess_faces --mode align --data-root data/raw --out data/preprocessed/align
```

These mirror the class/filename structure under `data/preprocessed/`. The split
csv remaps each raw filename onto the matching preprocessed file.

## Notes

- Generated folders (`data/preprocessed/`, `data/cache/`) and `splits/` are
  git-ignored.
- KOMNET / IMSFD / any Indonesian dataset is **out of scope** for this repo.
