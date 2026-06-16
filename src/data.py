"""Fixed, seeded train/val/test split + dataset for Niten Lama.

The split is created once into ``splits/split.csv`` and reused across all 16
experiments so every (model, preprocessing) combination sees identical
partitions. It is deterministic given the same dataset files and ``seed``.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

CLASSES = ["heart", "oblong", "oval", "round", "square"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _gather(data_root: Path) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = {}
    for label in CLASSES:
        cls_dir = data_root / label
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")
        paths = [p for p in cls_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
        if not paths:
            raise RuntimeError(f"No images in {cls_dir}")
        by_class[label] = sorted(paths)
    return by_class


def make_split(data_root, out_csv, ratios=(0.70, 0.15, 0.15), seed=42):
    """Create a stratified per-class split and write it to ``out_csv``."""
    data_root, out_csv = Path(data_root), Path(out_csv)
    rng = random.Random(seed)
    rows = []
    for label, paths in _gather(data_root).items():
        paths = list(paths)
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        for i, p in enumerate(paths):
            split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            rows.append((str(p), label, split))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        w.writerows(rows)
    return rows


def load_split(split_csv, split, source_root=None):
    """Return ``[(path, label), ...]`` for one split.

    If ``source_root`` is given, each raw filename is remapped onto
    ``source_root/<label>/<filename>`` so D2/D3/D4 read the precomputed
    face-cropped / aligned variants instead of the raw image.
    """
    split_csv = Path(split_csv)
    items = []
    with split_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            path = Path(row["path"])
            if source_root is not None:
                path = Path(source_root) / row["label"] / path.name
            items.append((path, row["label"]))
    return items


class FaceShapeDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), CLASS_TO_IDX[label]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create the fixed train/val/test split.")
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--out", default="splits/split.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = make_split(args.data_root, args.out, seed=args.seed)
    counts = {s: sum(1 for _, _, sp in rows if sp == s) for s in ("train", "val", "test")}
    print(f"Wrote {args.out}: {counts}")
