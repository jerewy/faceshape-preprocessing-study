"""Build the D1 resize cache: data/raw/<class>/*.jpg -> data/preprocessed/resize/.

D1 (resize only) reads from ``data/preprocessed/resize`` rather than ``data/raw`` so
that training does not re-decode full-size JPEGs every epoch. The crop and align caches
are produced by ``src/preprocess_faces.py``; this is the third cache the matrix needs.

Idempotent: files that already exist are skipped, so it is safe to re-run.

    python scripts/make_resize_cache.py
    python scripts/make_resize_cache.py --size 256 --data-root data/raw
"""
import argparse
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASSES = ["heart", "oblong", "oval", "round", "square"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--out", default="data/preprocessed/resize")
    # 256 rather than 224: leaves headroom for the RandomResizedCrop used by D3/D4,
    # and matches the cache the committed runs were trained against.
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    src, dst = Path(args.data_root), Path(args.out)
    made = skipped = failed = 0
    for cls in CLASSES:
        cls_dir = src / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")
        (dst / cls).mkdir(parents=True, exist_ok=True)
        for p in cls_dir.rglob("*"):
            if p.suffix.lower() not in EXTS:
                continue
            out = dst / cls / p.name
            if out.exists():
                skipped += 1
                continue
            try:
                Image.open(p).convert("RGB").resize((args.size, args.size)).save(out, quality=90)
                made += 1
            except Exception as exc:  # a few dataset images are truncated or corrupt
                print(f"skip {p.name}: {exc}")
                failed += 1

    total = sum(1 for c in CLASSES for _ in (dst / c).glob("*"))
    print(f"resize cache {dst} -> made {made}, existed {skipped}, failed {failed}, total {total}")


if __name__ == "__main__":
    main()
