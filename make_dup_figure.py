"""Publication figure for the near-duplicate audit (Section II.A).

Reviewer 2: "The assertion about near-duplicate images needs stronger support."

review_dups.py already produced the full 17-pair contact sheet for internal review;
this renders a small, captioned subset fit for the manuscript. Reads the audit result
in paper_assets/dup_pairs.csv and the raw images. Writes only to paper_assets/.

    python make_dup_figure.py            # 3 exact duplicates (Hamming = 0)
    python make_dup_figure.py --n 5      # widen to the nearest near-duplicates too
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
CLASSES = ["heart", "oblong", "oval", "round", "square"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUT = Path("../paper_assets")


def raw_index(raw_root):
    idx = {}
    for label in CLASSES:
        for p in (Path(raw_root) / label).rglob("*"):
            if p.suffix.lower() in EXTS:
                idx[(label, p.name)] = p
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", default="../paper_assets/dup_pairs.csv")
    ap.add_argument("--raw-root", default="data/raw")
    ap.add_argument("--n", type=int, default=3, help="pairs to show, closest first")
    args = ap.parse_args()

    pairs = sorted(csv.DictReader(open(args.pairs_csv, encoding="utf-8")),
                   key=lambda r: int(r["hamming"]))
    exact = sum(1 for r in pairs if int(r["hamming"]) == 0)
    print(f"audit: {len(pairs)} cross-split pairs at Hamming <= 5, {exact} exact duplicates")

    idx = raw_index(args.raw_root)
    show = pairs[:args.n]

    fig, axes = plt.subplots(len(show), 2, figsize=(4.0, 2.25 * len(show)))
    axes = axes.reshape(len(show), 2)
    for r, row in enumerate(show):
        h = int(row["hamming"])
        for c, side in enumerate(("A", "B")):
            path = idx.get((row["class"], row[f"file{side}"]))
            ax = axes[r, c]
            if path is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            else:
                ax.imshow(Image.open(path).convert("RGB").resize((224, 224)))
            ax.set_title(f"{row[f'split{side}']}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        tag = "identical" if h == 0 else f"Hamming = {h}"
        axes[r, 0].set_ylabel(f"{row['class']}\n({tag})", fontsize=9)

    fig.suptitle("Cross-split near-duplicates found by the dHash audit", fontsize=10)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "fig_near_duplicates"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {stem}.png/.pdf  ({len(show)} pairs shown)")
    for row in show:
        print(f"  [{row['class']}] H={row['hamming']}  "
              f"{row['splitA']}:{row['fileA']}  <->  {row['splitB']}:{row['fileB']}")


if __name__ == "__main__":
    main()
