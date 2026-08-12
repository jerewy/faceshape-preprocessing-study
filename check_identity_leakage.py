"""Identity-level leakage audit (extends the image-level dHash audit in Section II.A).

The dHash audit compares *raw files*. It cannot catch the same person photographed on
a different day: two different photographs have very different framing, so their raw
hashes diverge, yet after face cropping and alignment they become near-identical
inputs. That is the leakage mode that matters for a celebrity dataset, because the
model can memorise the face instead of learning the shape.

Embeds every image with InceptionResnetV1 (pretrained on VGGFace2, via facenet-pytorch)
and measures cross-split identity overlap. Read-only with respect to runs/ and splits/.

Statistics reported, and why:
  * a random-pair null, so thresholds are calibrated against this dataset rather than
    assumed from the face-recognition literature;
  * counts at conservative thresholds only. Per-image "max similarity to any training
    image" is a maximum over ~3,500 comparisons, so it is inflated by multiple
    comparisons and is NOT quoted as a leakage rate;
  * connected-component identity clusters that span a split boundary, which is the
    quantity the deduplication was supposed to drive to zero.

    python check_identity_leakage.py
    python check_identity_leakage.py --threshold 0.97   # stricter
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from PIL import Image, ImageFile
from scipy.sparse.csgraph import connected_components
from torch.utils.data import DataLoader, Dataset

from src.data import load_split

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("../paper_assets")
REPORT_THRESHOLDS = [0.99, 0.97, 0.95]


class ImageList(Dataset):
    """Faces at 160x160, the input size InceptionResnetV1 expects."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB").resize((160, 160))
        x = torch.from_numpy(np.asarray(img, np.float32)).permute(2, 0, 1)
        return (x - 127.5) / 128.0  # facenet's own normalisation


def collect(split_csv, align_root, crop_root):
    """Every split row, pointed at its aligned crop (falling back to the plain crop)."""
    rows = []
    for split in ("train", "val", "test"):
        for path, label in load_split(split_csv, split):
            name = Path(path).name
            src = Path(align_root) / label / name
            if not src.exists():
                src = Path(crop_root) / label / name
            if src.exists():
                rows.append({"path": src, "label": label, "split": split, "file": name})
    return rows


@torch.no_grad()
def embed(paths, batch_size=64):
    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
    dl = DataLoader(ImageList(paths), batch_size=batch_size, num_workers=0)
    out = []
    for i, x in enumerate(dl, 1):
        out.append(model(x.to(DEVICE)).cpu().numpy())
        if i % 20 == 0:
            print(f"  embedded {i * batch_size}/{len(paths)}")
    emb = np.concatenate(out)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def null_baseline(emb, n=200_000, seed=0):
    """Cosine distribution of random pairs — the dataset's own 'different face' null."""
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(emb), n), rng.integers(0, len(emb), n)
    m = i != j
    return (emb[i[m]] * emb[j[m]]).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-csv", default="splits/split.csv")
    ap.add_argument("--align-root", default="data/preprocessed/align")
    ap.add_argument("--crop-root", default="data/preprocessed/crop")
    ap.add_argument("--threshold", type=float, default=0.95,
                    help="conservative same-identity cut-off (validated visually)")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--cache", default="", help="optional .npy of precomputed embeddings")
    args = ap.parse_args()

    rows = collect(args.split_csv, args.align_root, args.crop_root)
    if args.cache and Path(args.cache).exists():
        emb = np.load(args.cache)
        print(f"loaded cached embeddings {emb.shape}")
    else:
        print(f"embedding {len(rows)} images on {DEVICE} ...")
        emb = embed([r["path"] for r in rows])
        if args.cache:
            np.save(args.cache, emb)

    split = np.array([r["split"] for r in rows])
    label = np.array([r["label"] for r in rows])
    tr, ho = np.where(split == "train")[0], np.where(split != "train")[0]

    null = null_baseline(emb)
    print(f"\nrandom-pair null: mean {null.mean():.3f}, sd {null.std():.3f}, "
          f"p99.9 {np.percentile(null, 99.9):.3f}")
    print("  (a discriminative embedding puts unrelated faces near 0 — it does)")

    sim = emb[tr] @ emb[ho].T

    print("\ncross-split identity overlap at conservative thresholds:")
    print(f"{'cos':>6} {'pairs':>7} {'held-out imgs':>14} {'expected by chance':>19}")
    n_pairs = sim.size
    for t in REPORT_THRESHOLDS:
        hits = int((sim >= t).sum())
        imgs = int((sim.max(0) >= t).sum())
        chance = float((null >= t).mean()) * n_pairs
        print(f"{t:6.2f} {hits:7d} {imgs:6d} ({imgs / len(ho) * 100:4.1f}%) {chance:19.0f}")

    print("\nNOTE: 'held-out imgs' is a max over ~%d comparisons per image, so it "
          "overstates\n      the leakage rate. The cluster count below is the "
          "defensible quantity." % len(tr))

    # Identity clusters that straddle the split boundary — what dedup should zero out.
    S = emb @ emb.T
    np.fill_diagonal(S, 0)
    ncomp, comp = connected_components(sp.csr_matrix(S >= args.threshold), directed=False)
    spanning, conflicting, leaked = 0, 0, 0
    for c in range(ncomp):
        idx = np.where(comp == c)[0]
        if len(idx) < 2:
            continue
        splits = set(split[idx])
        if len(splits) > 1:
            spanning += 1
            leaked += int(np.sum(split[idx] != "train"))
        if len(set(label[idx])) > 1:
            conflicting += 1

    print(f"\nidentity clusters at cos >= {args.threshold} (connected components):")
    print(f"  {ncomp} clusters over {len(rows)} images")
    print(f"  {spanning} clusters span more than one split  <-- cross-split identity leakage")
    print(f"  {leaked} held-out images sit in such a cluster "
          f"({leaked / len(ho) * 100:.1f}% of val+test)")
    print(f"  {conflicting} clusters carry MORE THAN ONE face-shape label "
          f"(annotation noise, independent of leakage)")

    ii, jj = np.where(sim >= args.threshold)
    order = np.argsort(-sim[ii, jj])
    pairs = [(tr[ii[k]], ho[jj[k]], float(sim[ii[k], jj[k]])) for k in order]

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "identity_leakage.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cosine", "train_file", "train_class",
                    "heldout_file", "heldout_class", "heldout_split", "same_class"])
        for a, b, s in pairs:
            w.writerow([round(s, 4), rows[a]["file"], rows[a]["label"],
                        rows[b]["file"], rows[b]["label"], rows[b]["split"],
                        label[a] == label[b]])

    if not pairs:
        print(f"\nno cross-split matches at cos >= {args.threshold} — the image-level "
              "audit was sufficient. That is itself a reportable result.")
        return

    # Sample across the similarity range, not just the top, so the figure is honest
    # about what a borderline match looks like.
    picks = [pairs[k] for k in np.linspace(0, len(pairs) - 1, min(args.top, len(pairs))).astype(int)]
    fig, axes = plt.subplots(len(picks), 2, figsize=(4.0, 2.3 * len(picks)),
                             layout="constrained")
    axes = np.array(axes).reshape(len(picks), 2)
    for r, (a, b, s) in enumerate(picks):
        for c, k in enumerate((a, b)):
            axes[r, c].imshow(Image.open(rows[k]["path"]).convert("RGB"))
            axes[r, c].set_title(f"{rows[k]['split']} · {rows[k]['label']}", fontsize=8)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(f"cos {s:.3f}", fontsize=8)
    fig.suptitle("Same identity across train and held-out splits", fontsize=10)
    stem = OUT / "fig_identity_leakage"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nwrote {stem}.png/.pdf and identity_leakage.csv ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
