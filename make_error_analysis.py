"""Error analysis figure + table for Section III.D.

Reviewer 1: "Additional error analysis using representative misclassified samples
would also strengthen the discussion."

Shows the most confidently wrong test images for the deployment model, concentrating
on the oval/round/oblong band the paper identifies as genuinely ambiguous. Reads the
archived checkpoints; writes only to paper_assets/.

    python make_error_analysis.py                       # deployment model, D3
    python make_error_analysis.py --model swint --version D4
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader

from src.data import CLASSES, FaceShapeDataset, load_split
from src.models import build_model
from src.preprocessing import build_transform, source_root_for

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("../paper_assets")


@torch.no_grad()
def predict_with_confidence(run_dir, model_name, version, split_csv, data_root, pre_root):
    """Return (paths, y_true, y_pred, prob_of_predicted_class)."""
    src = source_root_for(version, data_root, pre_root)
    items = load_split(split_csv, "test", source_root=src)
    ds = FaceShapeDataset(items, build_transform(version, "test", 224))
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(model_name, num_classes=len(CLASSES), pretrained=False).to(DEVICE).eval()
    model.load_state_dict(torch.load(Path(run_dir) / f"{model_name}_{version}" / "best.pt",
                                     map_location=DEVICE, weights_only=True))
    probs, tgts = [], []
    for x, y in dl:
        probs.append(torch.softmax(model(x.to(DEVICE)), dim=1).cpu().numpy())
        tgts.append(y.numpy())
    probs = np.concatenate(probs)
    return ([p for p, _ in items], np.concatenate(tgts),
            probs.argmax(1), probs.max(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenetv3small", help="deployment model by default")
    ap.add_argument("--version", default="D3")
    ap.add_argument("--run-dir", default="runs")
    ap.add_argument("--split-csv", default="splits/split.csv")
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--preprocessed-root", default="data/preprocessed")
    ap.add_argument("--n", type=int, default=8, help="misclassified samples to display")
    args = ap.parse_args()

    paths, y_true, y_pred, conf = predict_with_confidence(
        args.run_dir, args.model, args.version,
        args.split_csv, args.data_root, args.preprocessed_root)
    wrong = np.where(y_true != y_pred)[0]
    print(f"{args.model}_{args.version}: {len(wrong)}/{len(y_true)} misclassified "
          f"({len(wrong) / len(y_true) * 100:.1f}%)")

    # Which confusions dominate — the quantitative half of the error analysis.
    pairs = {}
    for i in wrong:
        key = (CLASSES[y_true[i]], CLASSES[y_pred[i]])
        pairs[key] = pairs.get(key, 0) + 1
    ranked = sorted(pairs.items(), key=lambda kv: -kv[1])
    print("\ntop confusions (true -> predicted):")
    for (t, p), n in ranked[:8]:
        print(f"  {t:8s} -> {p:8s}  {n:3d}  ({n / len(wrong) * 100:4.1f}% of errors)")

    with (OUT / "error_analysis_confusions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true", "predicted", "count", "pct_of_errors"])
        for (t, p), n in ranked:
            w.writerow([t, p, n, round(n / len(wrong) * 100, 2)])

    # Most confidently wrong first — these are the informative failures.
    picks = wrong[np.argsort(-conf[wrong])][:args.n]
    cols = 4
    rows = int(np.ceil(len(picks) / cols))
    # constrained_layout keeps the two-line per-axes titles from colliding with the
    # row above, which tight_layout does not handle here.
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.75 * rows),
                             layout="constrained")
    for ax, i in zip(np.ravel(axes), picks):
        ax.imshow(Image.open(paths[i]).convert("RGB").resize((224, 224)))
        ax.set_title(f"true {CLASSES[y_true[i]]}\npred {CLASSES[y_pred[i]]} ({conf[i]:.2f})",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in np.ravel(axes)[len(picks):]:
        ax.axis("off")
    fig.suptitle(f"Most confident misclassifications — {args.model}, {args.version}",
                 fontsize=10)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"fig_error_analysis_{args.model}_{args.version}"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nwrote {stem}.png/.pdf and error_analysis_confusions.csv")


if __name__ == "__main__":
    main()
