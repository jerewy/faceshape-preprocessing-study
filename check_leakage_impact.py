"""Does identity leakage change the *relative* preprocessing comparison?

The identity audit (``check_identity_leakage.py``) establishes that some held-out
images share an identity with a training image. That inflates absolute accuracy.
The claim the paper actually rests on is the weaker one: that leakage inflates
**every preprocessing arm equally**, so D2-D1 / D3-D2 / D4-D3 survive.

That claim is not self-evident. Cropping and alignment normalise away the framing
differences between two photographs of the same person (see the module docstring
of ``check_identity_leakage.py``), so D2/D3/D4 may memorise identities *better*
than D1 does. If so the D1->D3 gain is partly a leakage artefact and the central
result is overstated.

This script tests it directly: re-score every committed run on the subset of the
test set that has **no** identity match in training, and compare the contrasts.
No retraining and no dataset needed — the per-sample predictions are committed.
It needs only the audit's ``identity_leakage.csv``.

    python check_identity_leakage.py          # once, needs the dataset
    python check_leakage_impact.py            # then this, needs only the CSV
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

MODELS = ["resnet50", "efficientnetv2s", "mobilenetv3small", "swint"]
VERSIONS = ["D1", "D2", "D3", "D4"]
SEEDS = [42, 123, 2025]
SOURCES = {42: "runs", 123: "results_seed123", 2025: "results_seed2025"}
ABLATION_DIR = "results_align_ablation"
CONTRASTS = [("D1", "D2"), ("D2", "D3"), ("D3", "D4"), ("D1", "D3")]


def test_items(split_csv):
    """(label, filename) for each test row, in the order the predictions were written.

    ``load_split`` keeps split.csv row order and the test DataLoader runs with
    shuffle=False, so row k of the CSV's test rows is index k of every y_pred.
    """
    with Path(split_csv).open(encoding="utf-8") as f:
        return [(r["label"], Path(r["path"].replace("\\", "/")).name)
                for r in csv.DictReader(f) if r["split"] == "test"]


def leaked_test_keys(leakage_csv):
    """(class, filename) of every TEST image matched to a training image by the audit.

    identity_leakage.csv holds train x held-out pairs above the audit threshold.
    Val rows are ignored here: they cannot affect test accuracy.
    """
    keys = set()
    with Path(leakage_csv).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["heldout_split"] == "test":
                keys.add((r["heldout_class"], r["heldout_file"]))
    return keys


def load_predictions():
    """{(model, version, seed): y_pred} plus the shared targets, from committed files.

    runs/ stores one bundled npz; the follow-up directories store one per run.
    """
    preds, targets = {}, None
    bundle = np.load("runs/test_predictions.npz")
    targets = bundle["targets"]
    for m in MODELS:
        for v in VERSIONS:
            preds[(m, v, 42)] = bundle[f"{m}_{v}"]
    for seed in (123, 2025):
        for m in MODELS:
            for v in VERSIONS:
                z = np.load(f"{SOURCES[seed]}/{m}_{v}/test_predictions.npz")
                assert np.array_equal(z["y_true"], targets), f"target order differs: {m}_{v}@{seed}"
                preds[(m, v, seed)] = z["y_pred"]
    for m in MODELS:
        for v in ("D3n", "D4n"):
            z = np.load(f"{ABLATION_DIR}/{m}_{v}/test_predictions.npz")
            assert np.array_equal(z["y_true"], targets), f"target order differs: {m}_{v}"
            preds[(m, v, 42)] = z["y_pred"]
    return preds, targets


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leakage-csv", default="../paper_assets/identity_leakage.csv")
    ap.add_argument("--split-csv", default="splits/split.csv")
    ap.add_argument("--out", default="../paper_assets/leakage_impact.md")
    ap.add_argument("--null-draws", type=int, default=2000,
                    help="random subsets of the same size, to calibrate the shift")
    args = ap.parse_args()

    if not Path(args.leakage_csv).exists():
        raise SystemExit(
            f"missing {args.leakage_csv}\n"
            "Run `python check_identity_leakage.py` first — it needs the dataset and\n"
            "the crop/align caches. This script needs nothing but its CSV output."
        )

    items = test_items(args.split_csv)
    preds, targets = load_predictions()
    if len(items) != len(targets):
        raise SystemExit(f"split.csv has {len(items)} test rows but predictions have "
                         f"{len(targets)} — split.csv does not match the committed runs")

    leaked_keys = leaked_test_keys(args.leakage_csv)
    leaked = np.array([k in leaked_keys for k in items])
    clean = ~leaked
    n, n_leak, n_clean = len(targets), int(leaked.sum()), int(clean.sum())
    if n_clean == 0:
        raise SystemExit("every test image is flagged as leaked — nothing left to score")

    def acc(key, mask):
        return (preds[key][mask] == targets[mask]).mean() * 100

    L = [f"# Does identity leakage change the preprocessing comparison?\n",
         f"Test set: **{n}** images — **{n_leak}** ({n_leak / n * 100:.1f}%) share an identity "
         f"with a training image at the audit threshold, **{n_clean}** are clean.\n",
         f"Source: `{args.leakage_csv}`. Accuracies are 3-seed means (42 / 123 / 2025).\n"]

    print(f"test images: {n}   leaked: {n_leak} ({n_leak / n * 100:.1f}%)   clean: {n_clean}\n")

    L.append("\n## Per-arm accuracy, full test set vs leakage-free subset\n")
    L.append("| Model | Version | full % | clean % | drop |")
    L.append("|---|---|---|---|---|")
    print(f"{'model':17s} {'ver':4s} {'full':>7s} {'clean':>7s} {'drop':>7s}")
    for m in MODELS:
        for v in VERSIONS:
            f_ = np.mean([acc((m, v, s), slice(None)) for s in SEEDS])
            c_ = np.mean([acc((m, v, s), clean) for s in SEEDS])
            L.append(f"| {m} | {v} | {f_:.2f} | {c_:.2f} | {f_ - c_:+.2f} |")
            print(f"{m:17s} {v:4s} {f_:7.2f} {c_:7.2f} {f_ - c_:+7.2f}")

    L.append("\n## The contrasts the paper reports\n")
    L.append("If leakage inflates every arm equally, each `clean` column matches its "
             "`full` column and the paper's relative claims stand unchanged.\n")
    L.append("| Model | Contrast | full Δ | clean Δ | shift |")
    L.append("|---|---|---|---|---|")
    print(f"\n{'model':17s} {'contrast':10s} {'fullD':>7s} {'cleanD':>7s} {'shift':>7s}")
    worst = 0.0
    for m in MODELS:
        for a, b in CONTRASTS:
            f_ = np.mean([acc((m, b, s), slice(None)) - acc((m, a, s), slice(None)) for s in SEEDS])
            c_ = np.mean([acc((m, b, s), clean) - acc((m, a, s), clean) for s in SEEDS])
            worst = max(worst, abs(f_ - c_))
            L.append(f"| {m} | {b} vs {a} | {f_:+.2f} | {c_:+.2f} | {c_ - f_:+.2f} |")
            print(f"{m:17s} {b+' vs '+a:10s} {f_:+7.2f} {c_:+7.2f} {c_ - f_:+7.2f}")

    L.append("\n## Alignment ablation, de-confounded (seed 42 only)\n")
    L.append("| Model | full Δ | clean Δ | shift |")
    L.append("|---|---|---|---|")
    print(f"\n{'model':17s} {'D4n-D3n':>10s} {'clean':>8s} {'shift':>8s}")
    for m in MODELS:
        f_ = acc((m, "D4n", 42), slice(None)) - acc((m, "D3n", 42), slice(None))
        c_ = acc((m, "D4n", 42), clean) - acc((m, "D3n", 42), clean)
        L.append(f"| {m} | {f_:+.2f} | {c_:+.2f} | {c_ - f_:+.2f} |")
        print(f"{m:17s} {f_:+10.2f} {c_:+8.2f} {c_ - f_:+8.2f}")

    # Null: dropping ANY 121-image subset perturbs the contrasts a little. Calibrate
    # against random subsets of the same size, so the verdict compares the observed
    # shift to subsetting noise rather than to a number picked by eye.
    rng = np.random.default_rng(0)
    null = []
    for _ in range(args.null_draws):
        mask = np.zeros(n, bool)
        mask[rng.choice(n, n_clean, replace=False)] = True
        shifts = []
        for m in MODELS:
            for a, b in CONTRASTS:
                f_ = np.mean([acc((m, b, s), slice(None)) - acc((m, a, s), slice(None)) for s in SEEDS])
                c_ = np.mean([acc((m, b, s), mask) - acc((m, a, s), mask) for s in SEEDS])
                shifts.append(abs(c_ - f_))
        null.append(max(shifts))
    p95 = float(np.percentile(null, 95))
    verdict = ("within subsetting noise — the 'leakage affects all arms equally' "
               "assumption holds and the paper's relative claims stand"
               if worst <= p95 else
               "LARGER than subsetting noise — leakage is not equal across arms; "
               "check whether D1->D3 shrank, and report it")

    L.append(f"\n## Verdict\n\nLargest shift in any reported contrast: **{worst:.2f} points**.\n")
    L.append(f"Dropping a *random* subset of the same size ({n_clean} of {n}) shifts some "
             f"contrast by up to **{p95:.2f} points** at the 95th percentile over "
             f"{args.null_draws} draws. That is the noise floor from subsetting alone.\n")
    L.append(f"**{worst:.2f} vs {p95:.2f} → {verdict}.**\n")
    L.append("\nThe question this answers is narrow and worth stating precisely: leakage "
             "inflates *absolute* accuracy on this dataset regardless of the outcome above. "
             "What is tested here is only whether it inflates the preprocessing arms "
             "*unequally* — which is what the relative comparison depends on.\n")
    print(f"\nlargest observed shift : {worst:.2f} points")
    print(f"random-subset null (p95): {p95:.2f} points over {args.null_draws} draws")
    print(f"verdict: {verdict}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
