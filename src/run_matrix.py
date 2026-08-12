"""Run the full 16-experiment matrix and collect runs/summary.csv.

Resumable: any experiment that already has a results.json is skipped.

    python -m src.run_matrix --output-dir runs
    python -m src.run_matrix --models mobilenetv3small --versions D1 D2   # subset
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

MODELS = ["resnet50", "efficientnetv2s", "mobilenetv3small", "swint"]
VERSIONS = ["D1", "D2", "D3", "D4"]
# Lower batch size for the memory-hungry nets so they fit on smaller GPUs.
BATCH = {"resnet50": 32, "efficientnetv2s": 16, "mobilenetv3small": 32, "swint": 16}
# ``seed`` onwards are absent from the original 16 runs and come out blank there.
SUMMARY_KEYS = ["model", "preprocess", "test_accuracy", "macro_f1",
                "macro_precision", "macro_recall", "best_val_macro_f1",
                "seed", "epochs_run", "train_seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="runs")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--versions", nargs="*", default=VERSIONS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra args passed through to src.train (after --extra)")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        for ver in args.versions:
            exp = f"{model}_{ver}"
            if (out / exp / "results.json").exists():
                print(f"[skip] {exp} already complete")
                continue
            cmd = [
                sys.executable, "-m", "src.train",
                "--model", model, "--preprocess", ver,
                "--batch-size", str(BATCH.get(model, 32)),
                "--epochs", str(args.epochs),
                "--output-dir", args.output_dir,
            ] + args.extra
            print("[run]", " ".join(cmd))
            subprocess.run(cmd, check=True)

    rows = []
    for model in args.models:
        for ver in args.versions:
            res = out / f"{model}_{ver}" / "results.json"
            if res.exists():
                rows.append(json.loads(res.read_text(encoding="utf-8")))
    if rows:
        with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_KEYS)
            w.writeheader()
            for d in rows:
                w.writerow({k: d.get(k) for k in SUMMARY_KEYS})
        print(f"[summary] wrote {out / 'summary.csv'} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
