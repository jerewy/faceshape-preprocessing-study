"""Run the full reviewer-response matrix from a terminal (not a notebook).

An 18-hour job should not depend on a Jupyter kernel staying alive. This is the same
plan as the Run All cell in run_followup.ipynb, as a plain script that streams to both
the console and followup_log.txt.

Fully resumable: src.run_matrix skips any experiment that already has a results.json,
so an interruption costs only the run that was in flight. Re-run the same command.

    .venv\\Scripts\\python.exe run_followup.py              # all three tiers
    .venv\\Scripts\\python.exe run_followup.py --tier 1     # just the ablation
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

LOG = Path("followup_log.txt")
ALIGN_DIR = "results_align_ablation"
SEED_DIRS = {123: "results_seed123", 2025: "results_seed2025"}

# (tier, label, output dir, preprocessing versions, training seed)
PLAN = [
    (1, "Tier 1  alignment ablation", ALIGN_DIR, ["D3n", "D4n"], 42),
    (2, "Tier 2a seed 123", SEED_DIRS[123], ["D1", "D2", "D3", "D4"], 123),
    (3, "Tier 2b seed 2025", SEED_DIRS[2025], ["D1", "D2", "D3", "D4"], 2025),
]
MEAN_MINUTES_PER_RUN = 28  # measured from the original 16-run matrix


def log(msg=""):
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def guard():
    """Refuse to start if the original results are missing or would be overwritten."""
    originals = sorted(p.parent.name for p in Path("runs").glob("*/results.json"))
    if len(originals) != 16:
        sys.exit(f"ABORT: expected 16 original runs in runs/, found {len(originals)}")
    for _, _, d, _, _ in PLAN:
        if Path(d).resolve() == Path("runs").resolve():
            sys.exit(f"ABORT: {d} would overwrite runs/")
    return {
        d: round(json.loads(Path(f"runs/{d}/results.json").read_text())["test_accuracy"], 6)
        for d in originals
    }


def verify(fingerprint):
    now = {
        d: round(json.loads(Path(f"runs/{d}/results.json").read_text())["test_accuracy"], 6)
        for d in sorted(p.parent.name for p in Path("runs").glob("*/results.json"))
    }
    if now == fingerprint:
        log(f"OK - all {len(now)} original runs unchanged")
    else:
        changed = {k for k in set(now) | set(fingerprint) if now.get(k) != fingerprint.get(k)}
        log(f"WARNING — original runs changed: {changed}")


def run_tier(label, out_dir, versions, seed):
    cmd = [sys.executable, "-m", "src.run_matrix",
           "--output-dir", out_dir, "--versions", *versions,
           "--epochs", "30",
           "--extra", "--seed", str(seed), "--num-workers", "6"]
    log(f"\n{'=' * 72}\n{label}  ->  {out_dir}/   (seed {seed}, {len(versions) * 4} runs)")
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}\n{'=' * 72}")

    # Stream the child's output so progress is visible live and captured in the log.
    # PYTHONUNBUFFERED is inherited by src.run_matrix *and* the src.train processes it
    # spawns; without it Python block-buffers because its stdout is a pipe, not a
    # terminal, and progress would only surface in 8 KB chunks hours apart.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1,
                            env=env)
    for line in proc.stdout:
        log(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        log(f"\n{label} FAILED with exit code {proc.returncode}. "
            "Fix the cause and re-run this script — finished runs are skipped.")
        sys.exit(proc.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, choices=[1, 2, 3], default=None,
                    help="run only one tier (default: all three, in order)")
    args = ap.parse_args()

    plan = [p for p in PLAN if args.tier is None or p[0] == args.tier]
    total_runs = sum(len(v) * 4 for _, _, _, v, _ in plan)
    eta = timedelta(minutes=total_runs * MEAN_MINUTES_PER_RUN)

    fingerprint = guard()
    log(f"\n{'#' * 72}")
    log(f"# follow-up matrix: {total_runs} runs, estimated {eta} "
        f"(~{MEAN_MINUTES_PER_RUN} min/run)")
    log(f"# expected finish  : {datetime.now() + eta:%Y-%m-%d %H:%M}")
    log(f"# log file         : {LOG.resolve()}")
    log(f"# safe to interrupt: finished runs are skipped on restart")
    log(f"{'#' * 72}")

    t0 = time.time()
    try:
        for _, label, out_dir, versions, seed in plan:
            run_tier(label, out_dir, versions, seed)
            log(f"\n[{label}] complete - {(time.time() - t0) / 3600:.1f} h elapsed")
    except KeyboardInterrupt:
        log("\nInterrupted. Re-run this script to resume; completed runs are skipped.")
        sys.exit(130)

    log(f"\nALL TIERS COMPLETE in {(time.time() - t0) / 3600:.1f} h")
    verify(fingerprint)
    log("Next: open run_followup.ipynb and run the analysis cells (sections 5 onward).")


if __name__ == "__main__":
    main()
