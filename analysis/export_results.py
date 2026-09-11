"""Export a cluster wave's results to tidy CSV, for plotting in the thesis.

**Why this exists.** Until 2026-09-11 every result lived only in Slurm logs and
unsynced W&B run directories on the cluster's scratch filesystem. That is a
single point of failure for the thesis's evidence, and it is not a form anyone
can plot from. These CSVs are small, version-controllable, and independent of
the cluster continuing to exist.

Two files, because the thesis needs two kinds of figure:

`<prefix>_eval.csv` — one row per evaluation (every checkpoint save)
    run, arm, seed, step, win_rate, hideout_uncert, copdist, arrest_pct,
    timeout_pct, difficulty
    Drives: final-performance bar charts with per-seed points; eval win rate
    vs training step; the behavioural mechanism (copdist / arrest_pct).

`<prefix>_training.csv` — one row per update, downsampled
    run, arm, seed, step, train_win_rate, difficulty
    Drives: the curriculum ramp figure — difficulty against step, which is what
    shows the controller engaging (or not).

**`win_rate` in the eval file is Director-free** (invariant 2), so it is
comparable across arms. `train_win_rate` in the training file is NOT — for a
Director-ON run it is the controller's setpoint. Do not plot them on one axis.

Stdlib only, so it runs on the cluster login node (which cannot import numpy).

Usage:
    uv run python -m analysis.export_results 'logs/wc-train_1807209_*.out' \
        --prefix director_branch --out-dir docs/completion/results
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path

from analysis.sweep_report import _UPD_FIELDS, _arm_key, parse_log

EVAL_COLUMNS = [
    "run", "arm", "seed", "step", "win_rate", "hideout_uncert",
    "copdist", "arrest_pct", "timeout_pct", "difficulty",
]
TRAIN_COLUMNS = ["run", "arm", "seed", "step", "train_win_rate", "difficulty"]


def export(
    paths: list[str], prefix: str, out_dir: Path | str, every: int = 10
) -> dict[str, int]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = [parse_log(p) for p in sorted(paths)]
    runs = [r for r in runs if r.evals or r.steps]

    eval_path = out / f"{prefix}_eval.csv"
    train_path = out / f"{prefix}_training.csv"
    n_eval = n_train = 0

    with eval_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=EVAL_COLUMNS)
        w.writeheader()
        for r in runs:
            for row in r.eval_rows:
                w.writerow({"run": r.name, "arm": _arm_key(r), "seed": r.seed, **row})
                n_eval += 1

    with train_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=TRAIN_COLUMNS)
        w.writeheader()
        for r in runs:
            rows = _training_rows(r.log_path)
            # Downsample: 3,432 updates x 15 runs is 51k rows for a line that
            # reads identically at every 10th point. Always keep the last row so
            # the final difficulty is exact rather than whatever the stride hit.
            kept = rows[::every] + ([rows[-1]] if rows and len(rows) % every else [])
            for step, wr, diff in kept:
                w.writerow({
                    "run": r.name, "arm": _arm_key(r), "seed": r.seed,
                    "step": step, "train_win_rate": wr, "difficulty": diff,
                })
                n_train += 1

    return {"runs": len(runs), "eval_rows": n_eval, "training_rows": n_train}


def _training_rows(path: str) -> list[tuple[int, float, float | None]]:
    """(step, training win rate, difficulty) for every update line in a log."""
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        m = _UPD_FIELDS.match(line)
        if m:
            d = re.search(r"\bdifficulty=([-+]?[\d.]+)", line)
            rows.append(
                (
                    int(m.group("steps").replace(",", "")),
                    float(m.group("win")),
                    float(d.group(1)) if d else None,
                )
            )
    return rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("logs", nargs="+", help="Slurm .out files (globs accepted)")
    p.add_argument("--prefix", required=True, help="output filename prefix")
    p.add_argument("--out-dir", default="docs/completion/results")
    p.add_argument(
        "--every", type=int, default=10, help="keep every Nth update row (default 10)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    expanded: list[str] = []
    for pattern in args.logs:
        expanded.extend(glob.glob(pattern) or [pattern])

    stats = export(expanded, args.prefix, args.out_dir, args.every)
    print(
        f"{stats['runs']} runs -> {stats['eval_rows']} eval rows, "
        f"{stats['training_rows']} training rows in {args.out_dir}/"
    )
