"""Pick a base run's branch point for the Director branch wave.

The branch point is the first checkpoint at or after the moment TRAINING win
rate **sustains** above the curriculum band top for RUN_LEN consecutive updates
— the point where the P-controller would first have started hardening
difficulty.

Sustained, not first-crossing, on purpose. Workstream 09's runs each touched
above the band exactly once, on the final update, and that was enough to make
the wave look like the curriculum had engaged when 975/976 of training had run
at a constant difficulty. One spike is not an operating regime.

Stdlib only, so it runs on the cluster login node (which cannot import numpy —
its CPU lacks x86-64-v2).

Usage:
    uv run python tools/branch_point.py 'logs/wc-train_1807190_*.out'
    uv run python tools/branch_point.py 'logs/*.out' --band-top 0.6 --run-len 20
"""

from __future__ import annotations

import argparse
import glob
import re

# "update=123/2604 steps=1,234,567 sps=... win_rate=0.612 difficulty=-1.000 ..."
_UPD = re.compile(r"^update=(\d+)/\d+ steps=([\d,]+).*?win_rate=([\d.]+)")
_RUN = re.compile(r"--wandb-run (\S+)")

# train.py saves a checkpoint every 50 updates (train.py: `update % 50 == 0`).
CKPT_EVERY = 50


def updates(text: str) -> list[tuple[int, int, float]]:
    """(update, global_step, training win rate) for every update line."""
    out = []
    for line in text.splitlines():
        m = _UPD.match(line)
        if m:
            out.append(
                (int(m.group(1)), int(m.group(2).replace(",", "")), float(m.group(3)))
            )
    return out


def sustained_from(
    rows: list[tuple[int, int, float]], band_top: float, run_len: int
) -> tuple[int, int] | None:
    """First (update, step) beginning a run of `run_len` updates above band_top."""
    streak = 0
    for i, (upd, step, wr) in enumerate(rows):
        streak = streak + 1 if wr > band_top else 0
        if streak >= run_len:
            return rows[i - run_len + 1][0], rows[i - run_len + 1][1]
    return None


def report(paths: list[str], band_top: float, run_len: int) -> int:
    missing = 0
    for path in sorted(paths):
        text = open(path, errors="replace").read()
        name_match = _RUN.search(text)
        name = name_match.group(1) if name_match else "?"
        rows = updates(text)
        if not rows:
            print(f"{name:14} no update lines — still buffered or crashed?")
            missing += 1
            continue

        hit = sustained_from(rows, band_top, run_len)
        if hit is None:
            best = max(wr for _, _, wr in rows)
            print(
                f"{name:14} NEVER sustained >{band_top} for {run_len} updates "
                f"(peak {best:.3f}) — DO NOT BRANCH, extend this base"
            )
            missing += 1
            continue

        upd, step = hit
        # Round up to the next checkpoint, which is the first saved policy that
        # is at or past the crossing.
        ck_upd = -(-upd // CKPT_EVERY) * CKPT_EVERY
        ck_step = next((s for u, s, _ in rows if u == ck_upd), rows[-1][1])
        print(
            f"{name:14} sustained from update {upd:>5} (step {step:>10,})  ->  "
            f"agent_{ck_step:010d}.pt"
        )
    return missing


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("logs", nargs="+", help="Slurm .out files (globs accepted)")
    p.add_argument("--band-top", type=float, default=0.60)
    p.add_argument("--run-len", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    expanded: list[str] = []
    for pattern in args.logs:
        expanded.extend(glob.glob(pattern) or [pattern])
    missing = report(expanded, args.band_top, args.run_len)
    if missing:
        print()
        print(f"{missing} run(s) did not reach a sustained crossing. Branching one of")
        print("those reproduces workstream 09: the arms would differ in flags that")
        print("the controller never gets to act on.")
