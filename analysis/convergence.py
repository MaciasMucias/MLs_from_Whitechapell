"""Sample efficiency and end-state quality across training runs.

Final win rate hides two things the thesis needs:

* **How fast** an arm got there. The curriculum reaches 80% in ~32% fewer steps
  than no curriculum, and reaches 85% at all where the baseline never does
  inside the same 15M budget.
* **How it loses** when it loses. Two arms at the same win rate can differ
  sharply in arrest vs timeout, which says whether a policy is reckless or
  dawdling.

**Stitched curves.** Branch runs (`--branch-from`) inherit their first ~4.5M
steps from a base run and continue the global step counter, so their own logs
start mid-training. This module prepends the base run's evaluations before the
branch point, giving the full trajectory. Global step is then directly
comparable across branched and from-scratch arms at equal total compute — which
is what makes steps-to-threshold a fair measure rather than a flattering one.

**On `best` vs `final`.** `sweep_report` ranks on each run's best evaluation,
which is a maximum over ~69 noisy samples and therefore optimistically biased.
This module reports best, final and a last-k mean side by side so a conclusion
can be checked against the more conservative statistics. (For the Director
result they agree; the ordering does not change.)

Stdlib only, so it runs on the cluster login node.

Usage:
    uv run python -m analysis.convergence
    uv run python -m analysis.convergence --thresholds 50 70 80 85
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

RESULTS = Path("docs/completion/results")

# Where each seed's branch runs forked from its base (tools/branch_point.py).
BRANCH_POINT = {"27": 4_454_400, "28": 4_454_400, "29": 4_915_200}
SEEDS = ("27", "28", "29")

# arm -> (wave file prefix, branched?, human-readable label)
ARMS = {
    "cap000-b4060": ("director_cap", True, "curriculum, no injection"),
    "dbr-off": ("director_branch", True, "hand-picked two-phase"),
    "fs-off": ("director_v3", False, "no curriculum"),
    "fs-on": ("director_v3", False, "unbounded Director"),
    "cap015-b4060": ("director_cap", True, "curriculum, ceiling +0.15"),
    "dbr-fix05": ("director_branch", True, "fixed -0.5"),
}


def load_wave(prefix: str, results_dir: Path) -> dict[tuple[str, str], list[dict]]:
    """(arm, seed) -> eval rows, oldest first."""
    out: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    path = results_dir / f"{prefix}_eval.csv"
    if not path.exists():
        return out
    for row in csv.DictReader(path.open()):
        out[(row["arm"], row["seed"])].append(row)
    for rows in out.values():
        rows.sort(key=lambda r: int(r["step"]))
    return out


def stitched_curve(
    arm: str, seed: str, waves: dict, base: dict, branched: bool
) -> list[dict]:
    """Full trajectory for one run, prepending the base run when it branched."""
    rows = waves.get((arm, seed), [])
    if not branched or not rows:
        return rows
    prefix = [
        r for r in base.get(("dbase", seed), []) if int(r["step"]) < BRANCH_POINT[seed]
    ]
    return prefix + rows


def steps_to(rows: list[dict], threshold: float) -> int | None:
    """First global step at which eval win rate reached `threshold` percent."""
    for row in rows:
        if float(row["win_rate"]) >= threshold:
            return int(row["step"])
    return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def report(results_dir: Path, thresholds: list[float], last_k: int = 5) -> None:
    waves: dict = {}
    for prefix in {p for p, _, _ in ARMS.values()}:
        waves.update(load_wave(prefix, results_dir))
    base = load_wave("director_base", results_dir)

    curves = {
        arm: [stitched_curve(arm, s, waves, base, br) for s in SEEDS]
        for arm, (_, br, _) in ARMS.items()
    }
    curves = {a: c for a, c in curves.items() if all(c)}
    if not curves:
        print(f"No result CSVs found in {results_dir}.")
        return

    print(f"\nEND STATE — eval win% (Director-free, 200 games per evaluation)")
    print(f"{'arm':<16}{'':28}{'best':>7}{'final':>7}{f'last-{last_k}':>9}")
    print("-" * 67)
    for arm, seeds in curves.items():
        w = [[float(r["win_rate"]) for r in c] for c in seeds]
        print(
            f"{arm:<16}{ARMS[arm][2]:<28}"
            f"{_mean([max(x) for x in w]):>6.1f}%"
            f"{_mean([x[-1] for x in w]):>6.1f}%"
            f"{_mean([_mean(x[-last_k:]) for x in w]):>8.1f}%"
        )
    print()
    print("  'best' is a maximum over ~69 evaluations and is optimistically biased.")
    print("  Check any conclusion against 'final' and the last-k mean.")

    print(f"\nSAMPLE EFFICIENCY — global steps to first reach each win rate")
    print(f"{'arm':<16}" + "".join(f"{t:>11.0f}%" for t in thresholds))
    print("-" * (16 + 12 * len(thresholds)))
    for arm, seeds in curves.items():
        cells = []
        for thr in thresholds:
            hits = [steps_to(c, thr) for c in seeds]
            reached = [h for h in hits if h is not None]
            cells.append(
                f"{_mean([float(h) for h in reached]) / 1e6:>11.1f}M"
                if len(reached) == len(SEEDS)
                else f"{'never':>12}"
            )
        print(f"{arm:<16}" + "".join(cells))
    print()
    print("  Branch arms are stitched onto their base run, so step counts are")
    print("  comparable at equal total compute rather than flattering the branch.")

    print(f"\nHOW THEY LOSE — final evaluation, mean over seeds")
    print(f"{'arm':<16}{'copdist':>9}{'arrest%':>9}{'timeout%':>10}{'hideout_u':>11}")
    print("-" * 55)
    for arm, seeds in curves.items():
        last = [c[-1] for c in seeds]
        vals = {
            k: _mean([float(r[k]) for r in last if r.get(k)])
            for k in ("copdist", "arrest_pct", "timeout_pct", "hideout_uncert")
        }
        print(
            f"{arm:<16}{vals['copdist']:>9.2f}{vals['arrest_pct']:>9.1f}"
            f"{vals['timeout_pct']:>10.1f}{vals['hideout_uncert']:>11.2f}"
        )
    print()
    print("  Equal copdist with fewer arrests AND fewer timeouts means the policy")
    print("  is playing better, not merely more cautiously.")
    print()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results-dir", default=str(RESULTS))
    p.add_argument("--thresholds", type=float, nargs="+", default=[50, 70, 80, 85])
    p.add_argument("--last-k", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    report(Path(a.results_dir), a.thresholds, a.last_k)
