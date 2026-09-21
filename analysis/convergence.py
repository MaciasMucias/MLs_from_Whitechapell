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
# Seeds of the branch-era waves. Later waves use their own (04: 31-33,
# 10: 41-43), so seeds are taken from the CSV per arm; this is only the
# fallback for arms whose rows are missing entirely.
SEEDS = ("27", "28", "29")

# arm -> (wave file prefix, branched?, human-readable label)
ARMS = {
    # 09 — the Director waves, under the legacy reward.
    "cap000-b4060": ("director_cap", True, "curriculum, no injection"),
    "dbr-off": ("director_branch", True, "hand-picked two-phase"),
    "fs-off": ("director_v3", False, "no curriculum"),
    "fs-on": ("director_v3", False, "unbounded Director"),
    "cap015-b4060": ("director_cap", True, "curriculum, ceiling +0.15"),
    "dbr-fix05": ("director_branch", True, "fixed -0.5"),
    # 10 — the reward wave, trained on the stealth objective. These carry the
    # `score` column; the arms above predate it and show "-".
    #
    # Each takes TWO prefixes: the original 3 seeds (41-43) and the top-up
    # (61-63, array 1808812), pooled to 6 per arm. obj-cur's second triple is
    # w11-ent003 (51-53, array 1808727), which is that configuration
    # flag-for-flag; ARM_ALIASES folds it in.
    "w10s-obj-cur": (
        ("reward", "reward_topup", "entropy"),
        False,
        "objective, curriculum",
    ),
    "w10s-dlt-cur": (("reward", "reward_topup"), False, "+delta, curriculum"),
    # The two shaping arms add a third triple (71-73, array 1809122) for 9 seeds.
    "w10s-shp-cur": (
        ("reward", "reward_topup", "reward_topup2"),
        False,
        "+delta+shaping, curriculum",
    ),
    "w10s-obj-off": (("reward", "reward_topup"), False, "objective, no curriculum"),
    "w10s-dlt-off": (("reward", "reward_topup"), False, "+delta, no curriculum"),
    "w10s-shp-off": (
        ("reward", "reward_topup", "reward_topup2"),
        False,
        "+delta+shaping, no curriculum",
    ),
}

# Runs that are another arm's configuration under a different name.
ARM_ALIASES = {"w11-ent003": "w10s-obj-cur"}


def load_wave(prefix: str, results_dir: Path) -> dict[tuple[str, str], list[dict]]:
    """(arm, seed) -> eval rows, oldest first."""
    out: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    path = results_dir / f"{prefix}_eval.csv"
    if not path.exists():
        return out
    for row in csv.DictReader(path.open()):
        arm = ARM_ALIASES.get(row["arm"], row["arm"])
        out[(arm, row["seed"])].append(row)
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


def seeds_for(arm: str, waves: dict) -> tuple[str, ...]:
    """Seeds present for an arm. Waves use different seed sets (27-29, 31-33, 41-43)."""
    found = sorted({seed for (a, seed) in waves if a == arm})
    return tuple(found) if found else SEEDS


def steps_to(rows: list[dict], threshold: float) -> int | None:
    """First global step at which eval win rate reached `threshold` percent."""
    for row in rows:
        if float(row["win_rate"]) >= threshold:
            return int(row["step"])
    return None


def values(rows: list[dict], key: str) -> list[float]:
    """The numeric column, skipping evaluations that lack it (older waves)."""
    return [float(r[key]) for r in rows if r.get(key) not in (None, "")]


def steps_to_value(rows: list[dict], key: str, target: float) -> int | None:
    for row in rows:
        v = row.get(key)
        if v not in (None, "") and float(v) >= target:
            return int(row["step"])
    return None


def tail_slope(rows: list[dict], key: str, frac: float = 0.2) -> float:
    """Least-squares slope of `key` per 1M steps over the last `frac` of training.

    The plateau test. Exact potential-based shaping cannot change which policy is
    optimal, so a final-score gap between shaped and unshaped arms is only
    consistent with theory while the arms are still improving. A slope at or
    below ~0 in the unshaped arm would mean it has converged lower, which would
    be a bug rather than a finding.
    """
    pts = [
        (int(r["step"]), float(r[key])) for r in rows if r.get(key) not in (None, "")
    ]
    if len(pts) < 4:
        return float("nan")
    pts = pts[-max(2, int(len(pts) * frac)) :]
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    denom = sum((x - mx) ** 2 for x, _ in pts)
    if denom == 0:
        return float("nan")
    return (sum((x - mx) * (y - my) for x, y in pts) / denom) * 1e6


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def report(
    results_dir: Path,
    thresholds: list[float],
    last_k: int = 5,
    score_thresholds: tuple[float, ...] = (1.0, 1.1, 1.2, 1.3),
) -> None:
    waves: dict = {}
    prefixes = set()
    for p, _, _ in ARMS.values():
        prefixes.update((p,) if isinstance(p, str) else p)
    for prefix in sorted(prefixes):
        # Pool, do not overwrite: an arm's seeds can live in several waves.
        for key, rows in load_wave(prefix, results_dir).items():
            waves.setdefault(key, []).extend(rows)
    for rows in waves.values():
        rows.sort(key=lambda r: int(r["step"]))
    base = load_wave("director_base", results_dir)

    curves = {
        arm: [stitched_curve(arm, s, waves, base, br) for s in seeds_for(arm, waves)]
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
                if len(reached) == len(seeds)
                else f"{'never':>12}"
            )
        print(f"{arm:<16}" + "".join(cells))
    print()
    print("  Branch arms are stitched onto their base run, so step counts are")
    print("  comparable at equal total compute rather than flattering the branch.")

    scored = {
        arm: seeds
        for arm, seeds in curves.items()
        if all(values(c, "score") for c in seeds)
    }
    if scored:
        print(
            "\nPARTICIPANT SCORE — the objective every agent and human is reported on"
        )
        print(f"{'arm':<16}{'':28}{'best':>7}{'final':>8}{f'last-{last_k}':>9}")
        print("-" * 68)
        for arm, seeds in scored.items():
            sc = [values(c, "score") for c in seeds]
            print(
                f"{arm:<16}{ARMS[arm][2]:<28}"
                f"{_mean([max(x) for x in sc]):>7.3f}"
                f"{_mean([x[-1] for x in sc]):>8.3f}"
                f"{_mean([_mean(x[-last_k:]) for x in sc]):>9.3f}"
            )
        print()

        print("SAMPLE EFFICIENCY — steps to reach a FIXED participant score")
        print("(comparable across arms; the fraction table below is not)")
        print(f"{'arm':<16}" + "".join(f"{t:>12.2f}" for t in score_thresholds))
        print("-" * (16 + 12 * len(score_thresholds)))
        for arm, seeds in scored.items():
            cells = []
            for thr in score_thresholds:
                hits = [steps_to_value(c, "score", thr) for c in seeds]
                reached = [h for h in hits if h is not None]
                cells.append(
                    f"{_mean([float(h) for h in reached]) / 1e6:>11.1f}M"
                    if len(reached) == len(seeds)
                    else f"{'never':>12}"
                )
            print(f"{arm:<16}" + "".join(cells))
        print()

        print(
            "SAMPLE EFFICIENCY — steps to reach a fraction of the arm's OWN last-k score"
        )
        print(f"{'arm':<16}{'80%':>12}{'90%':>12}{'95%':>12}{'slope/1M':>11}")
        print("-" * 63)
        for arm, seeds in scored.items():
            targets = [_mean(values(c, "score")[-last_k:]) for c in seeds]
            cells = []
            for frac in (0.8, 0.9, 0.95):
                hits = [
                    steps_to_value(c, "score", t * frac) for c, t in zip(seeds, targets)
                ]
                reached = [h for h in hits if h is not None]
                cells.append(
                    f"{_mean([float(h) for h in reached]) / 1e6:>11.1f}M"
                    if len(reached) == len(seeds)
                    else f"{'never':>12}"
                )
            slope = _mean([tail_slope(c, "score") for c in seeds])
            print(f"{arm:<16}" + "".join(cells) + f"{slope:>+11.4f}")
        print()
        print("  Each arm is measured against its own end state, so this is speed,")
        print("  not quality. slope/1M is the least-squares trend of score over the")
        print("  last 20% of training: clearly positive means the arm was still")
        print("  improving when the budget ran out, so a final-score gap is a")
        print("  sample-efficiency difference rather than a different optimum.")
        print()

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
    p.add_argument(
        "--score-thresholds",
        type=float,
        nargs="+",
        default=[1.0, 1.1, 1.2, 1.3],
        help="Absolute participant-score levels for the cross-arm speed table",
    )
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    report(Path(a.results_dir), a.thresholds, a.last_k, tuple(a.score_thresholds))
