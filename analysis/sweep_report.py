"""Rank a sweep's runs from their Slurm logs and pick the winning config.

Reads the `.out` files an array wave produced, extracts each task's flags and its
`eval:` lines, and ranks by **Director-free `eval/win_rate`** — never
`charts/win_rate`, which for a Director-ON run is the P-controller's setpoint
rather than a score (see docs/completion/09-director-tuning.md).

**Stdlib only, on purpose.** The cluster's login node cannot run numpy at all —
its CPU lacks x86-64-v2 and the wheel aborts — so anything that imports numpy or
torch has to go through `srun`. Pure text parsing runs anywhere.

Note the logs of a wave launched before `PYTHONUNBUFFERED=1` landed stay nearly
empty until each task exits and Python flushes. That is not a stall; see
docs/completion/slurm/README.md.

Usage:
    uv run python -m analysis.sweep_report logs/wc-train_1806901_*.out
    uv run python -m analysis.sweep_report 'logs/*.out' --fill \
        docs/completion/slurm/manifests/sweep_w2.txt
"""

from __future__ import annotations

import argparse
import glob
import re
from dataclasses import dataclass, field
from pathlib import Path

# "  eval: win_rate=61.5% turns=8.4 turns(W)=9.6 turns(L)=6.4 hideout_u=0.75 ..."
_EVAL = re.compile(
    r"eval:\s*win_rate=(?P<win>[\d.]+)%"
    r"(?:.*?hideout_u=(?P<unc>[\d.]+))?"
    r"(?:.*?copdist=(?P<copdist>[\d.]+))?"
    r"(?:.*?arrest=(?P<arrest>[\d.]+)%)?"
    r"(?:.*?timeout=(?P<timeout>[\d.]+)%)?"
)
_STEPS = re.compile(r"^update=\S+\s+steps=([\d,]+)")
# curriculum/difficulty, printed on every update line as e.g. "difficulty=-1.000".
# For a Director run this is the progress signal that charts/win_rate is not:
# pinned at -1.000 means the P-controller never ramped and the run trained
# against maximally-suppressed cops rather than a curriculum.
_DIFF = re.compile(r"\bdifficulty=([-+]?[\d.]+)")
_RUNLINE = re.compile(r"^run:\s+(.*)$")
# The (?!--) matters: without it a boolean flag swallows the next flag as its
# value, so "--no-curriculum --wandb-run x" loses the run name entirely.
_FLAG = re.compile(r"--(\S+)(?:\s+(?!--)(\S+))?")

# Below this spread the wave has not separated its configs and picking an argmax
# is picking noise. 3M steps at 200 eval games is not a precise instrument.
NOISE_PP = 3.0

# What counts as the curriculum actually running. -0.95 rejects dithering just
# off the -1.0 clamp; 5% of updates rejects a single late nudge. Workstream 09
# scored -0.987 on 1/976 updates and would fail both.
MIN_DIFFICULTY_MARGIN = -0.95
MIN_OFF_FLOOR_FRACTION = 0.05


@dataclass
class Run:
    log: str
    name: str = "?"
    lr: str = "?"
    ent: str = "?"
    curriculum: bool = True
    evals: list[tuple[int, float]] = field(default_factory=list)  # (step, win%)
    last_uncert: float | None = None
    last_copdist: float | None = None
    last_arrest: float | None = None
    steps: int = 0
    finished: bool = False
    seed: str = "?"
    final_difficulty: float | None = None
    max_difficulty: float | None = None
    start_difficulty: float | None = None
    updates: int = 0
    updates_off_floor: int = 0

    @property
    def off_floor_fraction(self) -> float:
        """Share of updates spent away from the starting difficulty."""
        return self.updates_off_floor / self.updates if self.updates else 0.0

    @property
    def curriculum_engaged(self) -> bool:
        """Did the curriculum actually happen, as opposed to twitching once?

        An earlier version returned True for any `max_difficulty > -1.0`. That
        was far too generous: workstream 09's runs moved difficulty on update
        976 of 976 and were reported as "curriculum engaged in 15/16 ON runs",
        when in fact 975/976 of training ran at a constant difficulty — which is
        why six configurations produced byte-identical policies.

        A run counts as engaged only if difficulty moved by a visible margin AND
        stayed off the floor for a non-trivial share of training. Both halves
        matter: the margin rejects numerical dithering, the fraction rejects a
        single late nudge.
        """
        if self.max_difficulty is None:
            return False
        moved = self.max_difficulty > MIN_DIFFICULTY_MARGIN
        return moved and self.off_floor_fraction >= MIN_OFF_FLOOR_FRACTION

    @property
    def best_win(self) -> float:
        return max((w for _, w in self.evals), default=float("nan"))

    @property
    def final_win(self) -> float:
        return self.evals[-1][1] if self.evals else float("nan")

    @property
    def arm(self) -> str:
        return "ON" if self.curriculum else "OFF"


def parse_log(path: str | Path) -> Run:
    run = Run(log=Path(path).name)
    text = Path(path).read_text(errors="replace")
    step = 0

    for line in text.splitlines():
        m = _RUNLINE.match(line)
        if m:
            flags = dict(
                (k, v) for k, v in _FLAG.findall(m.group(1)) if not k.startswith("-")
            )
            run.name = flags.get("wandb-run", "?")
            run.lr = flags.get("lr", "?")
            run.ent = flags.get("ent-coef", "?")
            run.seed = flags.get("seed", "27")  # train.py's default
            run.curriculum = "no-curriculum" not in flags
            continue

        m = _STEPS.match(line)
        if m:
            step = int(m.group(1).replace(",", ""))
            run.steps = max(run.steps, step)
            run.updates += 1
            d = _DIFF.search(line)
            if d:
                value = float(d.group(1))
                run.final_difficulty = value
                if run.max_difficulty is None:
                    # First update's value is the floor this run started from.
                    run.start_difficulty = value
                    run.max_difficulty = value
                else:
                    run.max_difficulty = max(run.max_difficulty, value)
                if run.start_difficulty is not None and value != run.start_difficulty:
                    run.updates_off_floor += 1
            continue

        m = _EVAL.search(line)
        if m:
            run.evals.append((step, float(m.group("win"))))
            for attr, key in (
                ("last_uncert", "unc"),
                ("last_copdist", "copdist"),
                ("last_arrest", "arrest"),
            ):
                if m.group(key) is not None:
                    setattr(run, attr, float(m.group(key)))

    run.finished = "Training complete." in text
    return run


def _fmt(value: float | None, spec: str = "6.2f") -> str:
    return "-" if value is None else format(value, spec)


def _arm_key(run: Run) -> str:
    """A run's configuration with the seed stripped out.

    Run names in a seeded wave are `<arm>-s27`/`-s28`/`-s29`, so dropping a
    trailing seed suffix groups replicates. Falls back to the full name, which
    simply leaves an unseeded run in a group of its own.
    """
    return re.sub(r"[-_]s?\d+$", "", run.name)


def _print_seed_groups(runs: list[Run]) -> None:
    """Aggregate replicates so a seeded wave can be read without eyeballing.

    Reports mean and half-range across seeds. Half-range rather than a standard
    deviation because three seeds is far too few for the latter to mean much,
    and the spread is what the reader actually needs against the noise floor.
    """
    groups: dict[str, list[Run]] = {}
    for r in runs:
        if r.evals:
            groups.setdefault(_arm_key(r), []).append(r)

    replicated = {k: v for k, v in groups.items() if len(v) > 1}
    if not replicated:
        return  # unseeded wave — the per-run table above is the whole story

    print("Across seeds (mean ± half-range of best eval win%):")
    col = f"  {'arm':<26} {'n':>2} {'mean':>7} {'range':>16} {'engaged':>8}"
    print(col)
    print("  " + "-" * (len(col) - 2))
    for key, members in sorted(
        replicated.items(),
        key=lambda kv: -sum(m.best_win for m in kv[1]) / len(kv[1]),
    ):
        wins = [m.best_win for m in members]
        mean = sum(wins) / len(wins)
        half = (max(wins) - min(wins)) / 2
        engaged = sum(m.curriculum_engaged for m in members)
        print(
            f"  {key:<26} {len(members):>2} {mean:>6.1f}% "
            f"{f'±{half:.1f} [{min(wins):.1f}-{max(wins):.1f}]':>16} "
            f"{f'{engaged}/{len(members)}':>8}"
        )
    print()
    singles = len(groups) - len(replicated)
    if singles:
        print(f"  ({singles} configuration(s) had only one seed — not aggregated.)")
        print()


def report(paths: list[str], noise_pp: float = NOISE_PP) -> list[Run]:
    runs = [parse_log(p) for p in sorted(paths)]
    runs = [r for r in runs if r.evals or r.steps]
    if not runs:
        print("\nNo parseable runs. If the wave is still going, the logs may be")
        print("buffered — check checkpoint mtimes instead (slurm/README.md).")
        return []

    unfinished = [r for r in runs if not r.finished]
    print(f"\n{len(runs)} runs, {len(runs) - len(unfinished)} finished")
    if unfinished:
        print(f"WARNING: {len(unfinished)} still running — ranking is provisional.")
    print()

    col = (
        f"{'run':<24} {'arm':>4} {'seed':>5} {'steps':>10} "
        f"{'best win%':>10} {'final':>7} {'diff_max':>9} {'diff_end':>9} "
        f"{'off_floor%':>11} {'hideout_u':>10} {'copdist':>8} {'arrest%':>8}"
    )
    print(col)
    print("-" * len(col))
    for r in sorted(runs, key=lambda r: (-(r.best_win == r.best_win), -r.best_win)):
        print(
            f"{r.name:<24} {r.arm:>4} {r.seed:>5} {r.steps:>10,} "
            f"{_fmt(r.best_win, '9.1f'):>10} {_fmt(r.final_win, '6.1f'):>7} "
            f"{_fmt(r.max_difficulty, '8.3f'):>9} "
            f"{_fmt(r.final_difficulty, '8.3f'):>9} "
            f"{r.off_floor_fraction * 100:>10.1f} "
            f"{_fmt(r.last_uncert):>10} {_fmt(r.last_copdist):>8} "
            f"{_fmt(r.last_arrest):>8}"
        )
    print()
    _print_seed_groups(runs)

    # For a Director wave this is the finding, ahead of any ranking: if the
    # controller never left its floor, the runs are "train at fixed suppression",
    # not curriculum learning, and the grid is not what needs changing.
    on_runs = [r for r in runs if r.arm == "ON" and r.max_difficulty is not None]
    if on_runs:
        engaged = [r for r in on_runs if r.curriculum_engaged]
        if not engaged:
            print(
                f"*** curriculum/difficulty stayed pinned at its floor in ALL "
                f"{len(on_runs)} ON runs. ***"
            )
            print("No run curricularised — each trained at a fixed suppression level.")
            print("The P-controller only ramps once win rate leaves the deadband from")
            print("above, so this usually means the runs are too short, not that the")
            print("Director settings are wrong. See 09-director-tuning.md.")
            print()
        else:
            print(
                f"curriculum engaged (difficulty left its floor) in "
                f"{len(engaged)}/{len(on_runs)} ON runs."
            )
            print()
    return runs


def pick(runs: list[Run], arm: str = "ON", noise_pp: float = NOISE_PP):
    """Best config in one arm, by eval win rate, with the noise caveat applied."""
    pool = [r for r in runs if r.arm == arm and r.evals]
    if not pool:
        print(f"No {arm}-arm runs with eval results — cannot pick.")
        return None

    ranked = sorted(pool, key=lambda r: -r.best_win)
    winner, spread = ranked[0], ranked[0].best_win - ranked[-1].best_win

    print(
        f"Best {arm}-arm config: {winner.name}  lr={winner.lr} ent-coef={winner.ent}"
        f"  (best eval win rate {winner.best_win:.1f}%)"
    )
    print(f"Spread across the {arm} arm: {spread:.1f} points")

    # A one-run arm has zero spread by definition; warning there is noise.
    if len(ranked) > 1 and spread < noise_pp:
        print()
        print(f"*** CAUTION: spread is under {noise_pp:.0f} points. ***")
        print("The wave has not separated these configs, so the top row is close to")
        print("an arbitrary pick. 03-ppo-sweep.md says the response is LONGER runs,")
        print("not a different grid. Consider re-running at 6M steps before")
        print("committing the final runs to this config.")
    if len(ranked) > 1:
        second = ranked[1]
        print(
            f"Runner-up: {second.name} (lr={second.lr} ent={second.ent}, "
            f"{second.best_win:.1f}%) — {winner.best_win - second.best_win:.1f} "
            f"points behind."
        )
    return winner


def fill_manifest(path: Path | str, lr: str, ent: str) -> int:
    """Substitute <LR>/<ENT> in a manifest. Returns the number of lines changed."""
    p = Path(path)
    text = p.read_text()
    filled = text.replace("<LR>", lr).replace("<ENT>", ent)
    changed = sum(1 for a, b in zip(text.splitlines(), filled.splitlines()) if a != b)
    p.write_text(filled)
    return changed


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("logs", nargs="+", help="Slurm .out files (globs accepted)")
    p.add_argument("--arm", default="ON", choices=["ON", "OFF"])
    p.add_argument("--noise-pp", type=float, default=NOISE_PP)
    p.add_argument(
        "--fill",
        metavar="MANIFEST",
        help="Write the winning lr/ent into this manifest's <LR>/<ENT> placeholders",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    expanded: list[str] = []
    for pattern in args.logs:
        expanded.extend(glob.glob(pattern) or [pattern])

    runs = report(expanded, args.noise_pp)
    winner = pick(runs, args.arm, args.noise_pp) if runs else None

    if winner and args.fill:
        n = fill_manifest(args.fill, winner.lr, winner.ent)
        print(
            f"\nFilled {n} lines in {args.fill} with lr={winner.lr} "
            f"ent-coef={winner.ent}"
        )
        print("Review it, then submit:")
        print(f"  sbatch --array=0-11%9 docs/completion/slurm/array.sbatch {args.fill}")
