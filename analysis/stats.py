"""Uncertainty for the human-vs-RL comparison.

**The point of this module is that the games are not independent.** Each
participant plays a three-map course, so 57 games come from 19 people. A plain
binomial interval over 57 games assumes 57 independent draws and reports an
interval that is too narrow — roughly by a factor of sqrt(design effect). With a
sample this small, overstating precision is the easiest way to turn "suggestive"
into an unsupportable claim.

So every interval here resamples **participants** with replacement, carrying all
of that participant's games along. That is the standard cluster bootstrap, and it
is the honest unit of analysis: another participant is the thing the study could
have sampled differently, not another game from the same person.

Stdlib only — no scipy dependency for what is a resampling loop.

The paired comparison also matters: the policy is replayed on *the same boards*
the humans faced, so the per-scenario difference removes scenario difficulty from
the contrast. Comparing two independent win rates would throw that away.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

DEFAULT_BOOTSTRAP = 10_000
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class Interval:
    """A point estimate with a percentile bootstrap interval."""

    point: float
    low: float
    high: float
    n_clusters: int
    n_items: int

    def __str__(self) -> str:
        return f"{self.point:.1%} [{self.low:.1%}, {self.high:.1%}]"

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2


def _percentiles(values: list[float], alpha: float) -> tuple[float, float]:
    values = sorted(values)
    n = len(values)
    lo = values[max(0, int((alpha / 2) * n) - 1)]
    hi = values[min(n - 1, int((1 - alpha / 2) * n))]
    return lo, hi


def cluster_bootstrap(
    clusters: dict[int, list[float]],
    n_boot: int = DEFAULT_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Interval:
    """Mean of all values, with a participant-clustered percentile interval.

    `clusters` maps participant -> that participant's per-game values (1.0 for a
    win, 0.0 for a loss; or a per-scenario policy win rate; or a paired
    difference). Each bootstrap replicate draws len(clusters) participants with
    replacement and pools their games.
    """
    keys = list(clusters)
    flat = [v for k in keys for v in clusters[k]]
    if not flat:
        return Interval(float("nan"), float("nan"), float("nan"), 0, 0)

    point = sum(flat) / len(flat)
    if len(keys) < 2:
        # One cluster: resampling it tells us nothing about between-participant
        # variation, so report the point estimate without a fake interval.
        return Interval(point, float("nan"), float("nan"), len(keys), len(flat))

    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        pooled: list[float] = []
        for _ in keys:
            pooled.extend(clusters[rng.choice(keys)])
        means.append(sum(pooled) / len(pooled))

    low, high = _percentiles(means, alpha)
    return Interval(point, low, high, len(keys), len(flat))


def paired_difference(
    clusters: dict[int, list[tuple[float, float]]],
    n_boot: int = DEFAULT_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Interval:
    """Interval for (policy - human) on the same boards, clustered by participant.

    `clusters` maps participant -> [(policy_value, human_value), ...] for that
    participant's games. Pairing is what makes this worth doing: both sides saw
    the identical scenario, so scenario difficulty cancels instead of inflating
    the variance of both arms.

    An interval excluding zero is the claim "the agent differs from the humans on
    the boards they actually played".
    """
    diffs = {k: [p - h for p, h in v] for k, v in clusters.items()}
    return cluster_bootstrap(diffs, n_boot, alpha, seed)


def design_effect(clusters: dict[int, list[float]]) -> float:
    """How much the clustering inflates the variance, as a ratio.

    ~1.0 means the games within a participant are no more alike than games
    across participants, and a naive per-game interval would have been fine.
    Above 1.0 quantifies how misleading that interval would have been — worth
    reporting, because it justifies the extra machinery.

    Uses the standard Kish approximation with the intraclass correlation
    estimated from between- vs within-cluster variance.
    """
    sizes = [len(v) for v in clusters.values() if v]
    flat = [x for v in clusters.values() for x in v]
    if len(sizes) < 2 or len(flat) < 2:
        return 1.0

    grand = sum(flat) / len(flat)
    total_var = sum((x - grand) ** 2 for x in flat) / (len(flat) - 1)
    if total_var == 0:
        return 1.0

    # Between-cluster variance of the cluster means, weighted by cluster size.
    between = sum(
        len(v) * ((sum(v) / len(v)) - grand) ** 2
        for v in clusters.values()
        if v
    ) / max(len(sizes) - 1, 1)
    icc = max(0.0, min(1.0, (between / len(flat)) / total_var))
    mean_size = sum(sizes) / len(sizes)
    return 1 + (mean_size - 1) * icc
