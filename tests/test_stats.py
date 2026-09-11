"""Clustered bootstrap for the human-vs-RL comparison (analysis/stats.py).

The property under test is the one that is easy to get wrong and invisible when
wrong: resampling must happen at the PARTICIPANT level. Three games from one
person carry less information than three games from three people, and an
interval that ignores that is too narrow.
"""

from __future__ import annotations

import math

import pytest

from analysis.stats import (
    Interval,
    cluster_bootstrap,
    design_effect,
    paired_difference,
)

FAST = 600  # bootstrap replicates; enough to be stable, fast enough for CI


def _clusters(*groups) -> dict[int, list[float]]:
    return {i: list(g) for i, g in enumerate(groups)}


# --- point estimate ---------------------------------------------------------


def test_point_estimate_is_the_pooled_mean():
    r = cluster_bootstrap(_clusters([1, 1, 0], [0, 0, 0]), n_boot=FAST)
    assert r.point == pytest.approx(2 / 6)
    assert r.n_clusters == 2 and r.n_items == 6


def test_interval_brackets_the_point():
    r = cluster_bootstrap(_clusters([1, 0], [1, 1], [0, 0], [1, 0]), n_boot=FAST)
    assert r.low <= r.point <= r.high


def test_empty_input_is_nan_not_a_crash():
    r = cluster_bootstrap({}, n_boot=FAST)
    assert math.isnan(r.point)


def test_single_cluster_reports_no_interval():
    """One participant says nothing about between-participant variation."""
    r = cluster_bootstrap(_clusters([1, 0, 1]), n_boot=FAST)
    assert r.point == pytest.approx(2 / 3)
    assert math.isnan(r.low) and math.isnan(r.high)


def test_zero_variance_gives_a_degenerate_interval():
    r = cluster_bootstrap(_clusters([1, 1], [1, 1], [1, 1]), n_boot=FAST)
    assert r.point == 1.0 and r.low == 1.0 and r.high == 1.0


# --- the load-bearing property ----------------------------------------------


def test_clustering_widens_the_interval():
    """THE test. Same 12 games, same mean; clustered data must be less certain.

    Left: 4 participants who each went all-wins or all-losses — the outcome is a
    property of the person, so there are really only 4 observations.
    Right: 12 games spread so each participant is mixed — much more information.
    """
    lumpy = cluster_bootstrap(
        _clusters([1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]), n_boot=FAST
    )
    even = cluster_bootstrap(
        _clusters([1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0]), n_boot=FAST
    )
    assert lumpy.point == pytest.approx(even.point)
    assert lumpy.half_width > even.half_width


def test_more_participants_narrows_the_interval():
    few = cluster_bootstrap(_clusters([1, 0], [1, 1], [0, 0]), n_boot=FAST)
    many = cluster_bootstrap(_clusters(*([[1, 0], [1, 1], [0, 0]] * 6)), n_boot=FAST)
    assert many.half_width < few.half_width


def test_bootstrap_is_reproducible():
    a = cluster_bootstrap(_clusters([1, 0], [1, 1], [0, 0]), n_boot=FAST, seed=7)
    b = cluster_bootstrap(_clusters([1, 0], [1, 1], [0, 0]), n_boot=FAST, seed=7)
    assert (a.low, a.high) == (b.low, b.high)


def test_different_seeds_give_similar_intervals():
    kw = dict(n_boot=4000)
    data = _clusters([1, 0], [1, 1], [0, 0], [1, 0], [0, 1])
    a = cluster_bootstrap(data, seed=1, **kw)
    b = cluster_bootstrap(data, seed=2, **kw)
    assert abs(a.half_width - b.half_width) < 0.10


# --- paired difference ------------------------------------------------------


def test_paired_difference_point_estimate():
    r = paired_difference({0: [(0.8, 1.0), (0.6, 0.0)], 1: [(0.5, 0.0)]}, n_boot=FAST)
    assert r.point == pytest.approx((-0.2 + 0.6 + 0.5) / 3)


def test_paired_difference_excluding_zero_means_a_real_gap():
    """Policy wins every board the humans lost — interval must clear zero."""
    clusters = {i: [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)] for i in range(5)}
    r = paired_difference(clusters, n_boot=FAST)
    assert r.low > 0


def test_paired_difference_straddles_zero_when_matched():
    clusters = {i: [(1.0, 1.0), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)] for i in range(5)}
    r = paired_difference(clusters, n_boot=FAST)
    assert r.low <= 0 <= r.high


# --- design effect ----------------------------------------------------------


def test_design_effect_is_about_one_when_clustering_does_not_matter():
    de = design_effect(_clusters([1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]))
    assert 0.5 < de < 2.0


def test_design_effect_rises_when_outcomes_track_the_participant():
    lumpy = design_effect(_clusters([1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]))
    even = design_effect(_clusters([1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0]))
    assert lumpy > even


def test_design_effect_degenerate_inputs():
    assert design_effect({}) == 1.0
    assert design_effect(_clusters([1, 1])) == 1.0
    assert design_effect(_clusters([1, 1], [1, 1])) == 1.0  # no variance at all


def test_interval_formats_as_percentages():
    assert "50.0%" in str(Interval(0.5, 0.4, 0.6, 3, 9))
