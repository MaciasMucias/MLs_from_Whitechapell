"""The RL reward is the participant score, and its shaping cannot change the optimum.

Three invariants from docs/completion/10-reward-design.md:

1. The terminal reward IS the score human participants were given, computed by
   the same functions the server uses and pinned against the browser's point
   values. Humans and agents are compared on one objective.
2. alpha/beta/zeta are exact potential-based shaping, so they telescope: the
   discounted sum over an episode is -Phi(s_0), whatever Jack does.
3. --reward-objective legacy reproduces the reward every run before 2026-09-14
   trained on, so earlier waves stay reproducible.

Everything is self-contained: the legacy golden trajectory is embedded inline,
recorded from the pre-change env, never loaded from mutable data.
"""

from __future__ import annotations

from dataclasses import replace
import random
import re
from pathlib import Path

import numpy as np
import pytest

from agents.base import AgentOutput, JackAgent
from agents.heuristic_cops import HeuristicCops
from engine.env import make_initial_state
from engine.game import run_game
from engine.graph_utils import jack_bfs_distances
from engine.metrics import (
    SCORE_PROGRESS_POINTS,
    SCORE_STEALTH_POINTS,
    SCORE_STUDY_V1_STEALTH,
    hideout_distances,
    normalized_distance,
    normalized_distance_for_state,
    participant_score,
)
from training.env import REWARD_TERMS, JackEnv
from training.eval import game_score

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class HideoutSeeker(JackAgent):
    """Mostly walks toward the hideout, so games include both wins and losses."""

    def __init__(self, rng: random.Random, greed: float = 0.85) -> None:
        self._rng = rng
        self._greed = greed

    def act(self, state, legal_edges, game_map) -> AgentOutput:
        if self._rng.random() < self._greed:
            dists = jack_bfs_distances(state.hideout, game_map)
            edge = min(legal_edges, key=lambda e: dists.get(e.destination.id, 999))
        else:
            edge = self._rng.choice(legal_edges)
        return AgentOutput(edge=edge)


def _seek(env: JackEnv, info: dict, rng: random.Random, greed: float = 0.85) -> int:
    legal = [int(j) for j in np.flatnonzero(info["action_mask"])]
    hideout = env._state.hideout
    if rng.random() < greed:
        return min(legal, key=lambda j: env._all_dists[j].get(hideout, 999))
    return rng.choice(legal)


def _server_normalized_distance(state, game_map) -> float:
    """server/routes.py's score_info formula as it stood before the refactor.

    Kept verbatim as the reference the shared function must reproduce, so the
    refactor cannot have changed what production reports.
    """
    _bfs = jack_bfs_distances(state.hideout, game_map)
    _max_dist = max(_bfs.values()) if _bfs else 1
    _curr_dist = _bfs.get(state.jack_pos, _max_dist)
    return _curr_dist / _max_dist if _max_dist > 0 else 0.0


# ---------------------------------------------------------------------------
# 1. The objective is the participant score
# ---------------------------------------------------------------------------


def test_point_values_match_the_browser():
    """game.js holds the point values humans saw; the RL objective must agree."""
    js = (ROOT / "frontend_participant/game.js").read_text(encoding="utf-8")
    progress = re.search(r"return Math\.round\(\(1 - normDist\) \* (\d+)\)", js)
    stealth = re.search(r"hideout_uncertainty \* (\d+)\)", js)
    assert progress and stealth, (
        "game.js scoring changed shape; re-check SCORE_STUDY_V1"
    )
    assert int(progress.group(1)) == SCORE_PROGRESS_POINTS
    assert int(stealth.group(1)) == SCORE_STEALTH_POINTS
    assert SCORE_STUDY_V1_STEALTH == 0.5


def test_participant_score_ranges():
    assert participant_score(0.0, False, 1.0) == 0.0  # uncertainty ignored on a loss
    assert participant_score(0.7, False, 0.0) == 0.7
    assert participant_score(0.7, True, 0.0) == 1.0  # a win is full progress
    assert participant_score(0.2, True, 1.0) == 1.5


def test_normalized_distance_matches_the_pre_refactor_server(gm):
    rng = random.Random(3)
    for _ in range(25):
        state = make_initial_state(gm, rng=rng)
        dists, d_max = hideout_distances(state.hideout, gm)
        for node in gm.jack_nodes:
            moved = replace(state, jack_pos=node.id)
            ref = _server_normalized_distance(moved, gm)
            assert normalized_distance_for_state(moved, gm) == ref
            assert normalized_distance(node.id, dists, d_max) == ref


def test_game_score_matches_browser_arithmetic(gm):
    """Replays the browser: maxScore over each move's score_info, plus the bonus."""
    rng = random.Random(11)
    cops = HeuristicCops()
    n_wins = n_losses = 0
    for _ in range(12):
        record = run_game(gm, HideoutSeeker(rng), cops, rng=rng)
        max_score = 0
        for rr in record.history:
            nd = _server_normalized_distance(rr.state_after_round, gm)
            max_score = max(max_score, round((1 - nd) * SCORE_PROGRESS_POINTS))
        score, u = game_score(record, gm)
        if record.winner == "jack":
            n_wins += 1
            max_score += round(u * SCORE_STEALTH_POINTS)
        else:
            n_losses += 1
            assert u is None
        # Only JS rounding separates them: at most 1 point per rounded term.
        assert abs(score * SCORE_PROGRESS_POINTS - max_score) <= 1.0
    assert n_wins and n_losses, "sample must exercise both outcomes"


def test_stealth_terminal_reward_is_minus_one_or_the_winning_score(gm):
    """The default objective: -1 on any loss, the participant score on a win.

    No partial credit for getting close — that is what separates it from the
    score objective, and why it is the default (docs/completion/10-reward-design.md).
    """
    env = JackEnv(gm, alpha=0, beta=0, delta=0, zeta=0, rng=random.Random(5))
    assert env._objective == "stealth"
    act_rng = random.Random(5)
    outcomes = set()
    for _ in range(15):
        _, info = env.reset()
        ret, done = 0.0, False
        while not done:
            _, r, done, _, info = env.step(_seek(env, info, act_rng))
            ret += r
        outcomes.add(info["winner"])
        if info["winner"] == "jack":
            # A win pays exactly the participant score, in [1, 1.5].
            assert ret == pytest.approx(info["score"], abs=1e-12)
            assert 1.0 <= ret <= 1.5
        else:
            assert ret == -1.0
            # ...while the score reported for the same loss still carries progress.
            assert 0.0 <= info["score"] < 1.0
    assert outcomes == {"jack", "cops"}


def test_score_objective_return_is_the_participant_score(gm):
    """objective="score": with every auxiliary term off, the return is the score."""
    env = JackEnv(
        gm, alpha=0, beta=0, delta=0, zeta=0, objective="score", rng=random.Random(5)
    )
    act_rng = random.Random(5)
    outcomes = set()
    for _ in range(15):
        _, info = env.reset()
        ret, done = 0.0, False
        while not done:
            _, r, done, _, info = env.step(_seek(env, info, act_rng))
            ret += r
        outcomes.add(info["winner"])
        assert ret == pytest.approx(info["score"], abs=1e-12)
        if info["winner"] == "jack":
            assert 1.0 <= info["score"] <= 1.5
        else:
            assert 0.0 <= info["score"] < 1.0
    assert outcomes == {"jack", "cops"}


def test_reward_terms_sum_to_the_return(gm):
    env = JackEnv(gm, rng=random.Random(8))
    act_rng = random.Random(8)
    for _ in range(8):
        _, info = env.reset()
        ret, done = 0.0, False
        while not done:
            _, r, done, _, info = env.step(_seek(env, info, act_rng))
            ret += r
        terms = info["reward_terms"]
        assert set(terms) == set(REWARD_TERMS)
        assert sum(terms.values()) == pytest.approx(ret, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. alpha/beta/zeta are exact potential-based shaping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("discount", [0.99, 0.9])
def test_shaping_telescopes(gm, discount):
    """sum_t discount^t F_t == -Phi(s_0), for any trajectory.

    With delta off, every step's reward is shaping, except the last step also
    carries the objective. Remove that and what remains must telescope. A
    policy cannot farm shaping that sums to a constant.
    """
    env = JackEnv(
        gm,
        alpha=0.3,
        beta=0.2,
        zeta=0.25,
        delta=0.0,
        discount=discount,
        rng=random.Random(21),
    )
    act_rng = random.Random(21)
    for _ in range(10):
        _, info = env.reset()
        weights = {"alpha": 0.3, "beta": 0.2, "zeta": 0.25}
        phi0 = env._potentials(env._state, {})
        phi0_weighted = sum(weights[k] * phi0[k] for k in weights)

        rewards, done = [], False
        while not done:
            _, r, done, _, info = env.step(_seek(env, info, act_rng, greed=0.6))
            rewards.append(r)
        rewards[-1] -= info["reward_terms"]["objective"]
        discounted = sum(discount**t * r for t, r in enumerate(rewards))
        assert discounted == pytest.approx(-phi0_weighted, abs=1e-12)


def test_discount_must_match_ppo_gamma():
    """train.py wires PPO's --gamma into the env; guard the wiring textually."""
    src = (ROOT / "training/train.py").read_text(encoding="utf-8")
    assert '"discount": args.gamma' in src


# ---------------------------------------------------------------------------
# delta: the decaying exploration bonus, and its coverage stats
# ---------------------------------------------------------------------------


def test_delta_decays_with_lifetime_visits(gm):
    """Counts persist across episodes by design, so the bonus shrinks with use.

    Per-episode sums are dominated by episode length, so test the definition
    exactly instead: everything delta ever paid equals sum over nodes of
    delta/sqrt(i) for i = 1..visits, which only holds if counts never reset.
    """
    delta = 0.01
    env = JackEnv(gm, alpha=0, beta=0, zeta=0, delta=delta, rng=random.Random(2))
    act_rng = random.Random(2)
    total_paid = 0.0
    per_step = []  # (delta paid this episode, non-terminal steps this episode)
    coverage = []
    for _ in range(60):
        _, info = env.reset()
        done, steps = False, 0
        while not done:
            _, _, done, _, info = env.step(_seek(env, info, act_rng, greed=0.5))
            steps += not done
        total_paid += info["reward_terms"]["delta"]
        per_step.append((info["reward_terms"]["delta"], steps))
        coverage.append(info["explore"]["coverage"])
        assert 0.0 <= info["explore"]["visit_entropy"] <= 1.0

    expected = sum(
        delta / (i**0.5) for n in env._visit_counts.values() for i in range(1, n + 1)
    )
    assert total_paid == pytest.approx(expected, rel=1e-12)

    def mean_bonus(block):
        paid = sum(p for p, _ in block)
        steps = sum(s for _, s in block)
        return paid / steps

    assert mean_bonus(per_step[-20:]) < mean_bonus(per_step[:20])
    assert all(b >= a for a, b in zip(coverage, coverage[1:]))  # never shrinks
    assert 0.0 < coverage[-1] <= 1.0


# ---------------------------------------------------------------------------
# 3. legacy reproduces the pre-2026-09-14 reward
# ---------------------------------------------------------------------------

# Recorded from training/env.py at commit c2b4f65 (the last pre-change env):
# JackEnv(whitechapel, rng=Random(7)), actions from a hideout seeker on Random(123).
# (actions, per-step rewards, winner)
LEGACY_GOLDEN = [
    (
        [110, 92, 74, 57, 39, 56, 55],
        [
            0.005833333333333333,
            0.02,
            0.02840909090909091,
            0.01887398935121278,
            0.0015902083678047234,
            0.009708553485924048,
            1.0909090909090908,
        ],
        "jack",
    ),
    (
        [4, 14, 33, 54, 68, 69, 68, 102, 69, 86],
        [
            -1.734723475976807e-18,
            0.02648148148148148,
            0.01973875661375661,
            0.018665541459302167,
            0.02670103085339213,
            0.018374279086231037,
            -0.01258552285344729,
            0.012355425525696292,
            0.015704160542310268,
            1.2435897435897436,
        ],
        "jack",
    ),
    (
        [81, 61, 97, 117, 118, 152, 150, 175],
        [
            0.0028571428571428576,
            0.022773487773487776,
            0.011772470144563169,
            0.01022600720602686,
            0.014870892018779343,
            0.011927860696517413,
            0.010225383151357308,
            -1.0,
        ],
        "cops",
    ),
    (
        [4, 14, 33, 54, 101],
        [
            -0.0029289321881345275,
            0.023552549293346957,
            0.016746048915418005,
            0.015157029469255703,
            -1.0,
        ],
        "cops",
    ),
    (
        [28, 65, 83, 85, 68, 102],
        [
            0.013333333333333332,
            0.031033755274261603,
            0.010556458793016685,
            0.026733672827362435,
            0.022447724794645952,
            1.2777777777777777,
        ],
        "jack",
    ),
    (
        [78, 96, 115, 150, 175, 187],
        [
            -0.003333333333333334,
            0.006216216216216217,
            0.019974259974259972,
            0.01854725828805595,
            0.021196969696969697,
            1.1964285714285714,
        ],
        "jack",
    ),
    ([19, 17, 16], [0.008333333333333331, 0.034035087719298245, -1.0], "cops"),
    (
        [141, 124, 121, 97, 79, 77],
        [
            0.012083333333333333,
            0.015087209302325582,
            0.019360988895872615,
            0.015154920895718559,
            0.027004553079319434,
            1.193877551020408,
        ],
        "jack",
    ),
    ([99], [-1.0], "cops"),
    ([19, 17], [0.005404401145198806, -1.0], "cops"),
]


def test_legacy_objective_reproduces_the_old_reward(gm):
    env = JackEnv(gm, rng=random.Random(7), objective="legacy")
    for actions, expected, winner in LEGACY_GOLDEN:
        env.reset()
        got, info = [], {}
        for a in actions:
            _, r, done, _, info = env.step(a)
            got.append(r)
        assert done
        assert info["winner"] == winner
        # Only float summation order differs from the pre-change code.
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_unknown_objective_is_rejected(gm):
    with pytest.raises(ValueError):
        JackEnv(gm, objective="winloss")
