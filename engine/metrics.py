from __future__ import annotations

from engine.graph import Map
from engine.graph_utils import jack_bfs_distances
from engine.state import GameState

# ---------------------------------------------------------------------------
# The participant score — SCORE_STUDY_V1
# ---------------------------------------------------------------------------
#
# What human participants were scored on, and told they were scored on
# (frontend_participant/index.html, the scoring tutorial slide):
#
#   progress  10,000 * (1 - d / d_max), the best value reached over the game
#   stealth    5,000 * hideout_uncertainty, added only when Jack escapes
#
# The browser holds the running max and the point values
# (frontend_participant/game.js: computeDistanceScore, the +5000 bonus); the
# server supplies normalized_distance and hideout_uncertainty (server/routes.py).
# Both halves call the functions below, and tests/test_participant_score.py pins
# the constants against game.js so the two cannot drift apart.
#
# FROZEN, like COPS_STUDY_V2: participants already played under it. The RL agent
# maximises this same quantity so the human-vs-RL comparison is one objective,
# not two. A different scoring rule is a new named preset, never an edit here.

SCORE_PROGRESS_POINTS = 10_000
SCORE_STEALTH_POINTS = 5_000
# Stealth weight in normalised units (progress term = 1.0 on a win). This is the
# RL reward's --reward-gamma default; it is inherited, not tuned.
SCORE_STUDY_V1_STEALTH = SCORE_STEALTH_POINTS / SCORE_PROGRESS_POINTS


def hideout_uncertainty(
    hideout_zone: frozenset[int], position_pmf: dict[int, float]
) -> float:
    """
    Fraction of hideout zone nodes that still carry nonzero position PMF mass
    at night end. Returns a value in [0, 1].

    1.0 = cops have no information about which zone node is the hideout.
    0.0 = cops have fully eliminated every zone node.
    Only meaningful when Jack won.
    """
    if not hideout_zone:
        return 0.0
    nonzero = sum(1 for node_id in hideout_zone if position_pmf.get(node_id, 0.0) > 0.0)
    return nonzero / len(hideout_zone)


def hideout_distances(hideout: int, game_map: Map) -> tuple[dict[int, int], int]:
    """BFS distances from the hideout, and their maximum (d_max).

    d_max is per hideout — the farthest reachable Jack node from *this* hideout —
    not the map diameter. That is what the participant score normalises by.
    Callers that score many positions against one hideout should call this once.
    """
    dists = jack_bfs_distances(hideout, game_map)
    return dists, (max(dists.values()) if dists else 1)


def normalized_distance(jack_pos: int, dists: dict[int, int], d_max: int) -> float:
    """d / d_max in [0, 1]; 0.0 at the hideout. Unreachable nodes score d_max."""
    d = dists.get(jack_pos, d_max)
    return d / d_max if d_max > 0 else 0.0


def normalized_distance_for_state(state: GameState, game_map: Map) -> float:
    """The server's score_info["normalized_distance"] for one state."""
    dists, d_max = hideout_distances(state.hideout, game_map)
    return normalized_distance(state.jack_pos, dists, d_max)


def participant_score(best_progress: float, won: bool, uncertainty: float) -> float:
    """The participant score in normalised units: points / SCORE_PROGRESS_POINTS.

    best_progress  max over the game of (1 - normalized_distance), in [0, 1].
                   Taken over the positions after each of Jack's moves, starting
                   from 0 — the browser's maxScore starts at 0 and is updated
                   only after a move, so the start position is never scored.
    won            Jack reached the hideout.
    uncertainty    hideout_uncertainty at the end; ignored unless won.

    Range: [0, 1) on a loss, [1, 1.5] on a win.
    """
    if won:
        # Standing on the hideout is normalized_distance 0, so progress is 1.
        return 1.0 + SCORE_STUDY_V1_STEALTH * uncertainty
    return best_progress
