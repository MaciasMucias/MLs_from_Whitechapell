from __future__ import annotations

import math
import random

import numpy as np

from agents.base import Director
from agents.heuristic_cops import HeuristicCops
from agents.random_agents import NoOpDirector
from engine.env import (
    end_of_round,
    legal_jack_edges,
    make_initial_state,
    step_cop,
    step_jack,
)
from engine.graph import Map
from engine.metrics import (
    SCORE_STUDY_V1_STEALTH,
    hideout_distances,
    hideout_uncertainty,
    normalized_distance,
    participant_score,
)
from engine.state import GameState
from training.obs import build_obs, precompute_distances

# Reward objectives. "score" is the default and the only one to use for new runs.
#
#   score   SCORE_STUDY_V1 — the participant score (engine/metrics.py) as the
#           terminal reward, alpha/beta/zeta as exact potential-based shaping,
#           delta as a decaying exploration bonus. From 2026-09-14.
#   legacy  The reward every run before 2026-09-14 trained on: terminal +/-1 plus
#           gamma * hideout_uncertainty on a win, and shaping that was potential
#           *differences* without the discount and with nothing at the terminal
#           step. Kept so earlier waves stay reproducible — identical to the
#           pre-change env up to float summation order (~1e-17, pinned by
#           tests/test_reward.py). Pass --reward-objective legacy to rerun one.
REWARD_OBJECTIVES = ("score", "legacy")

# Shaping terms in the order they are reported. "objective" is the terminal
# reward itself; the rest are the four auxiliary terms.
REWARD_TERMS = ("objective", "alpha", "beta", "zeta", "delta")


class JackEnv:
    """
    Gym-style environment for training Jack's RL policy against heuristic cops.

    Action space:  Discrete(n_jack) with per-step action masking.
                   Action i = move to Jack node i. Only legal neighbours are
                   unmasked; all others must be set to -inf before softmax.
    Observation:   1,416-dim float32 vector (see training/obs.py).

    Reward (objective="score", the default) — see docs/completion/10-reward-design.md:

      objective  Terminal. The participant score: best progress on a loss, in
                 [0, 1); 1 + gamma * hideout_uncertainty on a win, in [1, 1.5].
                 gamma defaults to SCORE_STUDY_V1_STEALTH (0.5), inherited from
                 the study's scoring rule — it defines the goal, it is not tuned.
      alpha      Potential-based shaping, Phi = -d(hideout) / d_max.
      beta       Potential-based shaping, Phi = -P(cops place Jack here).
      zeta       Potential-based shaping, Phi = nearest-cop distance / diameter.
      delta      Count-based exploration bonus, delta / sqrt(n(node)).

    alpha/beta/zeta are F = w * (discount * Phi(s') - Phi(s)) with Phi = 0 at a
    terminal state (Ng, Harada & Russell 1999), so they can change how fast the
    policy learns but not which policy is optimal. Choose their weights on sample
    efficiency, never on final score. `discount` MUST equal PPO's --gamma for that
    guarantee to hold.

    delta is deliberately NOT potential-based. Visit counts are per env instance
    and persist across episodes, so the bonus front-loads exploration of the map
    and decays towards zero with experience (the MBIE-EB / pseudo-count form).
    It changes what is rewarded early in training, by design; it was found in
    development to raise final score by giving Jack a broader base understanding
    of the map. Judge it on final score and map coverage.
    """

    def __init__(
        self,
        game_map: Map,
        alpha: float = 0.1,
        beta: float = 0.05,
        delta: float = 0.01,
        gamma: float = SCORE_STUDY_V1_STEALTH,
        zeta: float = 0.1,
        discount: float = 0.99,
        objective: str = "score",
        blocking: bool = False,
        rng: random.Random | None = None,
        director: Director | None = None,
    ) -> None:
        if objective not in REWARD_OBJECTIVES:
            raise ValueError(
                f"objective must be one of {REWARD_OBJECTIVES}, got {objective!r}"
            )
        self._map = game_map
        self._alpha = alpha
        self._beta = beta
        self._delta = delta
        self._gamma = gamma
        self._zeta = zeta
        self._discount = discount
        self._objective = objective
        self._blocking = blocking  # EXTEND(blocking): _all_dists is topology-only (no cop positions);
        # with blocking enabled it underestimates true distance when cops wall off the short route.
        # Action masking enforces blocking correctly — this is a known reward-shaping approximation.
        self._rng = rng or random.Random()
        # EXTEND(multinight): reset() currently starts a fresh single-night game;
        # multi-night would chain states across nights here, and the stealth term
        # (hideout_uncertainty) would carry information forward to later nights

        self._all_dists, self._diameter = precompute_distances(game_map)
        self._n_jack = len(game_map.jack_nodes)

        # Separate RNG stream so cop decisions don't consume Jack's seed
        self._cops = HeuristicCops()
        self._director: Director = director if director is not None else NoOpDirector()

        # Per-env visit counts — intentionally persist across episodes (see the
        # class docstring on delta). EXTEND: --branch-from and resume start fresh
        # worker processes, so counts restart at zero and a branched run gets a
        # second exploration burst. No current reward-design arm branches.
        self._visit_counts: dict[int, int] = {}

        self._state: GameState | None = None
        # PMF from the most recent cops.act(); empty at episode start
        self._pmf: dict[int, float] = {}

        # Per-episode state for the participant score and for reporting.
        self._hideout_dists: dict[int, int] = {}
        self._hideout_d_max: int = 1
        self._best_progress: float = 0.0
        self._ep_terms: dict[str, float] = dict.fromkeys(REWARD_TERMS, 0.0)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = random.Random(seed)

        self._state = make_initial_state(self._map, rng=self._rng)
        self._cops.on_episode_start(self._state, self._map)
        self._director.on_episode_start(self._state, self._map)
        self._pmf = {}
        self._hideout_dists, self._hideout_d_max = hideout_distances(
            self._state.hideout, self._map
        )
        # The browser's maxScore starts at 0 and only updates after a move.
        self._best_progress = 0.0
        self._ep_terms = dict.fromkeys(REWARD_TERMS, 0.0)

        legal = legal_jack_edges(self._state, self._map, blocking=self._blocking)
        obs = build_obs(self._state, self._map, self._all_dists, self._diameter)
        return obs, {"action_mask": self._action_mask(legal)}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._state is not None, "call reset() before step()"

        state = self._state
        prev_pos = state.jack_pos
        phi_prev = self._potentials(state, self._pmf)

        # Resolve action index -> JackEdge
        legal = legal_jack_edges(state, self._map, blocking=self._blocking)
        edge = next((e for e in legal if e.destination.id == action), None)
        assert edge is not None, f"action {action} is not a legal move from {prev_pos}"

        # Jack moves
        state, terminated, winner = step_jack(state, edge)
        # The server reports score_info after every move, including the one that
        # ends the game, so progress counts from the post-move position.
        self._best_progress = max(
            self._best_progress,
            1.0
            - normalized_distance(
                state.jack_pos, self._hideout_dists, self._hideout_d_max
            ),
        )
        if terminated:
            return self._finish(state, winner, phi_prev)

        # Terminate early if Jack cannot mathematically reach the hideout in time.
        # After step_jack, state.turn is still T (end_of_round hasn't run yet), so
        # Jack has turn_limit - 1 - T remaining moves after this one.
        # Training-only: the server plays on to the turn limit. Under the score
        # objective the loss is still worth the progress reached so far. A human
        # could have crept closer in the remaining turns, so this can slightly
        # under-credit a loss, never over-credit it.
        dist_to_hideout = self._all_dists[state.jack_pos].get(
            state.hideout, self._diameter
        )
        if dist_to_hideout > self._map.turn_limit - 1 - state.turn:
            return self._finish(state, "cops", phi_prev)

        # Director intercepts cop knowledge (no-op for now)
        state = self._director.filter_knowledge(state, self._map)

        # Cops plan — updates last_position_pmf
        cop_turns, _ = self._cops.act(state, self._map)
        curr_pmf = self._cops.last_position_pmf

        # Execute cop turns
        for cop_turn in cop_turns:
            state, terminated, winner, _ = step_cop(state, cop_turn, self._map)
            if terminated:
                return self._finish(state, winner, phi_prev)

        # End-of-round check (turn limit / no legal moves when blocking)
        state, terminated, winner = end_of_round(
            state, self._map, blocking=self._blocking
        )
        if terminated:
            return self._finish(state, winner, phi_prev)

        # Non-terminal: shaping and exploration only; the objective is terminal.
        curr_pos = state.jack_pos
        terms = self._shaping(phi_prev, self._potentials(state, curr_pmf))
        self._visit_counts[curr_pos] = self._visit_counts.get(curr_pos, 0) + 1
        terms["delta"] = self._delta / math.sqrt(self._visit_counts[curr_pos])
        reward = self._accumulate(terms)

        self._state = state
        self._pmf = curr_pmf
        next_legal = legal_jack_edges(state, self._map, blocking=self._blocking)
        obs = build_obs(state, self._map, self._all_dists, self._diameter)
        return obs, reward, False, False, {"action_mask": self._action_mask(next_legal)}

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _potentials(self, state: GameState, pmf: dict[int, float]) -> dict[str, float]:
        """Unweighted potentials Phi_k(s) for the three shaping terms.

        pmf is the cops' position belief for this state (empty before the cops'
        first plan of the episode, which scores 0, as the legacy term did).
        """
        pos = state.jack_pos
        min_cop_dist = min(
            self._all_dists[pos].get(jn.id, self._diameter)
            for cp in state.cop_positions
            for jn in self._map.cop_nodes[cp].jack_neighbours
        )
        if self._objective == "legacy":
            # Hideout distance normalised by the map diameter, as before.
            d = self._all_dists[pos].get(state.hideout, self._diameter)
            d_norm = d / self._diameter
        else:
            # Per-hideout d_max, matching the participant score's progress term.
            d_norm = normalized_distance(pos, self._hideout_dists, self._hideout_d_max)
        return {
            "alpha": -d_norm,
            "beta": -pmf.get(pos, 0.0),
            "zeta": min_cop_dist / self._diameter,
        }

    def _shaping(
        self, phi_prev: dict[str, float], phi_next: dict[str, float] | None
    ) -> dict[str, float]:
        """Weighted shaping F_k for one transition. phi_next=None means terminal.

        score:  F = w * (discount * Phi(s') - Phi(s)), with Phi(terminal) = 0.
        legacy: F = w * (Phi(s') - Phi(s)) on non-terminal steps, 0 at terminal.
        """
        weights = {"alpha": self._alpha, "beta": self._beta, "zeta": self._zeta}
        if self._objective == "legacy":
            if phi_next is None:
                return dict.fromkeys(weights, 0.0)
            return {k: w * (phi_next[k] - phi_prev[k]) for k, w in weights.items()}
        terms = {}
        for k, w in weights.items():
            next_value = self._discount * phi_next[k] if phi_next is not None else 0.0
            terms[k] = w * (next_value - phi_prev[k])
        return terms

    def _accumulate(self, terms: dict[str, float]) -> float:
        for k, v in terms.items():
            self._ep_terms[k] += v
        return sum(terms.values())

    def _finish(
        self, state: GameState, winner: str, phi_prev: dict[str, float]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        won = winner == "jack"
        uncertainty = 0.0
        if won:
            # Recompute position PMF on the current state — Jack just moved so
            # the cached value from the previous cops.act() is stale.
            # Fraction of hideout zone nodes still carrying position PMF mass
            # (cops uncertain which zone node Jack is in).
            position_pmf = HeuristicCops.compute_pmf(state, self._map)
            uncertainty = hideout_uncertainty(state.hideout_zone, position_pmf)

        if self._objective == "legacy":
            objective = (1.0 + self._gamma * uncertainty) if won else -1.0
        else:
            # Progress part of the participant score, plus the stealth part at
            # this env's gamma. At the default gamma (SCORE_STUDY_V1_STEALTH)
            # this is exactly participant_score(); any other gamma is a
            # deliberate, logged deviation from the study's objective.
            objective = participant_score(self._best_progress, won, 0.0)
            if won:
                objective += self._gamma * uncertainty

        terms = self._shaping(phi_prev, None)
        terms["objective"] = objective
        reward = self._accumulate(terms)

        self._cops.on_episode_end(state, winner)
        self._director.on_game_end(winner, state.turn)
        self._state = state
        obs = build_obs(state, self._map, self._all_dists, self._diameter)
        info = {
            "winner": winner,
            "reward_terms": dict(self._ep_terms),
            # Always the study's score, whatever objective this env trains on,
            # so legacy and score runs are comparable on the same axis.
            "score": participant_score(self._best_progress, won, uncertainty),
            "explore": self._explore_stats(),
        }
        return obs, reward, True, False, info

    def _explore_stats(self) -> dict[str, float]:
        """Map coverage from this env's lifetime visit counts — delta's mechanism.

        coverage       fraction of Jack nodes visited at least once.
        visit_entropy  entropy of the visit distribution, normalised to [0, 1]
                       by log(n_jack): 1.0 is perfectly even coverage.
        """
        counts = self._visit_counts
        total = sum(counts.values())
        if total == 0 or self._n_jack < 2:
            return {"coverage": 0.0, "visit_entropy": 0.0}
        entropy = -sum((c / total) * math.log(c / total) for c in counts.values())
        return {
            "coverage": len(counts) / self._n_jack,
            "visit_entropy": entropy / math.log(self._n_jack),
        }

    def set_director_difficulty(self, value: float) -> None:
        if hasattr(self._director, "set_difficulty"):
            self._director.set_difficulty(value)

    def _action_mask(self, legal_edges: list) -> np.ndarray:
        mask = np.zeros(self._n_jack, dtype=bool)
        for edge in legal_edges:
            mask[edge.destination.id] = True
        return mask
