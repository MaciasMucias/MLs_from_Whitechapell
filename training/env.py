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
from engine.state import GameState
from training.obs import build_obs, precompute_distances


class JackEnv:
    """
    Gym-style environment for training Jack's RL policy against heuristic cops.

    Action space:  Discrete(n_jack) with per-step action masking.
                   Action i = move to Jack node i. Only legal neighbours are
                   unmasked; all others must be set to -inf before softmax.
    Observation:   1,416-dim float32 vector (see training/obs.py).
    Reward:        Terminal ±1.0 dominant; shaped by hideout-distance progress,
                   cop-distance delta, PMF potential delta, and count-based
                   exploration bonus.
    """

    def __init__(
        self,
        game_map: Map,
        alpha: float = 0.1,
        beta: float = 0.05,
        delta: float = 0.01,
        gamma: float = 0.5,
        zeta: float = 0.1,
        blocking: bool = False,
        rng: random.Random | None = None,
        director: Director | None = None,
    ) -> None:
        self._map = game_map
        self._alpha = alpha
        self._beta = beta
        self._delta = delta
        self._gamma = gamma
        self._zeta = zeta
        self._blocking = blocking  # EXTEND(blocking): _all_dists is topology-only (no cop positions);
        # with blocking enabled it underestimates true distance when cops wall off the short route.
        # Action masking enforces blocking correctly — this is a known reward-shaping approximation.
        self._rng = rng or random.Random()
        # EXTEND(multinight): reset() currently starts a fresh single-night game;
        # multi-night would chain states across nights here

        self._all_dists, self._diameter = precompute_distances(game_map)
        self._n_jack = len(game_map.jack_nodes)

        # Separate RNG stream so cop decisions don't consume Jack's seed
        self._cops = HeuristicCops()
        self._director: Director = director if director is not None else NoOpDirector()

        # Per-worker visit counts — intentionally persist across episodes
        self._visit_counts: dict[int, int] = {}

        self._state: GameState | None = None
        # PMF from the most recent cops.act(); empty at episode start
        self._pmf: dict[int, float] = {}

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

        legal = legal_jack_edges(self._state, self._map, blocking=self._blocking)
        obs = build_obs(self._state, self._map, self._all_dists, self._diameter)
        return obs, {"action_mask": self._action_mask(legal)}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._state is not None, "call reset() before step()"

        state = self._state
        prev_pos = state.jack_pos
        prev_dist = self._all_dists[prev_pos].get(state.hideout, self._diameter)
        prev_pmf_at_prev = self._pmf.get(prev_pos, 0.0)
        prev_min_cop_dist = min(
            self._all_dists[prev_pos].get(jn.id, self._diameter)
            for cp in state.cop_positions
            for jn in self._map.cop_nodes[cp].jack_neighbours
        )

        # Resolve action index → JackEdge
        legal = legal_jack_edges(state, self._map, blocking=self._blocking)
        edge = next((e for e in legal if e.destination.id == action), None)
        assert edge is not None, f"action {action} is not a legal move from {prev_pos}"

        # Jack moves
        state, terminated, winner = step_jack(state, edge)
        if terminated:
            return self._finish(state, winner, +1.0)

        # Director intercepts cop knowledge (no-op for now)
        state = self._director.filter_knowledge(state, self._map)

        # Cops plan — updates last_position_pmf
        cop_turns, _ = self._cops.act(state, self._map)
        curr_pmf = self._cops.last_position_pmf

        # Execute cop turns
        for cop_turn in cop_turns:
            state, terminated, winner, _ = step_cop(state, cop_turn, self._map)
            if terminated:
                return self._finish(state, winner, -1.0)

        # End-of-round check (turn limit / no legal moves when blocking)
        state, terminated, winner = end_of_round(
            state, self._map, blocking=self._blocking
        )
        if terminated:
            return self._finish(state, winner, -1.0)

        # Shaping rewards
        curr_pos = state.jack_pos
        curr_dist = self._all_dists[curr_pos].get(state.hideout, self._diameter)
        curr_pmf_at_curr = curr_pmf.get(curr_pos, 0.0)
        curr_min_cop_dist = min(
            self._all_dists[curr_pos].get(jn.id, self._diameter)
            for cp in state.cop_positions
            for jn in self._map.cop_nodes[cp].jack_neighbours
        )

        reward = 0.0
        reward += self._alpha * (prev_dist - curr_dist) / self._diameter
        reward += self._beta * (prev_pmf_at_prev - curr_pmf_at_curr)
        reward += self._zeta * (curr_min_cop_dist - prev_min_cop_dist) / self._diameter
        self._visit_counts[curr_pos] = self._visit_counts.get(curr_pos, 0) + 1
        reward += self._delta / math.sqrt(self._visit_counts[curr_pos])

        self._state = state
        self._pmf = curr_pmf
        next_legal = legal_jack_edges(state, self._map, blocking=self._blocking)
        obs = build_obs(state, self._map, self._all_dists, self._diameter)
        return obs, reward, False, False, {"action_mask": self._action_mask(next_legal)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _finish(
        self, state: GameState, winner: str, reward: float
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if winner == "jack":
            # Recompute position PMF on the current state — Jack just moved so
            # the cached value from the previous cops.act() is stale.
            # Reward is proportional to how many hideout zone nodes still have
            # non-zero position PMF mass (cops uncertain which zone node Jack is in).
            # Naturally 0 early in the game when Jack hasn't entered the zone yet.
            position_pmf = HeuristicCops.compute_pmf(state, self._map)
            nonzero_in_zone = sum(
                1 for h in state.hideout_zone if position_pmf.get(h, 0.0) > 0.0
            )
            reward += self._gamma * (nonzero_in_zone / len(state.hideout_zone))

        self._cops.on_episode_end(state, winner)
        self._director.on_game_end(winner, state.turn)
        self._state = state
        obs = build_obs(state, self._map, self._all_dists, self._diameter)
        return obs, reward, True, False, {"winner": winner}

    def set_director_difficulty(self, value: float) -> None:
        if hasattr(self._director, "set_difficulty"):
            self._director.set_difficulty(value)

    def _action_mask(self, legal_edges: list) -> np.ndarray:
        mask = np.zeros(self._n_jack, dtype=bool)
        for edge in legal_edges:
            mask[edge.destination.id] = True
        return mask
