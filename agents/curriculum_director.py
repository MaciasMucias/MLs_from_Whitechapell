from __future__ import annotations

import random
from dataclasses import replace

from agents.base import Director
from engine.graph import Map
from engine.state import GameState

INITIAL_DIFFICULTY: float = (
    -1.0
)  # full suppression — cops nearly blind, easiest for Jack


class CurriculumDirector(Director):
    """
    Adjusts cop knowledge difficulty via a scalar in [-1.0, 1.0].

    difficulty < 0  →  suppress visited_at entries at rate |difficulty| (easier for Jack)
    difficulty = 0  →  no-op
    difficulty > 0  →  inject undiscovered true nodes at rate difficulty (harder for Jack)

    Difficulty is snapshotted at on_episode_start so it is constant for the
    entire episode — mid-episode changes would produce incoherent cop knowledge.

    set_difficulty() is called by the training loop (via the worker pipe protocol)
    between rollouts; the new value takes effect at the next episode start.
    """

    def __init__(
        self,
        initial_difficulty: float = INITIAL_DIFFICULTY,
        rng: random.Random | None = None,
    ) -> None:
        self._difficulty: float = float(initial_difficulty)
        self._episode_difficulty: float = float(initial_difficulty)
        self._rng = rng or random.Random()

    def set_difficulty(self, value: float) -> None:
        self._difficulty = float(max(-1.0, min(1.0, value)))

    @property
    def difficulty(self) -> float:
        return self._difficulty

    def on_episode_start(self, state: GameState, game_map: Map) -> None:
        self._episode_difficulty = self._difficulty

    def filter_knowledge(self, state: GameState, game_map: Map) -> GameState:
        d = self._episode_difficulty
        if d == 0.0:
            return state

        ck = state.cop_knowledge

        if d < 0.0:
            # Suppression: drop each entry with probability |d|.
            # depth=0 is the public starting position — never suppress.
            kept = tuple(
                entry
                for entry in ck.visited_at
                if entry[1] == 0 or self._rng.random() > abs(d)
            )
            new_visited = kept
        else:
            # Injection: reveal undiscovered true nodes with probability d.
            # Only injects real nodes from jack_trace — no ghost entries.
            known_nodes = {n for n, _ in ck.visited_at}
            # sorted() for deterministic RNG consumption given a seed
            undiscovered = sorted(state.jack_trace - known_nodes)
            injected = tuple(
                (node, state.jack_path.index(node))
                for node in undiscovered
                if self._rng.random() < d
            )
            new_visited = ck.visited_at + injected

        if new_visited is ck.visited_at:
            return state

        return replace(state, cop_knowledge=replace(ck, visited_at=new_visited))
