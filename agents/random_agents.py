from __future__ import annotations
import random

from engine.env import CopTurn
from engine.graph_utils import reachable_cop_nodes
from engine.graph import Map
from engine.state import GameState
from agents.base import AgentOutput, CopAgent, Director, JackAgent


class RandomJack(JackAgent):
    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def act(self, state, legal_edges, game_map) -> AgentOutput:
        return AgentOutput(edge=self._rng.choice(legal_edges))


class RandomCops(CopAgent):
    """Always searches; each cop moves to a random reachable node."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def act(self, state: GameState, game_map: Map) -> tuple[list[CopTurn], None]:
        turns = []
        for cop_idx, cop_pos in enumerate(state.cop_positions):
            reachable = list(reachable_cop_nodes(cop_pos, game_map))
            turns.append(
                CopTurn(
                    cop_idx=cop_idx,
                    destination=self._rng.choice(reachable),
                    search=True,
                )
            )
        return turns, None


class NoOpDirector(Director):
    def filter_knowledge(self, state: GameState, game_map: Map) -> GameState:
        return state
