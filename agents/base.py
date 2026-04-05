from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from engine.env import CopTurn
from engine.graph import JackEdge, Map
from engine.state import GameState


@dataclass
class AgentOutput:
    edge: JackEdge
    logprob: float | None = None  # log π(a|s), for PPO rollout buffer
    value: float | None = None    # V(s), for PPO advantage computation
    aux: dict[str, Any] | None = None


class JackAgent(ABC):
    @abstractmethod
    def act(self, state: GameState, legal_edges: list[JackEdge], game_map: Map) -> AgentOutput: ...
    def on_episode_start(self, state: GameState, game_map: Map) -> None: pass
    def on_episode_end(self, final_state: GameState, winner: str) -> None: pass


class CopAgent(ABC):
    """
    All cops plan simultaneously on the same pre-move state. Called once per
    round, returns one CopTurn per cop. Execution then proceeds sequentially
    through step_cop() to accumulate state changes, but planning is blind to
    intra-round position updates.
    """
    @abstractmethod
    def act(self, state: GameState, game_map: Map) -> list[CopTurn]: ...
    def on_episode_start(self, state: GameState, game_map: Map) -> None: pass
    def on_episode_end(self, final_state: GameState, winner: str) -> None: pass


class Director(ABC):
    """
    May only modify cop_knowledge.visited. Called once per round after Jack
    moves and before cops plan, so cops act on the manipulated knowledge.
    on_game_end() receives performance stats for curriculum adjustment.
    """
    @abstractmethod
    def filter_knowledge(self, state: GameState, game_map: Map) -> GameState: ...
    def on_game_end(self, winner: str, turns_survived: int) -> None: pass
