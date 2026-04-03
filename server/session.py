import random
import uuid
from dataclasses import dataclass

from engine.env import make_initial_state
from engine.graph import Map
from engine.state import GameState


@dataclass
class GameSession:
    game_id: str
    game_map: Map
    state: GameState
    terminated: bool
    winner: str | None
    rng: random.Random


_sessions: dict[str, GameSession] = {}


def new_session(game_map: Map, rng: random.Random | None = None) -> GameSession:
    if rng is None:
        rng = random.Random()
    game_id = str(uuid.uuid4())[:8]
    state = make_initial_state(game_map, rng)
    session = GameSession(
        game_id=game_id,
        game_map=game_map,
        state=state,
        terminated=False,
        winner=None,
        rng=rng,
    )
    _sessions[game_id] = session
    return session


def get_session(game_id: str) -> GameSession | None:
    return _sessions.get(game_id)
