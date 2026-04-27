import random
import uuid
from dataclasses import dataclass, field

from engine.env import legal_jack_edges, make_initial_state
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
    blocking: bool = False
    turn_limit: int | None = None   # overrides game_map.turn_limit when set
    history: list = field(default_factory=list)        # list[GameState], capped at 50
    round_history: list = field(default_factory=list)  # list[RoundRecord], full game


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


def register_session(session: GameSession) -> None:
    _sessions[session.game_id] = session


def get_session(game_id: str) -> GameSession | None:
    return _sessions.get(game_id)


def push_history(session: GameSession) -> None:
    session.history.append(session.state)
    if len(session.history) > 50:
        session.history.pop(0)


def state_view(session: GameSession) -> dict:
    """Serialise session to a JSON-safe dict for API responses."""
    s = session.state
    gm = session.game_map
    effective_limit = session.turn_limit if session.turn_limit is not None else gm.turn_limit

    legal: list[int] = []
    if not session.terminated:
        seen: set[int] = set()
        for e in legal_jack_edges(s, gm, blocking=session.blocking):
            d = e.destination.id
            if d not in seen:
                seen.add(d)
                legal.append(d)

    return {
        "game_id": session.game_id,
        "jack_pos": s.jack_pos,
        "cop_positions": list(s.cop_positions),
        "hideout": s.hideout,
        "turn": s.turn,
        "turn_limit": effective_limit,
        "legal_moves": legal,
        "visited": sorted(s.cop_knowledge.visited),
        "search_misses": [list(m) for m in s.cop_knowledge.search_misses],
        "arrest_misses": [list(m) for m in s.cop_knowledge.arrest_misses],
        "jack_trace": sorted(s.jack_trace),
        "blocking": session.blocking,
        "history_size": len(session.history),
        "terminated": session.terminated,
        "winner": session.winner,
        "hideout_zone_anchor": s.hideout_zone_anchor,
        "hideout_zone": sorted(s.hideout_zone),
    }
