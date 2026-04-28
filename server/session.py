import random
import uuid
from dataclasses import dataclass, field

from engine.env import legal_jack_edges, make_initial_state
from engine.game import StepContext
from engine.graph import Map
from engine.state import GameState


@dataclass
class GameSession:
    game_id: str
    ctx: StepContext
    rng: random.Random
    history: list = field(default_factory=list)  # GameState undo stack (admin panel)

    @property
    def state(self) -> GameState: return self.ctx.state
    @state.setter
    def state(self, v: GameState) -> None: self.ctx.state = v

    @property
    def terminated(self) -> bool: return self.ctx.terminated
    @terminated.setter
    def terminated(self, v: bool) -> None: self.ctx.terminated = v

    @property
    def winner(self) -> str | None: return self.ctx.winner
    @winner.setter
    def winner(self, v: str | None) -> None: self.ctx.winner = v

    @property
    def game_map(self) -> Map: return self.ctx.game_map

    @property
    def blocking(self) -> bool: return self.ctx.blocking
    @blocking.setter
    def blocking(self, v: bool) -> None: self.ctx.blocking = v

    @property
    def turn_limit(self) -> int | None: return self.ctx.turn_limit
    @turn_limit.setter
    def turn_limit(self, v: int | None) -> None: self.ctx.turn_limit = v


_sessions: dict[str, GameSession] = {}


def new_session(game_map: Map, rng: random.Random | None = None) -> GameSession:
    if rng is None:
        rng = random.Random()
    game_id = str(uuid.uuid4())[:8]
    state = make_initial_state(game_map, rng)
    ctx = StepContext(game_map=game_map, state=state, terminated=False, winner=None)
    session = GameSession(game_id=game_id, ctx=ctx, rng=rng)
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
        "visited": sorted(n for n, _ in s.cop_knowledge.visited_at),
        "visited_at": sorted(s.cop_knowledge.visited_at),
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
