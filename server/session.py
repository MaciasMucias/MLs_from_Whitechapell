import random
import uuid
from dataclasses import dataclass, field
from agents import HeuristicCops
from engine.env import legal_jack_edges, make_initial_state
from engine.game import StepContext
from engine.graph import Map


@dataclass
class GameSession:
    game_id: str
    ctx: StepContext
    rng: random.Random
    cop_agent: HeuristicCops
    history: list = field(default_factory=list)  # GameState undo stack (admin panel)


_sessions: dict[str, GameSession] = {}
_participant_meta: dict[str, dict] = {}


def set_participant_meta(game_id: str, meta: dict) -> None:
    _participant_meta[game_id] = meta


def pop_participant_meta(game_id: str) -> dict:
    return _participant_meta.pop(game_id, {})


def new_session(game_map: Map, rng: random.Random | None = None) -> GameSession:
    if rng is None:
        rng = random.Random()
    game_id = str(uuid.uuid4())[:8]
    state = make_initial_state(game_map, rng)
    ctx = StepContext(game_map=game_map, state=state, terminated=False, winner=None)
    cop_agent = HeuristicCops()
    cop_agent.on_episode_start(state, game_map)
    session = GameSession(game_id=game_id, ctx=ctx, rng=rng, cop_agent=cop_agent)
    _sessions[game_id] = session
    return session


def register_session(session: GameSession) -> None:
    _sessions[session.game_id] = session


def get_session(game_id: str) -> GameSession | None:
    return _sessions.get(game_id)


def push_history(session: GameSession) -> None:
    session.history.append(session.ctx.state)
    if len(session.history) > 50:
        session.history.pop(0)


def state_view(session: GameSession) -> dict:
    """Serialise session to a JSON-safe dict for API responses."""
    s = session.ctx.state
    gm = session.ctx.game_map
    effective_limit = (
        session.ctx.turn_limit if session.ctx.turn_limit is not None else gm.turn_limit
    )

    legal: list[int] = []
    if not session.ctx.terminated:
        seen: set[int] = set()
        for e in legal_jack_edges(s, gm, blocking=session.ctx.blocking):
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
        "visited_at": sorted(s.cop_knowledge.visited_at),
        "search_misses": [list(m) for m in s.cop_knowledge.search_misses],
        "arrest_misses": [list(m) for m in s.cop_knowledge.arrest_misses],
        "jack_trace": sorted(s.jack_trace),
        "jack_path": list(s.jack_path),
        "blocking": session.ctx.blocking,
        "arrest_all_enabled": session.ctx.arrest_all_enabled,
        "history_size": len(session.history),
        "terminated": session.ctx.terminated,
        "winner": session.ctx.winner,
        "hideout_zone_anchor": s.hideout_zone_anchor,
        "hideout_zone": sorted(s.hideout_zone),
    }
