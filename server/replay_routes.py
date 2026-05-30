import random
import uuid
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from agents import HeuristicCops
from engine.game import StepContext
from engine.state import CopKnowledge, GameState
from server.database import load_replay_from_db
from server.replay import ReplayRecord, _deserialize_record, list_replays, load_replay
from server.session import GameSession, register_session, state_view

replay_router = APIRouter()


class ForkAtTurnBody(BaseModel):
    turn: int


class LoadFromDbBody(BaseModel):
    db_path: str
    db_id: int


class ForkFromDbBody(BaseModel):
    db_path: str
    db_id: int
    turn: int


def _build_session_from_replay(record: ReplayRecord, turn: int, app) -> dict:
    """Reconstruct game state at end of `turn` and return a live session view."""
    gm = next(iter(app.state.game_maps.values()))

    hideout = record.hideout
    hideout_zone_anchor = record.hideout_zone_anchor
    hideout_zone = frozenset(record.hideout_zone)

    if turn == -1:
        state = GameState(
            jack_pos=record.initial_jack_pos,
            cop_positions=tuple(record.initial_cop_positions),
            hideout=hideout,
            hideout_zone_anchor=hideout_zone_anchor,
            hideout_zone=hideout_zone,
            turn=0,
            jack_trace=frozenset({record.initial_jack_pos}),
            jack_path=(record.initial_jack_pos,),
            cop_searched_hits=frozenset(),
            cop_searched_misses=frozenset(),
            cop_knowledge=CopKnowledge(jack_start=record.initial_jack_pos),
        )
    else:
        rnd = record.rounds[turn]

        cop_positions = list(record.initial_cop_positions)
        for r in record.rounds[: turn + 1]:
            for ca in r.cop_actions:
                cop_positions[ca.cop_idx] = ca.to_node

        jack_trace: set[int] = {record.initial_jack_pos}
        cop_searched_hits: set[int] = set()
        cop_searched_misses: set[int] = set()
        for r in record.rounds[: turn + 1]:
            jack_trace.add(r.jack_to)
            for ca in r.cop_actions:
                cop_searched_hits.update(ca.search_hits)
                cop_searched_misses.update(ca.search_miss_nodes)
        jack_path = tuple(
            [record.initial_jack_pos] + [r.jack_to for r in record.rounds[: turn + 1]]
        )

        ck = CopKnowledge(
            jack_start=record.initial_jack_pos,
            visited_at=tuple(tuple(m) for m in rnd.visited_at_after),
            search_misses=tuple(tuple(m) for m in rnd.search_misses_after),
            arrest_misses=tuple(tuple(m) for m in rnd.arrest_misses_after),
        )

        state = GameState(
            jack_pos=rnd.jack_to,
            cop_positions=tuple(cop_positions),
            hideout=hideout,
            hideout_zone_anchor=hideout_zone_anchor,
            hideout_zone=hideout_zone,
            turn=turn + 1,
            jack_trace=frozenset(jack_trace),
            jack_path=jack_path,
            cop_searched_hits=frozenset(cop_searched_hits),
            cop_searched_misses=frozenset(cop_searched_misses),
            cop_knowledge=ck,
        )

    ctx = StepContext(
        game_map=gm,
        state=state,
        terminated=False,
        winner=None,
        blocking=record.blocking,
        turn_limit=record.turn_limit,
    )
    cop_agent = HeuristicCops()
    cop_agent.on_episode_start(state, gm)
    session = GameSession(
        game_id=str(uuid.uuid4())[:8],
        ctx=ctx,
        rng=random.Random(),
        cop_agent=cop_agent,
    )
    register_session(session)
    return state_view(session)


@replay_router.get("")
async def get_replays():
    """Return metadata list for all saved replay slots."""
    return list_replays()


@replay_router.get("/{slot}")
async def get_replay(slot: int):
    """Return the full replay record for a given slot."""
    record = load_replay(slot)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No replay in slot {slot}")
    return asdict(record)


@replay_router.post("/{slot}/fork-at-turn")
async def fork_at_turn(slot: int, body: ForkAtTurnBody, request: Request):
    """Create a live game session from the state at the end of a replay turn."""
    record = load_replay(slot)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No replay in slot {slot}")
    if body.turn < -1 or body.turn >= len(record.rounds):
        raise HTTPException(
            status_code=400, detail=f"Turn out of range (-1–{len(record.rounds) - 1})"
        )
    return _build_session_from_replay(record, body.turn, request.app)


@replay_router.post("/load-from-db")
async def load_replay_from_db_endpoint(body: LoadFromDbBody):
    """Return the full replay record stored in a SQLite games table row."""
    data = load_replay_from_db(body.db_id, Path(body.db_path))
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No game found with id={body.db_id} in {body.db_path}",
        )
    return asdict(_deserialize_record(data))


@replay_router.post("/fork-from-db")
async def fork_replay_from_db(body: ForkFromDbBody, request: Request):
    """Create a live game session from a DB-stored replay at the given turn."""
    data = load_replay_from_db(body.db_id, Path(body.db_path))
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No game found with id={body.db_id} in {body.db_path}",
        )
    record = _deserialize_record(data)
    if body.turn < -1 or body.turn >= len(record.rounds):
        raise HTTPException(
            status_code=400, detail=f"Turn out of range (-1–{len(record.rounds) - 1})"
        )
    return _build_session_from_replay(record, body.turn, request.app)
