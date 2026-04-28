import random
import uuid
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from engine.game import StepContext
from engine.graph_utils import jack_bfs_distances
from engine.state import CopKnowledge, GameState
from server.replay import list_replays, load_replay
from server.session import GameSession, register_session, state_view

replay_router = APIRouter()


class ForkAtTurnBody(BaseModel):
    turn: int


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
        raise HTTPException(status_code=400, detail=f"Turn out of range (-1–{len(record.rounds) - 1})")

    gm = request.app.state.game_map

    # Hideout zone: approximate as BFS neighbourhood around the hideout
    hideout = record.hideout
    hideout_distances = jack_bfs_distances(hideout, gm)
    hideout_zone = frozenset(v for v, d in hideout_distances.items() if d <= gm.zone_radius)

    if body.turn == -1:
        # Fork from the initial state — no moves have happened yet
        state = GameState(
            jack_pos=record.initial_jack_pos,
            cop_positions=tuple(record.initial_cop_positions),
            hideout=hideout,
            hideout_zone_anchor=hideout,
            hideout_zone=hideout_zone,
            turn=0,
            jack_trace=frozenset({record.initial_jack_pos}),
            cop_knowledge=CopKnowledge(jack_start=record.initial_jack_pos),
        )
    else:
        rnd = record.rounds[body.turn]

        # Cop positions: apply all moves through the requested turn
        cop_positions = list(record.initial_cop_positions)
        for r in record.rounds[: body.turn + 1]:
            for ca in r.cop_actions:
                cop_positions[ca.cop_idx] = ca.to_node

        # Jack trace: every node Jack occupied through the requested turn
        jack_trace: set[int] = {record.initial_jack_pos}
        for r in record.rounds[: body.turn + 1]:
            jack_trace.add(r.jack_to)

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
            hideout_zone_anchor=hideout,
            hideout_zone=hideout_zone,
            turn=body.turn + 1,
            jack_trace=frozenset(jack_trace),
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
    session = GameSession(
        game_id=str(uuid.uuid4())[:8],
        ctx=ctx,
        rng=random.Random(),
    )
    register_session(session)
    return state_view(session)
