import random
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

from agents.heuristic_cops import HeuristicCops
from engine.env import legal_jack_edges
from engine.game import step_round
from engine.graph import Map
from engine.graph_utils import jack_bfs_distances
from server.session import (
    get_session,
    new_session,
    pop_participant_meta,
    set_participant_meta,
    state_view,
)

router = APIRouter()
debug_router = APIRouter()


class JackMoveRequest(BaseModel):
    destination: int


class NewGameRequest(BaseModel):
    gaming_habit: Literal["never_played", "played_few", "played_many", "unknown"] = (
        "unknown"
    )
    # The participant flow reserves its whole map order up front via
    # POST /api/course/new and then requests each map by name. map_name/
    # scenario_order are omitted only by debug/admin callers, which get a
    # random single map.
    map_name: str | None = None
    scenario_order: int | None = None


async def _new_game_impl(body: NewGameRequest, request: Request):
    game_maps = request.app.state.game_maps
    if body.map_name is not None and body.map_name in game_maps:
        map_name = body.map_name
        scenario_order = body.scenario_order if body.scenario_order is not None else -1
    else:
        map_name = random.choice(list(game_maps.keys()))
        scenario_order = -1
    session = new_session(game_maps[map_name], map_name=map_name)
    set_participant_meta(
        session.game_id,
        {
            "map_name": map_name,
            "scenario_order": scenario_order,
            "gaming_habit": body.gaming_habit,
        },
    )
    return state_view(session)


async def _jack_move_impl(game_id: str, body: JackMoveRequest, request: Request):
    session = get_session(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    if session.ctx.terminated:
        raise HTTPException(status_code=400, detail="Game already over")

    edges = legal_jack_edges(
        session.ctx.state, session.ctx.game_map, blocking=session.ctx.blocking
    )
    edge = next((e for e in edges if e.destination.id == body.destination), None)
    if edge is None:
        raise HTTPException(status_code=400, detail="Illegal move")

    events, terminated, winner = step_round(session.ctx, edge, session.cop_agent)

    if terminated:
        from dataclasses import asdict
        from server.replay import build_and_save_replay
        from server.database import ParticipantGame, save_game

        replay_record = build_and_save_replay(session)
        ctx = session.ctx
        meta = pop_participant_meta(session.game_id)
        effective_limit = (
            ctx.turn_limit if ctx.turn_limit is not None else ctx.game_map.turn_limit
        )
        save_game(
            ParticipantGame(
                game_id=session.game_id,
                map_name=meta.get("map_name", "unknown"),
                scenario_order=meta.get("scenario_order", -1),
                gaming_habit=meta.get("gaming_habit", "unknown"),
                outcome=ctx.winner or "unknown",
                turns_survived=len(ctx.history),
                turn_limit=effective_limit,
                move_sequence=[
                    {
                        "turn": r.turn,
                        "jack_to": r.state_after_round.jack_pos,
                        "winner": r.winner,
                    }
                    for r in ctx.history
                ],
                replay=asdict(replay_record),
            )
        )

    _state = session.ctx.state
    _game_map = session.ctx.game_map
    _bfs = jack_bfs_distances(_state.hideout, _game_map)
    _max_dist = max(_bfs.values()) if _bfs else 1
    _curr_dist = _bfs.get(_state.jack_pos, _max_dist)
    _norm_dist = _curr_dist / _max_dist if _max_dist > 0 else 0.0
    score_info: dict = {"normalized_distance": _norm_dist}
    if terminated and winner == "jack":
        _pmf = HeuristicCops.compute_pmf(_state, _game_map)
        _zone = _state.hideout_zone
        if _zone:
            _nonzero = sum(1 for h in _zone if _pmf.get(h, 0.0) > 0.0)
            score_info["hideout_uncertainty"] = _nonzero / len(_zone)

    view = state_view(session)
    view["events"] = events
    view["score_info"] = score_info
    return view


@router.post("/game/new")
@limiter.limit("5/minute")
async def new_game(body: NewGameRequest, request: Request):
    return await _new_game_impl(body, request)


@debug_router.post("/game/new")
async def debug_new_game(body: NewGameRequest, request: Request):
    return await _new_game_impl(body, request)


@router.post("/game/{game_id}/jack-move")
@limiter.limit("60/minute")
async def jack_move(game_id: str, body: JackMoveRequest, request: Request):
    return await _jack_move_impl(game_id, body, request)


@debug_router.post("/game/{game_id}/jack-move")
async def debug_jack_move(game_id: str, body: JackMoveRequest, request: Request):
    return await _jack_move_impl(game_id, body, request)


@router.get("/course")
@debug_router.get("/course")
async def get_course(request: Request):
    return request.app.state.course


def _reserve_course_order(request: Request):
    """Reserve one participant's full, counterbalanced map order.

    Returns the course metadata entries reordered for this participant, each
    annotated with its ``scenario_order`` (position in the course).
    """
    order = request.app.state.course_sequencer.next_order()
    by_name = {entry["name"]: entry for entry in request.app.state.course}
    return [{**by_name[name], "scenario_order": pos} for name, pos in order]


@router.post("/course/new")
@limiter.limit("5/minute")
async def new_course(request: Request):
    return _reserve_course_order(request)


@debug_router.post("/course/new")
async def debug_new_course(request: Request):
    return _reserve_course_order(request)


@router.get("/game/{game_id}")
@debug_router.get("/game/{game_id}")
async def get_game(game_id: str):
    session = get_session(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return state_view(session)


@router.get("/map")
@debug_router.get("/map")
async def get_map(request: Request, map_name: str | None = None):
    gm_dict: dict[str, Map] = request.app.state.game_maps
    gm: Map = (
        gm_dict[map_name]
        if map_name and map_name in gm_dict
        else next(iter(gm_dict.values()))
    )
    return {
        "jack_nodes": [
            {
                "id": n.id,
                "x": n.x,
                "y": n.y,
                "node_type": n.node_type,
                "edges": list(dict.fromkeys(e.destination.id for e in n.edges)),
                "edge_routes": [
                    {"destination": e.destination.id, "via": [c.id for c in e.via]}
                    for e in n.edges
                ],
            }
            for n in gm.jack_nodes
        ],
        "cop_nodes": [
            {
                "id": n.id,
                "x": n.x,
                "y": n.y,
                "edges": [nb.id for nb in n.edges],
                "jack_neighbours": [jn.id for jn in n.jack_neighbours],
            }
            for n in gm.cop_nodes
        ],
    }


@router.get("/map-svg")
@debug_router.get("/map-svg")
async def get_map_svg():
    return FileResponse("Mapa_v5.svg", media_type="image/svg+xml")
