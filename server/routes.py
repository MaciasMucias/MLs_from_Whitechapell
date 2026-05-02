from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from engine.env import legal_jack_edges
from engine.game import step_round
from engine.graph import Map
from server.session import get_session, new_session, pop_participant_meta, set_participant_meta, state_view

router = APIRouter()


class JackMoveRequest(BaseModel):
    destination: int


class NewGameRequest(BaseModel):
    map_name: str = "whitechapel"
    gaming_habit: str = "unknown"


@router.post("/game/new")
async def new_game(body: NewGameRequest, request: Request):
    game_maps = request.app.state.game_maps
    if body.map_name not in game_maps:
        raise HTTPException(status_code=400, detail=f"Unknown map: {body.map_name}")
    session = new_session(game_maps[body.map_name])
    set_participant_meta(session.game_id, {
        "map_name": body.map_name,
        "gaming_habit": body.gaming_habit,
    })
    return state_view(session)


@router.get("/course")
async def get_course(request: Request):
    return request.app.state.course


@router.get("/game/{game_id}")
async def get_game(game_id: str):
    session = get_session(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return state_view(session)


@router.post("/game/{game_id}/jack-move")
async def jack_move(game_id: str, body: JackMoveRequest, request: Request):
    session = get_session(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    if session.ctx.terminated:
        raise HTTPException(status_code=400, detail="Game already over")

    edges = legal_jack_edges(session.ctx.state, session.ctx.game_map, blocking=session.ctx.blocking)
    edge = next((e for e in edges if e.destination.id == body.destination), None)
    if edge is None:
        raise HTTPException(status_code=400, detail="Illegal move")

    events, terminated, winner = step_round(session.ctx, edge, session.cop_agent)

    if terminated:
        from server.replay import build_and_save_replay
        from server.database import ParticipantGame, save_game
        build_and_save_replay(session)
        ctx = session.ctx
        meta = pop_participant_meta(session.game_id)
        effective_limit = ctx.turn_limit if ctx.turn_limit is not None else ctx.game_map.turn_limit
        save_game(ParticipantGame(
            game_id=session.game_id,
            map_name=meta.get("map_name", "unknown"),
            gaming_habit=meta.get("gaming_habit", "unknown"),
            outcome=ctx.winner or "unknown",
            turns_survived=len(ctx.history),
            turn_limit=effective_limit,
            move_sequence=[
                {"turn": r.turn, "jack_to": r.state_after_round.jack_pos, "winner": r.winner}
                for r in ctx.history
            ],
        ))

    view = state_view(session)
    view["events"] = events
    return view


@router.get("/map")
async def get_map(request: Request, map_name: str | None = None):
    gm_dict: dict[str, Map] = request.app.state.game_maps
    gm: Map = gm_dict[map_name] if map_name and map_name in gm_dict else next(iter(gm_dict.values()))
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
async def get_map_svg():
    return FileResponse("Mapa_v5.svg", media_type="image/svg+xml")
