from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from engine.env import legal_jack_edges
from engine.game import step_round
from engine.graph import Map
from server.session import get_session, new_session, state_view

router = APIRouter()


class JackMoveRequest(BaseModel):
    destination: int


@router.post("/game/new")
async def new_game(request: Request):
    session = new_session(request.app.state.game_map)
    return state_view(session)


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

    events, terminated, winner = step_round(
        session.ctx, edge,
        request.app.state.cop_agent,
        request.app.state.director,
    )

    if terminated:
        from server.replay import build_and_save_replay
        build_and_save_replay(session)

    view = state_view(session)
    view["events"] = events
    return view


@router.get("/map")
async def get_map(request: Request):
    gm: Map = request.app.state.game_map
    return {
        "jack_nodes": [
            {
                "id": n.id,
                "x": n.x,
                "y": n.y,
                "node_type": n.node_type,
                "edges": list(dict.fromkeys(e.destination.id for e in n.edges)),
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
