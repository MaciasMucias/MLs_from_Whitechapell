import random

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from engine.env import (
    CopTurn,
    end_of_round,
    legal_jack_edges,
    reachable_cop_nodes,
    step_cop,
    step_jack,
)
from engine.graph import Map
from engine.state import GameState
from server.session import GameSession, get_session, new_session, state_view

router = APIRouter()


class JackMoveRequest(BaseModel):
    destination: int


def _random_cop_turn(
    state: GameState, cop_idx: int, game_map: Map, rng: random.Random
) -> CopTurn:
    reachable = list(reachable_cop_nodes(state.cop_positions[cop_idx], game_map))
    destination = rng.choice(reachable)
    return CopTurn(cop_idx=cop_idx, destination=destination, search=True)


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
async def jack_move(game_id: str, body: JackMoveRequest):
    session = get_session(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    if session.terminated:
        raise HTTPException(status_code=400, detail="Game already over")

    edges = legal_jack_edges(session.state, session.game_map, blocking=session.blocking)
    edge = next((e for e in edges if e.destination.id == body.destination), None)
    if edge is None:
        raise HTTPException(status_code=400, detail="Illegal move")

    events: list[dict] = []
    state, terminated, winner = step_jack(session.state, edge)

    if not terminated:
        for cop_idx in range(session.game_map.num_cops):
            cop_turn = _random_cop_turn(state, cop_idx, session.game_map, session.rng)
            state, terminated, winner = step_cop(state, cop_turn, session.game_map)
            cop_node = session.game_map.cop_nodes[cop_turn.destination - 1]
            events.append({
                "cop": cop_idx,
                "moved_to": cop_turn.destination,
                "action": "search",
                "jack_neighbours": [n.id for n in cop_node.jack_neighbours],
            })
            if terminated:
                break

        if not terminated:
            state, terminated, winner = end_of_round(
                state, session.game_map,
                blocking=session.blocking,
                turn_limit=session.turn_limit,
            )

    session.state = state
    session.terminated = terminated
    session.winner = winner

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
