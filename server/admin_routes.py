import uuid
from dataclasses import replace

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from engine.env import CopTurn, step_cop
from engine.state import CopKnowledge, GameState
from server.session import (
    GameSession,
    get_session,
    push_history,
    register_session,
    state_view,
)

admin_router = APIRouter()


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class TeleportJackBody(BaseModel):
    node: int

class TeleportCopBody(BaseModel):
    cop: int
    node: int

class CopActionBody(BaseModel):
    cop: int
    destination: int | None = None   # None = stay at current node
    search: bool = True
    arrest_target: int | None = None

class CopActionsBody(BaseModel):
    actions: list[CopActionBody]

class SetTurnBody(BaseModel):
    turn: int

class SetTurnLimitBody(BaseModel):
    turn_limit: int

class SetBlockingBody(BaseModel):
    blocking: bool

class InjectNodeBody(BaseModel):
    node: int

class SetKnowledgeBody(BaseModel):
    jack_start: int
    visited: list[int] = []
    search_misses: list[list[int]] = []
    arrest_misses: list[list[int]] = []

class SetTraceBody(BaseModel):
    nodes: list[int]

class NewFromStateBody(BaseModel):
    same_hideout: bool = False

class NodeInfoBody(BaseModel):
    cop_node: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_404(game_id: str) -> GameSession:
    session = get_session(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return session


def _run_cop_action(
    state: GameState,
    action: CopActionBody,
    session: GameSession,
) -> tuple[GameState, bool, str | None]:
    destination = action.destination if action.destination is not None else state.cop_positions[action.cop]
    cop_turn = CopTurn(
        cop_idx=action.cop,
        destination=destination,
        search=action.search,
        arrest_target=action.arrest_target,
    )
    return step_cop(state, cop_turn, session.game_map)


def _mutate_knowledge(
    session: GameSession,
    **kwargs,
) -> None:
    new_k = replace(session.state.cop_knowledge, **kwargs)
    session.state = replace(session.state, cop_knowledge=new_k)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@admin_router.post("/{game_id}/teleport-jack")
async def teleport_jack(game_id: str, body: TeleportJackBody):
    session = _get_or_404(game_id)
    gm = session.game_map
    if body.node < 1 or body.node > len(gm.jack_nodes):
        raise HTTPException(status_code=400, detail="Invalid jack node")
    push_history(session)
    session.state = replace(
        session.state,
        jack_pos=body.node,
        jack_trace=session.state.jack_trace | {body.node},
    )
    return state_view(session)


@admin_router.post("/{game_id}/teleport-cop")
async def teleport_cop(game_id: str, body: TeleportCopBody):
    session = _get_or_404(game_id)
    gm = session.game_map
    if body.cop < 0 or body.cop >= gm.num_cops:
        raise HTTPException(status_code=400, detail="Invalid cop index")
    if body.node < 1 or body.node > len(gm.cop_nodes):
        raise HTTPException(status_code=400, detail="Invalid cop node")
    push_history(session)
    positions = list(session.state.cop_positions)
    positions[body.cop] = body.node
    session.state = replace(session.state, cop_positions=tuple(positions))
    return state_view(session)


@admin_router.post("/{game_id}/cop-action")
async def cop_action(game_id: str, body: CopActionBody):
    session = _get_or_404(game_id)
    if body.cop < 0 or body.cop >= session.game_map.num_cops:
        raise HTTPException(status_code=400, detail="Invalid cop index")
    push_history(session)
    state, terminated, winner = _run_cop_action(session.state, body, session)
    session.state = state
    session.terminated = terminated
    session.winner = winner
    return state_view(session)


@admin_router.post("/{game_id}/cop-actions")
async def cop_actions(game_id: str, body: CopActionsBody):
    session = _get_or_404(game_id)
    gm = session.game_map
    push_history(session)

    provided = {a.cop: a for a in body.actions}
    state = session.state
    terminated = session.terminated
    winner = session.winner

    for cop_idx in range(gm.num_cops):
        if terminated:
            break
        action = provided.get(cop_idx) or CopActionBody(cop=cop_idx, search=True)
        state, terminated, winner = _run_cop_action(state, action, session)

    session.state = state
    session.terminated = terminated
    session.winner = winner
    return state_view(session)


@admin_router.post("/{game_id}/set-turn")
async def set_turn(game_id: str, body: SetTurnBody):
    session = _get_or_404(game_id)
    push_history(session)
    session.state = replace(session.state, turn=body.turn)
    return state_view(session)


@admin_router.post("/{game_id}/set-turn-limit")
async def set_turn_limit(game_id: str, body: SetTurnLimitBody):
    session = _get_or_404(game_id)
    session.turn_limit = body.turn_limit
    return state_view(session)


@admin_router.post("/{game_id}/set-blocking")
async def set_blocking(game_id: str, body: SetBlockingBody):
    session = _get_or_404(game_id)
    session.blocking = body.blocking
    return state_view(session)


@admin_router.post("/{game_id}/inject-visited")
async def inject_visited(game_id: str, body: InjectNodeBody):
    session = _get_or_404(game_id)
    push_history(session)
    k = session.state.cop_knowledge
    _mutate_knowledge(session, visited=k.visited | {body.node})
    return state_view(session)


@admin_router.post("/{game_id}/remove-visited")
async def remove_visited(game_id: str, body: InjectNodeBody):
    session = _get_or_404(game_id)
    push_history(session)
    k = session.state.cop_knowledge
    _mutate_knowledge(session, visited=k.visited - {body.node})
    return state_view(session)



@admin_router.post("/{game_id}/clear-knowledge")
async def clear_knowledge(game_id: str):
    session = _get_or_404(game_id)
    push_history(session)
    session.state = replace(
        session.state,
        cop_knowledge=CopKnowledge(jack_start=session.state.cop_knowledge.jack_start),
    )
    return state_view(session)


@admin_router.post("/{game_id}/set-knowledge")
async def set_knowledge(game_id: str, body: SetKnowledgeBody):
    session = _get_or_404(game_id)
    push_history(session)
    session.state = replace(
        session.state,
        cop_knowledge=CopKnowledge(
            jack_start=body.jack_start,
            visited=frozenset(body.visited),

            search_misses=tuple(tuple(m) for m in body.search_misses),
            arrest_misses=tuple(tuple(m) for m in body.arrest_misses),
        ),
    )
    return state_view(session)


@admin_router.post("/{game_id}/set-trace")
async def set_trace(game_id: str, body: SetTraceBody):
    session = _get_or_404(game_id)
    push_history(session)
    session.state = replace(session.state, jack_trace=frozenset(body.nodes))
    return state_view(session)


@admin_router.post("/{game_id}/undo")
async def undo(game_id: str):
    session = _get_or_404(game_id)
    if not session.history:
        raise HTTPException(status_code=400, detail="Nothing to undo")
    session.state = session.history.pop()
    session.terminated = False
    session.winner = None
    return state_view(session)


@admin_router.post("/{game_id}/new-from-state")
async def new_from_state(game_id: str, body: NewFromStateBody):
    session = _get_or_404(game_id)
    state = session.state
    gm = session.game_map
    jack_start = state.jack_pos

    if body.same_hideout:
        hideout = state.hideout
    else:
        distances: dict[int, int] = {jack_start: 0}
        queue = [jack_start]
        while queue:
            node_id = queue.pop(0)
            for edge in gm.jack_nodes[node_id - 1].edges:
                nb_id = edge.destination.id
                if nb_id not in distances:
                    distances[nb_id] = distances[node_id] + 1
                    queue.append(nb_id)
        candidates = [jid for jid, d in distances.items() if d >= gm.hideout_min_distance]
        hideout = session.rng.choice(candidates or list(distances.keys()))

    new_state = GameState(
        jack_pos=jack_start,
        cop_positions=state.cop_positions,
        hideout=hideout,
        turn=0,
        jack_trace=frozenset({jack_start}),
        cop_knowledge=CopKnowledge(jack_start=jack_start),
    )
    new_sess = GameSession(
        game_id=str(uuid.uuid4())[:8],
        game_map=gm,
        state=new_state,
        terminated=False,
        winner=None,
        rng=session.rng,
        blocking=session.blocking,
        turn_limit=session.turn_limit,
    )
    register_session(new_sess)
    return state_view(new_sess)


@admin_router.post("/{game_id}/node-info")
async def node_info(game_id: str, body: NodeInfoBody):
    session = _get_or_404(game_id)
    gm = session.game_map
    if body.cop_node < 1 or body.cop_node > len(gm.cop_nodes):
        raise HTTPException(status_code=400, detail="Invalid cop node")
    cop_node = gm.cop_nodes[body.cop_node - 1]
    # BFS 2 steps
    reachable: set[int] = {body.cop_node}
    frontier = {body.cop_node}
    for _ in range(2):
        nxt: set[int] = set()
        for cid in frontier:
            for nb in gm.cop_nodes[cid - 1].edges:
                if nb.id not in reachable:
                    reachable.add(nb.id)
                    nxt.add(nb.id)
        frontier = nxt
    return {
        "cop_node": body.cop_node,
        "reachable": sorted(reachable),
        "jack_neighbours": [n.id for n in cop_node.jack_neighbours],
    }
