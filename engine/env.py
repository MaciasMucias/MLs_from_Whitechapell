from dataclasses import dataclass
import random

from engine.graph import JackEdge, Map
from engine.state import CopKnowledge, GameState


@dataclass(frozen=True)
class CopTurn:
    """
    A single cop's action for one round.

    Attributes:
        cop_idx:        Index into GameState.cop_positions.
        destination:    Cop node ID to move to (may equal current position).
        search:         True = search all adjacent jack nodes.
                        False = arrest attempt on arrest_target.
        arrest_target:  Jack node ID to arrest. Required when search=False.
    """
    cop_idx: int
    destination: int
    search: bool
    arrest_target: int | None = None


# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------

def make_initial_state(
    map: Map,
    rng: random.Random | None = None,
) -> GameState:
    """
    Create a starting GameState for a new game.

    Jack's starting node is chosen randomly from map.jack_starts.
    Cop starting nodes are sampled (without replacement) from map.cop_starts.
    Hideout is chosen randomly from jack nodes at least
    map.hideout_min_distance hops from Jack's starting node.

    Args:
        map: Loaded Map (graph + config).
        rng: Optional seeded Random instance for reproducibility.
    """
    if rng is None:
        rng = random.Random()

    start = rng.choice(map.jack_starts)

    # BFS from start to find distances in the jack graph
    distances: dict[int, int] = {start: 0}
    queue = [start]
    while queue:
        node_id = queue.pop(0)
        seen: set[int] = set()
        for edge in map.jack_nodes[node_id - 1].edges:
            nb_id = edge.destination.id
            if nb_id not in distances and nb_id not in seen:
                seen.add(nb_id)
                distances[nb_id] = distances[node_id] + 1
                queue.append(nb_id)

    candidates = [jid for jid, d in distances.items() if d >= map.hideout_min_distance]
    hideout = rng.choice(candidates)

    cop_positions = tuple(rng.sample(map.cop_starts, map.num_cops))

    return GameState(
        jack_pos=start,
        cop_positions=cop_positions,
        hideout=hideout,
        turn=0,
        jack_trace=frozenset({start}),
        cop_knowledge=CopKnowledge(jack_start=start),
    )


# ---------------------------------------------------------------------------
# Legal move queries
# ---------------------------------------------------------------------------

def legal_jack_edges(
    state: GameState,
    map: Map,
    blocking: bool = False,
) -> list[JackEdge]:
    """
    Return legal JackEdge moves from Jack's current position.

    When blocking is enabled, edges where any traversal cop node is currently
    occupied are excluded.
    """
    jack_node = map.jack_nodes[state.jack_pos - 1]
    if not blocking:
        return list(jack_node.edges)
    occupied = set(state.cop_positions)
    return [e for e in jack_node.edges if not any(c.id in occupied for c in e.via)]


def reachable_cop_nodes(cop_id: int, map: Map, max_steps: int = 2) -> set[int]:
    """BFS: all cop node IDs reachable from cop_id within max_steps moves."""
    reachable = {cop_id}
    frontier = {cop_id}
    for _ in range(max_steps):
        next_frontier: set[int] = set()
        for cid in frontier:
            for nb in map.cop_nodes[cid - 1].edges:
                if nb.id not in reachable:
                    reachable.add(nb.id)
                    next_frontier.add(nb.id)
        frontier = next_frontier
    return reachable


# ---------------------------------------------------------------------------
# Sequential turn steps
# ---------------------------------------------------------------------------

def step_jack(
    state: GameState,
    jack_edge: JackEdge,
    map: Map,
    blocking: bool = False,
) -> tuple[GameState, bool, str | None]:
    """
    Apply Jack's move for this round.

    Terminates immediately if Jack reaches the hideout.

    Returns:
        (new_state, terminated, winner)
    """
    new_jack_pos = jack_edge.destination.id
    new_trace = state.jack_trace | {new_jack_pos}
    new_state = GameState(
        jack_pos=new_jack_pos,
        cop_positions=state.cop_positions,
        hideout=state.hideout,
        turn=state.turn,
        jack_trace=new_trace,
        cop_knowledge=state.cop_knowledge,
    )
    if new_jack_pos == state.hideout:
        return new_state, True, "jack"
    return new_state, False, None


def step_cop(
    state: GameState,
    cop_turn: CopTurn,
    map: Map,
) -> tuple[GameState, bool, str | None]:
    """
    Apply one cop's turn: move, then search or arrest.

    Terminates immediately if the arrest succeeds.
    The Director should post-process cop_knowledge.visited after all cops
    have acted and before returning state to cop agents.

    Returns:
        (new_state, terminated, winner)
    """
    cop_positions = list(state.cop_positions)
    cop_positions[cop_turn.cop_idx] = cop_turn.destination
    cop_node = map.cop_nodes[cop_turn.destination - 1]

    visited = set(state.cop_knowledge.visited)
    never_visited = set(state.cop_knowledge.never_visited)
    search_misses = list(state.cop_knowledge.search_misses)
    arrest_misses = list(state.cop_knowledge.arrest_misses)

    if cop_turn.search:
        for jack_nb in cop_node.jack_neighbours:
            jid = jack_nb.id
            if jid in state.jack_trace:
                visited.add(jid)
                never_visited.discard(jid)  # supersedes any prior miss
            else:
                never_visited.add(jid)
                search_misses.append((jid, state.turn))
    else:
        target = cop_turn.arrest_target
        if target == state.jack_pos:
            return (
                _build_state(
                    state.jack_pos, cop_positions, state, state.jack_trace,
                    visited, never_visited, search_misses, arrest_misses,
                ),
                True,
                "cops",
            )
        arrest_misses.append((target, state.turn))

    return (
        _build_state(
            state.jack_pos, cop_positions, state, state.jack_trace,
            visited, never_visited, search_misses, arrest_misses,
        ),
        False,
        None,
    )


def end_of_round(
    state: GameState,
    map: Map,
    blocking: bool = False,
) -> tuple[GameState, bool, str | None]:
    """
    Advance the turn counter and check end-of-round terminal conditions.

    Call this after all cops have acted. Checks turn limit and, if blocking
    is enabled, whether Jack has any legal moves next round.

    Returns:
        (new_state, terminated, winner)
    """
    new_state = GameState(
        jack_pos=state.jack_pos,
        cop_positions=state.cop_positions,
        hideout=state.hideout,
        turn=state.turn + 1,
        jack_trace=state.jack_trace,
        cop_knowledge=state.cop_knowledge,
    )
    if new_state.turn >= map.turn_limit:
        return new_state, True, "cops"
    if blocking and not legal_jack_edges(new_state, map, blocking=True):
        return new_state, True, "cops"
    return new_state, False, None


def _build_state(
    jack_pos: int,
    cop_positions: list[int],
    prev: GameState,
    jack_trace: frozenset[int],
    visited: set[int],
    never_visited: set[int],
    search_misses: list[tuple[int, int]],
    arrest_misses: list[tuple[int, int]],
    turn: int | None = None,
) -> GameState:
    return GameState(
        jack_pos=jack_pos,
        cop_positions=tuple(cop_positions),
        hideout=prev.hideout,
        turn=turn if turn is not None else prev.turn,
        jack_trace=jack_trace,
        cop_knowledge=CopKnowledge(
            jack_start=prev.cop_knowledge.jack_start,
            visited=frozenset(visited),
            never_visited=frozenset(never_visited),
            search_misses=tuple(search_misses),
            arrest_misses=tuple(arrest_misses),
        ),
    )
