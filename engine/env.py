from dataclasses import dataclass, replace
import random

from engine.graph import JackEdge, Map
from engine.graph_utils import jack_bfs_distances
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
    arrest_all: bool = False


# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------

def make_initial_state(
    game_map: Map,
    rng: random.Random | None = None,
) -> GameState:
    """
    Create a starting GameState for a new game.

    Jack's starting node is chosen randomly from game_map.jack_starts.
    Cop starting nodes are sampled (without replacement) from game_map.cop_starts.
    Hideout is chosen randomly from jack nodes at least
    game_map.hideout_min_distance hops from Jack's starting node.

    Args:
        game_map: Loaded Map (graph + config).
        rng: Optional seeded Random instance for reproducibility.
    """
    if rng is None:
        rng = random.Random()

    start = rng.choice(game_map.jack_starts)

    distances = jack_bfs_distances(start, game_map)
    base_candidates = [jid for jid, d in distances.items() if d >= game_map.hideout_min_distance]

    anchor = rng.choice(base_candidates)
    anchor_distances = jack_bfs_distances(anchor, game_map)
    zone = frozenset(v for v, d in anchor_distances.items() if d <= game_map.zone_radius)
    zone_candidates = [jid for jid in base_candidates if jid in zone]
    hideout = rng.choice(zone_candidates if zone_candidates else base_candidates)

    cop_positions = tuple(rng.sample(game_map.cop_starts, game_map.num_cops))

    return GameState(
        jack_pos=start,
        cop_positions=cop_positions,
        hideout=hideout,
        hideout_zone_anchor=anchor,
        hideout_zone=zone,
        turn=0,
        jack_trace=frozenset({start}),
        cop_knowledge=CopKnowledge(jack_start=start, visited_at=((start, 0),)),
    )


# ---------------------------------------------------------------------------
# Legal move queries
# ---------------------------------------------------------------------------

def legal_jack_edges(
    state: GameState,
    game_map: Map,
    blocking: bool = False,
) -> list[JackEdge]:
    """
    Return legal JackEdge moves from Jack's current position.

    When blocking is enabled, edges where any traversal cop node is currently
    occupied are excluded.
    """
    jack_node = game_map.jack_nodes[state.jack_pos - 1]
    if not blocking:
        return list(jack_node.edges)
    occupied = set(state.cop_positions)
    return [e for e in jack_node.edges if not any(c.id in occupied for c in e.via)]



# ---------------------------------------------------------------------------
# Sequential turn steps
# ---------------------------------------------------------------------------

def step_jack(
    state: GameState,
    jack_edge: JackEdge,
) -> tuple[GameState, bool, str | None]:
    """
    Apply Jack's move for this round.

    Terminates immediately if Jack reaches the hideout.

    Returns:
        (new_state, terminated, winner)
    """
    new_trace = state.jack_trace | {jack_edge.destination.id}
    new_state = replace(state, jack_pos=jack_edge.destination.id, jack_trace=new_trace)
    if new_state.jack_pos == state.hideout:
        return new_state, True, "jack"
    return new_state, False, None


def step_cop(
    state: GameState,
    cop_turn: CopTurn,
    game_map: Map,
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
    cop_node = game_map.cop_nodes[cop_turn.destination - 1]

    visited = set(state.cop_knowledge.visited)
    search_misses = list(state.cop_knowledge.search_misses)
    arrest_misses = list(state.cop_knowledge.arrest_misses)
    visited_at_dict: dict[int, int] = dict(state.cop_knowledge.visited_at)

    if cop_turn.search:
        for jack_nb in cop_node.jack_neighbours:
            jid = jack_nb.id
            if jid in state.jack_trace:
                visited.add(jid)
                visited_at_dict.setdefault(jid, state.turn + 1)
            else:
                search_misses.append((jid, state.turn + 1))
    else:
        if cop_turn.arrest_all:
            cop_node_neighbours = {jn.id for jn in cop_node.jack_neighbours}
        else:
            cop_node_neighbours = {cop_turn.arrest_target}

        if state.jack_pos in cop_node_neighbours:
            new_knowledge = replace(
                state.cop_knowledge,
                visited=frozenset(visited),
                search_misses=tuple(search_misses),
                arrest_misses=tuple(arrest_misses),
                visited_at=tuple(visited_at_dict.items()),
            )
            return replace(state, cop_positions=tuple(cop_positions), cop_knowledge=new_knowledge), True, "cops"
        for jid in cop_node_neighbours:
            arrest_misses.append((jid, state.turn + 1))

    new_knowledge = replace(
        state.cop_knowledge,
        visited=frozenset(visited),
        search_misses=tuple(search_misses),
        arrest_misses=tuple(arrest_misses),
        visited_at=tuple(visited_at_dict.items()),
    )
    return replace(state, cop_positions=tuple(cop_positions), cop_knowledge=new_knowledge), False, None


def end_of_round(
    state: GameState,
    game_map: Map,
    blocking: bool = False,
    turn_limit: int | None = None,
) -> tuple[GameState, bool, str | None]:
    """
    Advance the turn counter and check end-of-round terminal conditions.

    Call this after all cops have acted. Checks turn limit and, if blocking
    is enabled, whether Jack has any legal moves next round.

    Args:
        state:
        game_map:
        blocking:
        turn_limit: Optional override for game_map.turn_limit.

    Returns:
        (new_state, terminated, winner)
    """
    new_state = replace(state, turn=state.turn + 1)
    effective_limit = turn_limit if turn_limit is not None else game_map.turn_limit
    if new_state.turn >= effective_limit:
        return new_state, True, "cops"
    if blocking and not legal_jack_edges(new_state, game_map, blocking=True):
        return new_state, True, "cops"
    return new_state, False, None


