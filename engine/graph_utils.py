"""
Graph traversal utilities for the meshed Jack/Cop board.

Functions here operate on the Map structure but have no dependency on game
state — they are pure graph queries. Add new BFS/DFS primitives here as
the heuristic cop PMF and other algorithms require them.
"""

from engine.graph import Map


def jack_bfs_distances(start_id: int, game_map: Map) -> dict[int, int]:
    """BFS distances (in hops) from start_id to all reachable Jack nodes."""
    distances: dict[int, int] = {start_id: 0}
    queue = [start_id]
    while queue:
        node_id = queue.pop(0)
        for edge in game_map.jack_nodes[node_id].edges:
            nb_id = edge.destination.id
            if nb_id not in distances:
                distances[nb_id] = distances[node_id] + 1
                queue.append(nb_id)
    return distances


def jack_reachable_within(start_id: int, max_hops: int, game_map: Map) -> set[int]:
    """Set of Jack node IDs reachable from start_id in at most max_hops moves."""
    reachable = {start_id}
    frontier = {start_id}
    for _ in range(max_hops):
        next_frontier: set[int] = set()
        for nid in frontier:
            for edge in game_map.jack_nodes[nid].edges:
                nb = edge.destination.id
                if nb not in reachable:
                    reachable.add(nb)
                    next_frontier.add(nb)
        frontier = next_frontier
    return reachable


def reachable_cop_nodes(cop_id: int, game_map: Map, max_steps: int = 2) -> set[int]:
    """BFS: all cop node IDs reachable from cop_id within max_steps moves.

    One move = a direct cop-to-cop edge OR passing through a shared jack
    neighbour (two cop nodes are one step apart if they both border the same
    jack circle). The latter is the dominant movement type on this map — many
    cop nodes have no direct cop edges but are connected through jack circles.
    """
    # Precompute jack_id -> list of cop_ids that share it (for jack-mediated hops)
    jack_to_cops: dict[int, list[int]] = {}
    for cn in game_map.cop_nodes:
        for jn in cn.jack_neighbours:
            jack_to_cops.setdefault(jn.id, []).append(cn.id)

    reachable = {cop_id}
    frontier = {cop_id}
    for _ in range(max_steps):
        next_frontier: set[int] = set()
        for cid in frontier:
            cn = game_map.cop_nodes[cid]
            for nb in cn.edges:
                if nb.id not in reachable:
                    reachable.add(nb.id)
                    next_frontier.add(nb.id)
            for jn in cn.jack_neighbours:
                for nb_id in jack_to_cops.get(jn.id, []):
                    if nb_id not in reachable:
                        reachable.add(nb_id)
                        next_frontier.add(nb_id)
        frontier = next_frontier
    return reachable
