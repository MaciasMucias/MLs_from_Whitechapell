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
        for edge in game_map.jack_nodes[node_id - 1].edges:
            nb_id = edge.destination.id
            if nb_id not in distances:
                distances[nb_id] = distances[node_id] + 1
                queue.append(nb_id)
    return distances


def reachable_cop_nodes(cop_id: int, game_map: Map, max_steps: int = 2) -> set[int]:
    """BFS: all cop node IDs reachable from cop_id within max_steps moves."""
    reachable = {cop_id}
    frontier = {cop_id}
    for _ in range(max_steps):
        next_frontier: set[int] = set()
        for cid in frontier:
            for nb in game_map.cop_nodes[cid - 1].edges:
                if nb.id not in reachable:
                    reachable.add(nb.id)
                    next_frontier.add(nb.id)
        frontier = next_frontier
    return reachable
