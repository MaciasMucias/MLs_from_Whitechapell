import numpy as np

from engine.graph import Map
from engine.graph_utils import jack_bfs_distances
from engine.state import GameState


def precompute_distances(game_map: Map) -> tuple[dict[int, dict[int, int]], int]:
    """
    All-pairs BFS distances on Jack graph. Call once per map load (not per episode —
    distances are topology-only, independent of game state).

    Returns (all_dists, diameter) where:
      all_dists[src][dst] = shortest path length in hops
      diameter            = max finite shortest path; used to normalise scalars to [0, 1]
    """
    all_dists = {
        node.id: jack_bfs_distances(node.id, game_map) for node in game_map.jack_nodes
    }
    diameter = max(max(d.values()) for d in all_dists.values())
    return all_dists, diameter


def build_obs(
    state: GameState,
    game_map: Map,
    all_dists: dict[int, dict[int, int]],
    diameter: int,
) -> np.ndarray:
    """
    Build the 1,416-dim observation vector for Jack's RL policy.

    Layout (in order):
      [0:195]     Jack position (one-hot, Jack nodes)
      [195:390]   Hideout (one-hot, Jack nodes)
      [390:585]   Hideout zone (binary, Jack nodes)
      [585:780]   Jack's visited nodes (binary, Jack nodes)
      [780:975]   Cop-searched hits (binary, Jack nodes) — state.cop_searched_hits
      [975:1170]  Cop-searched misses (binary, Jack nodes) — state.cop_searched_misses
      [1170:1404] Cop presence on cop nodes (binary, Cop nodes)
      [1404]      Turn normalised (scalar)
      [1405]      Jack → hideout distance normalised (scalar)
      [1406:1411] Cops → Jack distances sorted nearest-first (5 scalars)
      [1411:1416] Cops → hideout distances, same sort order (5 scalars)

    All distance scalars are in [0, 1] via division by graph diameter.
    """
    n_jack = len(game_map.jack_nodes)  # 195
    n_cop = len(game_map.cop_nodes)  # 234

    # --- Jack-node binary / one-hot blocks ---
    jack_pos_oh = np.zeros(n_jack, dtype=np.float32)
    jack_pos_oh[state.jack_pos] = 1.0

    hideout_oh = np.zeros(n_jack, dtype=np.float32)
    hideout_oh[state.hideout] = 1.0

    zone_bin = np.zeros(n_jack, dtype=np.float32)
    zone_bin[list(state.hideout_zone)] = 1.0

    visited_bin = np.zeros(n_jack, dtype=np.float32)
    if state.jack_trace:
        visited_bin[list(state.jack_trace)] = 1.0

    hit_bin = np.zeros(n_jack, dtype=np.float32)
    if state.cop_searched_hits:
        hit_bin[list(state.cop_searched_hits)] = 1.0

    miss_bin = np.zeros(n_jack, dtype=np.float32)
    if state.cop_searched_misses:
        miss_bin[list(state.cop_searched_misses)] = 1.0

    # --- Cop-node binary block ---
    cop_pres = np.zeros(n_cop, dtype=np.float32)
    cop_pres[list(state.cop_positions)] = 1.0

    # --- Scalar features ---
    turn_norm = state.turn / game_map.turn_limit
    j2h = all_dists[state.jack_pos].get(state.hideout, diameter) / diameter

    def _cop_dist(cop_pos: int, target_jack: int) -> float:
        nbs = game_map.cop_nodes[cop_pos].jack_neighbours
        if not nbs:
            return 1.0
        d = min(all_dists.get(nb.id, {}).get(target_jack, diameter) for nb in nbs)
        return min(d / diameter, 1.0)

    cop_to_jack = [_cop_dist(cp, state.jack_pos) for cp in state.cop_positions]
    cop_to_hideout = [_cop_dist(cp, state.hideout) for cp in state.cop_positions]

    # sort both by cop-to-Jack distance so slot 0 = nearest cop
    paired = sorted(zip(cop_to_jack, cop_to_hideout))
    cop_j_arr = np.array([p[0] for p in paired], dtype=np.float32)
    cop_h_arr = np.array([p[1] for p in paired], dtype=np.float32)

    return np.concatenate(
        [
            jack_pos_oh,
            hideout_oh,
            zone_bin,
            visited_bin,
            hit_bin,
            miss_bin,
            cop_pres,
            [turn_norm, j2h],
            cop_j_arr,
            cop_h_arr,
        ]
    )
