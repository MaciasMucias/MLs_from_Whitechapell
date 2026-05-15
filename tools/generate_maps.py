"""
One-time map generation script.

Derives the full graph structure purely from Mapa_v5.svg:
  - Jack-cop direct adjacency  (BFS through paths, stop at first cop)
  - Cop-cop direct adjacency   (same, starting from each cop node)
  - Jack-jack adjacency        (BFS through cop graph from each jack node)
  - Traversal routes           (DFS through cop graph for each jack-jack edge)

Run from the project root:
    uv run python tools/generate_maps.py
"""

import json
import sys
from pathlib import Path
from typing import Any
from xml.dom import minidom

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.graph import CopNode, JackEdge, JackNode

# ---------------------------------------------------------------------------
# Whitechapel full-map configuration
# ---------------------------------------------------------------------------
HIDEOUT_MIN_DISTANCE = 4
NUM_COPS = 5
TURN_LIMIT = 15

SVG_PATH = ROOT / "Mapa_v5.svg"
OUTPUT_PATH = ROOT / "maps" / "whitechapel.json"

# Set to a jack node ID to only compute routes for that node (for debugging).
# Set to None to compute routes for all nodes.
DEBUG_ROUTES_FOR_NODE = None


# ---------------------------------------------------------------------------
# SVG path coordinate parser
# ---------------------------------------------------------------------------


def _parse_path_coordinates(path_elem):
    d = path_elem.getAttribute("d")
    args = d.split(" ")
    coordinates = None
    mode = ""
    n = 1
    for arg in args:
        if len(arg) == 1:
            mode = arg
            continue
        if "," not in arg:
            val = float(arg)
        else:
            val = list(map(float, arg.split(",")))
        if coordinates is None:
            coordinates = [val if isinstance(val, list) else [val, 0.0]]
            continue
        prev = coordinates[n - 1]
        if mode in ("M", "L"):
            coordinates.append(val)
        elif mode in ("m", "l"):
            coordinates.append([round(prev[0] + val[0], 8), round(prev[1] + val[1], 8)])
        elif mode == "H":
            coordinates.append([val, prev[1]])
        elif mode == "h":
            coordinates.append([round(prev[0] + val, 8), prev[1]])
        elif mode == "V":
            coordinates.append([prev[0], val])
        elif mode == "v":
            coordinates.append([prev[0], round(prev[1] + val, 8)])
        else:
            raise RuntimeError(f"Unexpected SVG path mode: {mode!r}")
        n += 1
    return [[round(c, 2) for c in pt] for pt in coordinates]


def _dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def _is_connected(coordinates_a, coordinates_b, threshold=5.0):
    return any(_dist(a, b) < threshold for a in coordinates_a for b in coordinates_b)


# ---------------------------------------------------------------------------
# SVG parsing — builds all adjacency structures
# ---------------------------------------------------------------------------


def _parse_svg(svg_path):
    """
    Returns:
      jack_coordinates  : {jack_id: (x, y)}
      jack_types   : {jack_id: 'jack' | 'jack_start'}
      cop_coordinates   : {cop_id: (x, y)}
      jack_cop_adj : {jack_id: set of cop_ids}  — direct adjacency
      cop_cop_adj  : {cop_id: set of cop_ids}   — direct adjacency
      cop_jack_adj : {cop_id: set of jack_ids}  — inverse of jack_cop_adj
    """
    doc = minidom.parse(str(svg_path))

    # --- Jack nodes ---
    scale, dx, dy = 0.26458333, 9.26376, -28.409268
    jack_coordinates, jack_types, jack_point = {}, {}, {}
    for g in doc.getElementsByTagName("g"):
        if "layer" in g.getAttribute("id"):
            continue
        ellipses = g.getElementsByTagName("ellipse")
        tspans = g.getElementsByTagName("tspan")
        if not ellipses or not tspans:
            continue
        e = ellipses[0]
        node_id = int(tspans[0].firstChild.nodeValue)
        x = round(float(e.getAttribute("cx")) * scale + dx, 2)
        y = round(float(e.getAttribute("cy")) * scale + dy, 2)
        jack_coordinates[node_id] = (x, y)
        jack_point[node_id] = [[x, y]]
        jack_types[node_id] = (
            "jack_start" if "fill:#ff0000" in e.getAttribute("style") else "jack"
        )

    # --- Cop nodes (<rect> elements, 1-indexed by order) ---
    cop_coordinates, cop_point, cop_types = {}, {}, {}
    for idx, rect in enumerate(doc.getElementsByTagName("rect")):
        cid = idx + 1
        cx = round(
            float(rect.getAttribute("x")) + float(rect.getAttribute("width")) / 2, 2
        )
        cy = round(
            float(rect.getAttribute("y")) + float(rect.getAttribute("height")) / 2, 2
        )
        cop_coordinates[cid] = (cx, cy)
        cop_point[cid] = [[cx, cy]]
        cop_types[cid] = (
            "cops_spawn" if "stroke:#ffff00" in rect.getAttribute("style") else "cops"
        )

    # --- Path segments ---
    path_coordinates = []
    for p in doc.getElementsByTagName("path"):
        try:
            path_coordinates.append(_parse_path_coordinates(p))
        except RuntimeError:
            pass

    n_paths = len(path_coordinates)
    n_jack = len(jack_coordinates)
    n_cop = len(cop_coordinates)
    print(
        f"  {n_jack} jack nodes, {n_cop} cop nodes, {n_paths} path segments", flush=True
    )

    # --- Pre-compute path-path adjacency (once) ---
    print("  Pre-computing path graph...", flush=True)
    path_adj = [[] for _ in range(n_paths)]
    for i in range(n_paths):
        for j in range(i + 1, n_paths):
            if _is_connected(path_coordinates[i], path_coordinates[j]):
                path_adj[i].append(j)
                path_adj[j].append(i)

    # --- Pre-compute which nodes each path touches ---
    print("  Pre-computing path-node touches...", flush=True)
    path_cop = [set() for _ in range(n_paths)]  # path_index -> set of cop IDs
    path_jack = [set() for _ in range(n_paths)]  # path_index -> set of jack IDs

    for i, pc in enumerate(path_coordinates):
        for cid, cc in cop_point.items():
            if _is_connected(pc, cc):
                path_cop[i].add(cid)
        for jid, jc in jack_point.items():
            if _is_connected(pc, jc):
                path_jack[i].add(jid)

    # Build reverse lookups: node -> touching path indices
    jack_paths = {jid: [] for jid in jack_coordinates}
    cop_paths = {cid: [] for cid in cop_coordinates}
    for i in range(n_paths):
        for jid in path_jack[i]:
            jack_paths[jid].append(i)
        for cid in path_cop[i]:
            cop_paths[cid].append(i)

    # --- BFS helper: find directly adjacent cops from a starting node ---
    def find_adjacent_cops(start_path_indices, exclude_jack_id=None):
        """
        BFS from the given starting path indices.
        Stop each branch at the first cop node found.
        Do not cross any jack node (other than exclude_jack_id).
        Returns set of cop IDs.
        """
        visited = set(start_path_indices)
        queue = list(start_path_indices)
        found = set()
        while queue:
            pi = queue.pop()
            if path_cop[pi]:
                found.update(path_cop[pi])
                continue  # stop — don't expand past cop
            blocked = path_jack[pi] - ({exclude_jack_id} if exclude_jack_id else set())
            if blocked:
                continue  # stop — don't cross other jack nodes
            for npi in path_adj[pi]:
                if npi not in visited:
                    visited.add(npi)
                    queue.append(npi)
        return found

    # --- Jack-cop direct adjacency ---
    print("  Computing jack-cop adjacency...", flush=True)
    jack_cop_adj = {}
    cop_jack_adj = {cid: set() for cid in cop_coordinates}
    for jid in jack_coordinates:
        cops = find_adjacent_cops(jack_paths[jid], exclude_jack_id=jid)
        jack_cop_adj[jid] = cops
        for cid in cops:
            cop_jack_adj[cid].add(jid)

    # --- Cop-cop direct adjacency ---
    print("  Computing cop-cop adjacency...", flush=True)
    cop_cop_adj = {cid: set() for cid in cop_coordinates}
    for cid in cop_coordinates:
        # BFS from this cop's paths; looking for adjacent cops.
        # Stop at first cop (other than self), don't cross jack nodes.
        seen = set(cop_paths[cid])
        frontier = list(cop_paths[cid])
        adj_cops: set[int] = set()
        while frontier:
            path_idx = frontier.pop()
            touching_other_cops = path_cop[path_idx] - {cid}
            if touching_other_cops:
                adj_cops.update(touching_other_cops)
                continue  # stop at cop
            if path_jack[path_idx]:
                continue  # stop at jack
            for next_path in path_adj[path_idx]:
                if next_path not in seen:
                    seen.add(next_path)
                    frontier.append(next_path)
        for other in adj_cops:
            cop_cop_adj[cid].add(other)
            cop_cop_adj[other].add(cid)

    return (
        jack_coordinates,
        jack_types,
        cop_coordinates,
        cop_types,
        jack_cop_adj,
        cop_cop_adj,
        cop_jack_adj,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_map():
    print("Parsing SVG...", flush=True)
    (
        jack_coordinates,
        jack_types,
        cop_coordinates,
        cop_types,
        jack_cop_adj,
        cop_cop_adj,
        cop_jack_adj,
    ) = _parse_svg(SVG_PATH)

    # --- Build node objects ---
    cop_by_id = {
        cid: CopNode(id=cid, x=x, y=y) for cid, (x, y) in cop_coordinates.items()
    }
    jack_by_id = {
        jid: JackNode(id=jid, x=x, y=y, node_type=jack_types[jid])
        for jid, (x, y) in jack_coordinates.items()
    }

    # Cop-cop movement edges
    cop_edges_seen = set()
    for cid, neighbours in cop_cop_adj.items():
        for nid in neighbours:
            key = (min(cid, nid), max(cid, nid))
            if key not in cop_edges_seen:
                cop_edges_seen.add(key)
                cop_by_id[cid].edges.append(cop_by_id[nid])
                cop_by_id[nid].edges.append(cop_by_id[cid])

    # Cop jack_neighbours
    for cid, jids in cop_jack_adj.items():
        for jid in jids:
            cop_by_id[cid].jack_neighbours.append(jack_by_id[jid])

    # --- Derive jack-jack adjacency by BFS through cop graph ---
    print("Deriving jack-jack adjacency...", flush=True)
    jack_jack_adj: dict[int, set[int]] = {jid: set() for jid in jack_coordinates}
    for jid in jack_coordinates:
        visited_cops = set()
        queue = list(jack_cop_adj[jid])
        visited_cops.update(queue)
        while queue:
            cid = queue.pop()
            for neighbour_jid in cop_jack_adj[cid]:
                if neighbour_jid != jid:
                    jack_jack_adj[jid].add(neighbour_jid)
            for next_cid in cop_cop_adj[cid]:
                if next_cid not in visited_cops:
                    visited_cops.add(next_cid)
                    queue.append(next_cid)

    # --- Find traversal routes for each jack-jack edge ---
    def remove_dominated_routes(routes: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
        """
        Remove routes whose cop-node set is a strict superset of another route's set.
        Such routes can never provide an alternative when the smaller-set route is blocked,
        because they share all the same cops plus more.
        Also deduplicates routes with identical cop-node sets (order irrelevant for blocking).
        """
        unique: dict[frozenset, tuple[int, ...]] = {}
        for r in routes:
            rs = frozenset(r)
            if rs not in unique:
                unique[rs] = r
        route_sets = list(unique.keys())
        kept_sets = [
            rs
            for rs in route_sets
            if not any(other < rs for other in route_sets if other != rs)
        ]
        return [unique[rs] for rs in kept_sets]

    def find_routes(jack_a: int, jack_b: int) -> list[tuple[int, ...]]:
        b_cops = jack_cop_adj[jack_b]
        result: list[tuple[int, ...]] = []
        for start_cop in jack_cop_adj[jack_a]:
            stack: list[tuple[Any, tuple[Any, ...], set]] = [
                (start_cop, (start_cop,), {start_cop})
            ]
            while stack:
                current, route, route_visited = stack.pop()
                if current in b_cops:
                    result.append(route)
                for next_cop in cop_cop_adj[current]:
                    if next_cop not in route_visited:
                        stack.append(
                            (next_cop, route + (next_cop,), route_visited | {next_cop})
                        )
        return result

    print("Finding traversal routes...", flush=True)
    jack_edges_seen = set()
    missing = []

    nodes_to_process = (
        {1} if DEBUG_ROUTES_FOR_NODE is not None else set(jack_coordinates)
    )

    for node_id in nodes_to_process:
        neighbours = jack_jack_adj[node_id]
        for nb_id in neighbours:
            key = (min(node_id, nb_id), max(node_id, nb_id))
            if key in jack_edges_seen:
                continue
            jack_edges_seen.add(key)

            routes = remove_dominated_routes(find_routes(node_id, nb_id))
            if not routes:
                missing.append((node_id, nb_id))
                print(f"  WARNING: no route for ({node_id}, {nb_id})", flush=True)
                continue

            for via_ids in routes:
                via = tuple(cop_by_id[c] for c in via_ids)
                jack_by_id[node_id].edges.append(
                    JackEdge(destination=jack_by_id[nb_id], via=via)
                )
                jack_by_id[nb_id].edges.append(
                    JackEdge(destination=jack_by_id[node_id], via=tuple(reversed(via)))
                )

    if missing:
        print(f"  {len(missing)} edges had no route and were skipped", flush=True)

    if DEBUG_ROUTES_FOR_NODE is not None:
        print(
            f"\nDebug mode: only processed routes for jack node {DEBUG_ROUTES_FOR_NODE}. Not saving.",
            flush=True,
        )
        return

    # --- Derive start node lists from SVG types ---
    jack_starts = sorted(jid for jid, t in jack_types.items() if t == "jack_start")
    cop_starts = sorted(cid for cid, t in cop_types.items() if t == "cops_spawn")

    # --- Serialise to JSON ---
    jack_nodes = [jack_by_id[jid] for jid in sorted(jack_by_id)]
    cop_nodes = [cop_by_id[cid] for cid in sorted(cop_by_id)]

    data = {
        "jack_starts": jack_starts,
        "cop_starts": cop_starts,
        "config": {
            "hideout_min_distance": HIDEOUT_MIN_DISTANCE,
            "num_cops": NUM_COPS,
            "turn_limit": TURN_LIMIT,
        },
        "jack_nodes": [
            {
                "id": jn.id,
                "x": jn.x,
                "y": jn.y,
                "edges": [
                    {"destination": e.destination.id, "via": [c.id for c in e.via]}
                    for e in jn.edges
                ],
            }
            for jn in jack_nodes
        ],
        "cop_nodes": [
            {
                "id": cn.id,
                "x": cn.x,
                "y": cn.y,
                "edges": [nb.id for nb in cn.edges],
                "jack_neighbours": [jn.id for jn in cn.jack_neighbours],
            }
            for cn in cop_nodes
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"  Jack nodes : {len(jack_nodes)}")
    print(f"  Cop nodes  : {len(cop_nodes)}")
    total_edges = sum(len(jn.edges) for jn in jack_nodes) // 2
    print(f"  Jack edges : {total_edges}")


if __name__ == "__main__":
    build_map()
