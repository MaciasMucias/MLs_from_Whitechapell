from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class CopNode:
    id: int
    x: float
    y: float
    edges: list["CopNode"] = field(default_factory=list, repr=False)
    jack_neighbours: list["JackNode"] = field(default_factory=list, repr=False)


@dataclass
class JackEdge:
    destination: "JackNode"
    via: tuple[CopNode, ...]  # ordered sequence of cop nodes traversed (1 or more)


@dataclass
class JackNode:
    id: int
    x: float
    y: float
    node_type: str  # 'jack' or 'jack_start'
    edges: list[JackEdge] = field(default_factory=list, repr=False)


@dataclass
class Map:
    jack_nodes: list[JackNode]  # indexed by id
    cop_nodes: list[CopNode]    # indexed by id
    jack_starts: list[int]      # possible Jack starting node IDs
    cop_starts: list[int]       # pool of cop spawn node IDs
    num_cops: int
    turn_limit: int
    hideout_min_distance: int
    zone_radius: int


def load_map(path: str | Path) -> Map:
    with open(path) as f:
        data = json.load(f)

    cfg = data["config"]
    # Convert 1-based JSON IDs to 0-based internal IDs at load time.
    # All IDs throughout the codebase are 0-based after this point.
    jack_starts: list[int] = [x - 1 for x in data["jack_starts"]]
    cop_starts: list[int] = [x - 1 for x in data["cop_starts"]]
    jack_starts_set = set(jack_starts)

    # Pass 1: create stub nodes (no edges yet)
    cop_by_id: dict[int, CopNode] = {}
    for cn in data["cop_nodes"]:
        node = CopNode(id=cn["id"] - 1, x=cn["x"], y=cn["y"])
        cop_by_id[node.id] = node

    jack_by_id: dict[int, JackNode] = {}
    for jn in data["jack_nodes"]:
        node_type = "jack_start" if jn["id"] - 1 in jack_starts_set else "jack"
        node = JackNode(id=jn["id"] - 1, x=jn["x"], y=jn["y"], node_type=node_type)
        jack_by_id[node.id] = node

    # Pass 2: wire edges (JSON neighbour IDs are still 1-based — subtract 1 on lookup)
    for cn in data["cop_nodes"]:
        cop = cop_by_id[cn["id"] - 1]
        cop.edges = [cop_by_id[nb_id - 1] for nb_id in cn["edges"]]
        cop.jack_neighbours = [jack_by_id[jid - 1] for jid in cn["jack_neighbours"]]

    for jn in data["jack_nodes"]:
        jack = jack_by_id[jn["id"] - 1]
        jack.edges = [
            JackEdge(
                destination=jack_by_id[e["destination"] - 1],
                via=tuple(cop_by_id[cid - 1] for cid in e["via"]),
            )
            for e in jn["edges"]
        ]

    return Map(
        jack_nodes=[jack_by_id[i] for i in sorted(jack_by_id)],
        cop_nodes=[cop_by_id[i] for i in sorted(cop_by_id)],
        jack_starts=jack_starts,
        cop_starts=cop_starts,
        num_cops=cfg["num_cops"],
        turn_limit=cfg["turn_limit"],
        hideout_min_distance=cfg["hideout_min_distance"],
        zone_radius=cfg["zone_radius"],
    )
