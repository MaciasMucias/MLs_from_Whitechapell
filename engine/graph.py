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
    node_type: str  # 'jack' or 'jack_kill'
    edges: list[JackEdge] = field(default_factory=list, repr=False)


@dataclass
class Map:
    jack_nodes: list[JackNode]  # indexed by id-1
    cop_nodes: list[CopNode]    # indexed by id-1
    jack_starts: list[int]      # possible Jack starting node IDs
    cop_starts: list[int]       # pool of cop spawn node IDs
    num_cops: int
    turn_limit: int
    hideout_min_distance: int


def load_map(path: str | Path) -> Map:
    with open(path) as f:
        data = json.load(f)

    cfg = data["config"]

    # Pass 1: create stub nodes (no edges yet)
    cop_by_id: dict[int, CopNode] = {}
    for cn in data["cop_nodes"]:
        node = CopNode(id=cn["id"], x=cn["x"], y=cn["y"])
        cop_by_id[node.id] = node

    jack_by_id: dict[int, JackNode] = {}
    for jn in data["jack_nodes"]:
        node = JackNode(id=jn["id"], x=jn["x"], y=jn["y"], node_type=jn["type"])
        jack_by_id[node.id] = node

    # Pass 2: wire edges
    for cn in data["cop_nodes"]:
        cop = cop_by_id[cn["id"]]
        cop.edges = [cop_by_id[nb_id] for nb_id in cn["edges"]]
        cop.jack_neighbours = [jack_by_id[jid] for jid in cn["jack_neighbours"]]

    for jn in data["jack_nodes"]:
        jack = jack_by_id[jn["id"]]
        jack.edges = [
            JackEdge(
                destination=jack_by_id[e["destination"]],
                via=tuple(cop_by_id[cid] for cid in e["via"]),
            )
            for e in jn["edges"]
        ]

    return Map(
        jack_nodes=[jack_by_id[i] for i in sorted(jack_by_id)],
        cop_nodes=[cop_by_id[i] for i in sorted(cop_by_id)],
        jack_starts=cfg["jack_starts"],
        cop_starts=cfg["cop_starts"],
        num_cops=cfg["num_cops"],
        turn_limit=cfg["turn_limit"],
        hideout_min_distance=cfg["hideout_min_distance"],
    )
