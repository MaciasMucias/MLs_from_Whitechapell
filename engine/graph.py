from dataclasses import dataclass, field
import json
import pickle
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
class MapBundle:
    jack_nodes: list[JackNode]  # indexed by id-1
    cop_nodes: list[CopNode]    # indexed by id-1


@dataclass
class MapConfig:
    jack_start: int
    hideout_min_distance: int
    num_cops: int
    turn_limit: int


def load_map(path: str | Path) -> MapBundle:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_config(path: str | Path) -> MapConfig:
    with open(path) as f:
        data = json.load(f)
    return MapConfig(
        jack_start=data["jack_start"],
        hideout_min_distance=data["hideout_min_distance"],
        num_cops=data["num_cops"],
        turn_limit=data["turn_limit"],
    )
