"""
Replay storage: 5-slot rotating JSON files in data/replays/.

index.json tracks metadata for all slots so the list endpoint is fast.
Full round data is in slot_N.json files.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from agents.base import CopDecisionInfo

if TYPE_CHECKING:
    from server.session import GameSession

REPLAY_DIR = Path("data/replays")
INDEX_PATH = REPLAY_DIR / "index.json"
NUM_SLOTS = 5


# ---------------------------------------------------------------------------
# Replay dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReplayCopAction:
    cop_idx: int
    from_node: int
    to_node: int
    action: str  # "search" | "arrest"
    searched_nodes: list[int]  # adjacent jack nodes (search only)
    search_hits: list[int]  # searched nodes that Jack visited
    search_miss_nodes: list[int]  # searched nodes that Jack did NOT visit
    arrest_target: list[int]
    arrest_success: bool
    role: str | None  # from CopDecisionInfo
    coverage_score: float | None
    direction_score: float | None


@dataclass
class ReplayRound:
    turn: int
    jack_from: int
    jack_to: int
    jack_via: list[int]  # cop node IDs traversed (via edge)
    jack_legal_moves: list[int]  # legal destinations at start of turn
    cop_actions: list[ReplayCopAction]
    position_pmf: dict[str, float] | None  # PMF at cop planning time (before cops act)
    pmf_after_cops: (
        dict[str, float] | None
    )  # PMF after all cops acted (search misses applied)
    hideout_pmf: dict[str, float] | None
    visited_at_after: list[tuple[int, int]]  # each inner list is [node_id, depth]
    search_misses_after: list[tuple[int, int]]
    arrest_misses_after: list[tuple[int, int]]
    terminated: bool
    winner: str | None


@dataclass
class ReplayRecord:
    game_id: str
    timestamp: str  # ISO-8601
    map_name: str
    winner: str
    turns_survived: int
    initial_jack_pos: int
    initial_cop_positions: list[int]
    hideout: int
    blocking: bool
    turn_limit: int
    hideout_zone_anchor: int
    hideout_zone: list[int]
    rounds: list[ReplayRound] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------


def _read_index() -> dict:
    if not INDEX_PATH.exists():
        return {"next_slot": 0, "slots": []}
    with INDEX_PATH.open() as f:
        return json.load(f)


def _write_index(index: dict) -> None:
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    with INDEX_PATH.open("w") as f:
        json.dump(index, f, indent=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_replays() -> list[dict]:
    """Return metadata for all saved slots (no full round data)."""
    index = _read_index()
    return index.get("slots", [])


def _deserialize_record(data: dict) -> ReplayRecord:
    rounds = [
        ReplayRound(
            turn=r["turn"],
            jack_from=r["jack_from"],
            jack_to=r["jack_to"],
            jack_via=r["jack_via"],
            jack_legal_moves=r["jack_legal_moves"],
            cop_actions=[
                ReplayCopAction(**{**a, "arrest_target": a.get("arrest_target") or []})
                for a in r["cop_actions"]
            ],
            position_pmf=r.get("position_pmf"),
            pmf_after_cops=r.get("pmf_after_cops"),
            hideout_pmf=r.get("hideout_pmf"),
            visited_at_after=r["visited_at_after"],
            search_misses_after=r["search_misses_after"],
            arrest_misses_after=r["arrest_misses_after"],
            terminated=r["terminated"],
            winner=r.get("winner"),
        )
        for r in data["rounds"]
    ]
    return ReplayRecord(
        game_id=data["game_id"],
        timestamp=data["timestamp"],
        map_name=data["map_name"],
        winner=data["winner"],
        turns_survived=data["turns_survived"],
        initial_jack_pos=data["initial_jack_pos"],
        initial_cop_positions=data["initial_cop_positions"],
        hideout=data["hideout"],
        blocking=data["blocking"],
        turn_limit=data["turn_limit"],
        hideout_zone_anchor=data["hideout_zone_anchor"],
        hideout_zone=data["hideout_zone"],
        rounds=rounds,
    )


def load_replay(slot: int) -> ReplayRecord | None:
    path = REPLAY_DIR / f"slot_{slot}.json"
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    return _deserialize_record(data)


def save_replay(record: ReplayRecord) -> int:
    """Write record to next available slot. Returns the slot number used."""
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    index = _read_index()
    slot = index.get("next_slot", 0) % NUM_SLOTS

    path = REPLAY_DIR / f"slot_{slot}.json"
    with path.open("w") as f:
        json.dump(asdict(record), f)

    # Update index metadata (one entry per slot)
    slots: list[dict] = index.get("slots", [])
    meta = {
        "slot": slot,
        "game_id": record.game_id,
        "timestamp": record.timestamp,
        "winner": record.winner,
        "turns_survived": record.turns_survived,
    }
    # Replace existing entry for this slot or append
    slots = [s for s in slots if s.get("slot") != slot]
    slots.append(meta)
    slots.sort(key=lambda s: s["slot"])

    index["next_slot"] = (slot + 1) % NUM_SLOTS
    index["slots"] = slots
    _write_index(index)
    return slot


def build_replay(session: "GameSession") -> ReplayRecord:
    """Build a ReplayRecord from session history (does not save)."""
    from engine.env import legal_jack_edges
    from agents.heuristic_cops import HeuristicCops

    gm = session.ctx.game_map
    effective_limit = (
        session.ctx.turn_limit if session.ctx.turn_limit is not None else gm.turn_limit
    )

    rounds: list[ReplayRound] = []
    for rr in session.ctx.history:
        # Jack info
        jack_from = rr.state_before.jack_pos
        jack_to = rr.state_after_jack.jack_pos
        jack_via = [cn.id for cn in rr.jack_edge.via]

        legal_moves = list(
            dict.fromkeys(
                e.destination.id
                for e in legal_jack_edges(
                    rr.state_before, gm, blocking=session.ctx.blocking
                )
            )
        )

        # Cop decision metadata sidecar
        decisions = rr.cop_decisions  # RoundCopDecisions | None
        decision_by_idx: dict[int, CopDecisionInfo] = {}
        if decisions is not None:
            for cd in decisions.cops:
                decision_by_idx[cd.cop_idx] = cd

        # Per-cop actions
        cop_actions: list[ReplayCopAction] = []
        for step in rr.cop_steps:
            ct = step.cop_turn
            cop_node = gm.cop_nodes[ct.destination]
            searched = list(step.search_results.keys()) if ct.search else []
            hits = [n for n, hit in step.search_results.items() if hit]
            misses = [n for n, hit in step.search_results.items() if not hit]

            if not ct.search:
                if ct.arrest_all:
                    arrest_nodes = [jn.id for jn in cop_node.jack_neighbours]
                else:
                    arrest_nodes = (
                        [ct.arrest_target] if ct.arrest_target is not None else []
                    )
            else:
                arrest_nodes = []

            cd = decision_by_idx.get(ct.cop_idx)
            cop_actions.append(
                ReplayCopAction(
                    cop_idx=ct.cop_idx,
                    from_node=rr.state_before.cop_positions[ct.cop_idx],
                    to_node=ct.destination,
                    action="search" if ct.search else "arrest",
                    searched_nodes=searched,
                    search_hits=hits,
                    search_miss_nodes=misses,
                    arrest_target=arrest_nodes,
                    arrest_success=(
                        step.terminated and step.winner == "cops" and not ct.search
                    ),
                    role=cd.role if cd else None,
                    coverage_score=cd.coverage_score if cd else None,
                    direction_score=cd.direction_score if cd else None,
                )
            )

        state_after = rr.state_after_round
        ck = state_after.cop_knowledge

        pos_pmf = (
            {str(k): v for k, v in decisions.position_pmf.items()}
            if decisions is not None
            else None
        )
        hid_pmf = (
            {str(k): v for k, v in decisions.hideout_pmf.items()}
            if decisions is not None
            else None
        )

        # PMF after all cops acted — uses final cop step's state which has all
        # search/arrest misses incorporated into cop_knowledge.
        if rr.cop_steps:
            final_state = rr.cop_steps[-1].state_after
            pmf_raw = HeuristicCops.compute_pmf(final_state, gm)
            pmf_after_cops: dict[str, float] | None = {
                str(k): v for k, v in pmf_raw.items()
            }
        else:
            pmf_after_cops = None

        rounds.append(
            ReplayRound(
                turn=rr.turn,
                jack_from=jack_from,
                jack_to=jack_to,
                jack_via=jack_via,
                jack_legal_moves=legal_moves,
                cop_actions=cop_actions,
                position_pmf=pos_pmf,
                pmf_after_cops=pmf_after_cops,
                hideout_pmf=hid_pmf,
                visited_at_after=sorted(ck.visited_at),
                search_misses_after=[tuple(m) for m in ck.search_misses],
                arrest_misses_after=[tuple(m) for m in ck.arrest_misses],
                terminated=rr.terminated,
                winner=rr.winner,
            )
        )

    initial = (
        session.ctx.history[0].state_before
        if session.ctx.history
        else session.ctx.state
    )
    record = ReplayRecord(
        game_id=session.game_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        map_name=session.map_name or "whitechapel",
        winner=session.ctx.winner or "unknown",
        turns_survived=len(session.ctx.history),
        initial_jack_pos=initial.jack_pos,
        initial_cop_positions=list(initial.cop_positions),
        hideout=initial.hideout,
        blocking=session.ctx.blocking,
        turn_limit=effective_limit,
        hideout_zone_anchor=initial.hideout_zone_anchor,
        hideout_zone=sorted(initial.hideout_zone),
        rounds=rounds,
    )
    return record


def build_and_save_replay(session: "GameSession") -> ReplayRecord:
    """Build a ReplayRecord from session history and save it to the rotating slots."""
    record = build_replay(session)
    save_replay(record)
    return record
