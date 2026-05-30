"""
Run a game where Jack follows a fixed, pre-defined path while heuristic cops
respond adaptively.

Useful for regression-testing cop behaviour across parameter changes or code
edits: replay a known Jack path and compare winner, turns survived, and search
hits against a saved baseline.

Usage:
    uv run tools/scripted_sim.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.env import legal_jack_edges
from engine.game import StepContext, step_round
from engine.graph import Map, load_map
from engine.state import CopKnowledge, GameState
from agents.heuristic_cops import HeuristicCops


def run_scripted_game(
    jack_script: list[int],
    initial_jack_pos: int,
    initial_cop_positions: tuple[int, ...],
    hideout: int,
    hideout_zone_anchor: int,
    hideout_zone: frozenset[int],
    turn_limit: int = 15,
    blocking: bool = False,
    cop_params: dict | None = None,
    map_path: str = "maps/whitechapel.json",
    game_map: Map | None = None,
) -> dict:
    if game_map is None:
        game_map = load_map(map_path)
    state = GameState(
        jack_pos=initial_jack_pos,
        cop_positions=tuple(initial_cop_positions),
        hideout=hideout,
        hideout_zone_anchor=hideout_zone_anchor,
        hideout_zone=frozenset(hideout_zone),
        turn=0,
        jack_trace=frozenset({initial_jack_pos}),
        jack_path=(initial_jack_pos,),
        cop_searched_hits=frozenset(),
        cop_searched_misses=frozenset(),
        cop_knowledge=CopKnowledge(jack_start=initial_jack_pos),
    )
    ctx = StepContext(
        game_map=game_map,
        state=state,
        terminated=False,
        winner=None,
        blocking=blocking,
        turn_limit=turn_limit,
    )
    cop_agent = HeuristicCops(**(cop_params or {}))
    cop_agent.on_episode_start(state, game_map)

    for target_id in jack_script:
        if ctx.terminated:
            break
        edges = {
            e.destination.id: e for e in game_map.jack_nodes[ctx.state.jack_pos].edges
        }
        if target_id not in edges:
            raise ValueError(
                f"target {target_id} unreachable from {ctx.state.jack_pos}"
            )
        step_round(ctx, edges[target_id], cop_agent, director=None)

    per_round = []
    all_hits: set[int] = set()
    for rr in ctx.history:
        hits = [
            jn
            for step in rr.cop_steps
            for jn, hit in step.search_results.items()
            if hit
        ]
        all_hits.update(hits)
        roles = (
            [cd.role for cd in sorted(rr.cop_decisions.cops, key=lambda c: c.cop_idx)]
            if rr.cop_decisions
            else []
        )
        per_round.append(
            {
                "turn": rr.turn,
                "jack_to": rr.state_after_jack.jack_pos,
                "cop_destinations": list(rr.state_after_round.cop_positions),
                "cop_roles": roles,
                "search_hits": hits,
            }
        )

    return {
        "winner": ctx.winner,
        "turns_survived": len(ctx.history),
        "search_hits_total": len(all_hits),
        "per_round": per_round,
    }


def run_policy_game(
    initial_jack_pos: int,
    initial_cop_positions: tuple[int, ...],
    hideout: int,
    hideout_zone_anchor: int,
    hideout_zone: frozenset[int],
    turn_limit: int = 15,
    blocking: bool = False,
    cop_params: dict | None = None,
    map_path: str = "maps/whitechapel.json",
    game_map: Map | None = None,
    jack_agent=None,
) -> dict:
    """Run a full game with a live JackAgent instead of a fixed script.

    jack_agent defaults to RandomJack when None.  Pass a PolicyAgent for
    evaluation against a trained policy.  Returns the same dict format as
    run_scripted_game so callers are interchangeable.
    """
    from agents.random_agents import RandomJack

    if game_map is None:
        game_map = load_map(map_path)
    if jack_agent is None:
        jack_agent = RandomJack()

    state = GameState(
        jack_pos=initial_jack_pos,
        cop_positions=tuple(initial_cop_positions),
        hideout=hideout,
        hideout_zone_anchor=hideout_zone_anchor,
        hideout_zone=frozenset(hideout_zone),
        turn=0,
        jack_trace=frozenset({initial_jack_pos}),
        jack_path=(initial_jack_pos,),
        cop_searched_hits=frozenset(),
        cop_searched_misses=frozenset(),
        cop_knowledge=CopKnowledge(
            jack_start=initial_jack_pos,
            visited_at=((initial_jack_pos, 0),),
        ),
    )
    ctx = StepContext(
        game_map=game_map,
        state=state,
        terminated=False,
        winner=None,
        blocking=blocking,
        turn_limit=turn_limit,
    )
    cop_agent = HeuristicCops(**(cop_params or {}))
    cop_agent.on_episode_start(state, game_map)
    jack_agent.on_episode_start(state, game_map)

    while not ctx.terminated:
        edges = legal_jack_edges(ctx.state, game_map, blocking=blocking)
        if not edges:
            break
        output = jack_agent.act(ctx.state, edges, game_map)
        step_round(ctx, output.edge, cop_agent, director=None)

    per_round = []
    all_hits: set[int] = set()
    for rr in ctx.history:
        hits = [
            jn
            for step in rr.cop_steps
            for jn, hit in step.search_results.items()
            if hit
        ]
        all_hits.update(hits)
        roles = (
            [cd.role for cd in sorted(rr.cop_decisions.cops, key=lambda c: c.cop_idx)]
            if rr.cop_decisions
            else []
        )
        per_round.append(
            {
                "turn": rr.turn,
                "jack_to": rr.state_after_jack.jack_pos,
                "cop_destinations": list(rr.state_after_round.cop_positions),
                "cop_roles": roles,
                "search_hits": hits,
            }
        )

    return {
        "winner": ctx.winner,
        "turns_survived": len(ctx.history),
        "search_hits_total": len(all_hits),
        "per_round": per_round,
    }
