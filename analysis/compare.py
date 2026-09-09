"""Human-vs-RL comparison on the exact boards participants played.

The thesis's actual deliverable. Nothing else in the repo joins
`data/games.sqlite` to policy evaluation.

**Why paired.** The course maps pin Jack's start and the hideout *zone anchor*,
but the hideout is still sampled within the zone and cop starts are drawn from a
pool (`engine/env.py`). Scenarios are therefore comparable across participants
but not bit-identical, so averaging at the map level confounds scenario
difficulty with player skill. Every stored replay carries a complete scenario
specification, so instead each policy is replayed on the precise board its human
counterpart faced.

Two analyses, from the same reconstruction:

* **Outcome** (E2) — replay each human scenario N times per checkpoint. Answers
  "would the agent have won the games this person lost?"
* **Move agreement** (E3) — at every state the human actually reached, ask what
  the policy would have chosen. No new games are played. Answers something the
  win rate cannot: *does the Director change how the agent plays, or only how
  often it wins?*

Usage:
    uv run python -m analysis.compare checkpoints/run/agent_best.pt
    uv run python -m analysis.compare ckpt_on.pt ckpt_off.pt --n-replays 20
    uv run python -m analysis.compare ckpt.pt --include-flagged
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import torch

from agents.heuristic_cops import HeuristicCops
from analysis.sessions import DEFAULT_DB, load_games, reconstruct_sessions
from engine.env import legal_jack_edges
from engine.game import run_game
from engine.graph import Map, load_map
from engine.state import CopKnowledge, GameState
from training.eval import PolicyAgent, load_checkpoint

MAPS_DIR = Path("maps")
COURSE_INDEX = MAPS_DIR / "course_participant.json"


# ---------------------------------------------------------------------------
# Scenario reconstruction
# ---------------------------------------------------------------------------


@dataclass
class HumanGame:
    """One participant game, with everything needed to replay its scenario."""

    row_id: int
    map_name: str
    gaming_habit: str
    outcome: str
    turns_survived: int
    initial_state: GameState
    jack_moves: list[int]  # the destinations the human actually chose, in order

    @property
    def human_won(self) -> bool:
        return self.outcome == "jack"


def load_map_registry() -> dict[str, Map]:
    """map_name -> Map, mirroring how the server registers them."""
    registry: dict[str, Map] = {}
    if COURSE_INDEX.exists():
        for entry in json.loads(COURSE_INDEX.read_text()):
            registry[entry["name"]] = load_map(MAPS_DIR / entry["file"])
    # Pre-course games ran on the base map, which is not in the course index.
    if "whitechapel" not in registry:
        registry["whitechapel"] = load_map(MAPS_DIR / "whitechapel.json")
    return registry


def scenario_from_replay(replay: dict) -> GameState:
    """Rebuild the exact opening position a participant was dealt.

    Mirrors `make_initial_state` — turn 0, Jack's start public (depth 0 in
    cop_knowledge), nothing searched yet.
    """
    start = replay["initial_jack_pos"]
    return GameState(
        jack_pos=start,
        cop_positions=tuple(replay["initial_cop_positions"]),
        hideout=replay["hideout"],
        hideout_zone_anchor=replay["hideout_zone_anchor"],
        hideout_zone=frozenset(replay["hideout_zone"]),
        turn=0,
        jack_trace=frozenset({start}),
        jack_path=(start,),
        cop_searched_hits=frozenset(),
        cop_searched_misses=frozenset(),
        cop_knowledge=CopKnowledge(jack_start=start, visited_at=((start, 0),)),
    )


def load_human_games(
    db_path: Path | str = DEFAULT_DB, include_flagged: bool = False
) -> list[HumanGame]:
    """Human games from usable sessions, with their scenarios reconstructed.

    Only sessions that survive `analysis.sessions` filtering are included, so the
    comparison inherits one exclusion policy rather than inventing a second.
    `include_flagged` widens to every genuine participant game (still never admin
    artifacts) — useful for checking whether a conclusion is an artefact of the
    small usable N, never for the headline number.
    """
    sessions, _ = reconstruct_sessions(load_games(db_path))
    wanted = {
        g.row_id for s in sessions if include_flagged or s.is_usable for g in s.games
    }
    if not wanted:
        return []

    path = Path(db_path)
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT id, map_name, gaming_habit, outcome, turns_survived, replay "
            "FROM games ORDER BY id"
        ).fetchall()

    games: list[HumanGame] = []
    for row_id, map_name, habit, outcome, turns, replay_json in rows:
        if row_id not in wanted:
            continue
        replay = json.loads(replay_json)
        if not replay.get("rounds"):
            continue  # nothing to replay
        games.append(
            HumanGame(
                row_id=row_id,
                map_name=map_name,
                gaming_habit=habit,
                outcome=outcome,
                turns_survived=turns,
                initial_state=scenario_from_replay(replay),
                jack_moves=[r["jack_to"] for r in replay["rounds"]],
            )
        )
    return games


# ---------------------------------------------------------------------------
# E2 — outcome on the human's own scenario
# ---------------------------------------------------------------------------


def policy_on_scenario(
    jack_agent,
    game_map: Map,
    initial_state: GameState,
    n_replays: int,
    seed: int,
) -> float:
    """Win rate over n_replays of the identical board.

    Repeats are needed because `PolicyAgent` samples rather than acting greedily;
    the scenario is fixed but the policy is not deterministic.
    """
    wins = 0
    for i in range(n_replays):
        record = run_game(
            game_map,
            jack_agent,
            HeuristicCops(),
            director=None,
            rng=random.Random(seed + i),
            initial_state=initial_state,
        )
        wins += record.winner == "jack"
    return wins / n_replays


# ---------------------------------------------------------------------------
# E3 — move-level agreement
# ---------------------------------------------------------------------------


@dataclass
class AgreementResult:
    decisions: int
    agreements: int
    # Turns the reconstruction could not follow — see below. A non-zero value
    # invalidates the agreement rate rather than merely reducing its precision.
    desyncs: int

    @property
    def rate(self) -> float:
        return self.agreements / max(self.decisions, 1)


def move_agreement(
    jack_agent,
    game_map: Map,
    human: HumanGame,
) -> AgreementResult:
    """Ask the policy what it would do at each state the human actually reached.

    The human's move sequence is walked forward through the engine. `HeuristicCops`
    is deterministic given a state, so re-running the cops reproduces the original
    game exactly; a desync means that assumption broke and the result must not be
    trusted for that game.

    No games are played here — this is analysis of states that already happened.
    """
    from engine.game import StepContext, step_round

    ctx = StepContext(
        game_map=game_map,
        state=human.initial_state,
        terminated=False,
        winner=None,
        blocking=False,
        turn_limit=None,
    )
    cops = HeuristicCops()
    cops.on_episode_start(human.initial_state, game_map)
    jack_agent.on_episode_start(human.initial_state, game_map)

    decisions = agreements = desyncs = 0
    for human_move in human.jack_moves:
        if ctx.terminated:
            break
        legal = legal_jack_edges(ctx.state, game_map, blocking=False)
        edge = next((e for e in legal if e.destination.id == human_move), None)
        if edge is None:
            # The human's recorded move is not legal from the state we
            # reconstructed: the replay and the engine have diverged.
            desyncs += 1
            break
        policy_choice = jack_agent.act(ctx.state, legal, game_map).edge
        decisions += 1
        agreements += policy_choice.destination.id == human_move
        step_round(ctx, edge, cops, director=None)

    return AgreementResult(decisions, agreements, desyncs)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def compare(
    checkpoint_paths: list[str],
    db_path: Path | str = DEFAULT_DB,
    n_replays: int = 20,
    seed: int = 0,
    include_flagged: bool = False,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maps = load_map_registry()
    humans = load_human_games(db_path, include_flagged)

    if not humans:
        print("\nNo usable human games in this snapshot — nothing to compare.")
        print("Run `uv run python -m analysis.sessions --verbose` to see why.")
        return {}

    human_wins = sum(h.human_won for h in humans)
    print(
        f"\nHuman games: {len(humans)}"
        f"{' (INCLUDING FLAGGED — not the headline number)' if include_flagged else ''}"
    )
    print(
        f"Human win rate: {human_wins}/{len(humans)} = {human_wins / len(humans):.1%}"
    )
    print(f"Policy replays per scenario: {n_replays}\n")

    results: dict[str, dict] = {}
    for path in checkpoint_paths:
        agent, step = load_checkpoint(path, device)
        label = Path(path).parent.name or Path(path).stem
        print(f"  {label} (step {step:,}) ...", flush=True)

        per_game, agree = [], AgreementResult(0, 0, 0)
        for h in humans:
            game_map = maps[h.map_name]
            jack = PolicyAgent(agent, game_map, device)
            per_game.append(
                policy_on_scenario(jack, game_map, h.initial_state, n_replays, seed)
            )
            a = move_agreement(jack, game_map, h)
            agree = AgreementResult(
                agree.decisions + a.decisions,
                agree.agreements + a.agreements,
                agree.desyncs + a.desyncs,
            )

        results[label] = {
            "step": step,
            "policy_win_rate": sum(per_game) / len(per_game),
            "per_scenario": per_game,
            "agreement_rate": agree.rate,
            "decisions": agree.decisions,
            "desyncs": agree.desyncs,
            # Split by what the human did on that same board. "The agent wins
            # 70% of the boards the humans lost" is the claim the thesis wants;
            # a single pooled win rate cannot make it.
            "where_human_lost": _mean(
                [p for p, h in zip(per_game, humans) if not h.human_won]
            ),
            "where_human_won": _mean(
                [p for p, h in zip(per_game, humans) if h.human_won]
            ),
        }

    _print_table(results, humans)
    return results


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _print_table(results: dict, humans: list[HumanGame]) -> None:
    n_lost = sum(not h.human_won for h in humans)
    n_won = len(humans) - n_lost
    print()
    col = (
        f"{'checkpoint':<24} {'step':>8} {'policy win%':>12} {'vs human':>9} "
        f"{f'human lost (n={n_lost})':>20} {f'human won (n={n_won})':>19} "
        f"{'agree%':>8} {'desync':>7}"
    )
    print(col)
    print("-" * len(col))
    human_rate = sum(h.human_won for h in humans) / len(humans)
    for label, m in results.items():
        delta = m["policy_win_rate"] - human_rate
        print(
            f"{label:<24} {m['step'] / 1e6:>7.2f}M {m['policy_win_rate']:>11.1%} "
            f"{delta:>+8.1%} {m['where_human_lost']:>19.1%} "
            f"{m['where_human_won']:>18.1%} "
            f"{m['agreement_rate']:>7.1%} {m['desyncs']:>7}"
        )
    print()
    print(f"'human lost/won' = the policy's win rate on the boards humans lost / won.")
    print(f"'agree%' = share of the humans' own decisions the policy would have made.")
    print()
    if any(m["desyncs"] for m in results.values()):
        print("WARNING: desyncs > 0 — the replay could not be followed through the")
        print("engine for some games. Move-agreement numbers are not trustworthy")
        print("until that is understood. Outcome numbers are unaffected.\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoints", nargs="+", metavar="CHECKPOINT")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--n-replays", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--include-flagged",
        action="store_true",
        help="include sessions excluded by analysis.sessions — robustness check "
        "only, never the reported number",
    )
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    compare(a.checkpoints, a.db, a.n_replays, a.seed, a.include_flagged)
