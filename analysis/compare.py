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
  "would the agent have won the games this person lost?" Reported both as win
  rate and as the **participant score** (engine/metrics.py, SCORE_STUDY_V1) —
  the quantity humans were told they were scored on and, from workstream 10,
  the quantity the agent is trained to maximise. The score is the like-for-like
  comparison; win rate is kept for continuity.
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
from analysis.stats import cluster_bootstrap, design_effect, paired_difference
from engine.env import legal_jack_edges
from engine.game import GameRecord, StepContext, run_game, step_round
from engine.graph import Map, load_map
from engine.metrics import SCORE_PROGRESS_POINTS
from engine.state import CopKnowledge, GameState
from training.eval import PolicyAgent, game_score, load_checkpoint

MAPS_DIR = Path("maps")
COURSE_INDEX = MAPS_DIR / "course_participant.json"


# ---------------------------------------------------------------------------
# Scenario reconstruction
# ---------------------------------------------------------------------------


@dataclass
class HumanGame:
    """One participant game, with everything needed to replay its scenario."""

    row_id: int
    # Which reconstructed participant this game belongs to. Games are clustered
    # — three per participant — so any interval must resample participants, not
    # games. Treating 57 games as 57 independent draws overstates precision.
    participant: int
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
    # row_id -> participant index, so the bootstrap can cluster correctly.
    wanted = {
        g.row_id: i
        for i, s in enumerate(sessions)
        if include_flagged or s.is_usable
        for g in s.games
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
                participant=wanted[row_id],
                map_name=map_name,
                gaming_habit=habit,
                outcome=outcome,
                turns_survived=turns,
                initial_state=scenario_from_replay(replay),
                jack_moves=[r["jack_to"] for r in replay["rounds"]],
            )
        )
    return games



HABIT_ORDER = ("never_played", "played_few", "played_many", "unknown")


def human_breakdown_by_habit(
    humans: list[HumanGame], human_scores: dict[int, float]
) -> dict[str, dict]:
    """Human score and win rate split by self-reported experience.

    The project promises a comparison against "human players of varying skill
    levels"; gaming_habit is the only skill proxy collected. Clustered by
    participant like every other interval here, so three games from one person
    do not read as three independent draws.
    """
    out: dict[str, dict] = {}
    for habit in HABIT_ORDER:
        members = [h for h in humans if h.gaming_habit == habit]
        if not members:
            continue
        clusters: dict[int, list[float]] = {}
        for h in members:
            if h.row_id in human_scores:
                clusters.setdefault(h.participant, []).append(human_scores[h.row_id])
        out[habit] = {
            "participants": len({h.participant for h in members}),
            "games": len(members),
            "score": cluster_bootstrap(clusters),
            "win_rate": sum(h.human_won for h in members) / len(members),
        }
    return out

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
    """Win rate over n_replays of the identical board."""
    return policy_on_scenario_scored(
        jack_agent, game_map, initial_state, n_replays, seed
    )[0]


def policy_on_scenario_scored(
    jack_agent,
    game_map: Map,
    initial_state: GameState,
    n_replays: int,
    seed: int,
) -> tuple[float, float]:
    """(win rate, mean participant score) over n_replays of the identical board.

    Repeats are needed because `PolicyAgent` samples rather than acting greedily;
    the scenario is fixed but the policy is not deterministic.
    """
    wins = 0
    score_total = 0.0
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
        score_total += game_score(record, game_map)[0]
    return wins / n_replays, score_total / n_replays


def replay_human(
    game_map: Map, human: HumanGame, cop_params: dict | None = None
) -> GameRecord | None:
    """Re-run the human's own moves through the engine, to score their game.

    The database stores outcomes and moves but not scores (the browser kept
    those in sessionStorage), so the score is recomputed. `HeuristicCops` is
    deterministic given a state, so this reproduces the original game — but
    ONLY under the cop configuration the human actually faced.

    cop_params defaults to COPS_STUDY_V2 (None), which is correct for every
    game from the 2026-06-11 retune on, and so for all production study data
    (2026-08-05 onward). An older game needs COPS_PRERETUNE_V1: under today's
    cops it silently becomes a different game (fixture row 46 is one).

    Returns None unless the replay reproduces the stored outcome AND length —
    an illegal recorded move, a different winner or a different number of
    rounds all mean this is not the game the human played, and a score for it
    would be a score for a game that never happened.
    """
    ctx = StepContext(
        game_map=game_map,
        state=human.initial_state,
        terminated=False,
        winner=None,
        blocking=False,
        turn_limit=None,
    )
    cops = HeuristicCops(**(cop_params or {}))
    cops.on_episode_start(human.initial_state, game_map)
    for human_move in human.jack_moves:
        if ctx.terminated:
            break
        legal = legal_jack_edges(ctx.state, game_map, blocking=False)
        edge = next((e for e in legal if e.destination.id == human_move), None)
        if edge is None:
            return None
        step_round(ctx, edge, cops, director=None)
    if (
        not ctx.terminated
        or ctx.winner != human.outcome
        or len(ctx.history) != human.turns_survived
    ):
        return None
    return GameRecord(game_map, human.initial_state, ctx.winner, ctx.history)


def human_score(
    game_map: Map, human: HumanGame, cop_params: dict | None = None
) -> float | None:
    """The participant score this human earned, or None if it cannot be replayed."""
    record = replay_human(game_map, human, cop_params)
    return None if record is None else game_score(record, game_map)[0]


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
    participants = {h.participant for h in humans}

    # Cluster by participant: three games per person, so N games are not N
    # independent observations. See analysis/stats.py.
    human_clusters: dict[int, list[float]] = {}
    for h in humans:
        human_clusters.setdefault(h.participant, []).append(float(h.human_won))
    human_ci = cluster_bootstrap(human_clusters)
    deff = design_effect(human_clusters)

    print(
        f"\nHuman games: {len(humans)} from {len(participants)} participants"
        f"{' (INCLUDING FLAGGED — not the headline number)' if include_flagged else ''}"
    )
    print(f"Human win rate: {human_wins}/{len(humans)} = {human_ci}")
    print(
        f"  95% CI clustered by participant; design effect {deff:.2f}, so a "
        f"per-game\n  interval would have been about {deff**0.5:.2f}x too narrow."
    )

    # Human scores, recomputed by replaying each game (see replay_human).
    human_scores: dict[int, float] = {}
    for h in humans:
        sc = human_score(maps[h.map_name], h)
        if sc is not None:
            human_scores[h.row_id] = sc
    score_unreplayable = len(humans) - len(human_scores)
    human_score_clusters: dict[int, list[float]] = {}
    for h in humans:
        if h.row_id in human_scores:
            human_score_clusters.setdefault(h.participant, []).append(
                human_scores[h.row_id]
            )
    human_score_ci = cluster_bootstrap(human_score_clusters)
    print(
        f"Human participant score: {human_score_ci}  "
        f"(x{SCORE_PROGRESS_POINTS:,} = points shown to participants)"
    )
    if score_unreplayable:
        print(
            f"  WARNING: {score_unreplayable} game(s) could not be replayed to "
            f"the stored outcome and are left out of every score comparison."
        )
    print(f"Policy replays per scenario: {n_replays}\n")

    by_habit = human_breakdown_by_habit(humans, human_scores)
    if len(by_habit) > 1:
        # The project's stated comparison is against "human players of varying
        # skill levels", so the pooled human figure is not the whole claim.
        # gaming_habit is self-reported at sign-up and constant within a session.
        print("Humans by self-reported experience (participant-clustered CI):")
        print(f"  {'experience':<14}{'n':>3}{'games':>7}{'score':>26}{'win%':>8}")
        for habit, stats in by_habit.items():
            print(
                f"  {habit:<14}{stats['participants']:>3}{stats['games']:>7}"
                f"{str(stats['score']):>26}{stats['win_rate']:>7.1%}"
            )
        print("  Ns are small - read these as descriptive, not as a test.")
        print()


    results: dict[str, dict] = {}
    for path in checkpoint_paths:
        agent, step = load_checkpoint(path, device)
        label = Path(path).parent.name or Path(path).stem
        print(f"  {label} (step {step:,}) ...", flush=True)

        per_game, per_game_score, agree = [], [], AgreementResult(0, 0, 0)
        for h in humans:
            game_map = maps[h.map_name]
            jack = PolicyAgent(agent, game_map, device)
            win_rate, mean_score = policy_on_scenario_scored(
                jack, game_map, h.initial_state, n_replays, seed
            )
            per_game.append(win_rate)
            per_game_score.append(mean_score)
            a = move_agreement(jack, game_map, h)
            agree = AgreementResult(
                agree.decisions + a.decisions,
                agree.agreements + a.agreements,
                agree.desyncs + a.desyncs,
            )

        # Paired, clustered by participant. Pairing is the point: both sides saw
        # the identical board, so scenario difficulty cancels instead of
        # inflating the variance of both arms. An interval clear of zero is the
        # claim "the agent differs from the humans on the boards they played".
        paired: dict[int, list[tuple[float, float]]] = {}
        policy_clusters: dict[int, list[float]] = {}
        for p, h in zip(per_game, humans):
            paired.setdefault(h.participant, []).append((p, float(h.human_won)))
            policy_clusters.setdefault(h.participant, []).append(p)

        paired_score: dict[int, list[tuple[float, float]]] = {}
        policy_score_clusters: dict[int, list[float]] = {}
        for p, h in zip(per_game_score, humans):
            if h.row_id in human_scores:
                paired_score.setdefault(h.participant, []).append(
                    (p, human_scores[h.row_id])
                )
                policy_score_clusters.setdefault(h.participant, []).append(p)

        results[label] = {
            "step": step,
            "policy_score": _mean(
                [p for p, h in zip(per_game_score, humans) if h.row_id in human_scores]
            ),
            "policy_score_ci": cluster_bootstrap(policy_score_clusters),
            "paired_score_diff": paired_difference(paired_score),
            "human_score": _mean(list(human_scores.values())),
            "policy_win_rate": sum(per_game) / len(per_game),
            "policy_ci": cluster_bootstrap(policy_clusters),
            "paired_diff": paired_difference(paired),
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
        f"{'checkpoint':<24} {'step':>8} {'score':>7} {'vs human':>9} "
        f"{'policy win%':>12} {'vs human':>9} "
        f"{f'human lost (n={n_lost})':>20} {f'human won (n={n_won})':>19} "
        f"{'agree%':>8} {'desync':>7}"
    )
    print(col)
    print("-" * len(col))
    human_rate = sum(h.human_won for h in humans) / len(humans)
    for label, m in results.items():
        delta = m["policy_win_rate"] - human_rate
        score_delta = m["policy_score"] - m["human_score"]
        print(
            f"{label:<24} {m['step'] / 1e6:>7.2f}M {m['policy_score']:>7.3f} "
            f"{score_delta:>+9.3f} "
            f"{m['policy_win_rate']:>11.1%} "
            f"{delta:>+8.1%} {m['where_human_lost']:>19.1%} "
            f"{m['where_human_won']:>18.1%} "
            f"{m['agreement_rate']:>7.1%} {m['desyncs']:>7}"
        )
    print()
    print(f"'human lost/won' = the policy's win rate on the boards humans lost / won.")
    print(f"'agree%' = share of the humans' own decisions the policy would have made.")
    print()

    # The headline claim needs an interval, not a delta. Paired and clustered by
    # participant — see analysis/stats.py for why the clustering is not optional.
    print("Paired policy - human PARTICIPANT SCORE, 95% CI clustered by participant:")
    for label, m in results.items():
        d = m["paired_score_diff"]
        verdict = (
            "excludes zero"
            if d.low > 0 or d.high < 0
            else "INCLUDES ZERO - not significant"
        )
        print(f"  {label:<24} {d}   {verdict}")
    print()
    print("Paired policy - human WIN RATE, 95% CI clustered by participant:")
    for label, m in results.items():
        d = m["paired_diff"]
        verdict = (
            "excludes zero"
            if d.low > 0 or d.high < 0
            else "INCLUDES ZERO - not significant"
        )
        print(f"  {label:<24} {d}   {verdict}")
    print()
    print("  An interval clear of zero supports 'the agent differs from the humans")
    print("  on the boards they actually played'. One spanning zero does not, no")
    print("  matter how large the point estimate looks.")
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
