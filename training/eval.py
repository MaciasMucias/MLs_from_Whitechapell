"""
Evaluate trained Jack policies against full-strength heuristic cops (no Director).

Core function eval_policy() returns a plain dict — can be called from the
training loop to log per-checkpoint metrics to wandb, or used standalone.

Checkpoint names: periodic saves are agent_<step:010d>.pt and the best-scoring
one is copied to agent_best.pt (see training/checkpoints.py). There is no
agent_final.pt — train.py has never written one.

Usage:
    uv run python -m training.eval checkpoints/run/agent_best.pt
    uv run python -m training.eval checkpoints/run/*.pt --n-games 500 --seed 42
    uv run python -m training.eval ckpt1.pt ckpt2.pt --no-baseline
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from agents.base import AgentOutput, JackAgent
from agents.heuristic_cops import COPS_PRERETUNE_V1, HeuristicCops
from agents.random_agents import RandomJack
from engine.game import run_game
from engine.graph import JackEdge, Map, load_map
from engine.metrics import hideout_uncertainty
from engine.state import GameState
from training.model import Agent
from training.obs import build_obs, precompute_distances


# ---------------------------------------------------------------------------
# PolicyAgent — wraps a trained Agent as a JackAgent for run_game
# ---------------------------------------------------------------------------


class PolicyAgent(JackAgent):
    def __init__(self, agent: Agent, game_map: Map, device: torch.device) -> None:
        self._agent = agent
        self._device = device
        self._all_dists, self._diameter = precompute_distances(game_map)
        self._n_jack = len(game_map.jack_nodes)

    def act(
        self, state: GameState, legal_edges: list[JackEdge], game_map: Map
    ) -> AgentOutput:
        obs = build_obs(state, game_map, self._all_dists, self._diameter)
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self._device)
        mask = np.zeros(self._n_jack, dtype=bool)
        for e in legal_edges:
            mask[e.destination.id] = True
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self._device)
        with torch.no_grad():
            action, _, _, _ = self._agent.get_action_and_value(obs_t, mask_t)
        edge = next(e for e in legal_edges if e.destination.id == action.item())
        return AgentOutput(edge=edge)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------


def _min_cop_distance(
    state: GameState,
    game_map: Map,
    all_dists: dict[int, dict[int, int]],
    diameter: int,
) -> int:
    """Jack's BFS distance to the nearest Jack node any cop currently covers.

    Mirrors the cop-distance term in training/env.py so the eval metric and the
    zeta shaping term measure the same thing.
    """
    return min(
        all_dists[state.jack_pos].get(jn.id, diameter)
        for cp in state.cop_positions
        for jn in game_map.cop_nodes[cp].jack_neighbours
    )


def eval_agent(
    jack_agent: JackAgent,
    game_map: Map,
    n_games: int,
    rng: random.Random | None = None,
    cop_params: dict | None = None,
) -> dict[str, float]:
    """
    Run n_games against full-strength HeuristicCops with no Director.
    Returns a plain dict suitable for console printing or wandb logging.

    cop_params overrides the cop configuration. Default (None) is
    COPS_STUDY_V2, the frozen study cops — that is what every reported number
    uses. Pass COPS_PRERETUNE_V1 to measure generalisation to cops the policy
    never trained against; see that preset's docstring.
    """
    rng = rng or random.Random()
    cops = HeuristicCops(**(cop_params or {}))
    all_dists, diameter = precompute_distances(game_map)

    wins = 0
    arrests = 0
    turns_all: list[int] = []
    turns_on_win: list[int] = []
    turns_on_loss: list[int] = []
    hideout_uncerts: list[float] = []
    cop_dists: list[float] = []

    for _ in range(n_games):
        record = run_game(game_map, jack_agent, cops, director=None, rng=rng)
        t = record.turns_survived
        turns_all.append(t)

        # Mean distance Jack kept from his nearest cop over the whole game.
        # The direct evasiveness measure: a risk-averse policy holds a wider
        # berth, and pays for it in detours against the turn limit.
        per_round = [
            _min_cop_distance(rr.state_after_round, game_map, all_dists, diameter)
            for rr in record.history
        ]
        if per_round:
            cop_dists.append(sum(per_round) / len(per_round))

        if record.winner == "jack":
            wins += 1
            turns_on_win.append(t)
            final = record.history[-1].state_after_round
            pmf = HeuristicCops.compute_pmf(final, game_map)
            hideout_uncerts.append(hideout_uncertainty(final.hideout_zone, pmf))
        else:
            turns_on_loss.append(t)
            # Losses split into "caught" and "ran out of time". A cop step that
            # terminated the game is an arrest; anything else ended in
            # end_of_round, i.e. the turn limit. The split is what distinguishes
            # a policy that dies bravely from one that dawdles.
            if any(
                cs.terminated and cs.winner == "cops"
                for rr in record.history
                for cs in rr.cop_steps
            ):
                arrests += 1

    losses = n_games - wins
    return {
        "win_rate": wins / n_games,
        "mean_turns": sum(turns_all) / n_games,
        "mean_turns_on_win": sum(turns_on_win) / max(len(turns_on_win), 1),
        "mean_turns_on_loss": sum(turns_on_loss) / max(len(turns_on_loss), 1),
        "mean_hideout_uncert": sum(hideout_uncerts) / max(len(hideout_uncerts), 1),
        "mean_min_cop_dist": sum(cop_dists) / max(len(cop_dists), 1),
        # Fractions of ALL games, so arrest_rate + timeout_rate + win_rate == 1.
        "arrest_rate": arrests / n_games,
        "timeout_rate": (losses - arrests) / n_games,
        # ...and of losses only, which is the risk-aversion signal: a cautious
        # policy shifts its losses from arrest to timeout without necessarily
        # losing less often.
        "arrest_share_of_losses": arrests / max(losses, 1),
        "n_wins": float(wins),
        "n_games": float(n_games),
    }


def eval_policy(
    agent: Agent,
    game_map: Map,
    n_games: int,
    device: torch.device,
    rng: random.Random | None = None,
    cop_params: dict | None = None,
) -> dict[str, float]:
    """Evaluate a trained Agent. Thin wrapper around eval_agent."""
    jack = PolicyAgent(agent, game_map, device)
    return eval_agent(jack, game_map, n_games, rng, cop_params)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint(path: str, device: torch.device) -> tuple[Agent, int]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    agent = Agent(ckpt["obs_dim"], ckpt["n_actions"]).to(device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    return agent, ckpt["step"]


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

_COL_W = 48  # checkpoint name column width


def _fmt_step(step: int) -> str:
    return f"{step / 1_000_000:.2f}M" if step else "-"


def _print_table(
    rows: list[tuple[str, int | None, dict[str, float]]],
    n_games: int,
    map_path: str,
    cops_label: str = "COPS_STUDY_V2 (frozen - the study cops)",
) -> None:
    header = (
        f"Eval  {n_games} games | stochastic | no director | {map_path}\n"
        f"Cops: {cops_label}"
    )
    print()
    print(header)
    print()
    col = (
        f"{'checkpoint':<{_COL_W}}  {'step':>7}  {'win%':>6}  {'turns':>6}  "
        f"{'turns(W)':>8}  {'turns(L)':>8}  {'hideout_u':>9}  "
        f"{'copdist':>7}  {'arrest%':>7}  {'timeout%':>8}"
    )
    print(col)
    print("-" * len(col))
    for label, step, m in rows:
        step_str = _fmt_step(step) if step is not None else "-"
        print(
            f"{label:<{_COL_W}}  {step_str:>7}  "
            f"{m['win_rate']:>5.1%}  "
            f"{m['mean_turns']:>6.1f}  "
            f"{m['mean_turns_on_win']:>8.1f}  "
            f"{m['mean_turns_on_loss']:>8.1f}  "
            f"{m['mean_hideout_uncert']:>9.2f}  "
            f"{m['mean_min_cop_dist']:>7.2f}  "
            f"{m['arrest_rate']:>6.1%}  "
            f"{m['timeout_rate']:>7.1%}"
        )
    print()
    # copdist / arrest% / timeout% exist to test the risk-aversion hypothesis in
    # 09-director-tuning.md: a Director-trained Jack that lost by being cautious
    # shows a HIGHER copdist and shifts losses from arrest% to timeout%. Same
    # win rate with a different split is a different failure, not the same one.


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Jack policy checkpoints")
    p.add_argument("checkpoints", nargs="*", metavar="CHECKPOINT")
    p.add_argument("--n-games", type=int, default=200)
    p.add_argument("--map", default="maps/whitechapel.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-baseline", action="store_true", default=False)
    p.add_argument(
        "--held-out-cops",
        action="store_true",
        default=False,
        help="Also score against COPS_PRERETUNE_V1, cops no current policy "
        "trained on. The gap between the two tables is the generalisation gap — "
        "a policy that only beats its own training cops has memorised a "
        "decision rule rather than learned to evade",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game_map = load_map(args.map)

    # Evaluate each checkpoint in order of step count
    entries: list[tuple[str, Agent, int]] = []
    for path in args.checkpoints:
        agent, step = load_checkpoint(path, device)
        entries.append((path, agent, step))
    entries.sort(key=lambda x: x[2])

    if not entries:
        print("No checkpoints specified. Pass at least one .pt path.")
        return

    cop_sets = [("COPS_STUDY_V2 (frozen - the study cops)", None)]
    if args.held_out_cops:
        cop_sets.append(
            (
                "COPS_PRERETUNE_V1 (held out - no current policy trained on these)",
                dict(COPS_PRERETUNE_V1),
            )
        )

    for label, cop_params in cop_sets:
        rows: list[tuple[str, int | None, dict[str, float]]] = []
        for path, agent, step in entries:
            print(
                f"  evaluating {path} (step {step:,}) vs {label.split()[0]} ...",
                flush=True,
            )
            rng = random.Random(args.seed)
            rows.append(
                (
                    path,
                    step,
                    eval_policy(agent, game_map, args.n_games, device, rng, cop_params),
                )
            )

        if not args.no_baseline:
            rng = random.Random(args.seed)
            rows.append(
                (
                    "[random]",
                    None,
                    eval_agent(
                        RandomJack(rng=rng), game_map, args.n_games, rng, cop_params
                    ),
                )
            )

        _print_table(rows, args.n_games, args.map, cops_label=label)

    if args.held_out_cops:
        print("The gap between the two tables is the generalisation gap. A policy")
        print("that scores well only in the first has learned to exploit one cop")
        print("decision rule rather than to evade cops. See COPS_PRERETUNE_V1.\n")


if __name__ == "__main__":
    main()
