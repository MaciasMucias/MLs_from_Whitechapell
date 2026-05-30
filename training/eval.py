"""
Evaluate trained Jack policies against full-strength heuristic cops (no Director).

Core function eval_policy() returns a plain dict — can be called from the
training loop to log per-checkpoint metrics to wandb, or used standalone.

Usage:
    uv run python -m training.eval checkpoints/run/agent_final.pt
    uv run python -m training.eval checkpoints/run/*.pt --n-games 500 --seed 42
    uv run python -m training.eval ckpt1.pt ckpt2.pt --no-baseline
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from agents.base import AgentOutput, JackAgent
from agents.heuristic_cops import HeuristicCops
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


def eval_agent(
    jack_agent: JackAgent,
    game_map: Map,
    n_games: int,
    rng: random.Random | None = None,
) -> dict[str, float]:
    """
    Run n_games against full-strength HeuristicCops with no Director.
    Returns a plain dict suitable for console printing or wandb logging.
    """
    rng = rng or random.Random()
    cops = HeuristicCops()

    wins = 0
    turns_all: list[int] = []
    turns_on_win: list[int] = []
    turns_on_loss: list[int] = []
    hideout_uncerts: list[float] = []

    for _ in range(n_games):
        record = run_game(game_map, jack_agent, cops, director=None, rng=rng)
        t = record.turns_survived
        turns_all.append(t)
        if record.winner == "jack":
            wins += 1
            turns_on_win.append(t)
            final = record.history[-1].state_after_round
            pmf = HeuristicCops.compute_pmf(final, game_map)
            hideout_uncerts.append(hideout_uncertainty(final.hideout_zone, pmf))
        else:
            turns_on_loss.append(t)

    return {
        "win_rate": wins / n_games,
        "mean_turns": sum(turns_all) / n_games,
        "mean_turns_on_win": sum(turns_on_win) / max(len(turns_on_win), 1),
        "mean_turns_on_loss": sum(turns_on_loss) / max(len(turns_on_loss), 1),
        "mean_hideout_uncert": sum(hideout_uncerts) / max(len(hideout_uncerts), 1),
        "n_wins": float(wins),
        "n_games": float(n_games),
    }


def eval_policy(
    agent: Agent,
    game_map: Map,
    n_games: int,
    device: torch.device,
    rng: random.Random | None = None,
) -> dict[str, float]:
    """Evaluate a trained Agent. Thin wrapper around eval_agent."""
    jack = PolicyAgent(agent, game_map, device)
    return eval_agent(jack, game_map, n_games, rng)


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
    rows: list[tuple[str, int | None, dict[str, float]]], n_games: int, map_path: str
) -> None:
    header = f"Eval  {n_games} games | stochastic | no director | {map_path}"
    print()
    print(header)
    print()
    col = f"{'checkpoint':<{_COL_W}}  {'step':>7}  {'win%':>6}  {'turns':>6}  {'turns(W)':>8}  {'turns(L)':>8}  {'hideout_u':>9}"
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
            f"{m['mean_hideout_uncert']:>9.2f}"
        )
    print()


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
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game_map = load_map(args.map)

    rows: list[tuple[str, int | None, dict[str, float]]] = []

    # Evaluate each checkpoint in order of step count
    entries: list[tuple[str, Agent, int]] = []
    for path in args.checkpoints:
        agent, step = load_checkpoint(path, device)
        entries.append((path, agent, step))
    entries.sort(key=lambda x: x[2])

    for path, agent, step in entries:
        print(f"  evaluating {path} (step {step:,}) ...", flush=True)
        rng = random.Random(args.seed)
        m = eval_policy(agent, game_map, args.n_games, device, rng)
        rows.append((path, step, m))

    if not args.no_baseline:
        print(f"  evaluating [random baseline] ...", flush=True)
        rng = random.Random(args.seed)
        m = eval_agent(RandomJack(rng=rng), game_map, args.n_games, rng)
        rows.append(("[random]", None, m))

    if not rows:
        print("No checkpoints specified. Pass at least one .pt path.")
        return

    _print_table(rows, args.n_games, args.map)


if __name__ == "__main__":
    main()
