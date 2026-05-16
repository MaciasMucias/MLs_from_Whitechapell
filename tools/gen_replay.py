"""
Generate a replay file from a trained Jack checkpoint.

Usage:
    uv run tools/gen_replay.py
    uv run tools/gen_replay.py --checkpoint checkpoints/agent_0002611200.pt
    uv run tools/gen_replay.py --seed 42 --greedy
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import uuid
from pathlib import Path

# Ensure the project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from agents.base import AgentOutput, JackAgent
from agents.heuristic_cops import HeuristicCops
from agents.random_agents import NoOpDirector
from engine.game import GameRecord, run_game
from engine.graph import Map, load_map
from engine.state import GameState
from server.replay import build_replay, save_replay
from training.obs import build_obs, precompute_distances
from training.train import Agent


# ---------------------------------------------------------------------------
# Trained Jack wrapper
# ---------------------------------------------------------------------------


class TrainedJack(JackAgent):
    def __init__(
        self,
        agent: Agent,
        all_dists: dict[int, dict[int, int]],
        diameter: int,
        n_actions: int,
        greedy: bool = False,
    ) -> None:
        self._agent = agent
        self._all_dists = all_dists
        self._diameter = diameter
        self._n_actions = n_actions
        self._greedy = greedy

    def act(self, state: GameState, legal_edges, game_map: Map) -> AgentOutput:
        obs = build_obs(state, game_map, self._all_dists, self._diameter)
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)

        mask = np.zeros(self._n_actions, dtype=bool)
        for e in legal_edges:
            mask[e.destination.id] = True
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        with torch.no_grad():
            if self._greedy:
                features = self._agent.trunk(obs_t)
                logits = self._agent.policy_head(features)
                logits = logits.masked_fill(~mask_t, float("-inf"))
                action = logits.argmax(dim=-1)
                logprob_val = 0.0
                value_val = self._agent.value_head(features).squeeze(-1).item()
            else:
                action, logprob, _, value = self._agent.get_action_and_value(
                    obs_t, mask_t
                )
                logprob_val = logprob.item()
                value_val = value.item()

        action_id = action.item()
        edge = next(e for e in legal_edges if e.destination.id == action_id)
        return AgentOutput(edge=edge, logprob=logprob_val, value=value_val)


# ---------------------------------------------------------------------------
# Shim so build_replay works without a live GameSession
# ---------------------------------------------------------------------------


class _Ctx:
    def __init__(self, record: GameRecord) -> None:
        self.game_map = record.game_map
        self.history = record.history
        self.blocking = False
        self.turn_limit = None
        self.state = record.initial_state
        self.winner = record.winner


class _Session:
    def __init__(self, record: GameRecord, game_id: str) -> None:
        self.ctx = _Ctx(record)
        self.game_id = game_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latest_checkpoint(checkpoint_dir: str) -> Path:
    ckpts = sorted(Path(checkpoint_dir).glob("agent_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return ckpts[-1]


def _pmf_entropy(pmf: dict) -> float:
    total = sum(pmf.values())
    if total == 0:
        return 0.0
    return -sum((p / total) * math.log(p / total + 1e-12) for p in pmf.values())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else _latest_checkpoint(args.checkpoint_dir)
    )
    print(f"checkpoint : {ckpt_path}")

    game_map = load_map(args.map)
    print(
        f"map        : {args.map}  ({len(game_map.jack_nodes)} jack nodes, turn_limit={game_map.turn_limit})"
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    obs_dim = ckpt["obs_dim"]
    n_actions = ckpt["n_actions"]
    agent = Agent(obs_dim, n_actions)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    print(
        f"agent      : step={ckpt['step']:,}  obs_dim={obs_dim}  n_actions={n_actions}"
    )
    print(f"mode       : {'greedy (argmax)' if args.greedy else 'stochastic (sample)'}")
    print()

    all_dists, diameter = precompute_distances(game_map)
    rng = random.Random(args.seed)

    jack = TrainedJack(agent, all_dists, diameter, n_actions, greedy=args.greedy)
    cops = HeuristicCops(rng=random.Random(rng.randint(0, 2**31)))
    director = NoOpDirector()

    record = run_game(game_map, jack, cops, director, rng=rng)

    # Print turn-by-turn summary
    for rr in record.history:
        decisions = rr.cop_decisions
        pmf_ent = (
            f"  PMF_H={_pmf_entropy(decisions.position_pmf):.2f}" if decisions else ""
        )
        hits_total = sum(
            len([n for n, hit in s.search_results.items() if hit]) for s in rr.cop_steps
        )
        jack_from = rr.state_before.jack_pos
        jack_to = rr.state_after_jack.jack_pos
        print(
            f"  turn {rr.turn:2d}: jack {jack_from}->{jack_to}"
            f"  cop_hits={hits_total}"
            f"{pmf_ent}" + (f"  [{rr.winner} wins]" if rr.terminated else "")
        )

    print()
    print(f"Result: {record.winner} wins in {record.turns_survived} turn(s)")
    print(f"hideout: {record.initial_state.hideout}")

    game_id = uuid.uuid4().hex[:8]
    replay = build_replay(_Session(record, game_id))
    slot = save_replay(replay)
    print(f"Replay saved to slot {slot}  (game_id={game_id})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a replay from a trained Jack checkpoint"
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Path to .pt checkpoint (default: latest)",
    )
    p.add_argument("--checkpoint-dir", default="checkpoints", metavar="DIR")
    p.add_argument("--map", default="maps/whitechapel.json")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--greedy", action="store_true", help="Take argmax action instead of sampling"
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
