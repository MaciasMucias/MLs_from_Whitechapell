"""
Run one game per course map with the latest checkpoint and save replays.
Usage: uv run python tools/run_and_save_replays.py
"""

from __future__ import annotations

import random
import types
import uuid

import torch

from agents.heuristic_cops import HeuristicCops
from engine.game import StepContext, run_game
from engine.graph import load_map
from server.replay import build_replay, save_replay
from training.eval import PolicyAgent, load_checkpoint

CHECKPOINT = "checkpoints/sparse-copdist-15m-low-entropy/agent_0019998720.pt"
MAPS = [
    ("maps/course_1.json", "course_1"),
    ("maps/course_2.json", "course_2"),
    ("maps/course_3.json", "course_3"),
]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, step = load_checkpoint(CHECKPOINT, device)
    print(f"Loaded checkpoint step={step:,} on {device}\n")

    for map_path, map_name in MAPS:
        game_map = load_map(map_path)
        jack = PolicyAgent(agent, game_map, device)
        cops = HeuristicCops()
        rng = random.Random()

        record = run_game(game_map, jack, cops, director=None, rng=rng)

        last_state = (
            record.history[-1].state_after_round
            if record.history
            else record.initial_state
        )
        ctx = StepContext(
            game_map=game_map,
            state=last_state,
            terminated=True,
            winner=record.winner,
            history=record.history,
            blocking=False,
            turn_limit=None,
        )
        session = types.SimpleNamespace(
            ctx=ctx,
            game_id=str(uuid.uuid4()),
            map_name=map_name,
        )

        replay = build_replay(session)
        slot = save_replay(replay)
        print(
            f"{map_name}: winner={record.winner}, turns={record.turns_survived}, saved to slot {slot}"
        )


if __name__ == "__main__":
    main()
