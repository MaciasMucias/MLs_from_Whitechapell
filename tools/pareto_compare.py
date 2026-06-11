"""
Compare Pareto-front cop configs by conditional win rate per turn.

Tests whether high-win-rate configs are fragile (only win via early arrest)
versus configs that maintain effectiveness throughout the game.

For each turn T, reports:
  P(@T)       = P(cops win on exactly turn T | Jack alive at turn T)
  P(>=T|alive) = P(cops win eventually      | Jack alive at turn T)

A steep drop in P(>=T|alive) as T increases → config relies on early arrests.

Usage:
    uv run tools/pareto_compare.py --configs configs.json
    uv run tools/pareto_compare.py --configs configs.json --pool 200 --seed 42
    uv run tools/pareto_compare.py --configs configs.json --jack-checkpoint checkpoints/8dgaqm9w

configs.json format — list of named cop param dicts:
    [
      {
        "name": "aggressive (3966)",
        "arrest_threshold": 0.063,
        "min_arrest_fraction": 0.477,
        "pursuit_fraction": 0.285,
        "pursuit_weight": 1.173,
        "searcher_prox_fraction": 0.168,
        "direction_certainty_threshold": 0.545,
        "arrest_discount": 0.480,
        "miss_discount_decay": 0.774,
        "hideout_blend_floor": 0.076,
        "hideout_blend": 0.093,
        "max_passes": 4
      },
      { "name": "knee (1979)", ... }
    ]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.graph import load_map
from tools.optuna_tune import make_live_pool, make_scripted_pool, resolve_checkpoint
from tools.scripted_sim import run_policy_game, run_scripted_game

MAP_PATH = "maps/whitechapel.json"
POOL_SEED = 42


def _run_config(
    pool: list[dict], params: dict, game_map, jack_agent=None
) -> list[tuple[str | None, int]]:
    if jack_agent is not None:
        runner = lambda game: run_policy_game(
            **game, cop_params=params, game_map=game_map, jack_agent=jack_agent
        )
    else:
        runner = lambda game: run_scripted_game(
            **game, cop_params=params, game_map=game_map
        )
    return [(r["winner"], r["turns_survived"]) for game in pool for r in [runner(game)]]


def _print_summary(configs: list[dict], all_results: list[list[tuple]]) -> None:
    print(
        f"\n{'Config':<30}  {'win%':>6}  {'mean_win_turn':>13}  {'cop_wins':>8}  {'jack_wins':>9}"
    )
    print("-" * 74)
    for cfg, results in zip(configs, all_results):
        cop_wins = [t for w, t in results if w == "cops"]
        jack_wins = len(results) - len(cop_wins)
        win_rate = len(cop_wins) / len(results)
        mean_wt = sum(cop_wins) / len(cop_wins) if cop_wins else float("nan")
        print(
            f"{cfg['name']:<30}  {win_rate * 100:>5.1f}%  {mean_wt:>13.2f}"
            f"  {len(cop_wins):>8}  {jack_wins:>9}"
        )


def _print_conditional(
    configs: list[dict], all_results: list[list[tuple]], turn_limit: int
) -> None:
    names = [cfg["name"] for cfg in configs]

    print(f"\nConditional win rates by turn:")
    name_header = f"{'':>4}"
    col_header = f"{'turn':>4}"
    divider = f"{'----':>4}"
    for name in names:
        trunc = name[:19]
        name_header += f"  {trunc:<21}"
        col_header += f"  {'P(@T)':>7}  {'P(>=T|alive)':>12}"
        divider += f"  {'-------':>7}  {'------------':>12}"
    print(name_header)
    print(col_header)
    print(divider)

    for T in range(1, turn_limit + 1):
        row = f"{T:>4}"
        any_alive = False
        for results in all_results:
            alive = [(w, t) for w, t in results if t >= T]
            if not alive:
                row += f"  {'---':>7}  {'---':>12}"
                continue
            any_alive = True
            wins_at = sum(1 for w, t in alive if w == "cops" and t == T)
            wins_from = sum(1 for w, t in alive if w == "cops")
            row += f"  {wins_at / len(alive):>7.3f}  {wins_from / len(alive):>12.3f}"
        print(row)
        if not any_alive:
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        required=True,
        help="JSON file: list of {name, ...cop_params} dicts",
    )
    parser.add_argument("--pool", type=int, default=200)
    parser.add_argument("--seed", type=int, default=POOL_SEED)
    parser.add_argument(
        "--jack-checkpoint",
        default=None,
        help="Path to a .pt checkpoint or directory; omit to use random Jack",
    )
    args = parser.parse_args()

    configs: list[dict] = json.loads(Path(args.configs).read_text())

    print("Loading map...")
    game_map = load_map(MAP_PATH)

    jack_agent = None
    if args.jack_checkpoint:
        import torch
        from training.eval import PolicyAgent, load_checkpoint

        ckpt_path = resolve_checkpoint(args.jack_checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading checkpoint {ckpt_path.name} on {device}...")
        agent_model, step = load_checkpoint(str(ckpt_path), device)
        jack_agent = PolicyAgent(agent_model, game_map, device)
        print(f"  trained for {step:,} steps")

    if jack_agent is not None:
        print(
            f"Generating live pool of {args.pool} initial states (seed={args.seed})..."
        )
        pool = make_live_pool(game_map, seed=args.seed, size=args.pool)
        mode = "policy Jack"
    else:
        print(f"Generating scripted pool of {args.pool} games (seed={args.seed})...")
        pool = make_scripted_pool(game_map, seed=args.seed, size=args.pool)
        mode = "random Jack"
    print(f"Mode: {mode}\n")

    all_results = []
    for cfg in configs:
        params = {k: v for k, v in cfg.items() if k != "name"}
        print(f"Running '{cfg['name']}'...")
        all_results.append(_run_config(pool, params, game_map, jack_agent))

    _print_summary(configs, all_results)
    _print_conditional(configs, all_results, game_map.turn_limit)


if __name__ == "__main__":
    main()
