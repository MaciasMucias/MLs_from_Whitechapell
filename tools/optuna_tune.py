"""
Optimise cop heuristic parameters with Optuna against a fixed game pool.

Multi-objective: maximise both mean search hits and cop win rate (Pareto front).

Searches all 11 float/int parameters of HeuristicCops:
    arrest_threshold, min_arrest_fraction, pursuit_fraction, pursuit_weight,
    searcher_prox_fraction, direction_certainty_threshold, arrest_discount,
    miss_discount_decay, hideout_blend_floor, hideout_blend (via delta), max_passes.

Pool games are generated once with a fixed seed (common random numbers).

With --jack-checkpoint: Jack plays a trained policy and reacts to each cop
config.  Pool stores only initial states; full live games run per trial.

Without --jack-checkpoint: Jack plays random moves; pool stores pre-generated
scripts (original behaviour).

Usage:
    uv run tools/optuna_tune.py
    uv run tools/optuna_tune.py --trials 150 --pool 100 --jack-checkpoint checkpoints/8dgaqm9w
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from engine.env import make_initial_state
from engine.graph import load_map
from tools.scripted_sim import run_policy_game, run_scripted_game

MAP_PATH = "maps/whitechapel.json"
POOL_SEED = 42

FLOAT_PARAMS = {
    "arrest_threshold": (0.05, 0.6),
    "min_arrest_fraction": (0.3, 1.0),
    "pursuit_fraction": (0.2, 0.6),
    "pursuit_weight": (0.1, 1.2),
    "searcher_prox_fraction": (0.1, 0.8),
    "direction_certainty_threshold": (0.05, 0.6),
    "arrest_discount": (0.0, 0.5),
    "miss_discount_decay": (0.1, 1.0),
    "hideout_blend_floor": (0.0, 0.5),
    # hideout_blend = hideout_blend_floor + delta, guaranteeing blend >= floor.
    "hideout_blend_delta": (0.0, 0.5),
}

INT_PARAMS = {
    "max_passes": (1, 10),
}


# ---------------------------------------------------------------------------
# Pool generation
# ---------------------------------------------------------------------------


def make_scripted_pool(game_map, seed: int, size: int) -> list[dict]:
    """Pool with pre-generated random Jack scripts (no checkpoint needed)."""
    rng = random.Random(seed)
    pool = []
    while len(pool) < size:
        state = make_initial_state(game_map, rng)
        jack_script: list[int] = []
        pos = state.jack_pos
        for _ in range(game_map.turn_limit):
            pos = rng.choice(game_map.jack_nodes[pos].edges).destination.id
            jack_script.append(pos)
            if pos == state.hideout:
                break
        pool.append(
            dict(
                initial_jack_pos=state.jack_pos,
                initial_cop_positions=state.cop_positions,
                hideout=state.hideout,
                hideout_zone_anchor=state.hideout_zone_anchor,
                hideout_zone=state.hideout_zone,
                turn_limit=game_map.turn_limit,
                blocking=False,
                jack_script=jack_script,
            )
        )
    return pool


def make_live_pool(game_map, seed: int, size: int) -> list[dict]:
    """Pool with initial states only; Jack moves are decided per-trial."""
    rng = random.Random(seed)
    pool = []
    while len(pool) < size:
        state = make_initial_state(game_map, rng)
        pool.append(
            dict(
                initial_jack_pos=state.jack_pos,
                initial_cop_positions=state.cop_positions,
                hideout=state.hideout,
                hideout_zone_anchor=state.hideout_zone_anchor,
                hideout_zone=state.hideout_zone,
                turn_limit=game_map.turn_limit,
                blocking=False,
            )
        )
    return pool


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


def _suggest_params(trial: optuna.Trial) -> dict:
    params: dict = {
        name: trial.suggest_float(name, lo, hi)
        for name, (lo, hi) in FLOAT_PARAMS.items()
    }
    params.update(
        {name: trial.suggest_int(name, lo, hi) for name, (lo, hi) in INT_PARAMS.items()}
    )
    params["hideout_blend"] = (
        params.pop("hideout_blend_delta") + params["hideout_blend_floor"]
    )
    return params


def build_scripted_objective(pool: list[dict], game_map):
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        params = _suggest_params(trial)
        total_hits = cop_wins = 0
        for game in pool:
            r = run_scripted_game(**game, cop_params=params, game_map=game_map)
            total_hits += r["search_hits_total"]
            if r["winner"] == "cops":
                cop_wins += 1
        return total_hits / len(pool), cop_wins / len(pool)

    return objective


def build_policy_objective(pool: list[dict], game_map, jack_agent):
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        params = _suggest_params(trial)
        total_hits = cop_wins = 0
        for game in pool:
            r = run_policy_game(
                **game, cop_params=params, game_map=game_map, jack_agent=jack_agent
            )
            total_hits += r["search_hits_total"]
            if r["winner"] == "cops":
                cop_wins += 1
        return total_hits / len(pool), cop_wins / len(pool)

    return objective


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def resolve_checkpoint(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_dir():
        pts = sorted(p.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No .pt files in {p}")
        return pts[-1]
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--pool", type=int, default=100)
    parser.add_argument("--seed", type=int, default=POOL_SEED)
    parser.add_argument(
        "--jack-checkpoint",
        default=None,
        help="Path to a .pt checkpoint file or directory (uses latest .pt); "
        "omit to use random Jack",
    )
    args = parser.parse_args()

    print("Loading map...")
    game_map = load_map(MAP_PATH)

    if args.jack_checkpoint:
        import torch
        from training.eval import PolicyAgent, load_checkpoint

        ckpt_path = resolve_checkpoint(args.jack_checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading checkpoint {ckpt_path.name} on {device}...")
        agent_model, step = load_checkpoint(str(ckpt_path), device)
        jack_agent = PolicyAgent(agent_model, game_map, device)
        print(f"  trained for {step:,} steps")

        print(
            f"Generating live pool of {args.pool} initial states (seed={args.seed})..."
        )
        pool = make_live_pool(game_map, seed=args.seed, size=args.pool)
        objective_fn = build_policy_objective(pool, game_map, jack_agent)
        mode = "policy Jack"
    else:
        print(f"Generating scripted pool of {args.pool} games (seed={args.seed})...")
        pool = make_scripted_pool(game_map, seed=args.seed, size=args.pool)
        objective_fn = build_scripted_objective(pool, game_map)
        mode = "random Jack"

    print(
        f"Running {args.trials} trials [{mode}, multi-objective: hits + win rate]...\n"
    )
    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.NSGAIISampler(seed=args.seed),
    )
    study.optimize(objective_fn, n_trials=args.trials, show_progress_bar=True)

    pareto = sorted(
        study.best_trials,
        key=lambda t: (t.values[1], t.values[0]),
        reverse=True,
    )

    print(f"\nPareto front ({len(pareto)} trials) — sorted by win rate then hits:")
    print(
        f"{'#':>4}  {'hits':>6}  {'win%':>6}"
        f"  {'a_thr':>5}  {'ma_fr':>5}  {'p_fr':>5}  {'p_wt':>5}"
        f"  {'s_prx':>5}  {'cert':>5}  {'a_dis':>5}  {'m_dec':>5}"
        f"  {'h_fl':>5}  {'h_bl':>5}  {'pass':>4}"
    )
    print("-" * 100)
    for t in pareto:
        hits, win_rate = t.values
        p = t.params
        h_blend = p["hideout_blend_floor"] + p.get("hideout_blend_delta", 0.0)
        print(
            f"{t.number:>4}  {hits:>6.3f}  {win_rate * 100:>5.1f}%"
            f"  {p['arrest_threshold']:>5.3f}"
            f"  {p['min_arrest_fraction']:>5.3f}"
            f"  {p['pursuit_fraction']:>5.3f}"
            f"  {p['pursuit_weight']:>5.3f}"
            f"  {p['searcher_prox_fraction']:>5.3f}"
            f"  {p['direction_certainty_threshold']:>5.3f}"
            f"  {p['arrest_discount']:>5.3f}"
            f"  {p['miss_discount_decay']:>5.3f}"
            f"  {p['hideout_blend_floor']:>5.3f}"
            f"  {h_blend:>5.3f}"
            f"  {p['max_passes']:>4}"
        )


if __name__ == "__main__":
    main()
