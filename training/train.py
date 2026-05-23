"""
PPO training loop for Jack's RL policy (CleanRL-style, single-file).

Usage:
    uv run python -m training.train
    uv run python -m training.train --total-steps 10_000_000 --n-envs 16
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import time
from collections import deque
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Categorical
from torch.optim import Adam

from agents.curriculum_director import INITIAL_DIFFICULTY
from engine.graph import load_map
from training.env import JackEnv
from training.eval import eval_policy
from training.model import Agent


# ---------------------------------------------------------------------------
# Worker process  (must be at module level for Windows spawn)
# ---------------------------------------------------------------------------


def _worker_fn(
    conn: Connection,
    map_path: str,
    seeds: list[int],
    use_curriculum: bool = False,
    initial_difficulty: float = INITIAL_DIFFICULTY,
) -> None:
    """
    Env worker. Owns len(seeds) independent JackEnv instances and steps them
    sequentially per message. Loads its own map copy (Map is not picklable).

    Auto-resets each env on termination; the reset obs/mask are returned in
    the same message so the main process never needs a separate round-trip.

    Protocol:
        recv: ("step", [action, ...])      ->  send: [(obs, reward, term, trunc, info), ...]
        recv: ("reset",)                   ->  send: [(obs, info), ...]
        recv: ("set_difficulty", float)    ->  no response (fire-and-forget)
        recv: ("close",)                   ->  exit
    """
    game_map = load_map(map_path)
    if use_curriculum:
        from agents.curriculum_director import CurriculumDirector

        director = CurriculumDirector(
            initial_difficulty=initial_difficulty,
            rng=random.Random(seeds[0] + 10_000_000),
        )
        envs = [
            JackEnv(game_map, rng=random.Random(s), director=director) for s in seeds
        ]
    else:
        envs = [JackEnv(game_map, rng=random.Random(s)) for s in seeds]

    try:
        while True:
            msg = conn.recv()
            cmd = msg[0]

            if cmd == "reset":
                conn.send([env.reset() for env in envs])

            elif cmd == "step":
                actions = msg[1]
                results = []
                for env, action in zip(envs, actions):
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        reset_obs, reset_info = env.reset()
                        merged_info = {**info, "action_mask": reset_info["action_mask"]}
                        results.append(
                            (reset_obs, reward, terminated, truncated, merged_info)
                        )
                    else:
                        results.append((obs, reward, terminated, truncated, info))
                conn.send(results)

            elif cmd == "set_difficulty":
                new_d = msg[1]
                for env in envs:
                    env.set_director_difficulty(new_d)
                # fire-and-forget: no conn.send()

            elif cmd == "close":
                break

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Async vector env
# ---------------------------------------------------------------------------


class AsyncVectorJackEnv:
    """
    n_workers processes each owning (n_envs // n_workers) JackEnv instances.
    Workers step their envs sequentially; workers themselves run in parallel.

    This decouples episode diversity (n_envs) from process count (n_workers),
    letting you keep large batches while leaving CPU cores free.
    """

    def __init__(
        self,
        map_path: str,
        n_envs: int,
        n_workers: int,
        seed: int,
        use_curriculum: bool = False,
        initial_difficulty: float = INITIAL_DIFFICULTY,
    ) -> None:
        assert n_envs % n_workers == 0, "n_envs must be divisible by n_workers"
        self.n = n_envs
        self._n_workers = n_workers
        self._epw = n_envs // n_workers  # envs per worker
        ctx = mp.get_context("spawn")
        self._procs: list[mp.Process] = []
        self._conns: list[Connection] = []

        for i in range(n_workers):
            parent_conn, child_conn = ctx.Pipe()
            worker_seeds = [seed + i * self._epw + j for j in range(self._epw)]
            proc = ctx.Process(
                target=_worker_fn,
                args=(
                    child_conn,
                    map_path,
                    worker_seeds,
                    use_curriculum,
                    initial_difficulty,
                ),
                daemon=True,
            )
            proc.start()
            child_conn.close()  # parent must close its copy of the child end
            self._procs.append(proc)
            self._conns.append(parent_conn)

    def reset(self) -> tuple[np.ndarray, list[dict]]:
        for conn in self._conns:
            conn.send(("reset",))
        # Each worker returns [(obs, info), ...] for its envs
        flat = [item for conn in self._conns for item in conn.recv()]
        return np.stack([r[0] for r in flat]), [r[1] for r in flat]

    def step(
        self, actions: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        # Chunk actions across workers, send simultaneously
        for i, conn in enumerate(self._conns):
            conn.send(("step", actions[i * self._epw : (i + 1) * self._epw]))
        # Each worker returns [(obs, reward, term, trunc, info), ...] for its envs
        flat = [item for conn in self._conns for item in conn.recv()]
        obs = np.stack([r[0] for r in flat])
        rewards = np.array([r[1] for r in flat], dtype=np.float32)
        term = np.array([r[2] for r in flat], dtype=bool)
        trunc = np.array([r[3] for r in flat], dtype=bool)
        infos = [r[4] for r in flat]
        return obs, rewards, term, trunc, infos

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("close",))
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()

    def set_difficulty(self, value: float) -> None:
        """Broadcast new curriculum difficulty to all workers. Call between rollouts."""
        for conn in self._conns:
            conn.send(("set_difficulty", value))
        # fire-and-forget: no recv()

    def __enter__(self) -> AsyncVectorJackEnv:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Load map in main process only to read n_actions and obs_dim.
    # Workers load their own copies (Map is not picklable).
    game_map = load_map(args.map)
    n_actions = len(game_map.jack_nodes)
    sample_env = JackEnv(game_map)
    sample_obs, _ = sample_env.reset()
    obs_dim = sample_obs.shape[0]
    del sample_env
    print(
        f"obs_dim={obs_dim}  n_actions={n_actions}  n_envs={args.n_envs}  n_workers={args.n_workers}"
    )

    batch_size = args.n_steps * args.n_envs
    n_updates = args.total_steps // batch_size

    ckpt = None
    resume_step = 0
    start_update = 1
    wandb_resume_id = None
    curriculum_difficulty: float = INITIAL_DIFFICULTY
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        resume_step = ckpt["step"]
        start_update = resume_step // batch_size + 1
        wandb_resume_id = ckpt.get("wandb_run_id")
        curriculum_difficulty = ckpt.get("curriculum_difficulty", INITIAL_DIFFICULTY)
        print(
            f"Resumed from {args.resume} at step {resume_step:,} (update {start_update}/{n_updates})"
        )

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        mode=args.wandb_mode,
        config=vars(args),
        id=wandb_resume_id,
        resume="must" if wandb_resume_id else None,
    )
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")

    agent = Agent(obs_dim, n_actions).to(device)
    optimizer = Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    if ckpt is not None:
        agent.load_state_dict(ckpt["agent"])
        optimizer.load_state_dict(ckpt["optimizer"])

    # Rollout buffers - allocated once, reused every update
    b_obs = torch.zeros(args.n_steps, args.n_envs, obs_dim, device=device)
    b_actions = torch.zeros(args.n_steps, args.n_envs, device=device, dtype=torch.long)
    b_logprobs = torch.zeros(args.n_steps, args.n_envs, device=device)
    b_rewards = torch.zeros(args.n_steps, args.n_envs, device=device)
    b_dones = torch.zeros(args.n_steps, args.n_envs, device=device)
    b_values = torch.zeros(args.n_steps, args.n_envs, device=device)
    b_masks = torch.zeros(
        args.n_steps, args.n_envs, n_actions, device=device, dtype=torch.bool
    )

    with AsyncVectorJackEnv(
        args.map,
        args.n_envs,
        args.n_workers,
        args.seed,
        use_curriculum=not args.no_curriculum,
        initial_difficulty=curriculum_difficulty,
    ) as envs:
        obs_np, infos = envs.reset()
        obs = torch.from_numpy(obs_np).float().to(device)
        masks = torch.from_numpy(np.stack([i["action_mask"] for i in infos])).to(device)
        dones = torch.zeros(args.n_envs, device=device)

        ep_returns: deque[float] = deque(maxlen=100)
        ep_wins: deque[bool] = deque(maxlen=100)
        ep_return_buf = np.zeros(args.n_envs)
        ep_length_buf = np.zeros(args.n_envs, dtype=int)

        ckpt_dir = Path(args.checkpoint_dir) / (wandb.run.name or wandb.run.id)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        global_step = resume_step
        start_time = time.time()
        last_update_time = time.time()

        for update in range(start_update, n_updates + 1):
            # Linear LR annealing to 0 over training
            frac = 1.0 - (update - 1) / n_updates
            optimizer.param_groups[0]["lr"] = args.lr * frac

            # -- Rollout collection -----------------------------------------
            for step in range(args.n_steps):
                global_step += args.n_envs
                b_obs[step] = obs
                b_dones[step] = dones  # 1 = this obs came from a fresh reset
                b_masks[step] = masks

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(obs, masks)
                b_actions[step] = action
                b_logprobs[step] = logprob
                b_values[step] = value

                # All workers receive their action simultaneously, then we
                # collect results - this is where parallel CPU execution happens.
                obs_np, rewards, terminated, truncated, infos = envs.step(
                    action.cpu().tolist()
                )
                b_rewards[step] = torch.from_numpy(rewards).to(device)

                ep_return_buf += rewards
                ep_length_buf += 1

                done_np = terminated | truncated
                dones = torch.from_numpy(done_np.astype(np.float32)).to(device)
                # Workers auto-reset: obs_np[i] is already the reset obs when done
                obs = torch.from_numpy(obs_np).float().to(device)
                # info["action_mask"] is the reset mask when done, next-step mask otherwise
                masks = torch.from_numpy(
                    np.stack([info["action_mask"] for info in infos])
                ).to(device)

                for i, (term, trunc, info) in enumerate(
                    zip(terminated, truncated, infos)
                ):
                    if term or trunc:
                        ep_returns.append(float(ep_return_buf[i]))
                        ep_wins.append(info.get("winner") == "jack")
                        ep_return_buf[i] = 0.0
                        ep_length_buf[i] = 0

            # -- GAE advantage computation ----------------------------------
            with torch.no_grad():
                next_value = agent.get_value(obs)
                advantages = torch.zeros_like(b_rewards)
                last_gae = torch.zeros(args.n_envs, device=device)
                for t in reversed(range(args.n_steps)):
                    if t == args.n_steps - 1:
                        next_non_terminal = 1.0 - dones
                        next_val = next_value
                    else:
                        next_non_terminal = 1.0 - b_dones[t + 1]
                        next_val = b_values[t + 1]
                    delta = (
                        b_rewards[t]
                        + args.gamma * next_val * next_non_terminal
                        - b_values[t]
                    )
                    last_gae = (
                        delta
                        + args.gamma * args.gae_lambda * next_non_terminal * last_gae
                    )
                    advantages[t] = last_gae
                returns = advantages + b_values

            # -- PPO minibatch updates -------------------------------------
            flat_obs = b_obs.view(-1, obs_dim)
            flat_actions = b_actions.view(-1)
            flat_logprobs = b_logprobs.view(-1)
            flat_advantages = advantages.view(-1)
            flat_returns = returns.view(-1)
            flat_masks = b_masks.view(-1, n_actions)

            clip_fracs: list[float] = []
            pg_losses: list[float] = []
            vf_losses: list[float] = []
            ent_losses: list[float] = []

            for _ in range(args.n_epochs):
                perm = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, args.minibatch_size):
                    mb = perm[start : start + args.minibatch_size]

                    mb_adv = flat_advantages[mb]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    _, new_logprob, entropy, new_value = agent.get_action_and_value(
                        flat_obs[mb], flat_masks[mb], flat_actions[mb]
                    )

                    ratio = torch.exp(new_logprob - flat_logprobs[mb])
                    clip_fracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                    pg_loss = torch.max(
                        -mb_adv * ratio,
                        -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef),
                    ).mean()
                    vf_loss = F.mse_loss(new_value, flat_returns[mb])
                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss + args.vf_coef * vf_loss - args.ent_coef * entropy_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                    pg_losses.append(pg_loss.item())
                    vf_losses.append(vf_loss.item())
                    ent_losses.append(entropy_loss.item())

            # -- Logging ---------------------------------------------------
            now = time.time()
            sps = int((global_step - resume_step) / (now - start_time))
            instant_sps = int(batch_size / (now - last_update_time))
            last_update_time = now
            recent_ret = ep_returns
            recent_wins = ep_wins
            mean_return = sum(recent_ret) / len(recent_ret) if recent_ret else 0.0
            win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0.0
            mean_pg = sum(pg_losses) / len(pg_losses)
            mean_vf = sum(vf_losses) / len(vf_losses)
            mean_ent = sum(ent_losses) / len(ent_losses)
            mean_clip = sum(clip_fracs) / len(clip_fracs)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"update={update}/{n_updates} "
                f"steps={global_step:,} "
                f"sps={sps} "
                f"instant_sps={instant_sps} "
                f"episodes={len(ep_returns)} "
                f"return={mean_return:.3f} "
                f"win_rate={win_rate:.3f} "
                f"difficulty={curriculum_difficulty:+.3f} "
                f"pg={mean_pg:.4f} "
                f"vf={mean_vf:.4f} "
                f"ent={mean_ent:.4f} "
                f"clip_frac={mean_clip:.3f} "
                f"lr={current_lr:.2e}"
            )
            wandb.log(
                {
                    "charts/win_rate": win_rate,
                    "charts/mean_return": mean_return,
                    "charts/episodes": len(ep_returns),
                    "charts/sps": sps,
                    "charts/instant_sps": instant_sps,
                    "losses/policy": mean_pg,
                    "losses/value": mean_vf,
                    "losses/entropy": mean_ent,
                    "losses/clip_frac": mean_clip,
                    "train/lr": current_lr,
                    "curriculum/difficulty": curriculum_difficulty,
                    "global_step": global_step,
                },
            )

            # -- Curriculum P-controller update --------------------------------
            if not args.no_curriculum and len(ep_wins) >= 10:
                target_centre = (
                    args.curriculum_target_low + args.curriculum_target_high
                ) / 2.0
                if (
                    win_rate < args.curriculum_target_low
                    or win_rate > args.curriculum_target_high
                ):
                    error = win_rate - target_centre
                    curriculum_difficulty = max(
                        -1.0,
                        min(1.0, curriculum_difficulty + args.curriculum_kp * error),
                    )
                    envs.set_difficulty(curriculum_difficulty)

            if update % 50 == 0 or update == n_updates:
                ckpt_path = ckpt_dir / f"agent_{global_step:010d}.pt"
                torch.save(
                    {
                        "agent": agent.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": global_step,
                        "obs_dim": obs_dim,
                        "n_actions": n_actions,
                        "wandb_run_id": wandb.run.id,
                        "curriculum_difficulty": curriculum_difficulty,
                    },
                    ckpt_path,
                )
                print(f"  checkpoint -> {ckpt_path}")
                if args.eval_games > 0:
                    agent.eval()
                    eval_results = eval_policy(
                        agent,
                        game_map,
                        args.eval_games,
                        device,
                        rng=random.Random(args.seed),
                    )
                    agent.train()
                    print(
                        f"  eval: win_rate={eval_results['win_rate']:.1%} "
                        f"turns={eval_results['mean_turns']:.1f} "
                        f"turns(W)={eval_results['mean_turns_on_win']:.1f} "
                        f"hideout_u={eval_results['mean_hideout_uncert']:.2f}"
                    )
                    wandb.log(
                        {
                            "eval/win_rate": eval_results["win_rate"],
                            "eval/mean_turns": eval_results["mean_turns"],
                            "eval/mean_turns_on_win": eval_results["mean_turns_on_win"],
                            "eval/mean_turns_on_loss": eval_results[
                                "mean_turns_on_loss"
                            ],
                            "eval/mean_hideout_uncert": eval_results[
                                "mean_hideout_uncert"
                            ],
                            "global_step": global_step,
                        }
                    )

    wandb.finish()
    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO training for Jack RL policy")
    p.add_argument("--map", default="maps/whitechapel.json")
    p.add_argument("--total-steps", type=int, default=5_000_000)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--n-envs", type=int, default=12)
    p.add_argument("--n-workers", type=int, default=6)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=27)
    p.add_argument("--checkpoint-dir", default="checkpoints/")
    p.add_argument("--resume", default=None, metavar="CHECKPOINT")
    p.add_argument(
        "--no-curriculum",
        action="store_true",
        default=False,
        help="Disable curriculum Director; all envs use NoOpDirector",
    )
    p.add_argument(
        "--curriculum-kp",
        type=float,
        default=0.1,
        help="Proportional gain for difficulty adjustment",
    )
    p.add_argument("--curriculum-target-low", type=float, default=0.4)
    p.add_argument("--curriculum-target-high", type=float, default=0.6)
    p.add_argument(
        "--eval-games",
        type=int,
        default=200,
        help="Games per eval run on checkpoint save (0 to disable)",
    )
    p.add_argument("--wandb-project", default="mls-from-whitechapel")
    p.add_argument("--wandb-run", default=None)
    p.add_argument(
        "--wandb-mode", default="offline", choices=["offline", "online", "disabled"]
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
