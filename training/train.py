"""
PPO training loop for Jack's RL policy (CleanRL-style, single-file).

Usage:
    uv run training/train.py
    uv run training/train.py --total-steps 10_000_000 --n-envs 16
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from engine.graph import Map, load_map
from training.env import JackEnv


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Agent(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, n_actions)
        self.value_head = nn.Linear(256, 1)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.trunk(x)).squeeze(-1)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        logits = self.policy_head(features)
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        dist = Categorical(logits=masked_logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        # entropy: 0 * log(0) = nan → treat as 0 (illegal actions contribute nothing)
        entropy = dist.entropy().nan_to_num(0.0)
        value = self.value_head(features).squeeze(-1)
        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# Synchronous vector env
# ---------------------------------------------------------------------------

class SyncVectorJackEnv:
    """Thin wrapper holding n independent JackEnv instances."""

    def __init__(self, envs: list[JackEnv]) -> None:
        self._envs = envs
        self.n = len(envs)

    def reset(self) -> tuple[np.ndarray, list[dict]]:
        results = [e.reset() for e in self._envs]
        return np.stack([r[0] for r in results]), [r[1] for r in results]

    def step(
        self, actions: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        results = [e.step(a) for e, a in zip(self._envs, actions)]
        obs = np.stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        terminated = np.array([r[2] for r in results], dtype=bool)
        truncated = np.array([r[3] for r in results], dtype=bool)
        infos = [r[4] for r in results]
        return obs, rewards, terminated, truncated, infos

    def reset_one(self, i: int) -> tuple[np.ndarray, dict]:
        return self._envs[i].reset()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_envs(game_map: Map, n_envs: int, seed: int) -> SyncVectorJackEnv:
    return SyncVectorJackEnv([
        JackEnv(game_map, rng=random.Random(seed + i))
        for i in range(n_envs)
    ])


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    game_map = load_map(args.map)
    n_actions = len(game_map.jack_nodes)

    envs = make_envs(game_map, args.n_envs, args.seed)

    # Infer obs_dim from a live reset
    sample_obs, _ = envs._envs[0].reset()
    obs_dim = sample_obs.shape[0]
    print(f"obs_dim={obs_dim}  n_actions={n_actions}")

    agent = Agent(obs_dim, n_actions).to(device)
    optimizer = Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    batch_size = args.n_steps * args.n_envs
    n_updates = args.total_steps // batch_size

    # Rollout buffers — allocated once, reused every update
    b_obs = torch.zeros(args.n_steps, args.n_envs, obs_dim, device=device)
    b_actions = torch.zeros(args.n_steps, args.n_envs, dtype=torch.long, device=device)
    b_logprobs = torch.zeros(args.n_steps, args.n_envs, device=device)
    b_rewards = torch.zeros(args.n_steps, args.n_envs, device=device)
    b_dones = torch.zeros(args.n_steps, args.n_envs, device=device)
    b_values = torch.zeros(args.n_steps, args.n_envs, device=device)
    b_masks = torch.zeros(args.n_steps, args.n_envs, n_actions, dtype=torch.bool, device=device)

    obs_np, infos = envs.reset()
    obs = torch.from_numpy(obs_np).float().to(device)
    masks = torch.from_numpy(
        np.stack([info["action_mask"] for info in infos])
    ).to(device)
    dones = torch.zeros(args.n_envs, device=device)

    ep_returns: list[float] = []
    ep_wins: list[bool] = []
    ep_return_buf = np.zeros(args.n_envs)
    ep_length_buf = np.zeros(args.n_envs, dtype=int)

    global_step = 0
    start_time = time.time()

    for update in range(1, n_updates + 1):
        # Linear LR annealing to 0 over training
        frac = 1.0 - (update - 1) / n_updates
        optimizer.param_groups[0]["lr"] = args.lr * frac

        # -- Rollout collection ---------------------------------------------
        for step in range(args.n_steps):
            global_step += args.n_envs
            b_obs[step] = obs
            b_dones[step] = dones  # done flag entering this step (1 = fresh episode)
            b_masks[step] = masks

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs, masks)
            b_actions[step] = action
            b_logprobs[step] = logprob
            b_values[step] = value

            obs_np, rewards, terminated, truncated, infos = envs.step(action.cpu().tolist())
            b_rewards[step] = torch.from_numpy(rewards).to(device)

            ep_return_buf += rewards
            ep_length_buf += 1

            done_np = terminated | truncated
            dones = torch.from_numpy(done_np.astype(np.float32)).to(device)
            obs = torch.from_numpy(obs_np).float().to(device)

            for i, (term, trunc, info) in enumerate(zip(terminated, truncated, infos)):
                if term or trunc:
                    ep_returns.append(float(ep_return_buf[i]))
                    ep_wins.append(info.get("winner") == "jack")
                    ep_return_buf[i] = 0.0
                    ep_length_buf[i] = 0
                    # Auto-reset: replace obs/mask for this env
                    new_obs, new_info = envs.reset_one(i)
                    obs[i] = torch.from_numpy(new_obs).float().to(device)
                    masks[i] = torch.from_numpy(new_info["action_mask"]).to(device)
                else:
                    masks[i] = torch.from_numpy(info["action_mask"]).to(device)

        # -- GAE advantage computation --------------------------------------
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
                delta = b_rewards[t] + args.gamma * next_val * next_non_terminal - b_values[t]
                last_gae = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae
                advantages[t] = last_gae
            returns = advantages + b_values

        # -- PPO minibatch updates ------------------------------------------
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
                clip_fracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()
                vf_loss = F.mse_loss(new_value, flat_returns[mb])
                entropy_loss = entropy.mean()

                loss = pg_loss + args.vf_coef * vf_loss - args.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(entropy_loss.item())

        # -- Logging -------------------------------------------------------
        sps = int(global_step / (time.time() - start_time))
        recent_ret = ep_returns[-100:]
        recent_wins = ep_wins[-100:]
        mean_return = sum(recent_ret) / len(recent_ret) if recent_ret else 0.0
        win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0.0
        print(
            f"update={update}/{n_updates} "
            f"steps={global_step:,} "
            f"sps={sps} "
            f"episodes={len(ep_returns)} "
            f"return={mean_return:.3f} "
            f"win_rate={win_rate:.3f} "
            f"pg={sum(pg_losses)/len(pg_losses):.4f} "
            f"vf={sum(vf_losses)/len(vf_losses):.4f} "
            f"ent={sum(ent_losses)/len(ent_losses):.4f} "
            f"clip_frac={sum(clip_fracs)/len(clip_fracs):.3f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # Checkpoint every 50 updates
        if update % 50 == 0 or update == n_updates:
            ckpt_dir = Path(args.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"agent_{global_step:010d}.pt"
            torch.save(
                {
                    "agent": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "obs_dim": obs_dim,
                    "n_actions": n_actions,
                },
                ckpt_path,
            )
            print(f"  checkpoint -> {ckpt_path}")

    print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO training for Jack RL policy")
    p.add_argument("--map", default="maps/whitechapel.json")
    p.add_argument("--total-steps", type=int, default=5_000_000)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--n-envs", type=int, default=8)
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
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
