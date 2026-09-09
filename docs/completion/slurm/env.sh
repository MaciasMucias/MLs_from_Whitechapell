#!/usr/bin/env bash
# Common environment for every cluster job. Sourced by the sbatch scripts.
set -euo pipefail

# uv was installed as a user (not system-wide) — make it findable.
export PATH="$HOME/.local/bin:$PATH"

# CRITICAL: never let a job touch the shared .venv.
# `uv run` syncs the environment by default. With 9 array tasks starting at once
# that is 9 processes mutating one .venv on shared storage simultaneously — the
# fastest way to lose a whole array. Observed 2026-09-09: a bare `uv run` inside
# srun reinstalled 11 packages before starting.
# Sync ONCE on the login node (`uv sync --extra training`), then let jobs read it.
export UV_NO_SYNC=1

# CRITICAL for CPU-only training.
# Rollout runs 12 worker *processes* on 8 CPUs; PyTorch defaults to one OMP thread per core
# *per process*, so 12 workers x 8 threads on an 8-CPU allocation thrashes badly
# and can be several times slower than single-threaded. The PPO update is tiny
# (4 epochs x 12 minibatches of 256 on a ~0.9M-param MLP), so nothing is lost.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# W&B: compute nodes may have no outbound HTTPS. Offline is already train.py's
# default; sync from the login node afterwards with `wandb sync wandb/offline-*`.
export WANDB_MODE=offline

# Line-buffer stdout. Slurm sends stdout to a file, so Python block-buffers it in
# 8 KB chunks; at ~200 bytes per update line that is ~40 updates (10+ minutes)
# before anything appears. Observed on array 1806901: a job 460k steps in still
# had a 963-byte log, which is indistinguishable from a hang while you are
# watching it. Costs nothing, saves a false alarm.
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR:?set PROJECT_DIR to the repo checkout on shared storage}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-none}"
echo "cpus=${SLURM_CPUS_PER_TASK:-?} cwd=$PWD"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "gpu: none (CPU-only)"
