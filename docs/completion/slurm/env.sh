#!/usr/bin/env bash
# Common environment for every cluster job. Sourced by the sbatch scripts.
set -euo pipefail

# uv was installed as a user (not system-wide) — make it findable.
export PATH="$HOME/.local/bin:$PATH"

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

cd "${PROJECT_DIR:?set PROJECT_DIR to the repo checkout on shared storage}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-none}"
echo "cpus=${SLURM_CPUS_PER_TASK:-?} cwd=$PWD"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "gpu: none (CPU-only)"
