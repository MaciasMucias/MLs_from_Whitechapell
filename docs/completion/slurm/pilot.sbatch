#!/usr/bin/env bash
#SBATCH --job-name=wc-pilot
#SBATCH --partition=student
#SBATCH --account=stud-2526-l-03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#
# Throughput pilot: decides CPU-only vs GPU for every later run.
#
# Submit FROM the repo root (SLURM_SUBMIT_DIR is used as PROJECT_DIR):
#     sbatch docs/completion/slurm/pilot.sbatch
#     sbatch --gres=gpu:rtx6000:1 --job-name=wc-pilot-gpu docs/completion/slurm/pilot.sbatch
#
# Do NOT pass --export: this site's Slurm fails user env retrieval and holds the
# job with "user env retrieval failed requeued held".
#
# DECISION RULE (see 02-C3): choose CPU-only unless GPU SPS > 4.5x CPU SPS.
# The CPU cap allows 9 concurrent runs; the GPU cap allows only 2.
#
# ALSO CHECK: a 15M-step final run must fit 24h, i.e. >= 174 SPS sustained.
# If CPU SPS < ~250, read the "Walltime risk" section of 04-final-runs.md.

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "$PROJECT_DIR/docs/completion/slurm/env.sh"
mkdir -p "$PROJECT_DIR/logs"

STEPS=200000
OUT="${SLURM_TMPDIR:-/tmp}/wc_pilot_$$"

echo "=== config A: current settings (n-envs 12, n-workers 6) ==="
uv run python -m training.train \
    --total-steps $STEPS \
    --n-envs 12 --n-workers 6 \
    --eval-games 0 \
    --wandb-mode disabled \
    --checkpoint-dir "$OUT/a" 2>&1 | tail -30

echo
echo "=== config B: scaled up (n-envs 24, n-workers 12) — scaling curve only ==="
echo "NOTE: this changes batch size (n_steps x n_envs) and so the effective"
echo "hyperparameters. Measured for information, NOT adopted by default."
uv run python -m training.train \
    --total-steps $STEPS \
    --n-envs 24 --n-workers 12 \
    --eval-games 0 \
    --wandb-mode disabled \
    --checkpoint-dir "$OUT/b" 2>&1 | tail -30

rm -rf "$OUT"
echo
echo "Record the sps / instant_sps figures for each config in 02-C3."
