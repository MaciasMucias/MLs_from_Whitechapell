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
# Submit twice — once as-is (CPU-only), once with:
#     sbatch --gres=gpu:rtx6000:1 --job-name=wc-pilot-gpu pilot.sbatch
#
# DECISION RULE (see 02-C3): choose CPU-only unless GPU SPS > 4.5x CPU SPS.
# The CPU cap allows 9 concurrent runs; the GPU cap allows only 2.
#
# ALSO CHECK: a 15M-step final run must fit 24h, i.e. >= 174 SPS sustained.
# If CPU SPS < ~250, read the "Walltime risk" section of 04-final-runs.md.

# NOTE: under sbatch, $0 is Slurm's spool copy, not this file — resolve via PROJECT_DIR.
: "${PROJECT_DIR:?set PROJECT_DIR (e.g. --export=ALL,PROJECT_DIR=$PWD)}"
source "$PROJECT_DIR/docs/completion/slurm/env.sh"
mkdir -p "$PROJECT_DIR/logs"

STEPS=200000

echo "=== config A: current settings (n-envs 12, n-workers 6) ==="
uv run python -m training.train \
    --total-steps $STEPS \
    --n-envs 12 --n-workers 6 \
    --eval-games 0 \
    --wandb-mode disabled \
    --checkpoint-dir "${SLURM_TMPDIR:-/tmp}/pilot_a_$$" 2>&1 | tail -30

echo
echo "=== config B: scaled up (n-envs 24, n-workers 12) — scaling curve only ==="
echo "NOTE: this changes batch size (n_steps x n_envs) and so the effective"
echo "hyperparameters. It is measured for information, NOT adopted by default."
uv run python -m training.train \
    --total-steps $STEPS \
    --n-envs 24 --n-workers 12 \
    --eval-games 0 \
    --wandb-mode disabled \
    --checkpoint-dir "${SLURM_TMPDIR:-/tmp}/pilot_b_$$" 2>&1 | tail -30

echo
echo "Record charts/instant_sps (or the printed SPS) for each config in 02-C3."
