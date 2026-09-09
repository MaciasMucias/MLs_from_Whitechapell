#!/usr/bin/env bash
#SBATCH --job-name=wc-train
#SBATCH --partition=student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=23:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-8%9
#
# Runs one line of $MANIFEST per array task.
#
#   sbatch --export=ALL,PROJECT_DIR=$PWD,MANIFEST=slurm/manifests/final.txt \
#          --array=0-8%9 slurm/array.sbatch
#
# LIMITS (see 02-C3):
#   --array size <= 20     each task counts against MaxSubmitJobs, pending included
#   %9                     9 concurrent x 8 CPUs = 72, the per-person CPU cap
#   Add --gres=gpu:rtx6000:1 ONLY if the pilot showed GPU > 4.5x faster
#   (and then %2, since the GPU cap is 2).

mkdir -p logs
source "$(dirname "$0")/env.sh"

: "${MANIFEST:?set MANIFEST to a run manifest file}"

# Strip blank lines and comments, then take the task's line.
CMD=$(grep -vE '^\s*(#|$)' "$MANIFEST" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

if [ -z "$CMD" ]; then
    echo "ERROR: no manifest line for task $SLURM_ARRAY_TASK_ID in $MANIFEST" >&2
    echo "manifest has $(grep -cvE '^\s*(#|$)' "$MANIFEST") runs; array indices are 0-based" >&2
    exit 1
fi

echo "manifest: $MANIFEST"
echo "run:      $CMD"
echo "---"
eval "$CMD"
