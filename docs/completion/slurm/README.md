# Slurm job scripts

Scheduler confirmed **Slurm** (2026-09-09). Templates for the runs in
[03](../03-ppo-sweep.md) and [04](../04-final-runs.md).

## Before you submit anything — one-time setup

**Sync the environment on the login node first.** Do not let jobs do it.

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /path/to/MLs_from_Whitechapel
uv sync --extra training --no-dev
```

If nine concurrent jobs each run `uv sync` against the same `.venv`, they race and corrupt it. This
is the single easiest way to lose a whole array. Sync once, then submit.

## Fill these in

Partition is set to `student` (the only one available). No `--account` is required — the QoS
`student` carries the limits.


Set `PROJECT_DIR` to the repo checkout **on shared storage**, not node-local scratch, or checkpoints
vanish when the job ends.

## Order of operations

```bash
# 1. Throughput pilot — decides CPU vs GPU. ~minutes.
sbatch --export=ALL,PROJECT_DIR=$PWD slurm/pilot.sbatch

# 2. Sweep (after 01, 02-C1, 02-C2 land, and the pilot decides the partition)
sbatch --export=ALL,PROJECT_DIR=$PWD,MANIFEST=slurm/manifests/sweep.txt slurm/array.sbatch

# 3. Final runs (after the sweep picks hyperparameters)
sbatch --export=ALL,PROJECT_DIR=$PWD,MANIFEST=slurm/manifests/final.txt \
       --array=0-8%9 slurm/array.sbatch
```

## The limits that shape these scripts

Per person: **72 CPUs, 2 rtx6000 GPUs, 1 TB RAM, 20 submitted jobs, 24h walltime.**

- `--cpus-per-task=8` (6 workers + main + overhead) -> **9 concurrent runs** within the 72-CPU cap.
- **Array size is capped at 20 tasks, not 20 running.** Each array task counts individually against
  `MaxSubmitJobs`, pending included — the `%` throttle only limits concurrent *execution*. See
  [02-C3](../02-training-reproducibility.md) for the documentation quotes. Wider sweeps go in waves
  chained with `--dependency=afterany:<jobid>`.
- `--time=23:00:00` leaves margin under the 24h cap. A 15M-step run needs **>=174 SPS** to fit at
  all; the pilot verifies this.

## Manifest pattern

`array.sbatch` reads one command per line from `$MANIFEST` and runs the line matching
`$SLURM_ARRAY_TASK_ID`. Blank lines and `#` comments are skipped. The manifest doubles as the
thesis's experiment table — it *is* the record of what was run.
