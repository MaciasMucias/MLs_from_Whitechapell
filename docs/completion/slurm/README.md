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

Partition `student` and **`--account=stud-2526-l-03`** are both set already.

**The account is mandatory.** The default association is account `null` with `MaxSubmitJobs=0`, so
any job submitted without `--account` is rejected with `AssocMaxSubmitJobLimit`.


Set `PROJECT_DIR` to the repo checkout **on shared storage** (`~/MLs_from_Whitechapel` on
`/mnt/evafs`), never node-local scratch. `/tmp` is per-node — a file staged there from the login
node is invisible to compute nodes, and checkpoints written there
vanish when the job ends.

**Never run Python on the login node.** It is a KVM VM lacking x86-64-v2, so numpy aborts with a
confusing `NumPy was built with baseline optimizations: (X86_V2)` error. `uv sync` there is fine
(downloads only); everything else goes through `srun`/`sbatch`.

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
