# Slurm job scripts

Scheduler confirmed **Slurm** (2026-09-09). Templates for the runs in
[03](../03-ppo-sweep.md) and [04](../04-final-runs.md).

## Before you submit anything — one-time setup

**Sync the environment on the login node first.** Do not let jobs do it.

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /path/to/MLs_from_Whitechapel
uv sync --extra training          # NOT --no-dev, see below
```

If nine concurrent jobs each sync against the same `.venv`, they race and corrupt it. This is the
single easiest way to lose a whole array. Sync once, then submit.

**`env.sh` sets `UV_NO_SYNC=1`** so jobs physically cannot do it. That matters because `uv run`
syncs *by default* — a bare `uv run` is enough, no explicit `uv sync` required. Verified 2026-09-09:
`uv run` inside `srun` reinstalled 11 packages before starting.

**Sync without `--no-dev`.** `uv run` compares the venv against the full default set, so a venv
built with `--no-dev` looks stale to it and it re-syncs. Since `UV_NO_SYNC=1` now blocks that, a
`--no-dev` venv would instead make jobs fail on a missing import. The dev extra is a few small
packages; install them and keep the environment consistent.

## Fill these in

Partition `student` and **`--account=stud-2526-l-03`** are both set already.

**The account is mandatory.** The default association is account `null` with `MaxSubmitJobs=0`, so
any job submitted without `--account` is rejected with `AssocMaxSubmitJobLimit`.


`PROJECT_DIR` is taken from `SLURM_SUBMIT_DIR`, so just submit from the repo root on shared storage
(`~/MLs_from_Whitechapel` on
`/mnt/evafs`), never node-local scratch. `/tmp` is per-node — a file staged there from the login
node is invisible to compute nodes, and checkpoints written there
vanish when the job ends.

**Never run Python on the login node.** It is a KVM VM lacking x86-64-v2, so numpy aborts with a
confusing `NumPy was built with baseline optimizations: (X86_V2)` error. `uv sync` there is fine
(downloads only); everything else goes through `srun`/`sbatch`.

## Never pass `--export`

This site's Slurm cannot retrieve the user environment; a job submitted with `--export=...` is
requeued and held with:

```
(user env retrieval failed requeued held)
```

Both scripts therefore take `PROJECT_DIR` from `SLURM_SUBMIT_DIR` and the manifest as a positional
argument. **Submit from the repo root.**

## Order of operations

```bash
cd ~/MLs_from_Whitechapel

# 1. Throughput pilot — decides CPU vs GPU. ~15-20 min.
sbatch docs/completion/slurm/pilot.sbatch
sbatch --gres=gpu:rtx6000:1 --job-name=wc-pilot-gpu docs/completion/slurm/pilot.sbatch

# 2. Sweep wave 1 — lr x ent-coef, both Director arms. 18 tasks, ~2.5h.
#    (01, 02-C1 and 02-C2 have landed; the pilot chose CPU-only.)
J1=$(sbatch --parsable --array=0-17%9 docs/completion/slurm/array.sbatch \
            docs/completion/slurm/manifests/sweep_w1.txt)

# 3. Sweep wave 2 — reward coefficients. EDIT <LR>/<ENT> from wave 1 FIRST.
#    Chained because 18 + 12 = 30 tasks exceeds the 20-job submit cap.
sbatch --dependency=afterany:$J1 --array=0-11%9 docs/completion/slurm/array.sbatch \
       docs/completion/slurm/manifests/sweep_w2.txt

# 4. Director tuning — 17 tasks. EDIT <LR>/<ENT> first. See 09-director-tuning.md.
#    Must precede the final runs: 04's headline ON arm is 3 seeds x 15M steps and
#    is not worth spending on an untuned Director.
sbatch --array=0-16%9 docs/completion/slurm/array.sbatch \
       docs/completion/slurm/manifests/director.txt

# 5. Final runs — 9 tasks. EDIT <LR>/<ENT> and the chosen Director flags in
#    final.txt first.
sbatch --array=0-8%9 docs/completion/slurm/array.sbatch \
       docs/completion/slurm/manifests/final.txt
```

Step 3 cannot be submitted before step 2's values are known, so in practice you
will submit wave 1, read it, edit `sweep_w2.txt`, then submit wave 2 without the
dependency. The `--dependency` form above is for firing both at once and editing
the manifest while wave 1 runs — `array.sbatch` reads the manifest at task start,
not at submit time.

## The limits that shape these scripts

Per person: **72 CPUs, 2 rtx6000 GPUs, 1 TB RAM, 20 submitted jobs, 24h walltime.**

- `--cpus-per-task=8` (12 workers, deliberately oversubscribed — see 02-C3) -> **9 concurrent runs** within the 72-CPU cap.
- **Array size is capped at 20 tasks, not 20 running.** Each array task counts individually against
  `MaxSubmitJobs`, pending included — the `%` throttle only limits concurrent *execution*. See
  [02-C3](../02-training-reproducibility.md) for the documentation quotes. Wider sweeps go in waves
  chained with `--dependency=afterany:<jobid>`.
- `--time=23:00:00` leaves margin under the 24h cap. A 15M-step run needs **>=174 SPS** to fit at
  all; the pilot verifies this.

## Manifest pattern

`array.sbatch` takes the manifest as its first argument and runs the line matching
`$SLURM_ARRAY_TASK_ID`. Blank lines and `#` comments are skipped. The manifest doubles as the
thesis's experiment table — it *is* the record of what was run.
