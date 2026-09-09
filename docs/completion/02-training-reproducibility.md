# 02 — Training reproducibility + cluster readiness

**Status:** not started
**Blocks:** 03, 04 — a sweep over unlogged values is worthless
**Blocked by:** nothing. Can run in parallel with 01.

---

## Goal

Make a training run fully described by its logged config, stop the checkpoint dir from growing
without bound, and (if cluster access is confirmed) make runs queueable.

## Why it matters

Right now you **cannot write up what your existing runs did.** The reward coefficients are hardcoded
constructor kwargs, so they never reach `wandb.config` — the runs named `sparse-copdist-15m` and
`no-timeout-penalty` differ in reward structure in ways the logged config cannot distinguish. Any
sweep (03) inherits this problem unless it is fixed first.

---

## C1 — Expose the reward coefficients as CLI flags

### Current state

`training/env.py:36-47` — hardcoded `JackEnv.__init__` defaults:

```python
alpha: float = 0.1    # hideout-distance progress
beta:  float = 0.05   # cop PMF potential
delta: float = 0.01   # count-based exploration
gamma: float = 0.5    # terminal hideout-uncertainty bonus
zeta:  float = 0.1    # cop-distance delta
```

Reward structure, for reference when sweeping:

- Terminal (`env.py:111-142`): `+1.0` Jack reaches hideout; `-1.0` arrest / turn limit / the
  **early-abort** rule at `env.py:118-122` (if `dist_to_hideout > turn_limit - 1 - turn`, terminate
  immediately as a cop win — mathematically unreachable).
- Per-step shaping (`env.py:154-159`): distance progress, PMF potential, cop-distance delta, and a
  `delta / sqrt(visit_count)` exploration bonus. `_visit_counts` **persists across episodes** within
  a worker (deliberate, `env.py:68-69`).
- Win bonus (`env.py:174-184`): recomputes the PMF on the final state and adds
  `gamma * (nonzero_zone_nodes / |hideout_zone|)` — rewards leaving cops uncertain which zone node
  is the hideout.

### Steps

1. Add `--alpha/--beta/--delta/--gamma/--zeta` to `parse_args()` (`training/train.py:535-579`).
2. Thread them through `_worker_fn` (`train.py:39-106`) into each `JackEnv`. Note `_worker_fn` is
   module-level for Windows `spawn` and each worker loads its own `load_map()` copy because `Map` is
   not picklable (`train.py:59`).
3. `config=vars(args)` at `wandb.init` (`train.py:244-253`) then picks them up automatically.

### Verification

Launch a 50k-step run; confirm all five appear in `wandb.config`.

---

## C2 — Best-checkpoint tracking + pruning

### Current state

- Saves every 50 updates (~153,600 steps) plus final, to
  `checkpoints/<wandb run name or id>/agent_<step:010d>.pt` (`train.py:291`, `train.py:480-494`).
- Payload: `agent`, `optimizer`, `step`, `obs_dim`, `n_actions`, `wandb_run_id`,
  `curriculum_difficulty`.
- **No best-checkpoint tracking.** No `best.pt`, no comparison, nothing pruned. 7 run dirs, 66-132
  files each, **7.4 GB total**. Each `.pt` is ~11 MB, mostly Adam state (2x params).
- Consumers resolve "latest" by lexicographic sort — `tools/optuna_tune.py:186-193`
  `resolve_checkpoint()` takes `sorted(p.glob("*.pt"))[-1]`. Only correct because of `:010d` padding.
- `training/eval.py:8`'s docstring references `agent_final.pt`, which `train.py` never writes.
- Resume is fully supported (`train.py:229-260`) and restores step, optimizer, curriculum difficulty,
  and the same W&B run via `resume="must"`.

### Steps

1. Eval already runs at every checkpoint save when `--eval-games > 0` (`train.py:497-503`) — track
   the best `eval/win_rate` and write `agent_best.pt` alongside.
2. Keep only the last N periodic checkpoints (N configurable, default ~5). Never prune
   `agent_best.pt`.
3. Fix the `agent_final.pt` reference in `training/eval.py:8`.

### Verification

Confirm `agent_best.pt` is written, tracks the best `eval/win_rate`, survives pruning, and that
`--resume` still works from a pruned directory.

---

## C3 — Cluster readiness

**Access confirmed and probed 2026-09-09.** Verified on the cluster, not assumed:

| | |
|---|---|
| Host / scheduler | `eden.eden`, Linux 6.8, **Slurm** |
| Partition | **`student`** — the only one. No `--account` needed. |
| Partition capacity | **2 nodes, 96 CPUs total**: `stud-1` (48 CPU, no GPU), `stud-2` (48 CPU, 2x rtx6000) |
| QoS `student` limits | `MaxSubmitJobsPU=20`, `MaxTRESPU=cpu=72,gres/gpu:rtx6000=2,mem=1T` |
| Walltime | `MaxTime=1-00:00:00` (24h), and that is also the *default* |
| `uv` | 0.11.21 at `~/.local/bin/uv` — **not on the non-interactive PATH**, `env.sh` fixes it |
| System Python | 3.12.3 — too old, but `cpython-3.13.14` is downloadable by uv |
| Storage | `/mnt/evafs` Lustre, **no per-user quota enforced**, 193 GB free |
| Outbound HTTPS | works from the login node (`api.wandb.ai` responded) |
| Repo | not yet cloned |

### Verified working 2026-09-09

Repo cloned to `~/MLs_from_Whitechapel` on `/mnt/evafs` (shared). `uv sync --extra training --no-dev`
succeeded: **Python 3.13.14, numpy 2.4.4, torch 2.11.0+cu128, wandb 0.27.0**. On a compute node all
engine/agents/training modules import, the map loads (195 jack / 234 cop nodes), `HeuristicCops()`
reports `arrest_threshold=0.209`, and **all 26 tests pass**.

### Three cluster gotchas — each one blocks everything until fixed

**1. `--account=stud-2526-l-03` is mandatory.** The default association is account `null` with
`MaxSubmitJobs=0`, so any job submitted without an explicit account is rejected:

```
srun: error: AssocMaxSubmitJobLimit
srun: error: Unable to allocate resources: Job violates accounting/QOS policy
```

Both `.sbatch` files now set it. (This corrects the earlier note that no account was needed.)

**2. Python cannot run on the login node at all.** It is a VM reporting `Common KVM processor`,
missing `popcnt`/`sse4_1`/`sse4_2`/`ssse3`, so the prebuilt numpy wheel aborts:

```
RuntimeError: NumPy was built with baseline optimizations: (X86_V2)
but your machine doesn't support: (X86_V2).
```

Compute nodes are **Intel Xeon 6520P** with full x86-64-v2 + AVX2, where it works fine. So:
`uv sync` on the login node is fine (downloads only), but **every** smoke test, eval, or analysis
must go through `srun`/`sbatch`. Never debug Python on the login node — the failure is confusing and
has nothing to do with your code.

**3. `/tmp` is node-local, not shared.** Staging a script to `/tmp` on the login node leaves it
invisible to the compute node. Keep everything under the repo on `/mnt/evafs`, and note the same
applies to checkpoint output.

Minor: the project is a flat layout with no installed package, so a script run by path needs
`PYTHONPATH=$PWD` (or use `python -m`). `pytest` works unaided.

**Two corrections to earlier assumptions in this file:**

1. **The `student` partition is small — 96 CPUs total, not a slice of a 1,000-CPU cluster.** The
   per-person cap of 72 CPUs is *75% of the entire partition*, and the 2-GPU cap is *100% of its
   GPUs*. Nine concurrent jobs is therefore the whole partition in practice. Legitimate under the
   QoS, but expect real contention in term time.
2. **Disk is not a constraint.** No per-user quota is enforced and 193 GB is free; the full
   unpruned 9-run set is ~10 GB. C2's *best-checkpoint tracking* is still needed (there is otherwise
   no way to identify the best checkpoint), but its *pruning* is a tidiness win, not a blocker.

At the time of probing **both nodes were idle and the queue was empty** — a good window to run.

### The decisive consequence: run on CPU, not GPU

The GPU cap is **2**; the CPU cap is **72**. At ~8 CPUs per run (matching the pilot-chosen shape, at the
current `--n-envs 12 --n-workers 12`) that is:

```
GPU path:  2 concurrent runs
CPU path:  72 / 8 = 9 concurrent runs
```

So **GPU only wins if a GPU run is more than ~4.5x faster than a CPU run.** It almost certainly is
not. The model is a `1416 -> 512 -> 256 -> 256` MLP with two small heads
(`training/model.py:8-41`); the PPO update is 4 epochs x 12 minibatches of 256 — trivial arithmetic.
The rollout, meanwhile, runs the Python game engine and the heuristic cops' DP-based PMF in worker
processes at ~490 SPS. The environment dominates, the GPU accelerates only the update, and Amdahl's
law caps the achievable speedup well below 4.5x.

**Working assumption: CPU-only, 9 concurrent runs.** Confirm with the pilot below before committing.

### This also settles `--n-envs`

Keep **`--n-envs 12 --n-workers 12`**, request ~8 CPUs per job, and spend the CPU budget on
*concurrency* rather than on bigger individual runs. This is the right call twice over:

- It maximises throughput (9 concurrent runs vs 5 at `--n-envs 24`).
- It **preserves the batch size** (`n_steps * n_envs` = 256 x 12 = 3,072) used by the existing runs,
  so the sweep's conclusions stay comparable and nothing about the PPO dynamics shifts underneath
  you.

Whatever is chosen, it must be **identical across the 03 sweep and the 04 final runs**.
`AsyncVectorJackEnv` asserts `n_envs % n_workers == 0` (`train.py:114-202`); workers are processes,
so never exceed the allocated CPU count.

Memory is a non-constraint — 1 TB against runs that need single-digit GB. Request modestly
(~8-16 GB/job) rather than claiming a share you will not use.

### Pilot results — RUN 2026-09-09. Decision: **CPU-only, `--n-envs 12 --n-workers 12`, 8 CPUs**

200k steps per config, `--eval-games 0`, wandb disabled. SPS is full PPO throughput (rollout +
gradient updates) as reported by `train.py`.

**CPU vs GPU** (`--cpus-per-task=8`):

| Config | CPU (`stud-1`, Xeon 6520P) | GPU (`stud-2`, RTX PRO 6000 Blackwell) |
|---|---|---|
| A — 12 envs / 6 workers (batch 3,072) | **608 SPS** | 506 SPS |
| B — 24 envs / 12 workers (batch 6,144) | **838 SPS** | 778 SPS |

**The GPU is slower than the CPU** — 0.83x and 0.93x. Not merely under the 4.5x threshold: negative.
The workload is environment-bound, not network-bound. The model is ~0.9M parameters
(`1416 -> 512 -> 256 -> 256` + two small heads); the rollout does 256 *sequential* forward passes at
batch 12, so per-step host/device transfer, sync and kernel-launch overhead cost more than the
matmuls save, while the Python game engine and the cops' DP-based PMF run on CPU either way.

**Decision: CPU-only. Never request `--gres`.** Combined with the caps (9 concurrent CPU runs vs 2
GPU runs) this is roughly an order of magnitude more aggregate throughput.

**Choosing the run shape** — what matters is *aggregate* throughput across concurrent jobs, since the
72-CPU cap fixes how many fit:

| Config | envs / workers | CPUs | batch | SPS | concurrent | **aggregate** |
|---|---|---|---|---|---|---|
| A | 12 / 6 | 8 | 3,072 | 608 | 9 | 5,472 |
| B | 24 / 12 | 8 | **6,144** | 838 | 9 | **7,542** |
| **C** | **12 / 12** | **8** | **3,072** | **682** | **9** | **6,138** |
| C16 | 12 / 12 | 16 | 3,072 | 811 | 4 | 3,244 |

Three things fall out:

1. **The speedup came from more workers, not the bigger batch.** C matches most of B's gain with the
   batch unchanged — the 6-worker config simply leaves cores idle while the main process runs the
   PPO update.
2. **Spend the CPU budget on concurrency, not on individual runs.** C16 is 19% faster per run than C
   but only 53% of its aggregate, because it halves how many jobs fit under the cap.
3. **Oversubscription helps.** 12 workers on 8 CPUs beats 6 workers on 8 CPUs, because workers block
   on IPC and the engine rather than saturating a core.

**Adopted: config C** — `--n-envs 12 --n-workers 12`, `--cpus-per-task=8`, 9 concurrent. Written
into both run manifests. It keeps
batch size at 3,072, the value the existing architecture and hyperparameters were tuned around, for
12% more throughput than A. Config B is 23% better on aggregate but doubles the batch, which changes
PPO dynamics and would need its own sweep to justify; revisit only if throughput becomes binding.

**The walltime risk is resolved.** 15M steps at 682 SPS is **6.1h**, well inside the 24h cap — no
chaining needed. Both cluster figures beat the ~490 SPS reference from the local RTX 5080.

Incidental: `difficulty` stayed pinned at -1.000 throughout, because win rate at 200k steps is 3-8%,
far below the `[0.4, 0.6]` deadband, so the P-controller pushes toward "easier" and clamps. Expected
this early — but it means **the curriculum cannot help beyond full suppression**, worth remembering
when reading early training curves in 04.

### Queue strategy

Walltime is a non-issue: a 15M-step run is ~8.5h on an RTX 5080 and fits inside 24h with room to
spare. **Do not chain jobs** — resume-chaining works (`--resume` restores step, optimizer, curriculum
difficulty and rejoins the same W&B run, `train.py:229-260`) but each segment pays a fresh queue
wait. Keep it in reserve for jobs that die.

#### How the 20-job cap interacts with arrays (checked against Slurm docs)

**Every array task counts individually, pending ones included.** From SchedMD's
[job array documentation](https://slurm.schedmd.com/job_array.html):

> **NOTE**: Job array tasks still act like regular jobs, including in the enforcement of job-related
> limits (e.g., **MaxJobs**, **MaxSubmitJobs**).

The relevant limit definitions ([resource limits](https://slurm.schedmd.com/resource_limits.html)):

- **`MaxJobs`** — "The total number of jobs able to run at any given time for the given association."
- **`MaxSubmitJobs`** — "The maximum number of jobs able to be submitted to the system at any given
  time from the given association." This covers **pending *and* running**.

The access email's "max 20 simultaneous jobs submitted to the queue" maps to `MaxSubmitJobs`.

**Consequence — the `%` throttle does not help here.** `--array=0-31%9` submits **32 jobs** and
merely limits 9 to *run* concurrently. Against a cap of 20 that array is rejected outright, with
reason code `AssocMaxSubmitJobLimit`.

The fact that pending tasks are *displayed* collapsed in `squeue` (e.g. `1080_[5-1024]` on one line)
is a display optimisation only — Slurm creates one job record per task lazily, but the limit is
enforced per task regardless.

**So size arrays by the submit cap, not the concurrency cap:**

```bash
#SBATCH --array=0-19%9      # 20 tasks total (at the cap), at most 9 running at once
```

Keep total tasks **≤ 20 per wave**, and use `%9` on top to respect the 72-CPU limit. For anything
larger than 20, submit in waves, or chain waves with `--dependency=afterany:<jobid>` so the next
wave only enters the queue as the previous drains.

### Remaining unknowns

**Scheduler confirmed Slurm (2026-09-09)** — job scripts are written and live in
[`slurm/`](slurm/README.md).

Still to settle, via [`cluster_probe.sh`](cluster_probe.sh) on the login node (read-only; submits and
installs nothing):

- **Partition names** — the `<FILL>` in every `.sbatch`.
- **Outbound HTTPS from compute nodes** — decides W&B online vs offline. `slurm/env.sh` forces
  offline, which is safe either way; `wandb sync wandb/offline-*` from the login node afterwards.
- **Storage quota** — see Disk below.
- Whether an `--account` is required.

### Disk

Not a constraint after all: no per-user quota is enforced on `/mnt/evafs` and 193 GB is free. A
15M-step run writes ~1.1 GB, so the full 9-run set is ~10 GB. C2 is still worth doing for
**best-checkpoint tracking** (needed to pick a checkpoint at all), but its pruning is now optional.

### Run manifest

Written: [`slurm/manifests/`](slurm/manifests/). `array.sbatch` reads one command per line and runs
the line matching `$SLURM_ARRAY_TASK_ID`, so the manifest **is** the record of what was run — it
doubles as the thesis's experiment table.

- `final.txt` — the 9 final runs (3 arms x 3 seeds). `<LR>`/`<ENT>` filled in after the sweep.
- `sweep.txt` — example grid; keep each wave to <=20 lines.

### Verification

Beyond the pilot: confirm a job queues, finds its resources, writes checkpoints to the intended
filesystem (not node-local scratch that vanishes at job end), and that its W&B data syncs back.

## Session log

- 2026-09-09 — cluster access confirmed. `uv` installed as user; 24h walltime; no schedule pressure.
  Concluded walltime is a non-issue and chaining is counterproductive under contention.
- 2026-09-09 — per-person limits found (72 CPU / 2 rtx6000 GPU / 1 TB / 20 queued jobs). The 2-GPU
  cap vs 72-CPU cap makes **CPU-only the default plan** (9 concurrent runs vs 2), and settles
  `--n-envs 12 --n-workers 12` — which also preserves the existing batch size. Pilot still needed to
  confirm the 4.5x threshold is not met.
- 2026-09-09 — checked Slurm docs on array accounting. Each array task counts individually against
  `MaxSubmitJobs`, pending included, so the `%` throttle does **not** reduce the submitted count.
  Arrays must be sized <=20 tasks per wave, not merely throttled. Corrects earlier guidance that
  suggested `--array=0-31%9`.
- 2026-09-09 — scheduler confirmed Slurm. Wrote `slurm/` job scripts: `env.sh` (incl. the
  `OMP_NUM_THREADS=1` fix, critical for CPU-only rollouts), `pilot.sbatch`, `array.sbatch` +
  manifests for the sweep and the 9 final runs. Verified array indexing and shell syntax locally.
  Flagged the `uv sync` race: sync once on the login node, never from 9 concurrent jobs.
- 2026-09-09 — SSH access established (see [08](08-cluster-access.md)) and the cluster probed
  directly. Partition is `student` (2 nodes / 96 CPU / 2 GPU); QoS confirms 20 submitted jobs,
  72 CPU, 2 GPU, 1 TB, 24h. `uv` 0.11.21 present. **Corrected two assumptions:** the partition is
  small (72 CPU = 75% of it, so 9 concurrent jobs is the whole thing), and disk is not a constraint
  (no quota, 193 GB free), so C2 pruning is optional while best-checkpoint tracking is not.
  Partition filled into both `.sbatch` files.
- 2026-09-09 — cluster environment set up and verified. Repo cloned to `~/MLs_from_Whitechapel`;
  `uv sync --extra training` OK (py3.13.14 / numpy 2.4.4 / torch 2.11.0+cu128); 26/26 tests pass on
  a compute node. Found three blockers and fixed/documented all: `--account=stud-2526-l-03` is
  mandatory (default account has MaxSubmitJobs=0); the login node's CPU lacks x86-64-v2 so numpy
  cannot run there at all; `/tmp` is node-local. Partition + account written into both `.sbatch`.
- 2026-09-09 — pilot run. **GPU is slower than CPU** (506 vs 608 SPS), so CPU-only is settled with
  ~10x the aggregate throughput. Ran a second pilot separating worker count from batch size: the
  gain comes from workers, so adopted `--n-envs 12 --n-workers 12` at 8 CPUs (682 SPS, batch
  unchanged at 3,072, 9 concurrent). 15M steps = 6.1h, so the 24h walltime risk is closed. Also
  hit and fixed two more Slurm traps: `$0` is the spool copy under sbatch, and `--export` gets jobs
  held with "user env retrieval failed".
