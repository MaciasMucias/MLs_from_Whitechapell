# Completion plan — technical work

Working state for finishing the dissertation's technical work. One file per workstream. Each is
self-contained: a session can open any single file and execute it without reading the others.

**These docs are the working state, not a retrospective.** Updating the relevant file's Status and
Session log is part of finishing a piece of work.

Audit date: 2026-09-09. Branch `dev` @ `bea89b6`.

---

## Status board

| # | Workstream | Status | Blocks |
|---|---|---|---|
| [01](01-freeze-cops.md) | Freeze cop configuration | not started | 03, 04, 06 |
| [02](02-training-reproducibility.md) | Training reproducibility + cluster readiness | not started | 03, 04 |
| [03](03-ppo-sweep.md) | PPO hyperparameter sweep | not started | 04 |
| [04](04-final-runs.md) | Final runs — Director on/off | not started | 06 |
| [05](05-study-data.md) | Close out study data | not started | 06 |
| [06](06-comparison.md) | Human-vs-RL comparison | not started | — |
| [07](07-docs-cleanup.md) | Docs cleanup | not started | — |
| [08](08-cluster-access.md) | Cluster SSH access for automated work | **done** | — |

Critical path: **01 + 02 -> 03 -> 04 -> 06**. Workstreams 05 and 07, and the code parts of 06, are
laptop work that runs in parallel with cluster time.

**Compute: university cluster confirmed 2026-09-09.** Per-person limits: **72 CPUs, 2 rtx6000 GPUs,
1 TB RAM, 20 queued jobs, 24h walltime.** `uv` installed as user. No deadline pressure.

The 2-GPU vs 72-CPU asymmetry is decisive: at ~8 CPUs per run that is **9 concurrent CPU-only runs
against 2 GPU runs**, so GPU only wins if it is >4.5x faster per run — implausible for a small MLP
whose bottleneck is the Python game environment, not the network. **Plan of record is CPU-only**,
pending the pilot in [02-C3](02-training-reproducibility.md).

That capacity reshapes the experiment budget: the 03 sweep becomes a job array (sized ≤20 tasks —
each array task counts individually against the 20-job submit cap), and the
whole 04 final set — 2 Director arms + the reward ablation, **3 seeds each, 9 runs** — fits in one
concurrent window instead of running sequentially. The seeds matter most: PPO seed variance can
exceed the Director effect being claimed, so single-seed arms would not support the conclusion.

Next concrete step: the **02-C3 pilot** (~200k steps, minutes) to measure CPU vs GPU throughput.
It also settles one risk — 15M steps inside a 24h cap needs **≥174 SPS**, against ~490 SPS on GPU
today, so a CPU run must stay within ~2.8x of GPU speed. [`cluster_probe.sh`](cluster_probe.sh)
covers what remains: partition names, outbound internet for W&B, and quota. Slurm job scripts are
written and syntax-checked in [`slurm/`](slurm/README.md).

---

## Where the project actually stands

More is done than `CLAUDE.md` claims (see [07](07-docs-cleanup.md)). The engine, heuristic cops,
`CurriculumDirector`, the full PPO pipeline, the participant UI and the live Fly deployment are all
implemented and working.

Three findings reshape what "done" means:

1. **The curriculum has never been run.** Every training run since 2026-06-02 passed
   `--no-curriculum`; all W&B summaries show `curriculum/difficulty: -1` frozen at
   `INITIAL_DIFFICULTY`. The Director is the thesis's named contribution and the with/without
   comparison has no "with" arm. The code is written and wired — it has simply never been switched
   on. Invisible from the file tree. See [04](04-final-runs.md).
2. **Every checkpoint is stale.** All 7 runs finished by 2026-06-08; the cop retune landed
   2026-06-11 (`259ae5d`). No existing policy has played the cops participants faced.
3. **No comparison code exists.** Nothing joins `data/games.sqlite` to policy evaluation. This is the
   actual thesis deliverable. See [06](06-comparison.md).

---

## Invariants

Decisions that must not be silently reversed in a later session. Each one would quietly invalidate
the results rather than fail loudly.

1. **Cop parameters are frozen** at `COPS_STUDY_V2` — the values participants actually played
   against. A retune becomes a separate named v3 and never feeds the human comparison. See
   [01](01-freeze-cops.md).
2. **Evaluation is always Director-free**, for every arm. `eval_agent` passes `director=None`
   (`training/eval.py:84`); keep it that way. The Director shapes training difficulty only.
3. **Production is not modified.** The study is winding down. All analysis runs on a local snapshot.
4. **No PyTorch on the server.** The Fly VM is 256 MB; the comparison is offline by design.
5. **`uv` for everything** — `uv run python`, `uv add`. Never bare `python`/`pip`, never hand-edit
   `pyproject.toml`.

---

## Out of scope (decided, with reasons)

- **Server-side RL serving** — needs PyTorch on a 256 MB VM; not required by DESIGN_REQUIREMENTS §7.2.
- **Cop parameter retuning** — would invalidate the human data (invariant 1).
- **Board-size variants / the multi-size ablation (§7.3)** — never implemented; §4.3 is still TBD and
  `course_1/2/3` are same-size scenario variants, not size tiers. Report as a limitation.
- **Blocking rule, multi-night play** — tagged `EXTEND()` in `training/env.py:54-58`, deliberately
  unbuilt.
