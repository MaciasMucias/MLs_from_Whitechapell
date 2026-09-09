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
| [01](01-freeze-cops.md) | Freeze cop configuration | **done** | 03, 04, 06 |
| [02](02-training-reproducibility.md) | Training reproducibility + cluster readiness | **done** | 03, 04 |
| [03](03-ppo-sweep.md) | PPO hyperparameter sweep | **ready to submit** | 04, 09 |
| [09](09-director-tuning.md) | Director tuning | **ready to submit** (needs 03's lr/ent) | 04 |
| [04](04-final-runs.md) | Final runs — Director on/off | not started | 06 |
| [05](05-study-data.md) | Close out study data | **mostly done** (A1, A5 left) | 06 |
| [06](06-comparison.md) | Human-vs-RL comparison | **code done**, awaiting 04 | — |
| [07](07-docs-cleanup.md) | Docs cleanup | not started | — |
| [08](08-cluster-access.md) | Cluster SSH access for automated work | **done** | — |

Critical path: **01 + 02 -> 03 -> 09 -> 04 -> 06**. Workstreams 05 and 07, and the code parts of 06,
are laptop work that runs in parallel with cluster time.

**09 was added 2026-09-09** and inserted ahead of 04. The Director had exactly one configuration ever
tried — `INITIAL_DIFFICULTY` was a hardcoded constant, not a flag — and that one run underperformed
the no-Director arm on a fair Director-free evaluation. Committing 3 seeds x 15M steps to an untuned
ON arm would have bought an expensive negative result with no diagnosis. See
[09](09-director-tuning.md).

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

**Next concrete step (2026-09-09): submit sweep wave 1.** 01, 02 and the 02-C3 pilot are done; the
critical path is now pure cluster time. On the login node:

```bash
cd ~/MLs_from_Whitechapel && git pull
export PATH="$HOME/.local/bin:$PATH" && uv sync --extra training --no-dev   # once, never from jobs
sbatch --array=0-17%9 docs/completion/slurm/array.sbatch \
       docs/completion/slurm/manifests/sweep_w1.txt
```

Then read the ON block, fill `<LR>`/`<ENT>` into `sweep_w2.txt`, and submit wave 2. Full ordering in
[`slurm/README.md`](slurm/README.md). Remember: **never run Python on the login node** — its CPU
lacks x86-64-v2 and numpy aborts.

---

## Where the project actually stands

More is done than `CLAUDE.md` claims (see [07](07-docs-cleanup.md)). The engine, heuristic cops,
`CurriculumDirector`, the full PPO pipeline, the participant UI and the live Fly deployment are all
implemented and working.

Findings that reshape what "done" means. **Items 1 and 3 were superseded on 2026-09-09** — kept, with
corrections, because both were load-bearing assumptions elsewhere in these docs.

1. ~~**The curriculum has never been run.**~~ **Corrected:** it ran once, 2026-05-23
   (`offline-run-20260523_134803-mdw4ndpi`, 10M steps). Every run *since 2026-06-02* passed
   `--no-curriculum`. That single ON run is the source of the belief that the Director
   underperforms — and re-measuring it showed the Director **wins by 11 points on the cops it
   trained against** and loses only on the retuned cops it never saw. See
   [09](09-director-tuning.md).
2. **Every checkpoint is stale.** Still true. All 7 runs finished by 2026-06-08; the cop retune
   landed 2026-06-11 (`259ae5d`). No existing policy has played the cops participants faced.
3. ~~**No comparison code exists.**~~ **Built 2026-09-09.** `analysis/compare.py` reconstructs each
   participant's exact board from its stored replay and replays the policy on it, plus move-level
   agreement against the human's own decisions. Validated on all 33 human games with zero desyncs.
   Awaiting 04's checkpoints for the reported numbers. See [06](06-comparison.md).
4. **New — the study data was not where the plan assumed.** The live database and the local
   `data/games.sqlite` are **disjoint**: production holds 51 rows from 2026-08-05 onward, all
   post-fix and uncontaminated, while the local file ends 2026-07-20. Pulling the current snapshot
   took the usable N from **2 participants / 6 games to 11 / 33**. See [05](05-study-data.md).

---

## Invariants

Decisions that must not be silently reversed in a later session. Each one would quietly invalidate
the results rather than fail loudly.

1. **Cop parameters are frozen** at `COPS_STUDY_V2` — the values participants actually played
   against. A retune becomes a separate named v3 and never feeds the human comparison. See
   [01](01-freeze-cops.md). Implemented 2026-09-09 in `agents/heuristic_cops.py`; pinned by
   `tests/test_cop_config.py`.
2. **Evaluation is always Director-free**, for every arm. `eval_agent` passes `director=None`; keep
   it that way. The Director shapes training difficulty only.
2b. **Never rank Director-ON runs by `charts/win_rate`.** The P-controller drives difficulty until
   win rate sits inside its deadband, so every converged ON run reads ~0.5 regardless of policy
   quality — it is the controller's setpoint, not a score. Rank on Director-free `eval/win_rate`,
   and report final `curriculum/difficulty` beside it. See [09](09-director-tuning.md).
3. **Production is not modified.** The study is winding down. All analysis runs on a dated local
   snapshot under `data/study/`, pulled read-only with `fly ssh sftp get`. Reading is fine and is how
   the snapshot is refreshed; writing, migrating or "fixing" production data is not.
4. **No PyTorch on the server.** The Fly VM is 256 MB; the comparison is offline by design.
5. **`uv` for everything** — `uv run python`, `uv add`. Never bare `python`/`pip`, never hand-edit
   `pyproject.toml`.
6. **Reward coefficients are `--reward-alpha/-beta/-delta/-gamma/-zeta`, never `--alpha`/`--gamma`.**
   `--gamma` is PPO's discount factor; passing it to disable reward shaping sets the discount to
   zero instead, silently, in exactly the arm meant to isolate reward shaping. Enforced by
   `tests/test_run_manifests.py`. See [02-C1](02-training-reproducibility.md).
7. **`--n-envs 12 --n-workers 12`** across every sweep and final run, so batch size stays 3,072 and
   the runs stay comparable. Also enforced by `tests/test_run_manifests.py`.

---

## Out of scope (decided, with reasons)

- **Server-side RL serving** — needs PyTorch on a 256 MB VM; not required by DESIGN_REQUIREMENTS §7.2.
- **Cop parameter retuning** — would invalidate the human data (invariant 1).
- **Board-size variants / the multi-size ablation (§7.3)** — never implemented; §4.3 is still TBD and
  `course_1/2/3` are same-size scenario variants, not size tiers. Report as a limitation.
- **Blocking rule, multi-night play** — tagged `EXTEND()` in `training/env.py:54-58`, deliberately
  unbuilt.
