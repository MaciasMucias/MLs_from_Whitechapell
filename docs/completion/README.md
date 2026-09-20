# Completion plan — technical work

Working state for finishing the dissertation's technical work. One file per workstream. Each is
self-contained: a session can open any single file and execute it without reading the others.

**These docs are the working state, not a retrospective.** Updating the relevant file's Status and
Session log is part of finishing a piece of work.

Audit date: 2026-09-13 (cluster re-read after array `1807390` landed). Branch `dev`.

**[FINDINGS.md](FINDINGS.md) collects every experimental result worth putting in the thesis**, with
evidence and caveats. Read that before writing any chapter.

---

## Status board

| # | Workstream | Status | Blocks |
|---|---|---|---|
| [01](01-freeze-cops.md) | Freeze cop configuration | **done** | 03, 04, 06 |
| [02](02-training-reproducibility.md) | Training reproducibility + cluster readiness | **done** | 03, 04 |
| [03](03-ppo-sweep.md) | PPO hyperparameter sweep | **done** — lr 3e-4, ent 0.03 | 04, 09 |
| [09](09-director-tuning.md) | Director tuning | **done, closed** — `--curriculum-max-difficulty 0`, default band; floor + band both settled by `1807390` | 04 |
| [04](04-final-runs.md) | Final runs — Director on/off | **done** — Director +0.028 score on every seed; `sparse` trades stealth for win rate; no overfitting to training cops | — |
| [10](10-reward-design.md) | Reward design | **done** — curriculum >> shaping (they substitute); delta null; use objective-only + curriculum | 06 |
| [05](05-study-data.md) | Close out study data | **open** — recruitment continues; 20 participants / 60 games at the 09-19 pull | 06 |
| [06](06-comparison.md) | Human-vs-RL comparison | **result in (interim)** — agent +0.58 to +0.64 score over humans, intervals clear of zero; re-run when recruitment ends | — |
| [07](07-docs-cleanup.md) | Docs cleanup | **done** | — |
| [11](11-hyperparameters.md) | Hyperparameter validity | **done** — `ent-coef 0.03` stands (0.01 indistinguishable, 0.003 worse); lr safe; rest are CleanRL defaults | — |
| [08](08-cluster-access.md) | Cluster SSH access for automated work | **done** | — |

Critical path: **01 + 02 -> 03 -> 09 -> 04 -> 10 -> 06**. (10 was added 2026-09-14: 06's checkpoints now come from its wave, trained to win stealthily and scored on the participant score.) Workstreams 05 and 07, and the code parts of 06,
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


### Cluster waves run so far — all complete

| array | wave | tasks | outcome |
|---|---|---|---|
| `1806901` | 03 sweep w1 | 18 | **lr=3e-4** decisively, both arms. ~30-point spreads. |
| `1806936` | entropy boundary probe | 2 | **ent-coef 0.03** confirmed an *interior* optimum |
| `1806976` | 03 sweep w2 | 12 | reward coefficients: **all inside noise** (1 seed, 3M — could not detect anything; superseded by 10) |
| `1807070` | 09 director v1 | 17 | **null — runs ended before the curriculum starts** |
| `1807190` | 09 bases | 3 | the curriculum's first uptick is at 4.4–4.8M steps |
| `1807209` | 09 branch | 15 | "the Director does not help" — **later found confounded** |
| `1807292` | 09 from scratch + ramp shape | 18 | **correction:** ramp shape is irrelevant; the *warmup* is what helps |
| `1807359` | 09 ceilings | 18 | **the result: `--curriculum-max-difficulty 0.0`**, 92.8% |
| `1807390` | 09 schedule study | 18 | **both remaining knobs closed** — see below |
| `1807480` | 04 final runs | 9 | ON > OFF on every seed (93.2 vs 90.7 best eval); `sparse` arm confounded (zeroed the objective's stealth term) |
| `1807588` | 10 reward design | 18 | **the curriculum dominates; shaping substitutes for it; delta null** — see below |
| `1808727` | 11 entropy re-check | 9 | `ent-coef 0.03` stands; 0.01 indistinguishable, 0.003 worse |

**Chosen config: `--lr 3e-4 --ent-coef 0.03 --curriculum-max-difficulty 0.0`** (default band).

### What `1807390` settled (2026-09-13)

Two questions, one of which could have invalidated everything upstream of it. Full write-up in
[09](09-director-tuning.md).

- **The −1.0 floor is degenerate, but harmlessly so.** `random() > |d|` is never true at |d| = 1.0,
  so 0% of discovered nodes survive and an entire strategy class is unlearnable. Held permanently
  that costs 13.8 points (`fix095` 64.5% vs `fix10` 50.7%, non-overlapping). **As a warmup it is
  fine** — `fsc000-i100` (−1.0) 92.0% vs `fsc000-i095` (−0.95) 89.3%, inside the noise floor,
  because a degenerate floor is simply a longer warmup and the warmup is worth +17 points.
  **So the base runs stand and no downstream result is provisional.**
- **The target band at ceiling 0.0 is a null.** 90.5 / 91.8 / 92.5 / 92.8 across four bands — 2.3
  points of spread, non-monotone. Pacing only matters when it controls time spent somewhere harmful;
  at ceiling 0.0 the ceiling is the optimum, so it does not. Use the default band, do not tune it.

**09 is closed.** Every Director knob is either chosen or measured as a null.

### IN FLIGHT — array `1808812`, submitted 2026-09-20, 15 tasks, ~14h (two windows)

**Seed top-up: wave 10's arms from 3 to 6 seeds.** Not a new experiment — the same five
configurations on fresh seeds 61–63, flag-for-flag identical to their wave 10 lines.

**Why.** The per-seed SD of the last-5 score is 0.0401 pooled over all nine 3-seed arms, so at n=3
the smallest reliably detectable difference is ~0.092, and ~0.130 for an *interaction*. The
substitution claim is an interaction and measures +0.148 — it clears by 14%, which is the thesis's
most interesting result resting on its thinnest margin. At 6 seeds the interaction threshold falls
to 0.092.

`w10s-obj-cur` is absent because it already has 6 seeds: `w11-ent003` (array `1808727`) is that
configuration flag-for-flag, scoring 1.309 on seeds 41–43 and 1.280 on 51–53 (pooled **1.294**).
That 0.029 swing from seed choice alone is the calibration — see FINDINGS §6b.

Read it by pooling old and new seeds per arm; the export prefix is `reward_topup`.

### What `1807588` settled (2026-09-19) — workstream 10, the reward wave

18/18. Trained on the stealth objective (−1 / 1 + 0.5 × hideout uncertainty), reported on the
participant score. Humans average 0.679. Full write-up in [10](10-reward-design.md).

| reward | curriculum | no curriculum |
|---|---|---|
| **objective only** | **1.309** ±0.013 · 89.8% win | 1.141 ±0.051 · 70.7% |
| + delta | 1.302 ±0.024 · 88.8% | 1.161 ±0.045 · 70.7% |
| + delta + shaping | 1.265 ±0.062 · 85.2% | 1.245 ±0.039 · 82.0% |

- **The curriculum is the largest effect in the project:** +0.168 score, +19 win points,
  non-overlapping ranges, and 42% fewer steps to a score of 1.10 (7.9M vs 13.7M).
- **Shaping substitutes for it.** Worth +0.104 without the curriculum, nothing with it — and it
  triples the seed spread. **This is why 04's Director effect looked small: both its arms had
  shaping on.**
- **Delta is a null, because the mechanism is absent at this scale:** coverage hits 1.000 within ~1%
  of training in every arm, bonus or no bonus.
- **Nothing converged at 15M** (tail slopes all positive), so these are sample-efficiency results at
  a fixed budget. Say the budget with the number.

**Recommended configuration, and 06's checkpoints: `w10s-obj-cur` (objective only + curriculum).**

**Two measured numbers that govern how any of this is read.** The run-to-run noise floor is
**~9.5 points** of `eval/win_rate` (wave 1's `sw1-lr3e4-ent003-on` 39.5% vs wave 2's identical
`sw2-control` 30.0%) — most single-run differences are noise, which is why every 09 wave from
`1807209` on is seeded. And `--seed` controlled almost nothing until 2026-09-09: weight init, action
sampling and minibatch order all ran on torch's OS-seeded default, so any pre-fix "paired seed"
comparison was not paired.

W&B runs offline on the compute nodes — sync from the login node afterwards with
`wandb sync wandb/offline-*`. Remember: **never run Python on the login node**; its CPU lacks
x86-64-v2 and numpy aborts with a misleading error.

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

8. **Train on `--reward-objective stealth`; report on the participant score.** Training: −1 on a
   loss, 1 + 0.5 × hideout uncertainty on a win. Reporting: `SCORE_STUDY_V1` (`engine/metrics.py`),
   progress + 0.5 × hideout uncertainty on a win, which is what participants were scored on and told
   about. It's frozen like the cops and pinned against `frontend_participant/game.js` by
   `tests/test_reward.py`. The 0.5 stealth weight is shared by both and is never tuned. Progress credit
   on a loss is **not** trained on (user decision 2026-09-14: it made near-misses worth nearly as
   much as escaping). α/β/ζ must stay **exactly
   potential-based** (they may change learning speed, never the optimum); delta is the one sanctioned
   non-potential term, as a decaying exploration bonus. Every run before 2026-09-14 trained on
   `--reward-objective legacy`. See [10](10-reward-design.md).

---

## Out of scope (decided, with reasons)

- **Server-side RL serving** — needs PyTorch on a 256 MB VM; not required by DESIGN_REQUIREMENTS §7.2.
- **Cop parameter retuning** — would invalidate the human data (invariant 1).
- **Board-size variants / the multi-size ablation (§7.3)** — never implemented; §4.3 is still TBD and
  `course_1/2/3` are same-size scenario variants, not size tiers. Report as a limitation.
- **Blocking rule, multi-night play** — tagged `EXTEND()` in `training/env.py:54-58`, deliberately
  unbuilt.
