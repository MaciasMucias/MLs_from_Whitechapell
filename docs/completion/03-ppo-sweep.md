# 03 — PPO hyperparameter sweep

**Status:** **DONE** (2026-09-10). Waves 1+2 and the entropy probe all complete; chosen config
is `--lr 3e-4 --ent-coef 0.03`.
**Blocks:** 04
**Blocked by:** ~~01 (frozen cops), 02-C1 (coefficients must be logged)~~ — both landed 2026-09-09.

---

## Goal

Pick the training configuration for the final runs. Short runs (~3M steps), against frozen cops v2,
**curriculum ON**.

## Why it matters

The existing 7 runs were exploratory and are all stale (trained against pre-retune cops). The final
runs are expensive and their config should be a considered choice, not inherited from a test run.

## Current state

All hyperparameters are CLI args — there is no YAML/JSON training config. (`configs.json` at repo
root is unrelated: it holds *cop* params for `tools/pareto_compare.py`.)

Defaults, `training/train.py:535-579`:

| flag | default | | flag | default |
|---|---|---|---|---|
| `--total-steps` | 5,000,000 | | `--clip-coef` | 0.2 |
| `--n-steps` | 256 | | `--ent-coef` | 0.01 |
| `--n-envs` | 12 | | `--vf-coef` | 0.5 |
| `--n-workers` | 6 | | `--max-grad-norm` | 0.5 |
| `--n-epochs` | 4 | | `--seed` | 27 |
| `--minibatch-size` | 256 | | `--curriculum-kp` | 0.1 |
| `--lr` | 3e-4 | | `--curriculum-target-low/high` | 0.4 / 0.6 |
| `--gamma` | 0.99 | | `--eval-games` | 200 |
| `--gae-lambda` | 0.95 | | `--wandb-mode` | `offline` |

Batch = `n_steps * n_envs` = 3,072; 12 minibatches of 256, x4 epochs.

Architecture (`training/model.py:8-41`): MLP, shared trunk `obs_dim -> 512 -> 256 -> 256` (ReLU),
separate policy and value heads. No orthogonal init, no LayerNorm. Action masking via
`masked_fill(~mask, -inf)` with `entropy.nan_to_num(0.0)`.

Observation: 1,416 dims (`training/obs.py:24-109`). Cop distance vectors are sorted nearest-first
(`obs.py:92`) for permutation invariance.

Two deviations from the CleanRL reference the file claims to follow, worth knowing if reproducing:
**unclipped value loss**, and **no orthogonal init**. Not bugs.

## The grid — two waves, 30 runs

Written and validated. Both live in [`slurm/manifests/`](slurm/manifests/).

### Wave 1 — `sweep_w1.txt`, 18 tasks, `--array=0-17%9`, ~2.5h wall

`--lr {1e-4, 3e-4, 1e-3}` x `--ent-coef {0.003, 0.01, 0.03}`, **run on both Director arms**.

Sweeping both arms is what buys the answer to the fairness caveat below, and it is cheap: 3M steps
at the pilot's 682 SPS is ~1.2h per task, 18 tasks nine-wide is ~2.5h. Doing it any other way would
have meant defending a choice in the thesis rather than measuring it, for the sake of an afternoon.

### Wave 2 — `sweep_w2.txt`, 12 tasks, `--array=0-11%9`, ~1.6h wall

The five reward coefficients, one at a time off the defaults, at wave 1's chosen `lr`/`ent`.
Curriculum ON, because 04's reward ablation is an ON-arm run.

Line 1 is a **control**: wave 1's winner repeated unchanged. Its gap to the corresponding wave-1 run
is this sweep's run-to-run noise floor at a single seed — probes inside that gap are null results,
not small effects. Worth having, given that 04 exists precisely because PPO seed variance can swamp
the Director effect. Line 12 is the fully sparse config, run here at 3M steps as a cheap early
warning that 04's ablation arm may learn nothing at all.

### Why two waves and not one

18 + 12 = 30 exceeds the 20-job `MaxSubmitJobs` cap, which counts pending array tasks individually
(02-C3). Wave 2 also *cannot* be written until wave 1 answers — `<LR>`/`<ENT>` are placeholders. In
practice: submit wave 1, read it, edit `sweep_w2.txt`, submit wave 2. The
`--dependency=afterany:$J1` form in the [slurm README](slurm/README.md) is for firing both at once;
`array.sbatch` reads the manifest at task start, not at submit time, so the file can be edited while
wave 1 runs.

## Steps

1. ~~Sweep `--lr`, `--ent-coef`, and the reward coefficients exposed in 02-C1.~~ Grid above.
2. ~~3M steps per config. Curriculum ON, frozen cops v2.~~ Both encoded in the manifests. Frozen
   cops need no flag: `COPS_STUDY_V2` is `HeuristicCops()`'s default as of 01.
3. Submit wave 1, pick `lr`/`ent` from the ON block, edit `sweep_w2.txt`, submit wave 2.
4. Carry the winning `lr`/`ent` into `final.txt`'s `<LR>`/`<ENT>`, and the winning reward
   coefficients into 04 (as explicit `--reward-*` flags if they differ from the defaults).

**Flag names:** the reward coefficients are `--reward-alpha/-beta/-delta/-gamma/-zeta`. Not
`--gamma` — that is PPO's discount factor, and passing it would silently set the discount to zero.
See 02-C1; `tests/test_run_manifests.py` enforces it.

## RESULTS — wave 1, 2026-09-09 (array `1806901`, 18/18 complete)

All 3M steps, frozen cops v2, seed 27, batch 3,072. Ranked on Director-free `eval/win_rate`.
Reproduce with `uv run python -m analysis.sweep_report 'logs/wc-train_1806901_*.out'`.

### Both arms choose `lr=3e-4`, decisively

| arm | best config | win% | runner-up | spread across arm |
|---|---|---|---|---|
| ON | `lr 3e-4 / ent 0.03` | **39.5** | `3e-4 / 0.01` (36.0) | 32.5 |
| OFF | `lr 3e-4 / ent 0.003` | **35.5** | `3e-4 / 0.03` (34.5) | 30.0 |

Every top-five run in both arms uses `3e-4`. `1e-4` and `1e-3` are far behind. Spreads of ~30 points
are far outside noise, so **the sweep genuinely separated the configs** — this is signal, not an
argmax over a flat surface.

### The arms disagree on entropy, but only nominally — take 0.03

The OFF arm's top three are 35.5 / 34.5 / 34.0 across *all three* entropy values: a 1.5-point band.
That is a flat axis, not a preference. The ON arm's is 39.5 / 36.0 / 18.0 — a real gradient.

So: **entropy matters with the Director on and is close to irrelevant with it off.** `0.03` wins
outright in the ON arm and costs ~1 point in the OFF arm, which makes it the shared config on the
evidence rather than by fiat. **This resolves the fairness caveat below** — it did not have to be
argued away, the data settled it.

**Entropy interacts with `lr`; it is not monotone on its own.** At `3e-4` more entropy helps
(18.0 → 36.0 → 39.5). At `1e-3` it hurts badly (25.5 → 25.5 → 12.0). Do not carry "more entropy is
better" into 09 or 04 as a general rule.

### The arms are behaviourally different, in the direction 09 predicts

| arm | copdist | arrest% |
|---|---|---|
| ON (9 runs) | 0.99 – 1.11 | 48 – 78% |
| OFF (9 runs) | 1.19 – 1.49 | 29 – 50% |

Director-trained policies run **closer** to cops and are arrested more; no-Director policies keep a
wider berth. Same signature as the single 2026-05-23 pair, now with nine runs per arm — a much
stronger basis for [09](09-director-tuning.md)'s risk-aversion discussion, and again the *opposite*
of "the Director makes Jack cautious".

### The curriculum never engaged

`curriculum/difficulty` stayed at **−1.000 in all nine ON runs**. So the ON arm here is "training
against maximally-suppressed cops", not a curriculum. **The wave-1 ON/OFF result is therefore not
the Director comparison the thesis wants**; do not report it as one.

**Why (established 2026-09-10, and it is a property of the run length, not of the task):** the
historic 10M-step Director run's first difficulty uptick was at step **3,256,320**. Every run here
is 3M steps, so they all stop ~250k steps short of the point where the controller first acts. See
[09](09-director-tuning.md) — this is what its branch design exists to get past.

### Boundary probe (array `1806936`) — `0.03` is an interior optimum

`ent-coef 0.03` was the largest value in the grid, so the winner sat on the edge. Two extra runs at
`0.06` and `0.10` settled it. At `lr=3e-4` the curve is a clean inverted-U:

| ent-coef | 0.003 | 0.01 | **0.03** | 0.06 | 0.10 |
|---|---|---|---|---|---|
| eval win% | 18.0 | 36.0 | **39.5** | 26.5 | 16.5 |

It turns over sharply past 0.03, so the boundary was worth checking and the choice holds.

## RESULTS — wave 2, reward coefficients (array `1806976`, 12/12)

**Nothing here is significant, and the control is why.** `sw2-control` repeats wave 1's winner
byte-identically and scored **30.0%** against that run's **39.5%** — a 9.5-point gap between the
same command run twice. That is the sweep's run-to-run noise floor, and wave 2's entire spread is
13.0 points.

| run | win% | vs control |
|---|---|---|
| `gamma0` | 37.5 | +7.5 |
| `alpha0` / `alpha03` / `beta015` | 36.5 | +6.5 |
| **`sparse`** (all shaping off) | **33.0** | **+3.0** |
| **`control`** | **30.0** | — |
| `zeta0` | 25.0 | −5.0 |
| `gamma10` | 24.5 | −5.5 |

Every delta is inside the noise floor, so **keep the default reward coefficients** — wave 2 gives no
evidence for changing any of them. One observation does survive: the **fully sparse** config scored
*above* the control, so reward shaping cannot be shown to beat pure terminal reward at 3M steps.
That is a real result for 04's ablation arm, and not the one it was set up to expect.

The 9.5-point floor traced to a bug: `--seed` was not seeding torch, numpy or global `random`, so
same-seed runs were independent draws. Fixed 2026-09-09 (see 02); runs from `1807070` onward are
properly seeded.

## Judging

Primary: `eval/win_rate`. Secondary: `eval/mean_hideout_uncert` — a policy that wins by luck rather
than by staying unpredictable shows a low uncertainty score. Both are logged at every checkpoint save
(`train.py:497-503`).

**Expect materially lower win rates than the old runs.** The last run reached `eval/win_rate 0.92`
against the old, weaker cops. Against frozen v2 the numbers will drop — that is expected and is not
a regression.

## Fairness caveat, carried into 04

Tuning with the curriculum ON and reusing the config for the OFF arm mildly favours ON. **Resolved
by sweeping both arms** in wave 1 — compute allowed it easily. If the two arms pick different
`lr`/`ent`, that itself is a finding worth reporting, and 04 should then say which arm's config it
used and why.

## Verification

1. Every swept value appears in `wandb.config` — this is what 02-C1 buys, and it is the whole reason
   the sweep is worth running at all.
2. `eval/win_rate` separates the configs. If all 18 wave-1 runs land within noise of each other,
   3M steps is too short to rank them and the sweep needs lengthening, not re-gridding.

### `curriculum/difficulty` will probably stay at -1.0, and that is not a failure

An earlier draft of this file said that if difficulty stays pinned at -1.0 the Director is not
engaging and the sweep is measuring the wrong thing. **That is wrong**, and acting on it would
waste cluster time chasing a non-bug.

`INITIAL_DIFFICULTY` is -1.0, and in `CurriculumDirector` negative difficulty means *suppression*:
at -1.0 every discovered `visited_at` entry is dropped except the public depth-0 start
(`agents/curriculum_director.py:56-64`). Pinned at -1.0 the Director is engaging **maximally** —
cops are nearly blind. The ON and OFF arms are genuinely different games there.

What is pinned is the *ramp*, not the intervention. The P-controller only moves difficulty when win
rate leaves the `[0.4, 0.6]` deadband, and below 0.4 it pushes toward easier — already clamped. The
02-C3 pilot saw 3-8% win rates at 200k steps. So difficulty stays at -1.0 until Jack clears **60%**
against fully-suppressed cops, which 3M steps may well not reach.

The real thing to check: does any ON run's `charts/win_rate` cross 0.6 and pull
`curriculum/difficulty` off the clamp? If none does even in 04's 15M-step runs, then the curriculum
never actually curricularises, the ON arm reduces to "training against permanently blinded cops",
and **that is the honest finding to report** — not a bug to hide.

## Session log

- 2026-09-09 — cluster access confirmed; sweep widened from a hand-picked handful to a job array.
- 2026-09-09 — cluster caps (72 CPU / 2 GPU / 20 jobs) confirmed. Sweep runs 9-wide on CPU;
  `--n-envs 12 --n-workers 12` fixed here and must carry unchanged into 04.
- 2026-09-09 — **grid designed and manifests written; ready to submit.** Two waves: `sweep_w1.txt`
  (18 tasks, lr x ent on both Director arms) and `sweep_w2.txt` (12 tasks, reward coefficients one
  at a time, plus a control and the sparse config). Sweeping both arms was cheap enough (~2.5h) to
  close the fairness caveat outright rather than caveat it in the thesis. Every line is parsed by
  `tests/test_run_manifests.py` on the laptop, which also blocks the `--gamma` collision from 02-C1.
  **Corrected this file's own verification criterion:** difficulty pinned at -1.0 means the Director
  is suppressing maximally, not that it is disengaged — the previous wording would have sent a
  future session hunting a non-existent bug.
- 2026-09-09 — **wave 1 submitted: job array `1806901`, `--array=0-17%9`.** Both nodes were idle and
  the queue empty at submission. 9 tasks running immediately (5 on `stud-1`, 4 on `stud-2`), 9
  pending under the `%9` throttle with reason `JobArrayTaskLimit` — which is the throttle working,
  not an error. Task 0-8 map to the nine ON configs in manifest order, verified from the logs.
  Cluster is at `688966c`.
- 2026-09-09 — **found and fixed a real trap during pre-flight: `uv run` syncs by default.** A bare
  `uv run` inside `srun` reinstalled 11 packages before starting, so nine array tasks would each
  have mutated the shared `.venv` on Lustre concurrently — the exact failure `slurm/README.md`
  warns is "the single easiest way to lose a whole array", reachable *without* anyone writing
  `uv sync`. `env.sh` now exports **`UV_NO_SYNC=1`**, and the login-node sync must NOT use
  `--no-dev` (a `--no-dev` venv reads as stale to `uv run`, which with `UV_NO_SYNC=1` would surface
  as a missing import instead). Verified on a compute node: `UV_NO_SYNC=1`, `OMP_NUM_THREADS=1`,
  imports fine, nothing installed.
- 2026-09-09 — pre-flight smoke run on `stud-1` before committing the array: 12 envs / 12 workers,
  ~730 instant SPS (above the 682 the pilot measured), new eval diagnostics present,
  `agent_best.pt` written. **Never run this check on the login node** — its CPU lacks x86-64-v2 and
  numpy aborts with a misleading error.
