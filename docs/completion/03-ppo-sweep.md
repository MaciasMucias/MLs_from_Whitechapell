# 03 — PPO hyperparameter sweep

**Status:** not started
**Blocks:** 04
**Blocked by:** 01 (frozen cops), 02-C1 (coefficients must be logged)

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

## Steps

1. Sweep `--lr`, `--ent-coef`, and the reward coefficients exposed in 02-C1.
2. ~3M steps per config. Curriculum ON, frozen cops v2.
3. **Cluster: submit as job arrays of <=20 tasks** (`--array=0-19%9`) rather than the
   4-6 configs a local machine would allow. Per-person caps are 72 CPU / 20 queued jobs, giving 9
   concurrent CPU-only runs at `--n-envs 12 --n-workers 12`; a 3M-step run is ~5x shorter than a
   final run, so the sweep fits easily. Note each array task counts against the 20-job submit cap
   individually (see 02-C3), so a sweep wider than 20 configs must go out in waves, chained with
   `--dependency=afterany`. Sweep both Director arms if the array is cheap enough;
   that removes the fairness caveat below entirely.

## Judging

Primary: `eval/win_rate`. Secondary: `eval/mean_hideout_uncert` — a policy that wins by luck rather
than by staying unpredictable shows a low uncertainty score. Both are logged at every checkpoint save
(`train.py:497-503`).

**Expect materially lower win rates than the old runs.** The last run reached `eval/win_rate 0.92`
against the old, weaker cops. Against frozen v2 the numbers will drop — that is expected and is not
a regression.

## Fairness caveat, carried into 04

Tuning with the curriculum ON and reusing the config for the OFF arm mildly favours ON. Either sweep
both arms (if compute allows) or use the shared config and **state the choice explicitly in the
thesis**. Do not leave it implicit.

## Verification

Confirm every swept value appears in `wandb.config` (this is what 02-C1 buys) and that
`curriculum/difficulty` moves away from -1.0 during the runs — if it stays pinned, the Director is
not engaging and the sweep is measuring the wrong thing.

## Session log

- 2026-09-09 — cluster access confirmed; sweep widened from a hand-picked handful to a job array.
- 2026-09-09 — cluster caps (72 CPU / 2 GPU / 20 jobs) confirmed. Sweep runs 9-wide on CPU;
  `--n-envs 12 --n-workers 12` fixed here and must carry unchanged into 04.
