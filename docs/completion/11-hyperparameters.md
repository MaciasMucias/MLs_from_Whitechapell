# 11 — Hyperparameter validity

**Status:** **IN FLIGHT — array `1808727`**, 9 tasks, submitted 2026-09-19, ~7h.
**Blocks:** nothing. If 0.03 holds, nothing downstream changes.
**Blocked by:** nothing.

---

## Why this exists

An audit on 2026-09-19 asked a simple question — *have we actually tuned the training
hyperparameters?* — and the honest answer is "two of them, at a tenth of the final budget, under a
different reward".

| parameter | evidence | verdict |
|---|---|---|
| `lr` 3e-4 | 03 wave 1: beat 1e-4 and 1e-3 in **both** arms, every top-five run, ~30-point spreads against a 9.5-point noise floor | **safe** — a margin that size does not reverse because the reward changed |
| `ent-coef` 0.03 | 03 wave 1 ON-arm gradient 39.5 / 36.0 / 18.0; OFF arm flat (1.5-point band); probe `1806936` showed an interior optimum | **rationale void** — see below |
| `n-steps` 256, `n-epochs` 4, `minibatch-size` 256, `gamma` 0.99, `gae-lambda` 0.95, `clip-coef` 0.2, `vf-coef` 0.5, `max-grad-norm` 0.5, network 1416→512→256→256 | never varied | CleanRL defaults, reported as such |
| `n-envs`/`n-workers` 12/12 | fixed by decision, not evidence, to keep batch size comparable across waves | deliberate, documented |

**The entropy rationale does not hold.** 0.03 was chosen because entropy showed a steep gradient in
the ON arm and none in the OFF arm — but wave 1's ON arm had `curriculum/difficulty` pinned at
−1.000 for its whole run. The curriculum never engaged. That gradient describes training against
fully-suppressed cops, which is not a configuration used since.

Three further reasons nothing from 03 transfers cleanly: it ran at **3M steps** (the reported agents
train to 15M), **one seed per cell** against a ~9.5-point noise floor, and under the **legacy
reward**, ranked on win rate rather than the participant score. Strictly, **no hyperparameter had
been validated under the configuration the thesis recommends.**

## The experiment — `slurm/manifests/entropy.txt`

9 tasks: `ent-coef` ∈ {0.003, 0.01, 0.03} × seeds 51/52/53, everything else exactly the recommended
arm (`w10s-obj-cur`): stealth objective, curriculum at ceiling 0.0, no shaping, no delta, 15M steps,
`lr 3e-4`. One window, ~7h.

`lr` is not re-tested — its separation was decisive and consistent across arms and probes. The
untouched defaults are not swept either; that would need a multi-fidelity search rather than a grid,
and is out of scope unless this wave suggests the optimiser settings matter more than assumed.

**Decided before the data lands:**

- Rank on the **last-5 mean of `eval/score`**, never best-ever (in-training eval replays one fixed
  board set).
- The incumbent is `w10s-obj-cur`: **1.309** last-5, **1.326** re-scored on 2,000 fresh boards.
- A challenger must clear the incumbent by **more than the per-arm half-range** (0.013–0.05 in wave
  10) before anything is re-run on its account. Expect that bar not to be cleared.
- **If 0.03 holds, 03 made the right choice for the wrong reason** — write that down rather than
  quietly keeping the number.
- If 0.01 or 0.003 wins clearly, the recommended arm was mistuned; 06's comparison would need
  re-running on a retrained arm, and that is the only downstream consequence.

## Session log

- 2026-09-19 — workstream created after an audit against `DESIGN_REQUIREMENTS` §7 found that the
  entropy choice rested on an arm where the curriculum never engaged. Submitted the 9-task
  re-check under the final configuration as array `1808727`.
