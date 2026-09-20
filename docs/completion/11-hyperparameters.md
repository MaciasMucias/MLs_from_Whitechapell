# 11 — Hyperparameter validity

**Status:** **DONE (2026-09-20, array `1808727`, 9/9).** `--ent-coef 0.03` stands. 03 chose a
defensible value for a void reason; the value survives scrutiny, and **no experiment needs
re-running on its account**.

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

## RESULT (2026-09-20, array `1808727`, 9/9)

Last-5 mean participant score, 3 fresh seeds (51–53), everything else exactly the recommended arm:

| `ent-coef` | score | range | win% |
|---|---|---|---|
| 0.01 | **1.303** ±0.014 | [1.292–1.319] | 87.8% |
| **0.03** (incumbent) | 1.280 ±0.032 | [1.253–1.317] | 86.7% |
| 0.003 | 1.251 ±0.034 | [1.226–1.293] | 81.0% |

**Against the bar set before the data existed** — a challenger had to beat the incumbent's 1.309 by
more than the per-arm half-range — **0.01 does not clear it.** It scores 1.303, marginally *below*
the incumbent, with heavily overlapping ranges.

**The wave cannot separate 0.01 from 0.03, and here is the cleanest way to see it.** The *same*
configuration scored **1.309** on seeds 41–43 (`w10s-obj-cur`) and **1.280** on seeds 51–53
(`w11-ent003`) — those two arms are flag-for-flag identical. A **0.029** gap from seed choice alone,
against a 0.023 gap between the two entropy values. Pooling all six 0.03 seeds gives **1.294**.

**0.003 is genuinely worse** (−0.05 against 0.01, ranges nearly disjoint). So entropy does matter
under the curriculum, but only downward: there is an interior optimum somewhere in [0.01, 0.03] and
the wave cannot say where inside it. Either value is defensible; 0.03 is kept because it is the
incumbent and every reported result already uses it.

**Keep `--ent-coef 0.03`.** Nothing downstream changes — 04, 09, 10 and 06 all stand.

### A note on reading runs mid-flight

At 5.3M steps (a third of the budget) this wave read 1.020 for 0.003, 1.022 for 0.01 and 0.867 for
0.03 — i.e. exactly inverted from the final ordering, with the eventual worst arm apparently
leading. Rank on the last-5 mean at the full budget, as the pre-registered rule said; a mid-run
glance is not evidence.

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
- 2026-09-20 — **done, 9/9.** `ent-coef` 0.01 scores 1.303 and 0.03 scores 1.280 on fresh seeds, but
  the same config scored 1.309 and 1.280 on two different seed triples, so the wave cannot separate
  them; 0.003 is clearly worse (1.251). The incumbent stands and nothing is re-run. The entropy
  rationale from 03 was void, the value was not. Mid-run numbers at 5.3M had the ordering exactly
  inverted, which is the case for the pre-registered ranking rule.
- 2026-09-20 — **the wave paid for itself twice.** `w11-ent003` is `w10s-obj-cur` flag-for-flag, so
  the recommended arm now has 6 seeds for free, and the 0.029 gap between its two seed triples is
  the calibration that motivated the seed top-up (array `1808812`).
