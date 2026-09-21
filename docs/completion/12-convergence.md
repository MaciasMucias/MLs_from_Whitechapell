# 12 — Convergence: does the curriculum effect survive a longer budget?

**Status:** **IN FLIGHT — array `1809213`**, submitted 2026-09-21 ~22:45. 12 tasks, 9 then 3, ~37h —
expected down around **2026-09-23 12:00**. Walltime 23:55:00 confirmed on the tasks.
**Blocks:** nothing. Either outcome is a finding; neither invalidates a result.
**Blocked by:** nothing.

**Deadline:** cluster access may end around 2026-10-01. Everything this workstream needs from the
cluster must be pulled before then — see the pull at the bottom.

---

## Why this exists

Every result in the thesis is "at 15M steps", and the score was still rising in **every** wave-10
arm when the budget ran out (trend over the last 20% of training: +0.022 to +0.041 per 1M steps).

`training/train.py` anneals the learning rate **linearly to zero over `--total-steps`** (lines
~371–373). So 15M is where the schedule stopped the runs, not where they plateaued. That leaves the
headline — the curriculum is worth +0.114 score — ambiguous between two readings:

| | what it means |
|---|---|
| **the gap persists** | the curriculum produces a better *final* policy |
| **the gap closes** | the curriculum is a *sample-efficiency* method — the same destination, reached ~36% faster |

Both are findings the thesis can make. At 15M it cannot say which, and it is the most obvious
question an examiner could ask about the headline.

## Design — `slurm/manifests/convergence.txt`

`obj-cur` vs `obj-off` — the recommended arm and its matched control — **from scratch at 40M
steps** (2.7× the budget), 6 fresh seeds each (81–86), flags otherwise identical to wave 10.

- **From scratch, not resumed.** Resuming a 15M run would restart a learning-rate schedule that had
  already reached zero. That is not what a 40M run is, and it would confound the comparison.
- **6 seeds, not 3.** If the gap shrinks, 3 seeds (threshold ~0.111) could not distinguish "closed"
  from "smaller". 6 (~0.079) can bound it.
- **Walltime.** 40M at the slowest SPS seen here (~575) is ~19.3h. Submitted with
  `--time=23:55:00` (the cap is 24h) for margin.

## How to read it — decided before the data exists

Rank on the **last-5 mean of `eval/score`** at 40M, pooled over 6 seeds per arm, exactly as wave 10.
The 15M reference points are `obj-cur` **1.295** and `obj-off` **1.175** (gap +0.120).

1. **Gap at 40M ≥ the 6-seed threshold (~0.079)** → the curriculum improves the final policy. The
   headline stands as a statement about outcomes, not just speed.
2. **Gap below the threshold, and `obj-off` has risen toward `obj-cur`** → the gap closed. Report
   the curriculum as a sample-efficiency method: same destination, ~36% faster. The 15M headline is
   still true *at 15M* and should be stated with the budget attached.
3. **Gap below the threshold but neither arm moved much** → ambiguous; report the bound, not a
   conclusion.
4. **Check the tail slope at 40M too.** If both arms are *still* rising, 40M is not convergence
   either. Say so — "the gap at 2.7× the budget" is the honest claim then, not "at convergence".
5. **A 40M run is not a 15M run continued.** Its learning rate is higher at every step before the
   end, so do not compare a 40M run's 15M checkpoint to wave 10's 15M results.

## Session log

- 2026-09-21 — workstream created on learning that cluster access may end in ~10 days. The
  convergence question is the one open caveat on the headline that only the cluster can answer.
  Submitted 12 tasks; pulled every cluster-only checkpoint and the W&B offline runs in parallel.
