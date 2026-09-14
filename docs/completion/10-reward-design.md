# 10 — Reward design

**Status:** **ready to resubmit** with `--reward-objective stealth` (the array id is recorded in
[README](README.md)). Training rewards **winning stealthily**: −1 on a loss, 1 + 0.5 × hideout
uncertainty on a win. α/β/ζ are exact potential-based shaping, delta is kept as a decaying
exploration bonus, and every agent is **reported** on the full participant score. The first
submission, `1807573` with `--reward-objective score`, was cancelled ~1h in (see Finding 1b).

**Blocks:** 06 (its checkpoints come from this wave, not 04's)
**Blocked by:** 03 (lr/ent), 09 (`--curriculum-max-difficulty 0`)

---

## Why this workstream exists

The five reward weights (`training/env.py`) were set by game understanding, not measurement. The
one attempt to validate them, 03's wave 2, varied one weight at a time on 1 seed at 3M steps, against
a ~9.5-point noise floor, so it could not detect anything. 04's `sparse` arm then seemed to show
shaping doesn't help. Examined closely, that arm tested the wrong thing (below).

**Decision (user, 2026-09-14): Jack's objective is to win stealthily.**

## Finding 1 — the objective already existed: the participant score

Participants were not told only to win. The tutorial (`frontend_participant/index.html`, the scoring
slide) and the browser (`game.js`) scored them as:

| part | points | when |
|---|---|---|
| progress | 10,000 × (1 − d / d_max), **best value reached** in the game | every game |
| stealth | 5,000 × hideout uncertainty (share of zone nodes cops can't rule out) | only on escape |

`d_max` is the farthest Jack node **from this hideout**, not the map diameter. Normalised by 10,000,
the score is **progress + 0.5 · U · [win]**, in [0, 1) on a loss and [1, 1.5] on a win. The tutorial
even tells players that arriving too fast gives the hideout away.

So `--reward-gamma 0.5` was never really a guess: it's the study's own scoring rule. Like the cops
(`COPS_STUDY_V2`), it's **frozen** as `SCORE_STUDY_V1` in `engine/metrics.py`, and it's the
**measure** every agent and every human is reported on.

## Finding 1b — train on "win stealthily", report on the participant score

The participant score bundles two separate choices: the **stealth** weight, and **progress credit on
a loss**. Only the first is part of "win stealthily". The first implementation adopted both,
training on the full score, where a loss earns its best progress in [0, 1). That shrank the gap
between winning and dying one step from the hideout from ~2 to as little as ~0.1. Because progress
is the *best* reached, approaching early and then being caught kept the credit too. A 200k-step smoke
run scored 0.78 at a 7% win rate, earning mostly near-miss credit.

**Decision (user, 2026-09-14): option 2.** `--reward-objective stealth`, now the default:

| | loss | win |
|---|---|---|
| `stealth` (**trained on**) | −1 | 1 + 0.5 · U |
| participant score (**reported on**) | best progress, [0, 1) | 1 + 0.5 · U |
| `legacy` (every run before 2026-09-14) | −1 | 1 + 0.5 · U, with inexact shaping |

On a win the two are identical. They differ only in whether near-misses are paid. The comparison
with humans still uses the participant score, computed after the fact from each game. The small cost:
humans may have chased points (including progress on losses) while the agent chases escapes, so on
losing boards the comparison isn't quite like-for-like. The scoring screen was more feedback than
goal, so escaping is the better-grounded reading of what good Jack play is.

`--reward-objective score` stays available, and array `1807573` briefly ran it before cancellation.
Its checkpoints and logs are not results.

**The old RL terminal reward vs the study score.** A loss was a flat −1 while humans got progress
credit; `stealth` keeps that on purpose. The stealth bonus was already present.

## Finding 2 — the five terms are three different kinds of thing

By Ng, Harada & Russell (1999), a bonus of the form F = γΦ(s′) − Φ(s) cannot change which policy is
optimal, only how fast it's found.

| term | what it is | how to choose its weight |
|---|---|---|
| `gamma` 0.5 | **part of the objective** (stealth) | not chosen — inherited from `SCORE_STUDY_V1` |
| `alpha` 0.1 | potential-based shaping, Φ = −d/d_max | **sample efficiency only**, never final score |
| `beta` 0.05 | potential-based shaping, Φ = −P(cops place Jack here) | same |
| `zeta` 0.1 | potential-based shaping, Φ = nearest-cop distance / diameter | same |
| `delta` 0.01 | **decaying count-based exploration bonus**, δ/√n | final score and map coverage |

**α/β/ζ were only *almost* potential-based.** They used Φ(s′) − Φ(s) without the discount, and
added nothing at the terminal step, so they could in principle bias the optimum. Now exact:
`w · (γ·Φ(s′) − Φ(s))` with Φ(terminal) = 0 and γ = PPO's `--gamma`. `tests/test_reward.py` checks
the telescoping identity Σ γᵗ F_t = −Φ(s₀) on real trajectories.

**delta is deliberately not potential-based, and stays.** Visit counts are per env and persist across
episodes, so the bonus front-loads exploration and decays towards zero with experience. That is the
textbook form (MBIE-EB; Bellemare et al.'s pseudo-counts). With 195 Jack nodes, a typical node's
bonus is below 10% of its initial value after ~1.6% of training, while rarely visited nodes keep it
far longer. The user found in development that it raises final score by giving Jack a broader base
understanding of the map. It had never been measured in a seeded run, since wave 2 had no delta
variant, so this wave tests it as its own factor. It is judged on **map coverage** as well as score,
since coverage is the mechanism claimed.

> An earlier draft of this plan called delta "dead in practice" and proposed removing it, from the
> formula alone. That was wrong: front-loaded is not dead. Measure it; don't argue from the formula.

## Finding 3 — 04's `sparse` arm is not a shaping ablation

It set all five weights to zero, **including `gamma`**, so it removed part of the objective along
with the shaping. Its lower hideout uncertainty (0.65–0.69 vs 0.75–0.78 for shaped) is that removal
showing up, and it explains most of its apparent win-rate edge. A rough rescore of 04 on the
objective (final win rate × (1 + 0.5 · U), ignoring loss credit):

| 04 arm | ≈ objective |
|---|---|
| `director-on` | 1.27 |
| `sparse` | 1.26 |
| `director-off` | 1.23 |

The sparse advantage disappears. **Do not report "reward shaping doesn't help" from 04.** The
Director result (ON > OFF) holds under either reading.

## What was implemented (2026-09-14)

- **`engine/metrics.py`** — `SCORE_STUDY_V1` constants, `participant_score`, `hideout_distances`,
  `normalized_distance`, `normalized_distance_for_state`. One definition of the score.
- **`server/routes.py`** — `score_info` now calls those functions. It's a pure refactor: the old
  inline formula is kept verbatim in `tests/test_reward.py` and checked equal on every node of 25
  scenarios. Production behaviour is unchanged (invariant 3).
- **`training/env.py`** — `objective="stealth"` (default): terminal −1 / 1 + 0.5 · U, exact
  potential-based α/β/ζ (`discount` = PPO `--gamma`), delta unchanged. `objective="score"`: the same
  but a loss pays its best progress, exactly the participant score. `objective="legacy"`
  reproduces the pre-change reward for rerunning old waves. It's pinned by an embedded golden
  trajectory recorded from the old env (10 games, 5 wins); the only difference is ~1e-17
  float-summation order. Episode `info` now carries `score`, `reward_terms` and `explore`
  (coverage, visit entropy).
  - The env's early termination ("can't reach the hideout in time") is training-only; the server
    plays on. On such a loss the env credits progress so far, which can slightly under-credit
    compared with a human who could still creep closer, and never over-credits.
- **`training/train.py`** — `--reward-objective {score,legacy}` (default score); `--reward-gamma`
  defaults to `SCORE_STUDY_V1_STEALTH`; the env discount is wired to `--gamma`. Logs
  `charts/score`, `reward/<term>` and `reward/<term>_share`, and `explore/coverage` /
  `explore/visit_entropy`. Prints a `reward:` line every 10 updates, because on the cluster the
  Slurm log is the record.
- **`training/eval.py`** — `game_score()` scores a finished game exactly as a human's was. Eval
  reports `mean_score` as the first column and logs it as `eval/score`.
- **`analysis/sweep_report.py`, `export_results.py`** — parse `score=` and the `reward:` line. The
  per-seed aggregate for score uses the **last-5 eval mean**, not the best-ever value (see the eval
  caveat below). The CSVs gain `score`, `train_score`, `r_<term>`, `coverage` and `visit_entropy`.
  Old logs parse unchanged.
- **`analysis/compare.py`** — every human game is **replayed through the engine and scored**; the
  database stores moves and outcomes, not scores. The comparison reports participant score, and a
  paired, participant-clustered CI on the score difference, ahead of win rate.
  - A replay must reproduce the stored winner **and** number of rounds, or it is rejected rather
    than scored.
  - **The replay needs the cops the human faced.** A game from before the 2026-06-11 retune (fixture
    row 46) becomes a different game under `COPS_STUDY_V2`, so `replay_human` takes `cop_params`.
  - Verified on the real snapshot: **all 57 usable study games (19 participants) replay exactly
    under V2.** Humans average **0.679** (6,787 points): wins 1.381, losses 0.355.

Tests: `tests/test_reward.py` (12), `tests/test_score_logs.py` (5), additions to `test_compare.py`
and `test_train_args.py`. 862 passing.

### An eval caveat surfaced along the way

In-training evaluation seeds its RNG with `--seed` on **every** call (`train.py`, `eval_policy(...,
rng=random.Random(args.seed))`), so all ~69 evaluations of a run replay **the same 200 boards**.
`agent_best.pt` and any "best eval" figure are therefore selected on one fixed board set and biased
upward. That's why this wave reads **last-5 mean** score, and any headline number should come from a
fresh-seed re-evaluation of the final checkpoint. `agent_best.pt` selection is still by win rate; it
was left alone to keep resume and branch behaviour unchanged.

## The experiment — `slurm/manifests/reward.txt`

18 tasks = 3 nested reward conditions × 2 curriculum conditions × 3 fresh seeds (41/42/43), 15M
steps, `lr 3e-4`, `ent-coef 0.03`, `--reward-objective stealth`. Runs are named `w10s-*` so they
can't collide with the cancelled `1807573` runs' (`w10-*`) checkpoint directories.

| | curriculum (`--curriculum-max-difficulty 0.0`) | no curriculum |
|---|---|---|
| **obj** — objective only | `w10s-obj-cur` | `w10s-obj-off` |
| **dlt** — + delta | `w10s-dlt-cur` | `w10s-dlt-off` |
| **shp** — + delta + exact α/β/ζ | `w10s-shp-cur` | `w10s-shp-off` |

**How to read it, decided before the data exists:**

1. **obj → dlt measures delta.** Look at final score (last-5 eval) and `coverage`/`visit_entropy`
   from the training CSV. The user's prediction: higher early coverage, then higher final score,
   most visibly without the curriculum.
2. **dlt → shp measures α/β/ζ.** Look at steps to 80% / 90% of the arm's own final score, and the
   area under the eval score curve. A real final-score difference here would mean the shaping isn't
   potential-based after all, i.e. a bug to find, not a result to report.
3. **cur vs off within each row is 04 again under the true objective**, and gives three independent
   estimates of the curriculum effect. The hypothesis worth having is that **shaping and curriculum
   substitute for each other**, so shaping helps (if at all) mainly without the curriculum.
4. Use the ~9.5-point win-rate noise floor as a guide. In score units, compare per-seed half-ranges
   before believing any gap.
5. Winner's curse applies as in 04: if a condition is chosen from this wave, its number is a tuning
   estimate until re-measured on fresh seeds.

Submit:

```bash
ssh cluster 'cd ~/MLs_from_Whitechapel && git pull && \
  sbatch --array=0-17%9 docs/completion/slurm/array.sbatch \
         docs/completion/slurm/manifests/reward.txt'
```

Read and export with the usual recipe (`sweep_report`, then `export_results --prefix reward`, two
separate ssh calls).

## What comes after

- **delta helps** → keep 0.01. If α/β/ζ also help, tune delta's weight with them.
- **α/β/ζ speed learning up** → tune α, β, ζ (and δ) with Optuna TPE + ASHA on area under the score
  curve, ≥2 seeds per trial, ranges centred on the logged `reward/<term>_share`, and a fresh-seed
  confirmation of the winner. This needs a driver that submits trials as sbatch tasks; plan it as
  its own step.
- **α/β/ζ don't help** → drop them, report the null, and match DESIGN_REQUIREMENTS §6.1 (sparse
  baseline). Keep delta if item 1 says so.

## Session log

- 2026-09-14 — workstream created after 04. Established that the participant score is the objective
  (`gamma` inherited, frozen as `SCORE_STUDY_V1`) and that 04's sparse arm removed part of the
  objective, so it isn't a shaping ablation. Proposed removing delta from the formula alone; **the
  user corrected this** from development evidence, and delta became a tested factor instead.
- 2026-09-14 — implemented the score objective, exact potential-based α/β/ζ, a legacy mode pinned by
  a golden trajectory, score/term/coverage logging through to the CSVs, and human scoring in
  `compare.py`. Found and handled replay dependence on the cop preset (pre-retune games need
  `COPS_PRERETUNE_V1`); all 57 usable study games replay exactly under V2. Found that in-training
  eval replays one fixed board set, so "best" figures are selected; switched this wave's aggregate
  to the last-5 mean. 862 tests passing.
- 2026-09-14 — **200k-step local smoke run passed end to end** (4 envs, 401 SPS, no errors). The
  `reward:` line prints every 10 updates, eval carries `score=`, and both parsers export it (score in
  the eval CSV; `r_*`, `coverage`, `visit_entropy` in the training CSV). Two readings that confirm the
  implementation rather than the policy:
  - **The shaping terms behave as theory says they must.** Per-episode `alpha` sits at a flat ~0.07
    and `zeta` at ~−0.019 whatever the policy does, because exact potential-based shaping sums to
    −Φ(s₀) (≈ α·d₀/d_max and −ζ·copdist₀/diameter). A constant offset is the correct signature;
    the learning signal is in the per-step values. `beta` sums to ~0.0001, since Φ_β(s₀) = 0 (no
    cop belief before the first plan).
  - **delta decays as designed** (0.0068 → 0.0041 per episode), while `coverage` saturates at ~0.99
    within 200k steps. **Coverage is therefore uninformative at 15M; read `visit_entropy`**, which
    was still moving (0.86 → 0.79) as the policy concentrated on useful routes.
- **Risk to read the wave for, flagged in advance.** Under the score objective a loss earns its best
  progress, so dying next to the hideout (~0.9) is worth almost as much as a bare win (1.0). At 200k
  steps the smoke policy scored ~0.78 with only a 7% win rate, earning mostly progress credit. If
  the 15M arms plateau on approach-without-winning, their eval score will fall **below** 04's legacy
  arms on the same metric. So once 04 is re-evaluated with the score column, it's the control for
  whether the new terminal reward is harder to optimise. That would be a finding about the objective,
  not a reason to change it quietly.
- 2026-09-14 — **the approach-without-winning risk was the objective, so the objective changed.** The
  user asked why a loss earned partial credit. The first implementation had bundled progress credit
  on losses in with the stealth term; "win stealthily" requires only the latter. They are separate
  decisions and should have been put to the user separately. Cancelled `1807573` (10 tasks, ~1h in).
  Added `--reward-objective stealth` (−1 / 1 + 0.5 · U, exact shaping) as the default, kept `score`
  selectable, and renamed the wave's runs `w10s-*`. Agents are still reported on the participant
  score. 955 tests passing.
