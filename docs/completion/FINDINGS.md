# Findings worth putting in the thesis

Everything established experimentally between 2026-09-09 and 2026-09-13, with the
evidence and the caveats. Written to survive: if a later session has none of the
conversation, this file plus `results/` should be enough to write the chapters.

All win rates are **`eval/win_rate`: 200 games per evaluation, against
full-strength `COPS_STUDY_V2`, with `director=None`** (invariant 2), ~69
evaluations per run, 3 seeds per arm. Raw data in [`results/`](results/).

---

## 0. The headline, measured on the objective (2026-09-19, array `1807588`)

Everything in sections 1–5 was measured under the **legacy** reward (terminal ±1) and reported as win
rate. Workstream 10 re-ran the comparison under the reward the thesis actually argues for — win
stealthily — and reports the **participant score**, the quantity human participants were scored on
(`engine/metrics.py`, `SCORE_STUDY_V1`; humans average **0.679**). Prefer these numbers.

| reward | curriculum | no curriculum |
|---|---|---|
| **objective only** | **1.309** ±0.013 · 89.8% win | 1.141 ±0.051 · 70.7% |
| + delta | 1.302 ±0.024 · 88.8% | 1.161 ±0.045 · 70.7% |
| + delta + shaping | 1.265 ±0.062 · 85.2% | 1.245 ±0.039 · 82.0% |

3 seeds per arm, 15M steps, mean of the last five evaluations (not best-ever — in-training eval
replays one fixed board set, so a maximum over ~69 evaluations is selected on those boards).

**1. The curriculum is the largest effect in the project: +0.168 score, +19 win points,
non-overlapping ranges, and 42% fewer steps to reach a score of 1.10 (7.9M vs 13.7M).** It reaches
1.20 and 1.30 where the baseline never does inside the budget.

**2. Shaping and the curriculum are substitutes.** Shaping is worth +0.104 without the curriculum and
nothing with it (1.265 vs 1.309, and slower to every threshold), where it also triples the seed
spread and produced one collapsed seed (78.0% vs 91.5% win). **This is why 04's Director effect
looked small (+2.5 points): both of its arms had shaping on, so the OFF arm already had most of what
the curriculum provides.** The single most useful reframing to come out of this project.

**3. It is a sample-efficiency result, not a claim about optima.** No arm converged at 15M: the trend
over the last 20% of training is positive in every arm (+0.022 to +0.041 score per 1M steps). Exact
potential-based shaping cannot change which policy is optimal, so at a fixed pre-convergence budget
its effect appears as a score difference. **State the budget with every number.**

**4. Behaviour backs it up.** Curriculum arms hold a wider berth from cops (copdist 1.46 vs 1.34) and
lose less to *both* arrest (8.3% vs 19.8%) and the clock (1.8% vs 10.2%), while leaving the hideout
*more* ambiguous (0.79 vs 0.74). Better on both halves of the objective, not a trade.

**Recommended configuration:** objective only + curriculum. Simplest, best, tightest across seeds.

---

## 1. The headline: the curriculum works, if it is forbidden from injecting

| configuration | final eval win% | best | last-5 |
|---|---|---|---|
| **curriculum, ceiling 0.0** (`cap000`) | **92.0** | 92.8 | 91.2 |
| hand-picked two-phase (`dbr-off`) | 91.2 | 93.2 | 90.7 |
| fixed −0.5 (`dbr-fix05`) | 86.8 | 88.3 | 84.7 |
| **no curriculum** (`fs-off`) | 83.0 | 85.2 | 82.7 |
| unbounded Director (`fs-on`) | 82.2 | 83.2 | 81.3 |

**~+9 points over no curriculum**, robust to whether you report best, final or a
last-5 mean. `--curriculum-max-difficulty 0.0` is the recommended configuration.

**Replicated from scratch, which removes the table's one caveat.** `cap000` above
branches from a base trained at −1.0, so strictly it is "suppressed prefix, then
capped curriculum" while `fs-off` is a pure from-scratch control — a mismatch a
reader is entitled to object to. Array `1807390` ran the same configuration from
scratch: `fsc000-i100` reaches **92.0%** [91.0–93.0] best against `fs-off`'s
85.2% [81.5–88.5], **+6.8 points on a fully matched pair, paired per seed**. The
headline survives without the branch design.

> **Caveat on the exact number.** `--curriculum-max-difficulty 0.0` was *selected*
> as the best of ~6 ceiling arms, so 92.0% is a maximum over arms at a ~9.5-point
> noise floor and is biased upward — the usual winner's curse. The OFF arm was
> never selected on, so the *sign* of the effect is safe; the magnitude is not.
> Workstream 04 re-measures the chosen configuration on fresh seeds 31/32/33 and
> **04's figure is the one to quote in the thesis**, with this as the tuning
> estimate that chose the configuration.

### The dose-response peaks exactly at zero

Difficulty held constant after a suppressed warmup:

| fixed difficulty | −1.0 | −0.5 | **0.0** | +0.15 |
|---|---|---|---|---|
| eval win% | 50.7 | 88.3 | **93.2** | 72.3 |

A clean inverted-U. **Suppressing too long hurts; injecting hurts more per unit.**
Giving cops knowledge they never earned costs ~21 points at +0.15 and lands at
61.3% when an uncapped controller reached +0.54.

### Adaptive scheduling is not better than a hand-picked schedule

92.0% vs 91.2% — indistinguishable. **The value of adaptivity is that it finds the
transition point without tuning**, not that it finds a better one. That is a
defensible and interesting claim: an automatic curriculum matches a hand-designed
one, provided it cannot overshoot.

### Ramp shape is irrelevant, and here is why

`kp` (0.02 / 0.1 / 0.3), a per-update rate limit, and a monotone ratchet all
produced the same result (81.7–85.8%, overlapping). **Every uncapped adaptive arm
converged to difficulty +0.16 to +0.22 regardless of how it got there.** The
controller reaches the same equilibrium by different routes, so the route cannot
matter. Only the *ceiling* changes the destination.

---

## 2. Sample efficiency — the benefit the final win rate hides

Global steps to first reach a given eval win rate (branch arms stitched onto
their base run, so counts are comparable at equal total compute):

| arm | 50% | 70% | 80% | 85% |
|---|---|---|---|---|
| curriculum, no injection | 5.2M | 7.4M | **9.2M** | **11.2M** |
| hand-picked two-phase | 5.0M | 7.4M | 9.5M | 11.4M |
| no curriculum | 7.0M | 10.9M | 13.5M | **never** |
| unbounded Director | 5.4M | 8.0M | 13.3M | **never** |

**~32% fewer steps to 80%**, and the baseline never reaches 85% inside 15M while
the curriculum gets there at 11.2M with 4M to spare.

**Mechanistic detail worth reporting:** the unbounded Director converges *well
early* (5.4M to 50%) then stalls (13.3M to 80%, never 85%). Injection ramps up as
the agent improves, so it bites hardest in late training — exactly where a ceiling
removes it. This is the cleanest single illustration of why the bound matters.

Reproduce: `uv run python -m analysis.convergence`

---

## 3. How the policies lose, not just how often

Final evaluation, mean over seeds:

| arm | copdist | arrest% | timeout% |
|---|---|---|---|
| curriculum, no injection | 1.50 | **7.2** | **0.8** |
| no curriculum | 1.48 | 12.2 | 4.8 |

**Same cop distance, 40% fewer arrests, 6× fewer timeouts.** The curriculum agent
is not winning by being more cautious — it is simply playing better. Worth stating
explicitly, because "trained against weaker opponents" invites the opposite
assumption.

The failure-mode split is a sharp diagnostic elsewhere too. A policy trained at
permanent full suppression (`fix10`, 50.7%) shows copdist ~1.03 and **arrest ~52%**
— it never learned cops are dangerous, because in training they were not.

---

## 4. Order matters: suppress first, then relax

Same ceiling and band, differing only in whether the first 4.4M steps ran
suppressed:

| | eval win% |
|---|---|
| with suppressed warmup | 73.2 |
| from scratch | **56.2** |

**+17 points for the warmup.** Boosting the cops early, without first letting the
agent learn against weak ones, is the worst thing tested short of permanent full
suppression. Its signature is copdist ~1.12 and arrest ~43%.

---

## 5. Generalisation: policies overfit to the cop configuration they trained on

Measured on the project's only historic Director run and its matched control,
300 games, paired:

| scored against | Director ON | no Director |
|---|---|---|
| the cops both trained on (`COPS_PRERETUNE_V1`) | **74.7%** | 63.7% |
| the retuned cops neither saw (`COPS_STUDY_V2`) | 58.7% | **62.7%** |

The curriculum bought 11 points on home cops and cost 16 when the cops changed.
**Report a held-out cop configuration alongside the frozen one** —
`uv run python -m training.eval <ckpts> --held-out-cops` does this. A policy
scored only against its training opponent cannot be distinguished from one that
memorised that opponent's decision rule.

This is also a limitation to state plainly: `COPS_STUDY_V2` was itself tuned with
a trained Jack as adversary (provenance unrecorded), so historic policies may be
biased against it.

---

## 6. Human-vs-RL comparison

**N as of 2026-09-12: 19 usable participants, 57 usable games** — and still
growing (42 rows arrived in the three days to 09-12). **Re-pull before the final
analysis.** Human win rate ~30%. Course completion 48% (11 of 23 sessions at the
09-09 snapshot). Sample is novice-heavy: `played_many` is **n=1**, so any claim
about experienced humans rests on one person.

### Intervals are not optional here

Games are clustered — three per participant — so 57 games are not 57 independent
observations. `analysis/stats.py` resamples **participants**, not games.

Measured on the stale checkpoints:

| checkpoint | point estimate | 95% CI (paired, clustered) |
|---|---|---|
| `mdw4ndpi` | +7.4% | [−5.1%, 19.8%] |
| `atomic-feather-3` | +14.2% | [−0.4%, 28.2%] |

**Both look like the agent beating humans; neither survives an interval.**
Reporting "+14 points" would have been unsupportable. The design effect came out
at **1.01**, meaning clustering happened to matter little *here* — an honest
finding, and a reason to measure rather than assume.

### Move agreement says the agents do not play like better humans

~20–25% agreement with the humans' own decisions, roughly constant across
checkpoints whose win rates differ by 20 points. **The agents are not executing
better versions of human strategies — they are playing differently.** That is the
question `E3` exists to answer and it deserves its own section.

---

## 7. Methodological traps — worth a paragraph each in the writeup

Each of these silently corrupted a result before it was caught.

1. **`--seed` seeded almost nothing.** Weight init, action sampling and minibatch
   order ran on torch's OS-seeded default, so two runs with identical flags *and*
   identical seed were independent draws. This *is* the ~9.5-point noise floor
   (39.5% vs 30.0% on the same command), and it silently broke every "paired by
   seed" design. Fixed 2026-09-09.
2. **Runs that end before the phenomenon starts.** The curriculum's first
   difficulty movement is at ~3.3M steps (old cops) / 4.4–4.9M (frozen v2). Every
   3M-step run ended just short, producing a null result that looked like "the
   controller does nothing" — and six configurations produced *byte-identical*
   policies because 975/976 updates ran at a constant difficulty.
3. **`charts/win_rate` is the controller's setpoint, not a score.** A P-controller
   with a deadband drives difficulty until win rate sits inside the band, so a
   converged Director-ON run reads ~0.5 whatever its policy is worth. Rank on
   Director-free `eval/win_rate` only.
4. **Best-of-N evaluation is optimistically biased.** Ranking on the max of ~69
   evaluations flatters noisy runs. Conclusions here were re-checked on final and
   last-5 means; they held, but the check is the point.
5. **Binning across arms invents effects.** Binning evaluations by the difficulty
   in force suggested an interior optimum at +0.1 to +0.2. The per-arm test
   refuted it — the curve peaks at 0.0. Confounded aggregation, flagged at the
   time and worth a cautionary line.
6. **A shared prefix is a confound.** Branch designs are efficient but every arm
   inherits the base's training. "Director bad" turned out to be "warmup good,
   permanent suppression bad" — two different claims. Every arm needs a
   from-scratch control before an absolute claim.
7. **`--gamma` is PPO's discount.** The reward-shaping coefficients had to be
   renamed `--reward-*`; a bare `--gamma 0` intended to disable a reward term
   would silently have zeroed the discount, in the one arm meant to isolate reward
   shaping.

---

## 7b. An implementation boundary that looks like a hyperparameter

Worth a short subsection of its own: the most dangerous bug found in this project
was never a crash, and it sat inside the contribution being measured.

`CurriculumDirector` suppresses each discovered node when `rng.random() > |d|`.
`random()` returns [0, 1), so at **|d| = 1.0 that test is never true**:

| difficulty | −1.00 | −0.95 | −0.90 |
|---|---|---|---|
| % of discovered nodes surviving | **0.00** | 5.05 | 10.11 |

−1.0 is therefore not "very strong suppression" on a continuum but a
qualitatively different regime — the cops receive zero information ever, beyond
Jack's public start. Searching is pointless for them, so **Jack gets no training
signal that being seen matters at all**, and an entire strategy class is
unlearnable. Held constant for a whole run it costs 13.8 points:

| arm | training condition | mean | range | copdist | arrest% |
|---|---|---|---|---|---|
| `fix095` | pinned −0.95 | **64.5** | [62.5–68.5] | 1.13–1.21 | 33.5–40.0 |
| `fix10` | pinned −1.0 | 50.7 | [46.0–56.0] | 1.01–1.06 | 49.0–57.0 |

Non-overlapping, and the behavioural diagnostics move the right way — 5% of the
information surviving partly restores cop avoidance.

**It did not contaminate the results, and the reason is worth stating.** Every
warmup in this project started at exactly −1.0, including the runs behind the
headline. Tested as a *warmup* rather than a permanent setting, the effect
inverts and vanishes into the noise floor:

| arm | start, ramping to ceiling 0.0 | mean | range | % of training at the floor |
|---|---|---|---|---|
| `fsc000-i100` | −1.0 | **92.0** | [91.0–93.0] | ~26 |
| `fsc000-i095` | −0.95 | 89.3 | [87.0–91.0] | ~1 |

A degenerate floor is simply a *longer warmup*, and the warmup is worth +17
points (§4). The two effects point in opposite directions and roughly cancel.

The methodological point for the writeup: a parameter's extreme value can leave
its continuum and become a different mechanism, and no amount of sweeping the
interior would reveal it. It was found by reading the implementation and asking
what `random() > 1.0` returns — then tested, because the answer alone does not
say whether it matters.

---

## 8. Negative and null results worth reporting

- ~~**Reward shaping cannot be shown to help.**~~ **Retracted 2026-09-14 — do not report.** Both the
  3M sweep and 04's 15M `sparse` arm zeroed `gamma` too, and `gamma` is part of the objective (the
  participant score's stealth term), not shaping. Those arms removed part of the goal; that is why
  their hideout uncertainty fell (0.65–0.69 vs 0.75–0.78). Rescored roughly on the objective, 04's
  `sparse` ≈ 1.26 against `director-on` ≈ 1.27, so the apparent advantage disappears. Shaping is
  re-tested properly in [10](10-reward-design.md), with the objective held fixed. Original text:
  **Reward shaping cannot be shown to help.** At 3M steps the fully sparse
  configuration (all five coefficients zero) scored *above* the shaped control
  (33.0% vs 30.0%) — inside noise, but there is no evidence the shaping earns its
  complexity. Keep the defaults; do not claim they help.
- **PPO hyperparameters:** `lr=3e-4` wins decisively in both Director arms
  (~30-point spreads); `ent-coef=0.03` is an interior optimum confirmed by
  probing 0.06 (26.5%) and 0.10 (16.5%). Entropy interacts with lr and is **not**
  monotone on its own — at `lr=1e-3` more entropy hurts badly.
- **The curriculum's target band does not matter at ceiling 0.0.** Four bands
  from [0.10,0.20] to [0.60,0.80]: 90.5 / 91.8 / 92.5 / 92.8, **2.3 points of
  spread** against per-arm half-ranges of 1.0–2.2, and non-monotone. The band
  paces the ramp, so it only matters when it controls how long a run sits
  somewhere *harmful* — at ceiling +0.15 the same four bands spread 8.5 points
  and ordered monotonically (slower is better), and at ceiling 0.0, where the
  ceiling is the optimum, the effect vanishes rather than flipping sign. This is
  a band x ceiling interaction, and it is the more interesting way to report it.
  **Use the default band; do not tune it.**
- **The exploration bonus (delta) is a null on this map.** 1.302 vs 1.309 with the curriculum and
  1.161 vs 1.141 without — both smaller than the seed spread, in opposite directions. The mechanism
  is absent rather than weak: **coverage reaches 0.95 by 34k steps and 1.000 from ~1M in every arm**,
  with or without the bonus (entropy differs by ≤0.008, and by 0.001 at 15M). With 12 parallel
  environments on a 195-node map, exploration is not a bottleneck, so a front-loaded exploration
  bonus has nothing to add. Report it as scale-dependent, not as "count-based exploration doesn't
  work": it would be expected to matter on a larger board or a much shorter budget.
- **Board-size ablation** was never implemented (`course_1/2/3` are same-size
  scenario variants). Report as a limitation.

---

## Where the evidence lives

| | |
|---|---|
| Chart-ready CSVs | [`results/`](results/) — one row per evaluation and per update, all nine waves |
| Raw provenance | `results/slurm_logs.tar.gz` |
| Policies | `checkpoints/` (local, gitignored, ~14 GB) |
| Participant data | `data/study/games_<date>.sqlite` (gitignored — retention is an ethics decision) |
| Analysis | `analysis/convergence.py`, `sweep_report.py`, `compare.py`, `stats.py`, `sessions.py` |

**Backup gap, unresolved:** the checkpoints and the participant snapshots exist
only on one laptop. The CSVs and logs are in git; those two are not.
