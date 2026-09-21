# Findings worth putting in the thesis

Everything established experimentally between 2026-09-09 and 2026-09-21, with the evidence and the
caveats. Written to survive: if a later session has none of the conversation, this file plus
[`results/`](results/) should be enough to write the chapters.

**Two conventions changed partway through the project, so check which applies before quoting a
number:**

| | §0, §6, §6b (current) | §1–§5, §7–§8 (earlier waves) |
|---|---|---|
| metric | **participant score** (`SCORE_STUDY_V1`) | `eval/win_rate`, % |
| training reward | stealth objective (−1 / 1 + 0.5·U) | legacy (±1) |
| seeds per arm | **6** | 3, or 1 in the 03 sweeps |

Every evaluation everywhere is **Director-free against full-strength `COPS_STUDY_V2`** (invariant
2), 200 games per in-training evaluation and ~69 evaluations per run, unless a table says otherwise
(the 2,000-game re-scorings are marked).

**§0 is the headline. §1–§5 are the earlier waves**, kept because their *design* conclusions (the
injection ceiling, the suppressed warmup, the band null) still hold and are not re-measured
elsewhere — but their levels are win rates under the old reward and are not the thesis's numbers.

---

## 0. The headline, measured on the objective (6–9 seeds, 2026-09-21)

Everything in sections 1–5 was measured under the **legacy** reward and reported as win rate.
Workstream 10 re-ran the comparison under the reward the thesis argues for — win stealthily — and
reports the **participant score**, what human participants were scored on (`engine/metrics.py`,
`SCORE_STUDY_V1`; humans average **0.672**). Prefer these numbers.

**6 seeds per arm, 9 for the two shaping arms** (arrays `1807588` + `1808812` + `1809122`;
`obj-cur`'s second triple is `w11-ent003` from `1808727`, that configuration flag-for-flag). Mean of
the last five evaluations per seed.

| reward | curriculum | no curriculum | curriculum is worth |
|---|---|---|---|
| **objective only** | **1.295** | 1.175 | **+0.120** |
| + delta | 1.306 | 1.163 | **+0.143** |
| + delta + shaping | 1.272 | 1.183 | **+0.090** |

**1. The curriculum is the result, and it does not depend on anything else.** Pooled across all
three reward conditions (21 runs with it, 21 without) it is worth **+0.114 participant score, bootstrap
95% CI [+0.085, +0.142]** — clearing its detection threshold (0.042) by 169%. It also holds in each
condition separately: +0.120, +0.143 and +0.090, clearing their thresholds by 52%, 82% and 39%.
It is also a large speed effect — steps to reach a fixed score, pooled over all seeds:

| | → 1.00 | → 1.10 | → 1.20 |
|---|---|---|---|
| with curriculum (all three rewards) | 6.5–6.6M | **8.1–8.3M** | 9.9–11.4M |
| without curriculum | 10.5–10.9M | 12.7–13.0M (delta arm never) | **never** |

**~36% fewer steps to reach 1.10, and 1.20 is reached only with the curriculum.**

**2. Reward shaping does nothing, with or without the curriculum.** +0.008 without, −0.022 with;
both far inside their 0.072 threshold. Delta likewise (+0.012 and −0.012, threshold 0.079).

**3. The curriculum's benefit does not interact with the reward.** The shaping × curriculum
interaction is **+0.030 against a 0.102 threshold** — no evidence that shaping and the curriculum
substitute for one another, or that either changes what the other is worth. (An earlier 3-seed
reading claimed they did; see Appendix A.2.)

**4. The curriculum also stabilises training.** Per-seed SD is 0.019–0.040 in the curriculum arms
against 0.057–0.067 without — roughly a **1.5–3× reduction in seed variance**, on top of the level
difference.

**5. Behaviour backs it up.** Curriculum arms hold a wider berth from cops (copdist 1.46 vs 1.34)
and lose less to *both* arrest (8.3% vs 19.8%) and the clock (1.8% vs 10.2%), while leaving the
hideout *more* ambiguous (0.79 vs 0.74). Better on both halves of the objective, not a trade.

**Recommended configuration:** objective only + curriculum (`w10s-obj-cur`). `+delta` scores
marginally higher (1.306 vs 1.295, inside noise); the simpler arm is kept, and either is defensible.

### Confirmed on fresh boards (2,000 games, final checkpoints, 2026-09-19)

Everything above uses in-training evaluations, which replay one fixed board set. Re-scoring each
run's **final** checkpoint — no selection — on 2,000 unseen boards, all arms in one invocation.
**These were the 3-seed arms** (04 has 3 seeds; wave 10 had 3 at the time), so read them as
confirming the *ordering and the absolute level* on unseen boards, not as the 6-seed effect sizes:

| arm | score | win% | hideout_u |
|---|---|---|---|
| `director-on` (04, shaped, legacy reward) | **1.328** | 90.5 | 0.79 |
| `w10s-obj-cur` (10, no shaping, stealth objective) | **1.326** | 90.7 | 0.78 |
| `director-off` (04, shaped) | 1.300 | 87.0 | 0.78 |
| `sparse` (04, no shaping, no stealth term) | 1.299 | **93.9** | 0.68 |
| `w10s-shp-off` (10, shaping, no curriculum) | 1.243 | 80.4 | 0.75 |
| `w10s-obj-off` (10, neither) | 1.159 | 70.9 | 0.73 |
| random Jack (floor) | 0.399 | 0.1 | 0.64 |

**The two best arms — 1.328 and 1.326 — come from different rewards and are indistinguishable.**
`director-on` trained on the legacy reward with shaping; `w10s-obj-cur` on the stealth objective with
none. So the objective change cost nothing in absolute performance, and shaping adds nothing on top
of the curriculum. Both sit ~0.13 above `director-off` and ~0.17 above `w10s-obj-off`.

These rows were once read as evidence that the Director's value depends on the presence of shaping;
that reading is withdrawn (Appendix A.2).

### Win rate and the objective can disagree — 04's `sparse` arm

Highest win rate in its wave (93.9%) and **a lower score than both Director arms**, because it
dropped the stealth term and stopped hiding the hideout (uncertainty 0.68 vs 0.78–0.79). Paired
against `director-on`: **+3.4 win points, −0.029 score, on every seed.** Identical games, opposite
orderings. This is the concrete argument for reporting the participant score.

### No arm overfits to its training cops (retires a long-running worry)

Against `COPS_PRERETUNE_V1`, which no current policy trained on: `director-on` 1.328 → 1.209,
`director-off` 1.300 → 1.183, `sparse` 1.299 → 1.180 — **gaps of −0.119 / −0.117 / −0.119.**
Identical to three decimals, ordering unchanged. The 2026-09-09 belief that the Director overfits
to its own cops (the reason workstream 09 was created) does not survive a fair test; the gap belongs
to the cop retune, not the curriculum.

### The random-cops baseline says what the curriculum actually fixes (§7.2)

Against `RandomCops` — cops that move at random and always search, so they essentially never arrest:

| arm | vs study cops | vs random cops | timeouts vs random cops |
|---|---|---|---|
| `w10s-obj-cur` | 90.7% | 93.8% | 6.2% |
| `w10s-obj-off` | 70.9% | **73.7%** | **26.3%** |
| random Jack | 0.1% | 0.8% | 99.2% |

**Without the curriculum, a quarter of games are lost to the clock even against cops that cannot
catch anyone.** Its deficit is not evasion — it is failing to reach the hideout in time. So the
curriculum's main contribution is efficient routing under the turn limit, and only secondarily
evasion. Removing the cops entirely lifts the trained arms by only ~3 points, so by 15M steps
evasion is close to solved.

And a random Jack wins 0.8% of games with *harmless* cops: **the task is hard because of the turn
limit and the map, not only because the cops are good.**


---

## Sections 1–5 — the earlier waves (legacy reward, win rate, 3 seeds)

**Read these for the design conclusions, not for the headline numbers.** They established the
things §0 does not re-measure: that injection must be forbidden, that the suppressed warmup is
worth having, that the ramp shape and target band do not matter, and how policies lose. Their levels
are `eval/win_rate` under the legacy ±1 reward, with 3 seeds (1 in the 03 sweeps), so a difference
below ~9.5 points there is noise. §0 supersedes them on the size of the curriculum effect.

---

## 1. The curriculum works, if it is forbidden from injecting (legacy reward, win rate)

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
> **Quote §0's figure (+0.114 pooled, +0.090 to +0.143 per condition), not this one and not 04's.** 04
> re-measured the configuration on fresh seeds but has only 3 of them, so its +0.028 sits inside its
> own 0.118 detection threshold (§6b).

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

**The current version of this table is in §0** (6–9 seeds, participant score: ~36% fewer steps to a
score of 1.10, and 1.20 reached only with the curriculum). The table below is the legacy-reward,
win-rate version from the 09 waves, kept because it covers arms §0 does not — the unbounded
Director and the hand-picked two-phase schedule.

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

## 5. Generalisation: every arm loses the same on held-out cops

Scored against `COPS_PRERETUNE_V1`, which no current policy trained on — 04's nine final
checkpoints, 2,000 games each:

| arm | study cops | held-out `COPS_PRERETUNE_V1` | gap |
|---|---|---|---|
| `director-on` | 1.328 | 1.209 | −0.119 |
| `director-off` | 1.300 | 1.183 | −0.117 |
| `sparse` | 1.299 | 1.180 | −0.119 |

**The gaps are identical to three decimals and the ordering is unchanged.** Every policy loses about
0.12 of score when the opponent changes, so **the gap is a property of the cop retune, not of any
training condition**. (A 2026-09-09 reading of two historic single runs had it that the curriculum
overfits to its own cops — the reason workstream 09 exists. See Appendix A.1.)

**Report a held-out cop configuration alongside the frozen one** —
`uv run python -m training.eval <ckpts> --held-out-cops` does this. A policy scored only against its
training opponent cannot be distinguished from one that memorised that opponent's decision rule,
and that remains true whether or not any particular arm turns out to overfit.

This is also a limitation to state plainly: `COPS_STUDY_V2` was itself tuned with
a trained Jack as adversary (provenance unrecorded), so historic policies may be
biased against it.

---

## 6. Human-vs-RL comparison

**Measured 2026-09-21** on `games_20260921.sqlite` — **20 participants, 60 games**, 6 seeds per
arm. Interim: recruitment is still running, so re-run and re-date before submission. (The 09-21 pull
is byte-identical to 09-19; no new participants in between.)

Each policy replays the exact board its human counterpart faced, 20 times per scenario; humans are
scored by replaying their own recorded moves through the engine, so both sides go through one
scoring function. **Zero desyncs across all 60 games and three checkpoints.**

| | participant score | win rate | move agreement |
|---|---|---|---|
| **Humans** | **0.672** [0.578, 0.770] | **31.7%** [23.3%, 40.0%] | — |
| **`obj-cur`** (recommended), 6 seeds | **1.260** ±0.033 | 84.2% | 27.9% |
| `dlt-cur`, 6 seeds | 1.257 ±0.013 | 85.2% | 26.8% |
| `shp-cur`, 6 seeds | 1.248 ±0.038 | 83.5% | 27.5% |

**All 18 checkpoints beat the humans by +0.52 to +0.65 score and +47 to +57 win points. 36 of 36
paired intervals (18 score, 18 win rate), clustered by participant, exclude zero. Zero desyncs.**
Design effect 1.02, so clustering barely widens the interval here — but it is the correct interval.

This gap is ~0.58, an order of magnitude larger than the 0.02–0.15 between-arm differences that the
seed top-up overturned (§6b), so it was never at risk from seed noise. On human boards the three
reward conditions are indistinguishable (spread 0.012), consistent with shaping and delta being
nulls on the training map.

### The agent does not play like a better human

Move agreement is **27–29%**: at nearly three quarters of the decisions a participant actually made,
the policy would have chosen differently. A large win-rate gap with low agreement was already
visible in the 2026-09-09 smoke run (20–22% agreement, 20-point gap) and survives against the real
agents at a 52-point gap. **The agents win by playing a different game, not a tidier version of the
human one** — worth a paragraph, because it bears on what the comparison licenses you to say about
human play.

### Board difficulty does not explain the human losses

The policy wins **85.1%** of the boards its human lost and **83.4%** of the boards its human won. If
humans were mostly losing to unlucky scenarios, those two numbers would differ sharply.

### Experience shows no gradient

| self-reported experience | participants | games | score | win rate |
|---|---|---|---|---|
| never played | 13 | 39 | 0.662 [0.544, 0.775] | 30.8% |
| played a few | 5 | 15 | 0.689 [0.476, 0.939] | 33.3% |
| played many | 2 | 6 | 0.692 [0.548, 0.836] | 33.3% |

Flat, and with 2 participants in the top group this is descriptive only. **The project's "human
players of varying skill levels" framing cannot be supported beyond "self-reported experience was
collected and showed no clear gradient at this N."** If recruitment can be steered, experienced
players are the group that would change what this comparison can claim.

## 6b. How much precision the seeds buy — read this before quoting any effect

Pooled over every wave-10 arm (6 seeds each, 9 for the two shaping arms), the **per-seed SD of the
last-5 participant score is 0.0488**. It was 0.0401 estimated from 3-seed arms and 0.0516 at 6; it
settled as the noisy no-curriculum arms got better sampled. That fixes what these experiments can
resolve:

| seeds per arm | SE of an arm's mean | smallest detectable difference | …for an interaction |
|---|---|---|---|
| 3 | 0.028 | 0.111 | 0.158 |
| 6 | 0.020 | 0.079 | 0.111 |
| **9** | 0.016 | **0.064** | 0.091 |
| pooled, 21 v 21 | — | **0.042** | — |

(~80% power, 5% two-sided.)

| effect | 3 seeds | final | threshold | verdict |
|---|---|---|---|---|
| **Curriculum, pooled across rewards** | — | **+0.114** [+0.085, +0.142] | 0.042 | detectable, +169% |
| Curriculum, objective only | +0.168 | +0.120 | 0.079 | detectable, +52% |
| Curriculum, + delta | +0.141 | +0.143 | 0.079 | detectable, +82% |
| Curriculum, + shaping | +0.020 | +0.090 | 0.064 | detectable, +39% |
| Shaping, no curriculum | +0.104 | +0.008 | 0.072 | not detectable |
| Shaping, with curriculum | −0.044 | −0.022 | 0.072 | not detectable |
| Shaping × curriculum interaction | +0.148 | +0.030 | 0.102 | not detectable |
| delta, either way | ±0.007 | ±0.012 | 0.079 | not detectable |

**Two top-ups, two different outcomes — which is the point.** Going 3 → 6 seeds collapsed the
substitution claim (Appendix A.2): it had cleared its threshold by 14%. Going 6 → 9 on the shaping
arms tested the next-weakest claim, the curriculum effect with shaping on, which cleared by 12% — and
it **held**, moving only from +0.093 to +0.090 while its margin grew to 39%. A marginal result is not
necessarily wrong; it is untested. Both were checked the same way.

**The same configuration, two seed triples: 1.309 and 1.280** (`w10s-obj-cur` seeds 41–43 vs
`w11-ent003` seeds 51–53, identical flags). A 0.029 swing from seed choice alone.

Practical rules for the write-up:

1. **Lead with the pooled curriculum effect** (+0.114, CI [+0.085, +0.142]); give the per-condition
   figures as the robustness check.
2. **State the threshold beside every null** — "no difference larger than ~0.07 was detectable" is
   the honest form, and it sits alongside the ~9.5-point win-rate noise floor from 03.
3. **Be most suspicious of a small effect that is significant**, not of a null. Three of the four
   withdrawn claims in Appendix A were positive results from thin evidence; none was a null that
   later became an effect.

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

- **Reward shaping has no measurable effect**, with or without the curriculum: +0.010 and −0.017 at
  6 seeds against a 0.083 threshold. This is now a clean null on the *right* objective, and it
  supersedes both the 2026-09-14 retraction below and the 2026-09-19 "substitution" reading.
- **Reward shaping has no measurable effect on this task.** With the curriculum −0.022, without it
  +0.008, both far inside the 0.072 threshold (9 v 6 seeds); the interaction is +0.030 against 0.102.
  Two earlier readings of the shaping question were withdrawn on the way to this one (Appendix A.2,
  A.3).
- **PPO hyperparameters:** `lr=3e-4` wins decisively in both Director arms (~30-point spreads) and
  is the one hyperparameter this project can claim to have tuned. `ent-coef=0.03` was chosen on a
  rationale that **did not survive** (its ON arm's curriculum never engaged) and was re-checked in
  [11](11-hyperparameters.md): it stands, but only because 0.01 is indistinguishable from it.
  Entropy interacts with lr and is **not** monotone — at `lr=1e-3` more entropy hurts badly.
  **Everything else is a CleanRL default that was never varied** (`n-steps`, `n-epochs`,
  `minibatch-size`, `gamma`, `gae-lambda`, `clip-coef`, `vf-coef`, `max-grad-norm`, the network
  shape). Report that plainly.
- **The curriculum's target band does not matter at ceiling 0.0.** Four bands
  from [0.10,0.20] to [0.60,0.80]: 90.5 / 91.8 / 92.5 / 92.8, **2.3 points of
  spread** against per-arm half-ranges of 1.0–2.2, and non-monotone. The band
  paces the ramp, so it only matters when it controls how long a run sits
  somewhere *harmful* — at ceiling +0.15 the same four bands spread 8.5 points
  and ordered monotonically (slower is better), and at ceiling 0.0, where the
  ceiling is the optimum, the effect vanishes rather than flipping sign. This is
  a band x ceiling interaction, and it is the more interesting way to report it.
  **Use the default band; do not tune it.**
- **The exploration bonus (delta) is a null on this map.** At 6 seeds: 1.306 vs 1.295 with the
  curriculum (+0.012) and 1.163 vs 1.175 without (−0.012) — both far inside the 0.079 threshold, in
  opposite directions. The mechanism
  is absent rather than weak: **coverage reaches 0.95 by 34k steps and 1.000 from ~1M in every arm**,
  with or without the bonus (entropy differs by ≤0.008, and by 0.001 at 15M). With 12 parallel
  environments on a 195-node map, exploration is not a bottleneck, so a front-loaded exploration
  bonus has nothing to add. Report it as scale-dependent, not as "count-based exploration doesn't
  work": it would be expected to matter on a larger board or a much shorter budget.
- **`ent-coef` 0.01 and 0.03 are indistinguishable** under the final configuration (1.303 vs 1.280,
  3 fresh seeds each), while 0.003 is clearly worse (1.251). There is an interior optimum in
  [0.01, 0.03] and this project cannot locate it more precisely — the same config varies by 0.029
  across seed triples. 03's original rationale for 0.03 was void (its ON arm's curriculum never
  engaged); the value survives anyway. See [11](11-hyperparameters.md).
- **Board-size ablation** was never implemented (`course_1/2/3` are same-size
  scenario variants). Report as a limitation.

---

## Appendix A — retractions and corrections

Four claims in this project were stated, then withdrawn on better evidence. They are collected here
so the body reads as what is currently believed, and so the reasoning survives if anyone asks why
the write-up changed. **Nothing here is a current finding.**

Each entry: what was claimed, what it rested on, what overturned it, what replaced it.

### A.1 — "The curriculum overfits to the cops it trained against" (2026-09-09 → 2026-09-19)

**Claimed:** the Director bought 11 points against the cops it trained on and lost 16 when the cops
changed, i.e. the curriculum buys a policy that memorises one opponent.

| scored against | Director ON | no Director |
|---|---|---|
| the cops both trained on (`COPS_PRERETUNE_V1`) | 74.7% | 63.7% |
| the retuned cops neither saw (`COPS_STUDY_V2`) | 58.7% | 62.7% |

**Rested on:** two *historic single runs* (the project's only 2026-05 Director run and one control)
that differed in more than the Director, 300 games, no seed replication.

**Overturned by:** 04's nine final checkpoints at 2,000 games each — held-out gaps of −0.119 /
−0.117 / −0.119 across all three arms (§5).

**Replaced by:** the gap belongs to the cop retune, not to any training condition. **This claim is
why workstream 09 exists**, so it shaped months of work before it was tested properly — the single
most consequential wrong belief in the project.

### A.2 — "Reward shaping substitutes for the curriculum" (2026-09-19 → 2026-09-21)

**Claimed:** shaping is worth +0.104 without the curriculum and nothing with it (−0.044), an
interaction of **+0.148** against a 0.130 threshold — so the two buy the same thing, and that is why
04's Director effect (+0.028, both arms shaped) looked so much smaller than wave 10's (+0.167).

**Rested on:** 3 seeds per arm. The interaction cleared its threshold by 14%.

**Overturned by:** the seed top-up (array `1808812`) taking the arms to 6 seeds. `shp-off` fell
**−0.060** and `obj-off` rose **+0.034**; the interaction collapsed to **+0.026** against a 0.118
threshold. One lucky triple and one unlucky one, drifting in opposite directions.

**Replaced by:** shaping has no measurable effect either way, and the curriculum's benefit
(+0.090 to +0.143) holds in every reward condition (§0). A second top-up taking the shaping arms to 9
seeds put the interaction at +0.030 against 0.102 — confirming the collapse rather than reversing it. 04's small Director effect has the duller
explanation that 04 has 3 seeds and cannot resolve it.

**Why it matters methodologically:** this was written up as the project's most interesting result
before it was checked. It was caught only because the 14% margin prompted a top-up. See §6b.

### A.3 — "Reward shaping cannot be shown to help" (2026-09-11 → 2026-09-14)

**Claimed:** from 03's wave 2 (3M steps) and 04's `sparse` arm, the fully sparse configuration
scored at or above the shaped control, so shaping does not earn its complexity.

**Rested on:** arms that set all five reward coefficients to zero — **including `gamma`**, the
stealth term, which is part of the *objective* rather than the shaping. Those arms removed part of
the goal, which is why their hideout uncertainty fell (0.68 vs 0.78–0.79).

**Overturned by:** recognising that the participant score defines the objective (§0, workstream 10),
and re-running the shaping question with the objective held fixed.

**Replaced by:** A.2's conclusion — shaping is a null, measured properly. The *conclusion* was right
by accident; the evidence for it was not.

### A.4 — "`ent-coef` 0.03 because entropy matters with the Director on" (2026-09-09 → 2026-09-20)

**Claimed:** 0.03 beat 0.01 and 0.003 in the Director-ON arm (39.5 / 36.0 / 18.0) while the OFF arm
was flat, so entropy matters under the curriculum and 0.03 is the shared choice.

**Rested on:** 03 wave 1, 1 seed per cell, 3M steps — and an "ON" arm whose
`curriculum/difficulty` stayed pinned at −1.000 for the whole run. **The curriculum never engaged**,
so the gradient describes training against fully-suppressed cops.

**Overturned by:** workstream 11's re-check under the final configuration: 0.01 scores 1.303 and
0.03 scores 1.280 — indistinguishable, since the same configuration varies by 0.029 across seed
triples.

**Replaced by:** 0.03 stands, but as a value that cannot be distinguished from 0.01 rather than one
shown to be best; 0.003 *is* clearly worse (1.251), so an interior optimum exists in [0.01, 0.03].
A right answer for a wrong reason.

### What the four have in common

Three of the four were **positive claims from thin evidence** (1 seed, 3 seeds, or two single runs),
and the fourth (A.3) drew a right conclusion from a confounded arm. None was a null that later turned
out to be an effect. That asymmetry is the practical lesson, and it is why §6b says to be more
suspicious of a small significant result than of a null.

---

## Where the evidence lives

| | |
|---|---|
| Chart-ready CSVs | [`results/`](results/) — one row per evaluation and per update, all 13 waves |
| Raw provenance | `results/slurm_logs.tar.gz` |
| Policies | `checkpoints/` (local, gitignored, ~14 GB) |
| Participant data | `data/study/games_<date>.sqlite` (gitignored — retention is an ethics decision) |
| Analysis | `analysis/convergence.py`, `sweep_report.py`, `compare.py`, `stats.py`, `sessions.py` |
| Re-scorings | `results/final_reeval.txt`, `reward_reeval.txt` (2,000 games), `comparison_20260921.txt` |

**Backup gap, unresolved:** the checkpoints and the participant snapshots exist
only on one laptop. The CSVs and logs are in git; those two are not.
