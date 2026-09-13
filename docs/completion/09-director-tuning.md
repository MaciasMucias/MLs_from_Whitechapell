# 09 — Director tuning

**Status:** **DONE (2026-09-13). The curriculum works — but only if it is forbidden from
injecting.** Five waves: `1807070` (null, runs too short), `1807190` (bases), `1807209` (branch —
confounded), `1807292` (from scratch + ramp shape), `1807359` (ceilings — the result).

**Recommended configuration: `--curriculum-max-difficulty 0.0`.** Adaptive suppression that relaxes
to full strength and never injects: **92.8%** [92.5–93.0], tying the best number seen anywhere
(`dbr-off` 93.2%) and beating no-curriculum-at-all (`fs-off` 85.2%).

**Consequence for 04:** the ON arm is worth running after all, but as `--curriculum-max-difficulty 0`
rather than the original unbounded Director. The OFF arm stays as the baseline.

---

## FINAL RESULT (2026-09-13, array `1807359`, 18/18) — the curriculum works, if it never injects

Five waves and two reversals later, the picture is coherent. **The best configuration found is an
adaptive curriculum forbidden from injecting**, and it matches the best number seen anywhere.

### 1. The fixed dose-response peaks at zero

Difficulty held constant after the suppressed warmup:

| fixed difficulty | −1.0 | −0.5 | **0.0** | +0.15 |
|---|---|---|---|---|
| eval win% | 50.7 | 88.3 | **93.2** | 72.3 |
| range | [46.0–56.0] | [87.0–90.5] | [92.5–94.0] | [66.0–75.5] |

A clean inverted-U. Suppressing hurts, and so does injecting — **knowledge the cops never earned
costs ~21 points**. There is no interior optimum on the positive side.

> This kills the "interior optimum at +0.1 to +0.2" reported on 2026-09-12, which came from binning
> evals by difficulty *across* arms. It was flagged as confounded at the time; the per-arm test
> settles it. Do not quote the binned figure.

### 2. Adaptive control is fine — the ceiling is what matters

All branching from the same bases, so only the ceiling differs:

| ceiling | reached | eval win% |
|---|---|---|
| **0.0 (never inject)** | 0.0 | **92.8%** [92.5–93.0] |
| +0.15 | +0.15 | 81.7% [77.0–85.5] |
| uncapped | +0.19 | 81.3% [76.0–86.5] |
| uncapped | +0.54 | 61.3% [57.5–64.0] |

Two things fall out:

- **`cap000` (92.8%) ties `dbr-off` (93.2%) and fixed-0.0.** The controller finds the right schedule
  by itself, *provided it cannot overshoot*. Adaptivity neither helps nor hurts; forbidding
  injection is the whole game.
- **A ceiling only helps when it binds.** +0.15 against an uncapped arm that stopped at +0.19 changed
  nothing (81.7 vs 81.3) — predicted in advance. Against the arm that ran to +0.54 it recovered
  **+16.5 points** (77.8 vs 61.3).

### 3. The suppressed warmup is worth a lot

Same ceiling and band, differing only in whether the first 4.4M steps ran suppressed:

| | eval win% |
|---|---|
| `cap015-b1020` — with warmup | 73.2% [68.0–78.5] |
| `fscap015` — from scratch | **56.2%** [55.5–56.5] |

**+17 points for the warmup**, corroborating the +8 seen in `dbr-off` vs `fs-off`. Boosting the cops
early *without* first letting the agent learn against weak ones is the worst thing tested short of
permanent full suppression. `fscap015`'s behavioural signature says why: copdist ~1.12 and **arrest
~43%**, against ~1.40 and ~18% for the warmed-up arms.

### The claim for the thesis

1. **A curriculum helps: ~+8 points** over training at full strength throughout, via an early
   suppression phase that relaxes to full strength.
2. **Adaptive scheduling is neither better nor worse than a hand-picked two-phase schedule** — 92.8%
   vs 93.2%. Its value is that it finds the transition point without tuning.
3. **Injection is harmful in proportion to its magnitude**, and the Director's original unbounded
   design permitted it. `--curriculum-max-difficulty 0.0` is the fix.
4. **Order matters:** suppress first, then relax. Boosting early from scratch costs 17 points.

**Recommended configuration:** `--curriculum-max-difficulty 0.0` with the default band, from scratch.
That is the arm to carry into 04 and 06.

---

## CORRECTION (2026-09-12, array `1807292`, 18/18) — "the Director is detrimental" was wrong

The conclusion below is **superseded**. Tested without the branch design's
confound, the Director neither helps nor hurts, and the ramp shape does not
matter at all. From scratch, 15M steps, 3 seeds each:

| arm | mean | range | converged difficulty |
|---|---|---|---|
| `kp-002` (slowest ramp) | 85.8% | [79.5–89.5] | +0.18 to +0.21 |
| **`fs-off` (no Director)** | **85.2%** | [81.5–88.5] | — |
| `kp-030` (fastest ramp) | 84.0% | [78.0–88.5] | +0.20 to +0.22 |
| `fs-on` (Director as designed) | 83.2% | [80.5–85.5] | +0.18 to +0.21 |
| `kp-cap001` (rate-limited) | 82.7% | [76.5–86.0] | +0.16 to +0.21 |
| `kp-ratchet` | 81.7% | [76.5–86.0] | +0.18 to +0.21 |

Total spread across all six arms is **4.1 points** against per-arm half-ranges of
2.5–5.2. Everything overlaps everything. `fs-on` vs `fs-off` — the Director as
designed against no Director — differ by 2 points with heavily overlapping
ranges.

**The monotone prediction failed.** This file predicted `kp-002 < kp-010 < kp-030
< off`, on the reasoning that a slower ramp spends longer suppressed. Observed:
`kp-002` is *highest* and `kp-030` is *below* `off`. The ordering is noise.

**Why ramp shape is irrelevant — the mechanism.** Every adaptive arm converged to
difficulty **+0.16 to +0.22** regardless of kp (0.02 vs 0.3), rate limit, or
ratchet. The controller finds the same equilibrium by different routes, so how it
travels there cannot matter much. `off_floor%` is 100% for the branch arms and
72–74% for the from-scratch ones, so the controller was live throughout.

### What actually does have an effect: a fixed suppressed warmup

The one robust signal across every wave. Both arms below are "no Director for the
second phase", same seeds, same 15M total; they differ only in whether the first
~4.5M steps ran at difficulty −1.0:

| | mean | range |
|---|---|---|
| `dbr-off` — 4.5M suppressed warmup, then full-strength cops | **93.2%** | [92.5–94.0] |
| `fs-off` — full-strength cops throughout | 85.2% | [81.5–88.5] |

**+8.0 points, non-overlapping ranges**, and higher than anything in the
from-scratch wave. A hand-designed two-phase curriculum helps substantially. The
*adaptive* controller does not improve on no curriculum, because it settles at
~+0.2 injection, which is worth about the same as baseline.

### Where the earlier conclusion went wrong

The branch wave compared arms that all shared a suppressed warmup, so `dbr-off`
carried the warmup benefit while `dbr-fix10` stayed suppressed for all 15M. That
made the gap look like "Director bad" when it was really "warmup good, permanent
suppression bad". Two different things.

**This is the second misreading in this workstream** — the first was "the
controller has no operating range", which was actually runs ending too early.
Both were caught by building the experiment that could falsify them. The lesson
for the writeup: every Director claim here needed a from-scratch control, and the
branch design, while efficient, could not provide one.

### Revised claim for the thesis

- Adaptive difficulty control: **no measurable effect**, and insensitive to all
  of its tuning parameters, because it converges to the same equilibrium.
- A fixed two-phase curriculum (train easy, then hard): **+8 points**, robust
  across seeds.
- Over-hardening past ~+0.25: **catastrophic** (`b2035` at +0.54 → 61.3%).

---

## SUPERSEDED (2026-09-11, array `1807209`, 15/15) — "the Director does not help"

*Kept for the record; the confound is explained above.*

The branch design worked: **the curriculum engaged in both adaptive arms**, `off_floor% = 100` in
all six, ramping from −1.0 into *injection* territory (`diff_max` +0.19 and +0.54). This is the
first fair test of the Director, and it loses.

All arms branch from the same three base checkpoints and run to 15M. Eval is Director-free
(invariant 2), so every arm is scored against full-strength `COPS_STUDY_V2`.

| arm | training condition | mean best eval win% | range | engaged |
|---|---|---|---|---|
| **`off`** | **no Director** | **93.2%** | 92.5–94.0 | 0/3 |
| `fix05` | fixed −0.5 | 88.3% | 87.0–90.5 | 0/3 |
| `b4060` | adaptive, band [0.40,0.60] | 81.3% | 76.0–86.5 | 3/3 |
| `b2035` | adaptive, band [0.20,0.35] | 61.3% | 57.5–64.0 | 3/3 |
| `fix10` | fixed −1.0 | 50.7% | 46.0–56.0 | 0/3 |

**No Director wins outright**, by 4.9 points over the next arm, and the ranges do not overlap
(92.5–94.0 vs 87.0–90.5). Against a ~9.5-point single-run noise floor this is the first Director
comparison in the project with separation you can actually lean on.

### The mechanism is monotone: handicapping the cops during training hurts

Among the *fixed* levels the ordering is clean and monotone in how much help Jack got:

| training difficulty | −1.0 (cops blind) | −0.5 | 0.0 (full strength) |
|---|---|---|---|
| final eval win% | 50.7 | 88.3 | 93.2 |

The behavioural diagnostics say why. More suppression during training produces a policy that does
not respect cops at evaluation time:

| arm | copdist | arrest% |
|---|---|---|
| `off` | 1.46–1.57 | 6.0–7.5 |
| `fix05` | 1.42–1.46 | 9.0–15.0 |
| `fix10` | 1.01–1.06 | 49.0–57.0 |

A policy trained against blind cops walks straight past them and is arrested ~8x more often than one
trained against real ones.

### The adaptive arms do not rescue it, and over-hardening makes it worse

`b2035` ramped difficulty to **+0.54** — cops *better informed than the evaluation cops* — and scored
20 points below `b4060`, which only reached +0.19. Pushing past difficulty 0 is a train/test mismatch
in the opposite direction, and it costs.

So the curriculum's problem is not that it fails to engage (it does now) or that it is mistuned
(two bands, both worse than off). **Every form of information handicap tested made the final policy
worse.**

### The one caveat that bounds this claim

**All five arms share a base trained at difficulty −1.0 for 4.4–4.9M steps.** The comparison
*between* arms is clean — identical prefix, identical seeds — but `dbr-off` is "suppressed prefix,
then no Director", not a pure no-Director run. The absolute claim "the Director hurts" therefore
still needs a from-scratch `--no-curriculum` 15M run to compare against.

**That run is 04's OFF arm**, so 04 supplies it at no extra cost. If from-scratch OFF beats 93.2%,
the suppressed prefix hurt too and the result strengthens. Report the branch-wave numbers as a
*controlled* comparison and the 04 number as the absolute one.

---

## Why array `1807070` found nothing: the runs ended before the curriculum starts

**The 3M-step runs stop just short of the first difficulty uptick.** In the only historic Director
run (`wandb/offline-run-20260523_134803-mdw4ndpi`, 10M steps, old cops) the first uptick was at
step **3,256,320** — 1,059 of 3,255 updates were pinned at −1.0 before it moved. Every run in the 03
sweep and in this wave was 3M steps.

Everything odd about the wave follows from that:

- Difficulty left the floor on **update 976 of 976** in every `--initial-difficulty -1.0` config.
- So 975/976 of training was bit-identical across them, and **six configurations produced
  byte-identical eval metrics** (`dir-i10-kp010-CONTROL`, `dir-i10-kp002`, `dir-i10-kp030`,
  `dir-cap001`, `dir-ratchet`, `dir-cap001-ratchet` — all 35.0%, copdist 1.08, arrest 56.0%).
- `--curriculum-kp`, `--curriculum-max-step`, `--curriculum-ratchet` and the deadband width were
  therefore **untested, not disproven**. The controller never reached the regime where it acts.

Two results from the wave *are* meaningful:

- **`dir-i00-ratchet` and `dir-OFF-reference` have identical eval lines.** Difficulty 0.0 with a
  ratchet, and win rate never above the band, is exactly equivalent to no Director. It should be,
  and it is — a correctness check on the implementation.
- **The fixed suppression level moves results**: −0.5 → 37.0%, −1.0 → 35.0%, off → 27.0%. All n=1
  against a ~9.5-point noise floor, so suggestive only — this is what the branch wave tests.

Against frozen `COPS_STUDY_V2` the agent is weaker than that historic run (this wave's control spent
769/976 updates *below* the band), so the uptick should be expected **later** than 3.26M — plausibly
4–6M. Simply raising `--total-steps` would work but multiplies the pre-curriculum prefix across
every arm.

**Correction to an earlier reading of mine:** I first concluded "the adaptive controller has no
operating range on this task". That was wrong, and it was the user who spotted why — the runs were
too short, not the controller inert. The evidence above is consistent with a controller that simply
had not been reached yet.

## The redesign: branch from a post-uptick checkpoint

Train the shared prefix **once per seed**, then fork every Director arm from that checkpoint via
the new `--branch-from`. The pre-curriculum phase is paid 3 times instead of 15, and every arm
starts from an identical policy — a cleaner contrast than independent runs, since initialisation
variance drops out of the between-arm comparison.

| phase | manifest | tasks | what |
|---|---|---|---|
| 2 | `slurm/manifests/director_base.txt` | 3 | seeds 27/28/29 to 8M, difficulty pinned at −1.0 |
| 3 | `slurm/manifests/director_branch.txt` | 15 | 5 arms × 3 bases, branch → 15M |

Arms: fixed −1.0 (control), fixed −0.5, adaptive `[0.40,0.60]` (original), adaptive `[0.20,0.35]`
(lowered into the achievable range), and no Director.

**`--branch-from`, not `--resume`.** `--resume` forces the checkpoint's saved curriculum difficulty
(so the arms could not differ), rejoins its W&B run via `resume="must"` (15 forks colliding on one
run), and restores `best_eval_win_rate` (so a branch worse than its base never writes its own
`agent_best.pt`). `--branch-from` keeps weights, optimizer and the step counter — so LR annealing
stays on one schedule — while forking a fresh W&B run, honouring the curriculum flags on the command
line, and resetting the best-eval tracker. Provenance is saved as `branched_from`.

**`--curriculum-kp 0` pins difficulty** (delta becomes 0, so the update is skipped). A base pinned at
−1.0 is provably identical to a curriculum-ON base up to the uptick, so it costs nothing and makes
the base unambiguous.

**Branch point** = the first checkpoint where *training* win rate sustains > 0.60, i.e. where the
controller would first have hardened. `--keep-checkpoints 0` on the bases so pruning cannot discard
it. If no base crosses 0.60 by 8M, extend the bases — branching early reproduces this wave exactly.

---

## Goal

Find a Director configuration that keeps the curriculum's large advantage on the cops it trains
against **without** the generalisation loss that comes with it (see the measurement below: +11
points on home cops, −16 when the cops change). Failing that, establish the trade-off precisely —
it is a more interesting result than a single win-rate delta either way.

## Why this workstream exists

Added 2026-09-09, after the observation that historic Director runs underperformed. Chasing that
down turned up three things, and each one changes what the rest of the plan should do.

### 1. The Director was never tunable

`INITIAL_DIFFICULTY` was a module constant in `agents/curriculum_director.py`, not a CLI flag. The
03 sweep covers `lr`, `ent-coef` and the five reward coefficients and touches no curriculum
parameter at all. So the Director had exactly one configuration ever tried, and it was the first
one written down.

### 2. "The curriculum has never been run" was wrong

Both `04-final-runs.md` and the completion README said so. There is one ON run:

| | |
|---|---|
| Run | `wandb/offline-run-20260523_134803-mdw4ndpi`, 2026-05-23 |
| Config | 10M steps, `no_curriculum=False`, kp 0.1, band [0.4, 0.6], lr 3e-4 |
| Final | `charts/win_rate` 0.56, `curriculum/difficulty` **+0.252** |

It is the source of the Director's reputation, and it predates both the cop retune (`259ae5d`,
2026-06-11) and `eval/win_rate`. Its neighbours `fsc1zobb` (OFF, 10M) and `8dgaqm9w` (pre-flag, 10M)
finished at 0.67 and 0.83.

**The W&B summary comparison is not like-for-like**, and should not be quoted: the ON run's final
`charts/win_rate` of 0.56 was scored against `difficulty +0.252` cops — positive difficulty is
*injection*, cops strictly **better informed than baseline** — while the OFF runs' 0.67/0.83 were
against baseline cops.

**But the deficit is real and was measured properly.** The checkpoints were compared offline through
`training/eval.py`, which is Director-free on both sides (`eval_agent` passes `director=None`,
invariant 2), and the Director-trained policy still came out worse. That comparison is sound and is
the one to trust. Nothing below explains it away — the point of this workstream is to find out why.

### 3. The controller pins win rate at the band centre, by construction

Note what this does and does not explain. It makes `charts/win_rate` useless for *ranking* ON runs,
so it must not be used as the comparison metric in 04. It does **not** account for the offline
Director-free deficit above, which was measured the right way.

This is the important one, and it is not a bug — it is what a P-controller with a deadband does.

The trajectory from that run's `output.log`:

| steps | `charts/win_rate` | `curriculum/difficulty` |
|---|---|---|
| 3.20M | 0.51 | −1.000 |
| 3.57M | 0.67 | −0.926 |
| 3.94M | 0.44 | −0.616 |
| 4.30M | 0.51 | +0.005 |
| 6.51M | 0.45 | +0.161 |
| 10.0M | 0.56 | +0.252 |

Difficulty is driven until win rate re-enters `[0.4, 0.6]`, so the last 5.7M steps never leave
0.42–0.59. **The agent's improvement is absorbed into rising difficulty, not rising win rate.**

> **`charts/win_rate` on a Director-ON run is the controller's setpoint, not the policy's score.**
> Comparing it against an OFF run's win rate compares a thermostat setting to a temperature. Use
> `eval/win_rate`, which passes `director=None` (invariant 2), and report the final
> `curriculum/difficulty` alongside it — a higher final difficulty at equal eval win rate means the
> agent tolerated better-informed cops, which is exactly what the curriculum is supposed to buy.

Carry this into [04](04-final-runs.md) and [06](06-comparison.md).

### The ramp-speed hypothesis

Independently supported by the same trace. The entire −1.0 → 0.0 traverse happened between 3.3M and
4.3M steps — about 10% of the run — and win rate fell 0.67 → 0.44 across it, never recovering past
0.59 afterwards. Before it: 3.3M steps of flat full suppression, ample time to entrench a
shortest-path policy against near-blind cops.

The hypothesis is that unlearning shortest-path is harder than never learning it, so a policy raised
under full suppression is worse-positioned than one that met informed cops from the start. The core
grid below separates the two candidate causes — *where the ramp starts* against *how fast it moves*.

## MEASURED 2026-09-09 — the premise was wrong. The Director wins on its own cops and overfits to them.

Before designing anything further, the two 10M-step checkpoints were re-scored against **both** cop
configurations, 300 games, seed 11, paired (identical game sequence per row). The runs differ in
`no_curriculum` **and nothing else** — 22 of 26 config keys identical, same seed, same steps, same
`lr`. This is a clean controlled pair.

**vs `COPS_V1` — the pre-retune cops both policies actually trained against:**

| checkpoint | win% | turns | copdist | arrest% | timeout% |
|---|---|---|---|---|---|
| **ON** `mdw4ndpi` | **74.7%** | 9.1 | 1.09 | 19.7% | 5.7% |
| **OFF** `fsc1zobb` | 63.7% | 10.9 | 1.54 | 15.0% | 21.3% |

**vs `COPS_STUDY_V2` — the current frozen cops, which neither ever saw:**

| checkpoint | win% | turns | copdist | arrest% | timeout% |
|---|---|---|---|---|---|
| **ON** `mdw4ndpi` | 58.7% | 8.3 | 1.27 | 36.3% | 5.0% |
| **OFF** `fsc1zobb` | **62.7%** | 10.4 | 1.79 | 18.7% | 18.7% |

### What this actually says

1. **The Director works — by 11 points — on the cops it trained against.** The "Director
   underperforms" result was real but was measured against *retuned* cops the ON policy had never
   trained on. It is a train/test mismatch, not a training failure.
2. **The Director produces a BOLDER Jack, not a more risk-averse one — the inverse of the
   hypothesis below.** The ON policy runs close to cops (copdist 1.09), finishes fast (9.1 turns)
   and almost never times out (5.7%). It is the **OFF** policy that is cautious: wide berth (1.54),
   long games (10.9 turns), and 21.3% of all its games lost to the clock.
3. **The Director's policy generalises worse.** Change the cops and it drops 16 points (74.7 → 58.7)
   while the cautious OFF policy drops one (63.7 → 62.7). Boldness means threading gaps in a
   *specific* cop decision rule; caution is robust to which rule you face.

So the real question for this workstream is not "why does the Director lose" — it doesn't — but:
**can the Director be tuned to keep the boldness without the overfitting?** That is what the
ratchet and rate-limit probes are for: they change how long the policy spends at each difficulty,
which is exactly the knob on how narrowly it specialises.

### Consequences for 04

- **The historic deficit will not recur in 04.** Every arm there trains against `COPS_STUDY_V2` and
  is evaluated against `COPS_STUDY_V2`. The train/test mismatch that produced the deficit is gone,
  and the ON arm should be expected to *win*.
- **Report the generalisation gap — it is the most interesting result here.** "The curriculum buys
  +11 points against the cops you trained on and costs 16 when they change" is a sharper
  contribution than a single win-rate delta. **Tooling exists as of 2026-09-09:**

  ```bash
  uv run python -m training.eval <ckpts...> --held-out-cops --n-games 500
  ```

  prints two tables, the frozen `COPS_STUDY_V2` and the held-out `COPS_PRERETUNE_V1`. The second
  preset lives in `agents/heuristic_cops.py`, recovered from `259ae5d^`, and is pinned by
  `tests/test_cop_config.py`. It is a **test set only** — never the human comparison, which is
  `COPS_STUDY_V2` by invariant 1.
- **Caveat, stated plainly:** one seed per arm, n=300 (±~5.6pp). The 11-point V1 advantage is
  comfortably outside that; the 4-point V2 deficit is not significant. Re-running at n=200 with a
  different sample gave 62.5% vs 59.5% on V2 — the sign of that gap flips between runs, which is
  what "not significant" looks like in practice. The held-out gap did not flip. 04's three seeds
  exist for exactly this reason.
- **Possible contamination:** `COPS_STUDY_V2` came from an Optuna study that used *a trained Jack
  policy* as its adversary (see [01](01-freeze-cops.md) — the provenance is not recorded). If that
  adversary was one of these checkpoints, the cops were tuned to beat it and its V2 score is biased
  down. This does not affect 03/09/04, whose policies are all trained fresh against V2, but it must
  be stated as a limitation wherever these historic numbers appear.

## The superseded hypothesis: the Director may make Jack permanently risk-averse

**Measured false, 2026-09-09** — see above; the Director-trained policy is the *less* cautious of the
two on every metric, under both cop configurations. Kept here because the reasoning was sound, the
diagnostics it motivated are what produced the finding above, and the prediction should be re-checked
on 09's own runs rather than assumed settled from a single seed.

Raised 2026-09-09, and it does not depend on ramp speed at all. **A curriculum that spends its whole
life holding Jack at a 50% win rate teaches him to survive, not to win.** Jack is trained under cops
that periodically become better-informed than any cops he will be evaluated against, so the policy
that survives that regime is a cautious one — wide berths, safe detours. Evaluated against baseline
cops, a bolder policy that threads between them reaches the hideout in fewer turns and wins more,
while the cautious one detours into the turn limit.

If that is what is happening, **no amount of Director tuning fixes it**, because it is a property of
training against an adversary tuned to beat you half the time. That would be a genuine, reportable
finding about the contribution — but it is a conclusion to reach *after* the sweep, not a reason to
skip it. The two hypotheses are not exclusive either: a bad ramp could produce a risk-averse policy.

### It is now directly measurable

`eval_agent` gained three metrics on 2026-09-09 specifically to separate this from ordinary
underperformance. All are Director-free and logged under `eval/` at every checkpoint.

| metric | meaning |
|---|---|
| `mean_min_cop_dist` | Jack's mean BFS distance to his nearest cop, averaged over the game. The direct berth-width measure; mirrors the `zeta` shaping term. |
| `arrest_rate` | Fraction of **all** games lost to arrest. |
| `timeout_rate` | Fraction of **all** games lost to the turn limit. (`win_rate + arrest_rate + timeout_rate == 1`.) |
| `arrest_share_of_losses` | Of the losses only, the fraction that were arrests. The cleanest signal — it moves even when the win rate does not. |

**The prediction to test.** Risk aversion says the Director-trained policy loses *differently*, not
merely more: higher `mean_min_cop_dist`, higher `mean_turns_on_win` (detours cost turns), and losses
shifted from arrest toward timeout — a lower `arrest_share_of_losses`. If instead it shows the same
loss split and the same cop distance at a lower win rate, it is simply a weaker policy and the
curriculum's problem is optimisation, not induced caution.

The same columns are in the `training/eval.py` CLI table (`copdist`, `arrest%`, `timeout%`), so the
existing ON-vs-OFF checkpoint comparison can be re-run and read for this immediately — it does not
have to wait for the sweep.

## What was implemented (2026-09-09)

Three new flags in `training/train.py`. **All three default to the existing behaviour**, so nothing
already planned changes shape and the ON/OFF comparison stays interpretable.

| flag | default | effect |
|---|---|---|
| `--initial-difficulty` | −1.0 | Starting difficulty. Was the hardcoded `INITIAL_DIFFICULTY`. `--resume` still wins, restoring the saved value. |
| `--curriculum-max-step` | 2.0 | Cap on per-update movement. The difficulty range is 2.0 wide, so the default never binds. |
| `--curriculum-ratchet` | off | Difficulty may only increase; the controller cannot relax back toward suppression. |

The controller also no longer sends a `set_difficulty` broadcast when the value is unchanged
(clamped at ±1.0) — behaviour-neutral, saves pointless IPC.

## The sweep

[`slurm/manifests/director.txt`](slurm/manifests/director.txt) — 17 tasks, `--array=0-16%9`, 3M
steps each, ~2.5h wall. At 03's chosen `lr`/`ent-coef`.

- **Core grid (9):** `initial-difficulty {−1.0, −0.5, 0.0}` x `curriculum-kp {0.02, 0.1, 0.3}`.
  Separates "starts too suppressed" from "ramps too fast". Cell (−1.0, 0.1) is the **control** — the
  historic configuration exactly.
- **Ramp-shape probes (5):** rate limit, ratchet, both together, and two deadband widths
  (`[0.3, 0.7]` and `[0.45, 0.55]`), each one change off the control.
- **Combined candidates (2):** `initial-difficulty 0.0 --curriculum-ratchet` is the purest test of
  the hypothesis — the Director starts inactive and only ever gets harder, so there is no
  suppression phase to unlearn. Plus a maximally gentle everything-slowed config.
- **OFF reference (1):** same `lr`/`ent`, `--no-curriculum`. Every line above must beat it on
  `eval/win_rate` for the Director to be worth claiming.

## Judging

1. **Primary: `eval/win_rate`** (Director-free). Against the OFF reference in the same wave, not
   against historic runs — those faced pre-retune cops.
2. **Secondary: final `curriculum/difficulty`.** At equal eval win rate, higher is strictly better.
3. **Diagnostic: `eval/mean_min_cop_dist` and `eval/arrest_share_of_losses`.** These say *why* an arm
   lost, and are what separates the ramp-speed story from the risk-aversion one. Record them for
   every arm even when the win rate settles the ranking — if the Director loses, the thesis needs the
   mechanism, not just the number.
4. **Ignore `charts/win_rate`** for ranking ON runs. Read it only to confirm the controller
   converged into its band rather than saturating at a clamp.

At 3M steps the control may not leave difficulty −1.0 at all (see
[03](03-ppo-sweep.md) — the 200k-step pilot sat at 3–8% win rate, and the historic run needed 3.3M
steps to first move). **If no arm's difficulty moves, the wave has measured nothing** and the sweep
needs 6M steps rather than a different grid. Check this on the first finishing task rather than
after all 17.

## The outcome that is still a result

If nothing beats the OFF reference, the honest finding is that adaptive difficulty control does not
help this task — and by then the diagnostics should say which mechanism, which is what makes it a
finding rather than a shrug:

- **Induced caution** — losses shifted toward timeout, wider `mean_min_cop_dist`, longer wins. The
  curriculum taught survival where the task rewards arrival.
- **Optimisation damage** — same loss split, same cop distance, lower win rate. The ramp disrupted
  learning; a gentler arm in the grid should then have recovered some of it.
- **Neither, cleanly** — the arms are indistinguishable and the Director is simply inert at 3M steps.

A negative result with a mechanism is a perfectly good chapter, and much better than shipping an
untuned arm that loses for reasons nobody looked into.

## Session log

- 2026-09-09 — workstream created. Traced the "Director does worse" belief to
  `offline-run-20260523_134803-mdw4ndpi`, the project's only ON run, and found the comparison was
  never like-for-like (its 0.56 was scored against `difficulty +0.252` cops, i.e. *harder* than
  baseline, and it predates `eval/win_rate` entirely). Corrected the "never been run" claim in
  [04](04-final-runs.md) and the [README](README.md).
- 2026-09-09 — identified that the deadband pins ON-arm win rate at the band centre by construction,
  which makes `charts/win_rate` unusable for ranking ON runs and explains the apparent deficit.
  This changes the reporting metric for 04 and 06, not just this workstream.
- 2026-09-09 — implemented `--initial-difficulty`, `--curriculum-max-step` and
  `--curriculum-ratchet` (all defaulting to current behaviour) and wrote `director.txt`.
  Not submitted — cluster execution is out of scope for this session.
- 2026-09-09 — **re-measured the premise and it did not hold.** Scored both 10M checkpoints against
  the pre-retune cops as well as the frozen ones. The Director run beats the no-Director run by
  **11 points on the cops both trained against** (74.7% vs 63.7%) and loses by 4 on the retuned cops
  it never saw. The historic "Director is worse" finding was a train/test mismatch. The
  risk-aversion hypothesis is measured false in the strong sense — the ON policy is the *bolder* of
  the two on every diagnostic, under both cop sets. The real effect is a **generalisation gap**, and
  it is now this workstream's subject. Runs verified as a controlled pair: 22 of 26 config keys
  identical, differing only in `no_curriculum`.
- 2026-09-10 — **array `1807070` completed: null, and the design was at fault, not the grid.** All
  17 tasks finished. Difficulty left the floor on update 976/976, six configurations produced
  byte-identical eval metrics, and the ramp-shape parameters were never exercised. Root cause:
  3M-step runs end ~250k steps before the first uptick, which the historic 10M run put at step
  3,256,320. **The user diagnosed this; my first reading ("the controller has no operating range")
  was wrong.**
- 2026-09-10 — redesigned around branching. Added `--branch-from` to `training/train.py` so the
  shared pre-curriculum prefix is trained once per seed and every arm forks from it: 3 base runs
  instead of 15 prefixes, and arms that differ only in their Director settings rather than also in
  initialisation. Verified end to end — step continues, CLI curriculum flags win over the
  checkpoint's saved values, a fresh W&B run id is allocated, `best_eval_win_rate` resets, and
  `branched_from` records provenance. Also verified `--curriculum-kp 0` pins difficulty exactly.
- 2026-09-10 — **fixed a reporting bug of my own.** `sweep_report`'s `curriculum_engaged` returned
  True for any `max_difficulty > -1.0`, so this wave's single final-update nudge was reported as
  "curriculum engaged in 15/16 ON runs" — the opposite of the truth. It now requires a margin off
  the floor *and* a non-trivial share of updates spent there, measured against each run's own
  starting difficulty so a fixed −0.5 run correctly reads as never leaving its floor. Added an
  `off_floor%` column and seed-aware aggregation (mean ± half-range), since the branch wave is the
  first with replicates.
- 2026-09-11 — **branch wave `1807209` complete, 15/15. The Director does not help.** The redesign
  worked in the sense that mattered: both adaptive arms show `off_floor% = 100` and ramped from
  −1.0 into injection territory, so the controller was genuinely live. It still lost. No Director
  93.2%, fixed −0.5 88.3%, adaptive [0.40,0.60] 81.3%, adaptive [0.20,0.35] 61.3%, fixed −1.0
  50.7% — ranges on the top two do not overlap. Handicapping the cops during training is monotonically
  harmful, and the diagnostics say why: a policy raised against blind cops runs ~0.45 closer to them
  and is arrested ~8x more often at evaluation.
- 2026-09-11 — bounded the claim: all arms share a −1.0 prefix, so `dbr-off` is "suppressed prefix
  then no Director", not a pure control. 04's OFF arm supplies the from-scratch number; until then
  report this as a controlled between-arm comparison, not an absolute one.
