# 09 — Director tuning

**Status:** **RUNNING** — submitted 2026-09-09 23:24 as job array **`1807070`** (17 tasks,
`--array=0-16%9`, ~2.5h, ETA ~02:00). Filled with `lr=3e-4 --ent-coef 0.03` from
[03](03-ppo-sweep.md).

```bash
ssh cluster 'squeue -u $USER'
ssh cluster 'cd ~/MLs_from_Whitechapel && export PATH=$HOME/.local/bin:$PATH && \
    UV_NO_SYNC=1 uv run python -m analysis.sweep_report "logs/wc-train_1807070_*.out"'
ssh cluster 'scancel 1807070'     # abort
```

**Read it as a SCREEN, not a decision.** One seed per config against a measured **~9.5-point
noise floor** — wave 1's `sw1-lr3e4-ent003-on` scored 39.5% and wave 2's `sw2-control`, the
identical command, scored 30.0%. Most plausible Director effects are smaller than that. Take the
top 2–3 configs and confirm them at 3+ seeds before anything enters 04.

**The first thing to check is not the ranking**, it is whether `diff_max` leaves −1.000 in *any*
run. It did not in a single one of wave 1's nine ON runs. `sweep_report` now prints this
explicitly. If it stays pinned here too, the curriculum still never engages and the fix is **longer
runs, not a different grid**.

These are also the first runs with the 2026-09-09 seeding fix, so `--seed` finally controls weight
init, action sampling and minibatch order rather than only the env workers.
**Blocks:** 04 — the headline ON arm is not worth 3 seeds until the Director is tuned
**Blocked by:** 03 (needs the chosen `lr`/`ent-coef`)

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
