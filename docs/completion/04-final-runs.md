# 04 — Final runs: Director on/off

**Status:** **DONE (2026-09-19).** Array `1807480`, 9/9, exported and re-evaluated at 2,000 games
per checkpoint on fresh boards. The Director wins on every seed; the `sparse` arm buys win rate by
abandoning stealth; the generalisation gap is identical across arms. Results below.

> **The 2026-09-10 blocking caveat is cleared (2026-09-13).** It said the ON arm's Director config
> had never been tested in the regime where it operates. Five further waves did exactly that; see
> [09](09-director-tuning.md). The three concerns it raised, resolved:
>
> - *"Runs end before the curriculum does anything"* — fixed by `--branch-from` and by moving to 15M
>   steps. The first uptick is at 4.4–4.8M; every seeded wave since `1807209` shows the controller
>   live (`off_floor%` 72–100).
> - *"The three seeds are only now real"* — still true and still the binding constraint. The
>   ~9.5-point noise floor is why 04 runs 3 seeds per arm and why it re-measures on fresh ones.
> - *"The sparse ablation may not show what this file expects"* — confirmed. At 3M the sparse config
>   scored *above* the shaped control (33.0% vs 30.0%). Run it anyway at 15M and report it as a null
>   if that is what it is; "reward shaping cannot be shown to help" is a legitimate finding.

## RESULT (2026-09-19) — re-evaluated, 2,000 games, fresh seed, final checkpoints

Selection-free: the **final** checkpoint of each run (not `agent_best.pt`, which is chosen on one
fixed board set), 2,000 games on boards no training evaluation saw, all nine scored in one
invocation so every arm plays identical boards. Raw output in
[`results/final_reeval.txt`](results/final_reeval.txt).

| arm | participant score | win% | hideout_u | arrest% | timeout% |
|---|---|---|---|---|---|
| **`director-on`** | **1.328** [1.317–1.334] | 90.5 | 0.79 | 7.1 | 2.3 |
| `director-off` | 1.300 [1.285–1.324] | 87.0 | 0.78 | 9.4 | 3.7 |
| `sparse` | 1.299 [1.285–1.309] | **93.9** | 0.68 | 5.4 | 0.8 |
| random Jack (floor) | 0.399 | 0.1 | 0.64 | 93.7 | 6.2 |

**The Director wins on all three seeds: +0.028 score (+0.010 / +0.032 / +0.043), +3.5 win points.**
The in-training ordering survives re-measurement; the absolute numbers drop slightly, as expected
when a selected "best" is replaced by a final checkpoint on unseen boards.

### The `sparse` arm shows why win rate is the wrong headline

It has the **highest win rate in the wave (93.9%) and a lower score than both Director arms**,
because it zeroed `gamma` and stopped hiding the hideout: uncertainty 0.68 against 0.78–0.79.
Paired against `director-on` it is **+3.4 win points but −0.029 score on every seed**. Two metrics,
opposite orderings, identical games. Report the score.

This also replaces the rough estimate in [10](10-reward-design.md) (1.27 / 1.26 / 1.23) with
measured values: 1.328 / 1.299 / 1.300.

### Generalisation: no arm overfits to its training cops more than another

Scored again against `COPS_PRERETUNE_V1`, which no current policy trained on:

| arm | study cops | held-out cops | gap |
|---|---|---|---|
| `director-on` | 1.328 | 1.209 | −0.119 |
| `director-off` | 1.300 | 1.183 | −0.117 |
| `sparse` | 1.299 | 1.180 | −0.119 |

**The gaps are identical to three decimals.** The 2026-09-09 worry that the Director overfits to the
cops it trained against — the original reason workstream 09 existed — does not survive a fair test.
Every arm loses the same amount and the ordering is unchanged, so the gap is a property of the cop
retune, not of the curriculum.

### What 04 could not see, and 10 could

04's Director effect is **+0.028**, against the same comparison in workstream 10 at
**+0.093 to +0.143**.

**The explanation written here on 2026-09-19 — "both of 04's arms had shaping on, and shaping does
the curriculum's job" — is withdrawn.** That rested on the substitution finding, which was retracted
at 6 seeds ([10](10-reward-design.md) §2): shaping has no measurable effect either way.

The remaining explanation is simpler. **04 has 3 seeds, so its own detection threshold is 0.118**
(FINDINGS §6b) and +0.028 is far inside it — 04 cannot resolve the Director effect at all. Its three
seeds all favour ON, which is suggestive and no more. The Director claim rests on workstream 10's
6-seed measurement; 04's contribution is the `sparse` arm, the held-out-cops comparison and the
legacy-reward baseline, not the size of its Director gap.

Cross-checking the waves on the identical metric: `director-on` (legacy reward, shaped) scores
**1.328**, `w10s-obj-cur` (stealth objective, no shaping) scores **1.326**. Indistinguishable — the
objective change did not make training harder, and shaping adds nothing once the curriculum is on.

### Why 04 re-runs arms that already exist

`fsc000-i100` (array `1807390`) is this file's ON arm flag-for-flag — 15M steps, from scratch,
`lr 3e-4`, `ent-coef 0.03`, default band, `--curriculum-max-difficulty 0.0`, seeds 27/28/29. `fs-off`
(array `1807292`) is its OFF arm. Together they already give **92.0% [91.0–93.0] vs 85.2%
[81.5–88.5]**, paired per seed, **+6.8 points**.

That is the tuning estimate, not the headline. `--curriculum-max-difficulty 0.0` was *selected* as
the best of ~6 ceiling arms, and a maximum over 6 arms at a 9.5-point noise floor is biased upward.
The OFF arm carries no such bias, so the effect's *sign* is safe either way — its *magnitude* is
what 04 measures, on **fresh seeds 31/32/33**. The tuning study picks the configuration; 04 reports
the number. Keep both in the writeup and say which is which.

**Blocks:** 06
**Blocked by:** 01 (frozen cops), 02 (reproducibility), 03 (chosen config)

---

## Goal

Produce the headline result: a trained policy **with** the Director and a matched one **without**,
plus seed repeats, all against frozen cops v2.

## Why it matters — read this before deprioritising anything here

**The curriculum has been run exactly once, in May 2026, and never since.** Every training run since
2026-06-02 passed `--no-curriculum`; those W&B summaries show `curriculum/difficulty: -1`, frozen at
`INITIAL_DIFFICULTY`.

> **Correction (2026-09-09).** An earlier version of this file, and of the completion README, said
> the curriculum had *never* been switched on. That is wrong.
> `wandb/offline-run-20260523_134803-mdw4ndpi` (2026-05-23, 10M steps) has `no_curriculum=False`
> and ended at `difficulty=+0.252`. It is the only ON run in the project's history, it predates both
> the cop retune and `eval/win_rate`, and **it is the run that gave the Director its reputation for
> underperforming.** See [09](09-director-tuning.md) for what it actually shows.

The Director is the thesis's named contribution. The with/without comparison is a headline result,
not a side ablation, and it currently has no "with" arm at all.

## Current state

### The Director is implemented and wired

`agents/curriculum_director.py` (82 lines), `INITIAL_DIFFICULTY = -1.0` (full suppression, easiest
for Jack):

- `d < 0` -> suppression (`curriculum_director.py:56-64`): drop each `visited_at` entry with
  probability `|d|`, **except depth-0** (the public start position).
- `d > 0` -> injection (`curriculum_director.py:65-76`): reveal undiscovered nodes from `jack_trace`
  with probability `d`. Only real nodes, no ghosts. `sorted()` for deterministic RNG consumption.
- Difficulty is **snapshotted at `on_episode_start`** (`curriculum_director.py:46-47`) so it is
  constant within an episode.

The P-controller (`train.py:464-478`):

```python
if not args.no_curriculum and len(ep_wins) >= 10:
    target_centre = (args.curriculum_target_low + args.curriculum_target_high) / 2.0
    if win_rate < args.curriculum_target_low or win_rate > args.curriculum_target_high:
        error = win_rate - target_centre
        curriculum_difficulty = clamp(curriculum_difficulty + args.curriculum_kp * error, -1, 1)
        envs.set_difficulty(curriculum_difficulty)
```

Path: `envs.set_difficulty` (`train.py:192-196`) -> worker pipe (`train.py:96-99`) ->
`JackEnv.set_director_difficulty` (`env.py:192-194`) -> `CurriculumDirector.set_difficulty`, taking
effect at the next episode start. Deadband `[0.4, 0.6]` on a 100-episode deque win rate. Difficulty
is logged as `curriculum/difficulty` and persisted in the checkpoint.

`NoOpDirector` (`agents/random_agents.py:39-41`) is the default when `JackEnv` gets `director=None`.

### Existing checkpoints are all unusable for final results

7.4 GB across 7 runs, **all trained against pre-retune cops** (last run finished 2026-06-08; retune
landed 2026-06-11 in `259ae5d`). Keep them for the appendix at most.

---

## Why both arms are comparable by construction

State both of these explicitly in the writeup:

1. **Evaluation is already Director-free.** `eval_agent` runs against full-strength
   `HeuristicCops()` with `director=None` (`training/eval.py:84`), regardless of how the policy was
   trained. The Director shapes *training* difficulty only; both arms are scored on the identical
   task.
2. **Cops are frozen** at v2 for both arms, so the only difference between them is the Director.

Carried-over caveat from 03: if the sweep was run with the curriculum ON and its config reused for
the OFF arm, that mildly favours ON. Sweep both arms or declare the choice.

---

## Run budget

15M steps each. ~8.5h on an RTX 5080; CPU-only timing to be measured by the 02-C3 pilot.

Cluster limits (72 CPU / 2 GPU / 20 queued jobs per person) allow **9 concurrent CPU-only runs** at
`--n-envs 12 --n-workers 12`. The full final set fits that exactly:

| Run | Seeds | Concurrent slots | Purpose |
|---|---|---|---|
| `director-on` | 3 | 3 | Curriculum training. The first real curriculum run in the project. |
| `director-off` (`--no-curriculum`) | 3 | 3 | The paired comparison arm |
| `sparse` | 3 | 3 | Reward ablation (DESIGN_REQUIREMENTS §7.3) |
| | | **9 total** | fits the CPU cap and the 20-job limit |

**So the entire final run set can execute in a single window rather than sequentially** — submit it
as one array (`--array=0-8%9` — 9 tasks, comfortably under the 20-job submit cap) and the whole
thing lands together. That is what makes the
reward ablation affordable alongside the seeds, rather than a thing to cut.

Both Director arms **must** use identical hyperparameters and the *same seed values* (e.g. 27/28/29),
so the comparison is paired at the seed level rather than merely averaged. `--n-envs` and `--n-steps`
must match the 03 sweep exactly — batch size is `n_steps * n_envs`, so changing either would
invalidate the sweep's conclusions.

**Seed repeats are the point of having the cluster.** PPO seed variance can easily exceed the effect
being claimed, so a one-seed-per-arm gap would not support the conclusion regardless of how large it
looks. If anything has to be cut, cut the reward ablation — not the seeds.

Report the Director comparison with per-seed numbers as well as the mean, so the reader can see
whether the arms separate consistently or only on average.

### Walltime risk — check before submitting

15M steps must finish inside the **24h cap**. That is a floor of `15e6 / 86400` = **~174 SPS**, and
the current GPU-based figure is ~490 SPS. So a CPU-only run has to stay within ~2.8x of GPU speed to
fit at all, and within ~2x to have comfortable margin.

The 02-C3 pilot measures this. If CPU SPS lands below ~250, pick one of:

- run the final runs on GPU (2 concurrent, sequenced in waves) and keep CPU for the 03 sweep, whose
  3M-step runs are ~5x shorter and fit trivially;
- reduce total steps for the final runs (acceptable — but apply it uniformly to every arm);
- resume-chain only these runs, accepting the extra queue wait.

Decide from the measurement, not in advance, and record the choice here.
## Verification

```bash
uv run python -m training.eval checkpoints/<run>/agent_best.pt --n-games 500
```

- Confirm `curriculum/difficulty` **moves away from -1.0** in every `director-on` seed. If it stays
  pinned, the Director is not engaging and the headline result is void.
- Confirm it **stays at -1.0** in `director-off`, so the arms are genuinely distinct.
- `training/eval.py` accepts multiple checkpoints and evaluates them under an identically-seeded RNG
  — score both arms and all seeds in a **single invocation** to keep the comparison paired.
- `PolicyAgent` **samples** rather than acting greedily (`eval.py:37-56`; greedy exists only in
  `tools/gen_replay.py --greedy`). Pick one mode and use it consistently across every reported number.

## Session log

- 2026-09-09 — cluster access confirmed; seed repeats promoted from "if compute allows" to the plan
  of record (3 seeds per Director arm).
- 2026-09-09 — cluster limits (72 CPU / 2 GPU / 20 jobs) allow 9 concurrent CPU-only runs, so the
  full 9-run final set (2 Director arms + sparse, 3 seeds each) fits in one window. Reward ablation
  promoted from "cut this first" to affordable. Flagged the 174 SPS walltime floor as the one thing
  that could force the final runs onto GPU.
- 2026-09-13 — **unblocked and filled.** 09 closed with `--curriculum-max-difficulty 0.0` (default
  band). Filled `final.txt` with lr 3e-4 / ent 0.03 and the Director flags on seeds **31/32/33**,
  deliberately not the tuning study's 27/28/29: `fsc000-i100` and `fs-off` already constitute this
  file's two arms at 92.0% vs 85.2%, but the ON config was selected as the best of ~6 ceiling arms
  and so carries a winner's curse at the 9.5-point noise floor. 04 re-measures it clean. Validated
  with `tests/test_run_manifests.py` (671 passed) before submitting — the manifest test is what
  catches a `--gamma`/`--reward-gamma` slip or an off-spec `--n-envs` for free on the laptop.
- 2026-09-14 — **array `1807480` complete, 9/9, sanity gate passed** (ON left −1.0 in 3/3, OFF pinned
  in 3/3). In-training best eval: `sparse` 95.0% [93.0–96.5], `director-on` 93.2% [92.5–94.5],
  `director-off` 90.7% [89.5–91.5]; ON > OFF on every seed. The gap shrank from the tuning estimate
  (+6.8) because **OFF moved up** (85.2 → 90.7), not because ON was inflated: ON replicated.
  **The `sparse` arm is not a valid shaping ablation**: it zeroed `gamma`, which is part of the
  objective (see [10](10-reward-design.md)). The Director result stands under the legacy reward, but
  06's checkpoints now come from workstream 10's wave, trained to win stealthily and reported on
  the participant score.
  Export, fresh-seed re-evaluation and write-up are still pending.
- 2026-09-19 — **exported, re-evaluated and closed.** 2,000 games per checkpoint on fresh boards,
  final checkpoints only, all nine in one invocation. Director +0.028 score on every seed (+3.5 win
  points). `sparse` has the highest win rate and a lower score than both Director arms — the
  cleanest case in the project for reporting the objective rather than win rate. Generalisation gaps
  identical across arms (−0.117 to −0.119), which retires the Director-overfitting worry. The ON arm
  ties workstream 10's shaping-free curriculum arm (1.328 vs 1.326).
