# 04 — Final runs: Director on/off

**Status:** **IN FLIGHT — array `1807480`, submitted 2026-09-13 16:58, 9 tasks, ~5–7h.** `slurm/manifests/final.txt` is filled and validated
(`uv run pytest tests/test_run_manifests.py`, 671 passed). 09 is closed and supplied the ON arm's
configuration: `--initial-difficulty -1.0 --curriculum-max-difficulty 0.0`, default band.

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
