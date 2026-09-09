# 04 — Final runs: Director on/off

**Status:** not started
**Blocks:** 06
**Blocked by:** 01 (frozen cops), 02 (reproducibility), 03 (chosen config)

---

## Goal

Produce the headline result: a trained policy **with** the Director and a matched one **without**,
plus seed repeats, all against frozen cops v2.

## Why it matters — read this before deprioritising anything here

**The curriculum has never actually been run.** Every training run since 2026-06-02 passed
`--no-curriculum`. All four recent runs' W&B summaries show `curriculum/difficulty: -1`, frozen at
`INITIAL_DIFFICULTY`. The `CurriculumDirector` is implemented, wired end-to-end, and has **never
been switched on in a real run**.

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
