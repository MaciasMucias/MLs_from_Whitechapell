# 06 — Human-vs-RL comparison

**Status:** **RESULT IN (2026-09-19), interim by design.** The agent beats the participants on the
boards they actually played, by an interval clear of zero on both the participant score and win
rate, with zero desyncs. **Recruitment is still running**, so every number here is dated to the
`games_20260919.sqlite` snapshot (20 participants / 60 games) and should be re-run before
submission — one `fly ssh sftp get`, then one `analysis.compare` invocation.

---

## Built (2026-09-09)

| | |
|---|---|
| `engine/game.py` | **E1** — `run_game(..., initial_state=None)`. Omit it and behaviour is byte-identical; supply one and the game starts from precisely that board. |
| `analysis/compare.py` | **E2 + E3** — scenario reconstruction, paired replay, move-level agreement. |
| `tests/test_compare.py` | 29 tests, fixtures frozen in `tests/fixtures/participant_games.json`. |

```bash
uv run python -m analysis.compare ckpt_on.pt ckpt_off.pt \
    --db data/study/games_20260909.sqlite --n-replays 20
```

It takes a **list** of checkpoints so `director-on`, `director-off` and every seed are scored on the
identical set of human scenarios in one invocation, and it reads its exclusion policy from
`analysis.sessions` rather than inventing a second one.

### Validated, not merely written

The load-bearing claim is that a stored replay reconstructs the exact board a participant faced.
Three independent checks, all green on all six fixture games:

1. Starting position, cop placement, hideout, and hideout zone all match the stored replay.
2. The reconstructed state is a clean turn-0 opening — nothing searched, Jack's start public.
3. **The engine's legal-move set at turn 0 equals the one stored in the replay.** A mismatch would
   mean the replay was produced under different rules than the policy is evaluated under.

And at runtime, `move_agreement` walks each human's real move sequence forward through the engine.
`HeuristicCops` is deterministic given a state, so the walk should reproduce the original game
exactly; any divergence is counted as a **desync** and reported. **Across all 33 human games and
three checkpoints: zero desyncs.** That is the strongest evidence the reconstruction is faithful.

## RESULT (2026-09-19) — `w10s-obj-cur`, 20 participants / 60 games

Each policy replays **the exact board its human counterpart faced**, 20 times per scenario (the
policy samples rather than acting greedily). Humans are scored by replaying their own recorded moves
through the engine, so both sides are measured by the same function. Raw output in
[`results/comparison_20260919.txt`](results/comparison_20260919.txt).

| | participant score | win rate |
|---|---|---|
| **Humans** (20 participants, 60 games) | **0.672** [0.578, 0.770] | **31.7%** [23.3%, 40.0%] |
| `w10s-obj-cur-s41` | 1.256 | 84.6% |
| `w10s-obj-cur-s42` | 1.310 | 88.8% |
| `w10s-obj-cur-s43` | 1.261 | 83.8% |

**Paired, clustered by participant — every interval excludes zero:**

| seed | score difference | win-rate difference |
|---|---|---|
| s41 | **+0.584** [+0.471, +0.689] | +52.9 pts [+42.0, +63.1] |
| s42 | **+0.638** [+0.520, +0.751] | +57.1 pts [+46.3, +67.0] |
| s43 | **+0.589** [+0.458, +0.712] | +52.1 pts [+40.2, +63.4] |

**Zero desyncs across all 60 games and three checkpoints**, so every replayed board is the board the
human actually faced.

### Three things the headline number does not say

1. **The agent does not play like a better human.** Move agreement is **27–29%**: at nearly three
   quarters of the decisions a participant made, the policy would have gone somewhere else. A
   20-point win-rate gap with ~20% agreement was already visible in the 2026-09-09 smoke run against
   stale checkpoints; it survives against the real agents with a 52-point gap.
2. **Board difficulty does not explain human losses.** The policy wins **85.1%** of the boards its
   human lost and **83.4%** of the boards its human won (s41; the other seeds match). If humans lost
   mainly to unlucky boards, the first number would be much lower than the second. It is not.
3. **Experience barely separates the humans** — 66.2% / 68.9% / 69.2% score for never-played /
   played-a-few / played-many, and 30.8% / 33.3% / 33.3% win rate. With 13 / 5 / 2 participants this
   is descriptive only, but nothing in it suggests experienced players approach the agent.

### Caveats to carry into the write-up

- **Interim N.** Recruitment continues; re-run before submission and re-state the date.
- **The `played_many` group is two people.** The project's "varying skill levels" claim cannot be
  supported beyond "we collected self-reported experience and saw no clear gradient".
- **The design effect is 1.02**, so clustering by participant barely widens the intervals here — but
  it is still the correct interval, and the machinery is cheap.
- The policy scores 1.26–1.31 on the course maps against 1.326 on `whitechapel.json`: the course
  scenarios are a different, slightly harder board set. Compare agents to humans within this table,
  not across to the training-map numbers.

## Smoke result — NOT a finding

Run against the **stale** pre-retune checkpoints, purely to prove the pipeline. These policies never
played `COPS_STUDY_V2` and are not the thesis's agents.

```
Human games: 33          Human win rate: 9/33 = 27.3%

checkpoint            step   policy win%  vs human   human lost(24)  human won(9)  agree%  desync
mdw4ndpi            10.00M         46.7%    +19.4%           45.0%         51.1%   20.1%       0
atomic-feather-3    10.00M         49.2%    +22.0%           39.6%         75.0%   22.0%       0
sparse-copdist-15m  20.00M         47.3%    +20.0%           39.8%         67.2%   22.0%       0
```

Worth noticing for later: **move agreement is ~20-22% while win rates differ by 20 points.** The
agents do not win by playing like better humans — they play a substantially different game. That is
exactly the question E3 exists to answer, and it will be worth re-asking per Director arm.

## Steps

---

## Goal

The thesis's actual deliverable: a defensible comparison between human participants and the trained
agent, for **both Director arms**.

## Why it matters

Nothing in the codebase joins `data/games.sqlite` to policy evaluation. Human data collection works;
policy evaluation works; **no code connects them**. This is the one genuinely missing piece.

## Current state

- `training/eval.py` has no awareness of participant data — no reference to `games.sqlite`,
  `ParticipantGame`, or human outcomes anywhere in the repo.
- Closest existing thing: `tools/run_and_save_replays.py` runs a checkpoint once per course map and
  writes replays through the server's replay writer. It fakes a `GameSession` with
  `types.SimpleNamespace` (`run_and_save_replays.py:36-45`) — `build_replay` only touches
  `session.ctx`, `session.game_id`, `session.map_name`. **That is the seam to reuse.**
- `server/replay_routes.py:34` `_build_session_from_replay` already reconstructs a session from a
  stored replay — reuse this reconstruction pattern.

### What the stored replays actually contain

Richer than expected, and this is what makes the whole analysis possible. `ReplayRecord`
(`server/replay.py:67-81`) holds `initial_jack_pos`, `initial_cop_positions`, `hideout`,
`hideout_zone`, `hideout_zone_anchor`, `turn_limit`, `blocking`, `map_name` — a complete scenario
specification. Per round (`replay.py:47-64`) it stores Jack's from/to/via, **the legal move set**,
every cop's move/search/arrest with hits and misses and heuristic role/coverage/direction scores,
**the cop position PMF at planning time**, the PMF after cops acted, the hideout PMF, and cumulative
cop knowledge.

That is a per-decision dataset. It supports asking what the policy would have done at each state a
human faced.

### Why a paired design is required

The course maps pin Jack's start and the hideout *zone anchor*, but the hideout itself is still
sampled within the zone and cop starts are sampled from the pool (`engine/env.py:60-72`):

| map | jack_starts | cop_starts | hideout_min_distance | zone_radius | zone_anchor |
|---|---|---|---|---|---|
| whitechapel | 8 options | 7 | 4 | 3 | none (random) |
| course_1 | `[147]` | 5 | 6 | 3 | 112 |
| course_2 | `[27]` | 5 | 4 | 1 | 159 |
| course_3 | `[149]` | 6 | 4 | 4 | 23 |

So scenarios are comparable across participants but **not bit-identical**. Averaging at the map level
would confound scenario difficulty with player skill. Replaying each human's exact scenario removes
that confound.

---

## Steps

### E1 — Allow injecting an initial state into `run_game`

`engine/game.py:205-228` always calls `make_initial_state(game_map, rng)`. Add an optional
`initial_state: GameState | None = None` parameter. No behaviour change when omitted.

**This is the one engine change the whole comparison depends on.** Keep it minimal.

### E2 — `analysis/compare.py` — paired scenario replay

For each human game in the snapshot: reconstruct the exact scenario from its stored replay, then run
the policy on that identical scenario N times.

- Take a **list of checkpoints**, not one — `director-on`, `director-off` and every seed must be
  evaluated on the identical set of human scenarios.
- Output: win rate and survival, human vs RL, broken down by `gaming_habit` and by course map.

This gives the Director comparison a second, independent axis beyond win rate against heuristic
cops: which arm is closer to (or better than) human play on the exact boards humans faced.

### E3 — Move-level agreement

At every human decision point, ask what the policy would have chosen. The legal move set and the cop
PMF at planning time are already stored, so this is **analysis only — no new games need to be run**.

Produces: agreement rate, and a value-function comparison of the human's chosen move vs. the
policy's. Computed per Director arm, it answers something the win rate cannot: **does the Director
change how the agent plays, or only how often it wins** — do the arms diverge in move choice, or
agree on moves and differ only in outcome?

This is what turns a single win-rate number into a results chapter.

### E4 — Baselines table

`RandomJack`, humans by skill group, `director-on`, `director-off` — all against frozen cops v2.
`training/eval.py` already appends a `RandomJack` baseline row by default (`--no-baseline` suppresses
it).

Metrics available from `eval_agent` (`training/eval.py:96-104`): `win_rate`, `mean_turns`,
`mean_turns_on_win`, `mean_turns_on_loss`, `mean_hideout_uncert` (computed on wins only).
`hideout_uncertainty` (`engine/metrics.py:4-18`) is the fraction of hideout-zone nodes still carrying
nonzero PMF mass at night end — 1.0 means cops learned nothing about which zone node was the hideout.

**Consistency:** `PolicyAgent` samples rather than acting greedily (`eval.py:37-56`). Pick one mode
and use it for every reported number, including these.

## Verification

- **E1:** `run_game` with no `initial_state` reproduces current behaviour exactly under a fixed seed;
  with one supplied, the game starts from precisely that state.
- **E2:** re-running one known human replay's scenario reproduces that game's starting position, cop
  placement and hideout exactly, before any policy move is taken.
- Any test fixtures must be **self-contained** — copy replays into a stable fixtures dir, never load
  from the mutable `data/replays/`.

## Session log

- 2026-09-09 — **E1, E2 and E3 built, tested and validated end-to-end.** `run_game` gained an
  optional `initial_state`; `analysis/compare.py` reconstructs each human's board from its stored
  replay, replays the policy on it N times, and computes move-level agreement against the human's
  own decisions. 29 tests, fixtures frozen into `tests/fixtures/participant_games.json` rather than
  read from the mutable snapshot.
  Ran clean on all 33 human games from the fresh 05 snapshot with **zero desyncs**, which is the
  reconstruction's strongest validation. Only E4 (baselines table) remains, and it is a formatting
  job on numbers `training/eval.py` already produces.
- 2026-09-09 — the smoke run surfaces a question worth carrying into the writeup: move agreement is
  ~20% across every checkpoint while win rates differ by 20 points. The agents are not playing
  better human strategies, they are playing different ones. Re-ask this per Director arm once 04
  lands — "does the Director change *how* it plays or only how often it wins" is the E3 question and
  it now has a number attached.

## Session log

- 2026-09-19 — **ran the comparison; 06 has its result.** `w10s-obj-cur`, three seeds, against the
  `games_20260919.sqlite` snapshot (20 participants / 60 games). Agent beats humans by +0.584 to
  +0.638 participant score and +52 to +57 win points, every interval clear of zero, zero desyncs.
  Move agreement 27–29%, so the agents win by playing a different game rather than a better version
  of the human one. The policy wins the boards humans lost and the boards humans won at the same
  rate, so board difficulty does not explain the human losses. Interim: recruitment continues.
