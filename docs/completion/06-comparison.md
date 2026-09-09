# 06 — Human-vs-RL comparison

**Status:** not started
**Blocks:** nothing — this is the final deliverable
**Blocked by:** 05 (data snapshot) for results; 04 (trained policies) for results.
**The code can be written and tested against existing checkpoints before either lands.**

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

- _(empty)_
