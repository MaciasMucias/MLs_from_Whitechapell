# 05 — Close out study data

**Status:** not started
**Blocks:** 06
**Blocked by:** nothing. Laptop work — do it while the GPU is busy.

---

## Goal

Get the participant data off Fly into an immutable local snapshot, reconstruct which games belong to
which participant, and record which rows are excluded and why.

## Invariant

**Do not modify production.** The study is winding down; there is no participant-ID fix to deploy.
Everything here is local analysis.

## Current state

### Where the data lives

- `data/games.sqlite` on the Fly volume (`fly.toml` mounts `replay_data` -> `/app/data`) is the only
  durable copy.
- `data/replays/` is a **5-slot rotating buffer** (`server/replay.py:21-23`) that overwrites itself.
  It is a debugging convenience, **not an archive**. Do not treat it as data.
- The `replay` column holds the *entire* `ReplayRecord` as JSON, so the DB is fully self-sufficient
  for analysis (~2.1 MB for 64 rows locally).

### Schema — `server/database.py:25-42`

`id, game_id, map_name, scenario_order, gaming_habit, outcome, turns_survived, turn_limit,
move_sequence (json), replay (json), created_at`

**There is no participant/session ID.** This is the main gap and it cannot be fixed retroactively.

### Other things not persisted

The participant's score is computed **client-side only** (`game.js:108-110, 809-815`) and never
stored. If score is wanted for analysis, recompute it from the replay. `score_info`
(`routes.py:113-126`) is returned to the client but not saved.

### How rows get written

Only at termination (`routes.py:80-111`). An abandoned or interrupted game persists nothing, and a
server restart or the 4-hour session sweep (`session.py:59-64`) drops in-progress games silently.

## Steps

### A1 — Sanity-check the live deploy

User believes the course-duplicate fix is live. Confirm cheaply: `POST /api/course/new` against the
live URL should return **three distinct maps** with `scenario_order` 0/1/2.

### A2 — Pull and stabilise the snapshot

```bash
fly ssh console -C "cat /app/data/games.sqlite" > data/study/games_<YYYYMMDD>.sqlite
```

Treat the dated file as immutable. All analysis reads from it, never from a live pull.

### A3 — Reconstruct participant sessions

Write `analysis/sessions.py`. Group rows using:

- monotonic `scenario_order` runs 0 -> 1 -> 2
- `created_at` proximity
- consistent `gaming_habit`

**Flag ambiguous groupings rather than guessing.** Concurrent participants are exactly the case that
cannot be resolved from this schema, and a wrong grouping is worse than a dropped one.

### A4 — Filter contaminated rows, recording the reason for each

Three known classes:

1. `map_name='unknown'` / `scenario_order=-1` — admin-panel and replay-fork artifacts. Sessions
   created via `new-from-state` or a replay fork call `register_session` without
   `set_participant_meta`, so `save_game` still fires with the fallback values
   (`routes.py:96-98`). 15 such rows locally.
2. `map_name='whitechapel'` — pre-course-era games. Production now serves only `course_1/2/3`
   (`maps/course_participant.json`), so this is a clean marker for data collected before the course
   design existed. 32 rows locally.
3. Sessions collected before the duplicate fix (`594fba0`), identifiable by a **repeated `map_name`
   within one reconstructed session**.

Known contamination in the local snapshot, as an example of what class 3 looks like: rows 35->37 and
38->40 are both `whitechapel(0), course_1(1), whitechapel(2)` — same map twice, `course_2`/`course_3`
never served.

### A5 — Commit the untracked deployment artifacts

All of these are currently untracked: `Dockerfile`, `fly.toml`, `.dockerignore`, `.github/`,
`DEPLOYMENT_PLAN.md`, and — most importantly — **`uv.lock`**.

- The Dockerfile runs `uv sync --frozen`, which **cannot work from a clean checkout** without
  `uv.lock`. This is a live landmine.
- `.github/workflows/fly-deploy.yml` triggers on push to **`main`**, but this repo only has `master`
  and `dev`. CI has never fired and never could. Either point it at `dev` or delete it. (The live
  site is the product of a manual `fly deploy` from the working directory.)
- `master` is stale — its last commit is from 2024. `dev` is the real branch.

## The duplicate-course bug, for the writeup

The old `CourseQueue` handed out **one map per call** from a process-global shuffled queue that
reshuffled on drain, and `/game/new` called it on every request. Three failure modes: concurrent
participants interleaved draws from the shared queue; a participant straddling a refill got a fresh
permutation mid-course; and abandonment consumed draws and desynced the cycle position.
`scenario_order` was the *global* cycle position, not the participant's own index.

Fixed in `594fba0` by `CourseSequencer.next_order()` (`server/course_queue.py`), which reserves a
**complete cyclic-Latin-square permutation atomically** per participant. The client calls
`POST /api/course/new` once at course start and stores the whole order in `sessionStorage`
(`index.html:641-664`); it never calls back between games (`game.js:876-936`).

Residual limitations worth a footnote: the client is authoritative for `map_name` (the server never
checks it against what was reserved, `routes.py:47-52`), there is no participant identity, and
`_counter` is per-process (a second Fly machine would have an independent counter — costs
counterbalance quality, not correctness).

## Verification

Session reconstruction totals must reconcile: every retained row belongs to exactly one session, and
every exclusion has a recorded reason. Report the final usable N — it is small, and the thesis needs
an honest number.

## Session log

- _(empty)_
