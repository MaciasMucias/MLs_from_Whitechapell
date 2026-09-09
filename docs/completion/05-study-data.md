# 05 — Close out study data

**Status:** **mostly done** (2026-09-09) — snapshot pulled, sessions reconstructed, N established.
A2/A3/A4 done; A1 and A5 outstanding.
**Blocks:** 06
**Blocked by:** nothing. Laptop work — do it while the GPU is busy.

---

## READ THIS FIRST — the two snapshots are disjoint

Pulling the live snapshot on 2026-09-09 turned up something the plan did not anticipate.

| | rows | collected | content |
|---|---|---|---|
| `data/games.sqlite` (was already local) | 64 | 2026-05-24 → **2026-07-20** | mostly pre-course, 15 admin artifacts, duplicate-bug sessions |
| `data/study/games_20260909.sqlite` (pulled from Fly) | 51 | **2026-08-05** → 2026-09-08 | all course maps, zero admin artifacts, all post-fix |

**They share no rows.** Production starts two weeks after the local file ends, so the Fly volume was
evidently reset or recreated between 07-20 and 08-05. Consequences:

1. **The local file is the only surviving copy of the May–July data.** It is not on production and
   cannot be re-pulled.
2. **The 2026-09-09 pull is the only local copy of the August–September data**, which is the data the
   thesis actually wants — it is entirely post-fix and uncontaminated.
3. The duplicate fix `594fba0` is dated **2026-07-20 23:24**, after the last local row (21:22 the
   same day). So the local file is *entirely* pre-fix and the production file *entirely* post-fix.
   That is a clean split, and it is why the usable N differs so sharply between them.
4. **Both files are one laptop failure from gone.** `data/study/` has been added to `.gitignore`
   rather than committed: the records are anonymous (the schema collects no participant identity at
   all) but retention and backup of participant data is an ethics-approval question, not a call to
   make in passing. **Decide where these live.**

### The usable N

`uv run python -m analysis.sessions --db data/study/games_20260909.sqlite --verbose`

| | local (pre-fix) | **production (post-fix)** |
|---|---|---|
| rows | 64 | 51 |
| admin artifacts | 15 | **0** |
| reconstructed sessions | 36 | 23 |
| **usable participants** | **2** | **11** |
| **usable games** | **6** | **33** |

**11 participants, 33 games** — by `gaming_habit`: `never_played` 7, `played_few` 3, `played_many` 1.
Human win rate 9/33 = **27.3%**.

Two things to report honestly in the thesis:

- **Course completion rate is 48%** (11 of 23 sessions finished all three maps). Nothing is persisted
  for an abandoned game, so the 12 incomplete sessions are dropouts, not data loss.
- **The skill distribution is heavily novice-weighted** (7/11 never played). Any "RL vs experienced
  humans" claim rests on a single participant. Report per-group numbers and resist pooling.

Both snapshots should be kept: the pre-fix file still supports the duplicate-bug writeup below.

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

**Done.** `tests/test_sessions.py` asserts the partition reconciles (no row silently disappears), and
`report()` prints the reconciliation for each snapshot. The snapshot is opened
`file:...?mode=ro` so an analysis bug cannot mutate it — also asserted.

## What is left

- **A1 — sanity-check the live deploy.** Not done. Arguably moot: the production data is *evidence*
  the fix works, since all 23 reconstructed sessions serve three distinct course maps. A direct
  `POST /api/course/new` check would still confirm it cheaply.
- **A5 — commit the untracked deployment artifacts.** Not done. `uv.lock` in particular is a live
  landmine: the Dockerfile runs `uv sync --frozen`, which cannot work from a clean checkout without
  it. (Checked 2026-09-09: `uv.lock` **is** tracked; the rest of A5's list needs re-checking against
  the current tree, since it was written before commit `8528821` landed deployment artifacts.)
- **The study is still live** — the newest row is 2026-09-08, the day before this pull. If
  collection continues, re-pull before the final analysis and re-run `analysis.sessions`.

## Session log

- 2026-09-09 — **A2/A3/A4 done.** Pulled the live snapshot read-only
  (`fly ssh sftp get /app/data/games.sqlite`) to `data/study/games_20260909.sqlite`; integrity check
  passes. Discovered production and the existing local file are **disjoint** — see the top of this
  file. Wrote `analysis/sessions.py` (reconstruction + exclusion accounting, every reason recorded,
  ambiguous groupings flagged rather than guessed) and `tests/test_sessions.py` (15 tests, fixtures
  built in `tmp_path`, nothing read from mutable data).
  **Usable N went from 2 participants / 6 games to 11 / 33** purely by pulling the current data.
- 2026-09-09 — note for whoever writes A5: use `fly ssh sftp get`, not
  `fly ssh console -C "cat ..."`. Under Git Bash the latter mangles `/app/...` into a Windows path
  (MSYS path translation) and, being a text pipe, risks corrupting a binary file anyway. Run it from
  PowerShell or prefix `MSYS_NO_PATHCONV=1`.
