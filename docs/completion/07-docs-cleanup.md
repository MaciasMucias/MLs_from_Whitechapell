# 07 — Docs cleanup

**Status:** not started
**Blocks:** nothing
**Blocked by:** nothing. Do it before drafting the thesis.

---

## Goal

Stop `CLAUDE.md` from describing a version of the project that no longer exists.

## Why it matters

The thesis will be drafted against this description of the codebase. Right now large parts of it are
false in a direction that *understates* what was built — it claims the entire training pipeline and
the Director don't exist.

## Current state — what is actually wrong

| Location | Claim | Reality |
|---|---|---|
| `CLAUDE.md:71` | `training/` marked `[NOT STARTED]` | Full PPO pipeline implemented |
| `CLAUDE.md:110` | "Only `NoOpDirector` (pass-through stub) is implemented" | `CurriculumDirector` fully implemented, `agents/curriculum_director.py` |
| `CLAUDE.md:119` | "RL agent: Not implemented — only `RandomJack` stub" | `PolicyAgent` exists, `training/eval.py:37-56` |
| `CLAUDE.md:120` | Curriculum Director "planned but not started" | Implemented and wired to the P-controller |
| `CLAUDE.md:122-125` | "Training (`training/`) **Not yet implemented.** The `training/` module is empty" | 5 modules, ~1,400 lines |

Only the `## Current State` block (lines 37-63) is accurate. The stale text is the pre-implementation
design doc, never updated after the curriculum director landed.

`CLAUDE.md:59` is correct and important — it already records that a retrain against the updated cops
is required.

Also drifted, lower priority:

- `DEPLOYMENT_PLAN.md` is a historical spec, mostly executed. It says `skill_level` (the field became
  `gaming_habit`) and describes the debug frontend at `/debug/` (it became a separate ASGI app,
  `server/debug_main.py` on port 8001, not deployed).
- `server/database.py:17` docstring says `'never'|'sometimes'|'regularly'`; stored values are
  `never_played|played_few|played_many`.
- ~~`training/eval.py:8` references `agent_final.pt`, never written~~ — **fixed 2026-09-09** in
  02-C2, along with the same stale reference in the root `README.md`. Both now document the real
  `agent_<step>.pt` / `agent_best.pt` scheme.
- `CLAUDE.md` still says `training/` is `[NOT STARTED]` and the module is empty, in two places. It
  is the full PPO pipeline, and as of 2026-09-09 also has `checkpoints.py`. Same section claims the
  `training` extra has no dependencies.

## Steps

1. Rewrite the stale `CLAUDE.md` sections to match reality.
2. Mark `DEPLOYMENT_PLAN.md` as historical at the top rather than editing it — it is a useful record
   of what was planned vs built.
3. Fix the `gaming_habit` docstring in `server/database.py:17`.

## Verification

Read `CLAUDE.md` end to end and check each claim against the tree. No section should describe
something as unimplemented that exists.

## Session log

- _(empty)_
