# 01 — Freeze the cop configuration

**Status:** not started
**Blocks:** 03, 04, 06 — every run and every comparison depends on cops being pinned
**Blocked by:** nothing. Start here.

---

## Goal

Give the cop parameters that participants actually played against a name, a date, and a test that
guards them. Behaviour must not change at all — this is labelling, not tuning.

## Why it matters

Human participants played against `HeuristicCops()` with the defaults set on 2026-06-11 (`259ae5d`).
If those values ever change, the RL agent gets evaluated against cops **no human ever faced**, and
the headline human-vs-RL comparison silently becomes meaningless. Nothing in the codebase currently
signals that these numbers are study-critical — they look like ordinary defaults, and a future
"let's retune the cops" session would overwrite them without noticing.

## Current state

The 11 tuned parameters are bare constructor defaults at `agents/heuristic_cops.py:69-81`:

```python
arrest_threshold: float = 0.209
min_arrest_fraction: float = 0.357
pursuit_fraction: float = 0.271
pursuit_weight: float = 1.049
searcher_prox_fraction: float = 0.339
direction_certainty_threshold: float = 0.573
arrest_discount: float = 0.287
miss_discount_decay: float = 0.420
hideout_blend: float = 0.434
hideout_blend_floor: float = 0.076
max_passes: int = 7
cop_max_steps: int = 2
```

Every call site instantiates `HeuristicCops()` bare, with no parameterisation:
`training/env.py:65`, `training/eval.py:75`, `server/session.py:42`,
`server/admin_routes.py:350,365`, `server/replay_routes.py:106`,
`tools/run_and_save_replays.py:36`, `tests/test_game.py:16`.

Only `tools/scripted_sim.py:63,161` accepts overrides (`HeuristicCops(**(cop_params or {}))`) — that
is how Optuna and `pareto_compare` inject candidates.

There is **no config-file loading anywhere**. The tuned sets in `configs.json` are not wired into
anything; that file is only consumed by `tools/pareto_compare.py --configs`.

### Provenance (for the writeup)

Two Optuna rounds landed in these defaults:

1. `2d90bcb` (2026-05-25) — first round, changed 4 params.
2. `259ae5d` (2026-06-11) — replaced all 11, and simultaneously added the `zeta` cop-distance reward
   term to `training/env.py`.

The current values do **not** match any row in `optuna_result1.txt` (that is the *random-Jack*
Pareto front) nor either entry in `configs.json`. They came from a later policy-Jack study whose
output was not saved to a file. Worth noting as a provenance gap in the thesis.

## Steps

1. Add `COPS_STUDY_V2` to `agents/heuristic_cops.py` — a dict or frozen dataclass holding the 12
   values above, with a docstring recording: the date frozen (2026-06-11), the commit (`259ae5d`),
   and that this is the configuration deployed for the human study and must not be edited.
2. Default `HeuristicCops.__init__` from it. Keep the signature and behaviour byte-identical.
3. Add `tests/test_cop_config.py` pinning all 12 values literally. Self-contained — no fixture
   loading from `data/replays/`.
4. Note in the docstring that a future retune lands as a *new* named preset (`COPS_V3`), never by
   editing `COPS_STUDY_V2`.

## Verification

```bash
uv run python -m pytest tests/ -q          # 26 existing tests + the new pin
uv run python tools/scripted_sim.py        # same winner, turns, search hits as before
```

`scripted_sim.py` is the behavioural check: it replays a fixed Jack path against the cops, so any
accidental parameter change shows up as a different winner or hit count.

## Session log

- _(empty)_
