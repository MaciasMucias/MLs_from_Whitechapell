# 01 — Freeze the cop configuration

**Status:** **done** (2026-09-09)
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
uv run python -m pytest tests/ -q          # 26 existing tests + the 5 new pins
```

**The originally planned second check does not exist.** `tools/scripted_sim.py` has no `__main__`
block — it is a library module imported by `optuna_tune.py`, so `uv run python tools/scripted_sim.py`
runs and prints nothing, exit 0. It would have passed silently whatever the cop parameters were.

Behavioural equivalence was instead checked with a throwaway fingerprint: 60 seeded
`RandomJack`-vs-`HeuristicCops` games through `engine.game.run_game`, hashing winner, turns
survived, final cop positions and search hits per game. Identical before and after
(`fcc115dbfb843f64c99f38206fce5252ff118132b5cff0a39400ddd4726320e6`), confirming the change is pure
labelling. Rebuild it in five minutes if a future change needs the same assurance; it was not kept
because `tests/test_cop_config.py` now guards the parameters directly and a digest over
`run_game` output would also trip on unrelated engine changes.

## Session log

- 2026-09-09 — **done.** `COPS_STUDY_V2` added to `agents/heuristic_cops.py` as an immutable
  `MappingProxyType`, with the freeze date, commit, the "a retune becomes COPS_V3" rule and the
  provenance gap in its docstring. `HeuristicCops.__init__` defaults from it entry by entry, so the
  link is visible in the signature. `tests/test_cop_config.py` pins all 12 values literally and also
  asserts the preset's key set equals the constructor's parameter names — a future tunable added as
  a loose default fails the suite rather than escaping the freeze. 31/31 tests pass; behavioural
  digest unchanged.
- 2026-09-09 — found the planned `scripted_sim.py` verification is a no-op (no `__main__` block).
  Replaced with the seeded 60-game fingerprint described above. Worth knowing before trusting that
  command anywhere else in these docs.
