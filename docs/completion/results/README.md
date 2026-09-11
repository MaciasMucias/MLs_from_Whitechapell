# Experiment results — chart-ready data

Every cluster wave, exported to tidy CSV so the thesis's figures can be built
without the cluster. Regenerate with:

```bash
uv run python -m analysis.export_results 'logs/wc-train_<arrayid>_*.out' \
    --prefix <name> --out-dir results
```

**These files are the durable record.** Until 2026-09-11 the results existed only
as Slurm logs and unsynced W&B directories on cluster scratch — one cleanup away
from losing the evidence behind the thesis's main claim. `slurm_logs.tar.gz`
(2.9 MB, 74 files) is the raw provenance the CSVs were derived from.

## Waves

| prefix | array | runs | what it establishes |
|---|---|---|---|
| `sweep_w1` | 1806901 | 18 | `lr=3e-4` wins decisively in both Director arms |
| `ent_probe` | 1806936 | 2 | `ent-coef 0.03` is an *interior* optimum, not a grid edge |
| `sweep_w2` | 1806976 | 12 | reward coefficients: every change inside the noise floor |
| `director_v1` | 1807070 | 17 | null — runs ended before the curriculum starts |
| `director_base` | 1807190 | 3 | the curriculum's first uptick is at 4.4–4.8M steps |
| **`director_branch`** | **1807209** | **15** | **the headline: the Director does not help** |

## Schema

**`*_eval.csv`** — one row per evaluation (every checkpoint save).

| column | meaning |
|---|---|
| `run` | W&B run name, e.g. `dbr-fix05-s28` |
| `arm` | run name with the seed suffix stripped — group by this |
| `seed` | 27 / 28 / 29 |
| `step` | global environment step |
| `win_rate` | **Director-free** eval win rate, % — comparable across all arms |
| `hideout_uncert` | fraction of hideout-zone nodes still carrying PMF mass (wins only) |
| `copdist` | mean BFS distance Jack kept from his nearest cop |
| `arrest_pct` / `timeout_pct` | share of *all* games lost to arrest / the clock |
| `difficulty` | `curriculum/difficulty` in force when the eval ran |

**`*_training.csv`** — one row per update, every 10th kept (plus the last).

| column | meaning |
|---|---|
| `train_win_rate` | win rate **during training**, under whatever difficulty was active |
| `difficulty` | the curriculum state |

> **Do not plot `win_rate` and `train_win_rate` on one axis.** The first is
> Director-free and comparable. The second is measured under the arm's own
> training difficulty, and for a Director-ON run the P-controller drives it
> toward the target band — it is the controller's *setpoint*, not a score.
> See invariant 2b in `../README.md`.

## Figures these support

1. **The headline.** Final `win_rate` by arm from `director_branch_eval.csv`, one
   point per seed. Reproduces: off 93.2% [92.5–94.0], fix05 88.3% [87.0–90.5],
   b4060 81.3% [76.0–86.5], b2035 61.3% [57.5–64.0], fix10 50.7% [46.0–56.0].
   The top two ranges do not overlap.
2. **The dose-response.** Final `win_rate` against *fixed* training difficulty
   (−1.0 → 50.7, −0.5 → 88.3, none → 93.2) — monotone: handicapping the cops
   during training is monotonically harmful.
3. **The curriculum engaging.** `difficulty` vs `step` from
   `director_branch_training.csv`. Both adaptive arms ramp from −1.0 into
   *injection* territory (+0.199 and +0.542), which is what makes this a fair
   test where `director_v1` was not. Overlay `director_v1_training.csv` for the
   contrast: it never leaves −1.0.
4. **The mechanism.** `copdist` and `arrest_pct` by arm. A policy trained against
   blind cops holds ~1.03 and is arrested ~53% of the time; one trained against
   full-strength cops holds ~1.50 and is arrested ~7%. It never learns cops are
   dangerous.
5. **Learning curves.** `win_rate` vs `step`, mean over seeds per arm — the arms
   share a base and visibly diverge after the branch point (4.45M / 4.92M).

## Caveat to carry into any figure of the branch wave

All five arms branch from a base trained at difficulty −1.0 for 4.4–4.9M steps,
so `dbr-off` is "suppressed prefix, then no Director" — not a pure control. The
*between-arm* comparison is clean (identical prefix, identical seeds); the
absolute number needs a from-scratch `--no-curriculum` run, which is 04's OFF arm.
