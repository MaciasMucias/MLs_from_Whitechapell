"""Sweep log parsing and config selection (analysis/sweep_report.py).

Logs are synthesised here in the exact format train.py and array.sbatch emit, so
the test needs no cluster, no real run, and nothing from logs/.
"""

from __future__ import annotations

import pytest

from analysis.sweep_report import (
    NOISE_PP,
    fill_manifest,
    parse_log,
    pick,
    report,
)

HEADER = """host=stud-1 job=1806902 task=0
cpus=8 cwd=/home/u/MLs_from_Whitechapel
gpu: none (CPU-only)
manifest: docs/completion/slurm/manifests/sweep_w1.txt
run:      uv run python -m training.train --total-steps 3000000 --n-envs 12 --n-workers 12 --lr {lr} --ent-coef {ent} {extra}--wandb-run {name}
---
device: cpu
obs_dim=1416  n_actions=195  n_envs=12  n_workers=12
"""

UPDATE = (
    "update={u}/976 steps={steps} sps=640 instant_sps=651 episodes=100 "
    "return=-0.512 win_rate=0.480 difficulty=-1.000 pg=-0.01 vf=0.02 "
    "ent=1.90 clip_frac=0.05 lr=1.00e-04\n"
)

EVAL = (
    "  eval: win_rate={win}% turns=8.4 turns(W)=9.6 turns(L)=6.4 "
    "hideout_u={unc} copdist={cd} arrest={ar}% timeout=4.0%\n"
)


def _log(tmp_path, name, lr, ent, wins, *, off=False, finished=True, unc="0.75"):
    text = HEADER.format(
        lr=lr, ent=ent, name=name, extra="--no-curriculum " if off else ""
    )
    for i, w in enumerate(wins, start=1):
        text += UPDATE.format(u=i * 50, steps=f"{i * 153_600:,}")
        text += EVAL.format(win=w, unc=unc, cd="1.28", ar="34.0")
    if finished:
        text += "Training complete.\n"
    p = tmp_path / f"wc-train_{name}.out"
    p.write_text(text)
    return p


# --- parsing ----------------------------------------------------------------


def test_parses_flags_from_the_run_line(tmp_path):
    r = parse_log(_log(tmp_path, "sw1-a", "3e-4", "0.01", [40.0]))
    assert (r.name, r.lr, r.ent) == ("sw1-a", "3e-4", "0.01")
    assert r.curriculum is True and r.arm == "ON"


def test_detects_the_off_arm(tmp_path):
    r = parse_log(_log(tmp_path, "sw1-b", "1e-4", "0.03", [40.0], off=True))
    assert r.curriculum is False and r.arm == "OFF"


def test_collects_every_eval_and_its_step(tmp_path):
    r = parse_log(_log(tmp_path, "sw1-c", "3e-4", "0.01", [10.0, 30.0, 20.0]))
    assert [w for _, w in r.evals] == [10.0, 30.0, 20.0]
    assert r.best_win == 30.0
    assert r.final_win == 20.0  # final is NOT the best
    assert r.steps == 3 * 153_600


def test_parses_the_new_diagnostic_columns(tmp_path):
    r = parse_log(_log(tmp_path, "sw1-d", "3e-4", "0.01", [40.0]))
    assert r.last_uncert == 0.75
    assert r.last_copdist == 1.28
    assert r.last_arrest == 34.0


def test_unfinished_run_is_flagged(tmp_path):
    r = parse_log(_log(tmp_path, "sw1-e", "3e-4", "0.01", [40.0], finished=False))
    assert r.finished is False


def test_empty_log_does_not_crash(tmp_path):
    p = tmp_path / "empty.out"
    p.write_text("")
    r = parse_log(p)
    assert r.evals == [] and r.steps == 0


# --- selection --------------------------------------------------------------


def test_picks_the_highest_eval_win_rate(tmp_path):
    runs = [
        parse_log(_log(tmp_path, "lo", "1e-4", "0.01", [20.0])),
        parse_log(_log(tmp_path, "hi", "3e-4", "0.03", [45.0])),
        parse_log(_log(tmp_path, "mid", "1e-3", "0.01", [30.0])),
    ]
    winner = pick(runs, "ON")
    assert winner.name == "hi"
    assert (winner.lr, winner.ent) == ("3e-4", "0.03")


def test_pick_ranks_on_best_not_final(tmp_path):
    """A run that peaked then regressed still ranks on its peak."""
    runs = [
        parse_log(_log(tmp_path, "peaked", "3e-4", "0.01", [50.0, 10.0])),
        parse_log(_log(tmp_path, "steady", "1e-4", "0.01", [30.0, 30.0])),
    ]
    assert pick(runs, "ON").name == "peaked"


def test_pick_ignores_the_other_arm(tmp_path):
    runs = [
        parse_log(_log(tmp_path, "on-low", "1e-4", "0.01", [20.0])),
        parse_log(_log(tmp_path, "off-high", "3e-4", "0.01", [90.0], off=True)),
    ]
    assert pick(runs, "ON").name == "on-low"
    assert pick(runs, "OFF").name == "off-high"


def test_pick_returns_none_when_the_arm_is_empty(tmp_path):
    runs = [parse_log(_log(tmp_path, "on", "1e-4", "0.01", [20.0]))]
    assert pick(runs, "OFF") is None


def test_narrow_spread_is_called_out(tmp_path, capsys):
    """Picking an argmax out of noise is the failure mode worth shouting about."""
    runs = [
        parse_log(_log(tmp_path, "a", "1e-4", "0.01", [40.0])),
        parse_log(_log(tmp_path, "b", "3e-4", "0.01", [41.0])),
    ]
    pick(runs, "ON", noise_pp=NOISE_PP)
    assert "CAUTION" in capsys.readouterr().out


def test_wide_spread_is_not_called_out(tmp_path, capsys):
    runs = [
        parse_log(_log(tmp_path, "a", "1e-4", "0.01", [20.0])),
        parse_log(_log(tmp_path, "b", "3e-4", "0.01", [50.0])),
    ]
    pick(runs, "ON", noise_pp=NOISE_PP)
    assert "CAUTION" not in capsys.readouterr().out


# --- manifest filling -------------------------------------------------------


def test_fill_manifest_substitutes_placeholders(tmp_path):
    m = tmp_path / "sweep_w2.txt"
    m.write_text(
        "# comment with <LR> left alone? no - substituted everywhere\n"
        "uv run python -m training.train --lr <LR> --ent-coef <ENT> --wandb-run a\n"
        "uv run python -m training.train --lr <LR> --ent-coef <ENT> --wandb-run b\n"
    )
    changed = fill_manifest(m, "3e-4", "0.01")
    out = m.read_text()
    assert "<LR>" not in out and "<ENT>" not in out
    assert out.count("--lr 3e-4") == 2
    assert out.count("--ent-coef 0.01") == 2
    assert changed == 3


def test_fill_manifest_is_idempotent(tmp_path):
    m = tmp_path / "m.txt"
    m.write_text("--lr <LR> --ent-coef <ENT>\n")
    fill_manifest(m, "3e-4", "0.01")
    first = m.read_text()
    assert fill_manifest(m, "3e-4", "0.01") == 0
    assert m.read_text() == first


def test_report_handles_no_runs(capsys):
    assert report([]) == []
    assert "No parseable runs" in capsys.readouterr().out


# --- curriculum difficulty tracking -----------------------------------------
# For a Director wave this is the load-bearing signal: difficulty pinned at its
# floor means the run trained at fixed suppression rather than curricularising.


def _log_diff(tmp_path, name, diffs, win=30.0):
    """A log whose update lines carry the given difficulty trajectory."""
    text = HEADER.format(lr="3e-4", ent="0.03", name=name, extra="")
    for i, d in enumerate(diffs, start=1):
        text += (
            f"update={i * 50}/976 steps={i * 153_600:,} sps=640 instant_sps=651 "
            f"episodes=100 return=-0.5 win_rate=0.480 difficulty={d} pg=-0.01 "
            f"vf=0.02 ent=1.90 clip_frac=0.05 lr=3.00e-04\n"
        )
        text += EVAL.format(win=win, unc="0.70", cd="1.10", ar="50.0")
    text += "Training complete.\n"
    p = tmp_path / f"wc-train_{name}.out"
    p.write_text(text)
    return p


def test_tracks_max_and_final_difficulty(tmp_path):
    r = parse_log(
        _log_diff(tmp_path, "ramped", ["-1.000", "-0.400", "+0.252", "+0.100"])
    )
    assert r.max_difficulty == 0.252
    assert r.final_difficulty == 0.100  # final is not the max
    assert r.curriculum_engaged is True


def test_pinned_difficulty_is_not_engaged(tmp_path):
    r = parse_log(_log_diff(tmp_path, "pinned", ["-1.000", "-1.000", "-1.000"]))
    assert r.max_difficulty == -1.0
    assert r.curriculum_engaged is False


def test_barely_moving_does_not_count_as_engaged(tmp_path):
    """Dithering just off the -1.0 clamp is not a curriculum."""
    r = parse_log(_log_diff(tmp_path, "nudged", ["-1.000", "-0.974"]))
    assert r.curriculum_engaged is False


def test_one_late_nudge_does_not_count_as_engaged(tmp_path):
    """The workstream-09 shape: difficulty moves on the final update only.

    09 reported 'curriculum engaged in 15/16 ON runs' on exactly this pattern,
    while 975/976 of training ran at a constant difficulty. Guarding it.
    """
    diffs = ["-1.000"] * 40 + ["-0.500"]
    r = parse_log(_log_diff(tmp_path, "late", diffs))
    assert r.max_difficulty == -0.5  # moved a long way...
    assert r.off_floor_fraction < 0.05  # ...but for almost no training
    assert r.curriculum_engaged is False


def test_sustained_movement_counts_as_engaged(tmp_path):
    """Moved by a clear margin and stayed there — a real curriculum."""
    diffs = ["-1.000", "-0.800", "-0.400", "0.000", "0.250", "0.250"]
    r = parse_log(_log_diff(tmp_path, "ramped", diffs))
    assert r.off_floor_fraction > 0.5
    assert r.curriculum_engaged is True


def test_off_floor_is_measured_against_the_run_start_not_minus_one(tmp_path):
    """A run pinned at a fixed -0.5 never leaves its own floor."""
    r = parse_log(_log_diff(tmp_path, "fixed05", ["-0.500"] * 10))
    assert r.start_difficulty == -0.5
    assert r.off_floor_fraction == 0.0
    assert r.curriculum_engaged is False


def test_report_shouts_when_no_run_curricularised(tmp_path, capsys):
    runs = [
        _log_diff(tmp_path, "a", ["-1.000", "-1.000"]),
        _log_diff(tmp_path, "b", ["-1.000", "-1.000"]),
    ]
    report([str(p) for p in runs])
    out = capsys.readouterr().out
    assert "pinned at its floor in ALL 2 ON runs" in out


def test_report_reports_partial_engagement(tmp_path, capsys):
    runs = [
        _log_diff(tmp_path, "a", ["-1.000", "-1.000"]),
        _log_diff(tmp_path, "b", ["-1.000", "-0.200"]),
    ]
    report([str(p) for p in runs])
    assert (
        "curriculum engaged (difficulty left its floor) in 1/2"
        in capsys.readouterr().out
    )


def test_difficulty_absent_does_not_crash(tmp_path):
    """Older logs, or OFF runs, may not carry a difficulty field."""
    r = parse_log(_log(tmp_path, "nodiff", "3e-4", "0.01", [40.0]))
    assert r.max_difficulty is not None or r.max_difficulty is None
    assert r.best_win == 40.0


# --- seed grouping ----------------------------------------------------------
# A seeded wave has three runs per arm; without aggregation the reader is left
# eyeballing triplets across 15 rows, which is the error this tool exists to
# prevent.


def _seeded(tmp_path, arm, seed, win):
    text = HEADER.format(lr="3e-4", ent="0.03", name=f"{arm}-s{seed}", extra="")
    text = text.replace("--wandb-run", f"--seed {seed} --wandb-run")
    text += UPDATE.format(u=50, steps="153,600")
    text += EVAL.format(win=win, unc="0.70", cd="1.10", ar="50.0")
    text += "Training complete.\n"
    p = tmp_path / f"wc-train_{arm}-s{seed}.out"
    p.write_text(text)
    return str(p)


def test_seed_is_parsed(tmp_path):
    r = parse_log(_seeded(tmp_path, "fixed-05", 28, 33.0))
    assert r.seed == "28"


def test_seed_defaults_when_absent(tmp_path):
    """train.py's default is 27; an unseeded manifest line still groups."""
    assert parse_log(_log(tmp_path, "x", "3e-4", "0.01", [40.0])).seed == "27"


def test_replicates_are_aggregated(tmp_path, capsys):
    paths = [_seeded(tmp_path, "arm-a", s, w) for s, w in ((27, 30.0), (28, 36.0), (29, 33.0))]
    report(paths)
    out = capsys.readouterr().out
    assert "Across seeds" in out
    assert "arm-a" in out
    assert "33.0%" in out          # mean of 30/36/33
    assert "±3.0 [30.0-36.0]" in out


def test_arms_are_ranked_by_mean(tmp_path, capsys):
    paths = [_seeded(tmp_path, "low", s, 20.0) for s in (27, 28)]
    paths += [_seeded(tmp_path, "high", s, 45.0) for s in (27, 28)]
    report(paths)
    body = capsys.readouterr().out.split("Across seeds")[1]
    assert body.index("high") < body.index("low")


def test_unreplicated_wave_prints_no_aggregate(tmp_path, capsys):
    report([_log(tmp_path, "solo", "3e-4", "0.01", [40.0])])
    assert "Across seeds" not in capsys.readouterr().out
