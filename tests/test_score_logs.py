"""Workstream 10's log fields: participant score and the per-term reward line.

W&B runs offline on the cluster, so the Slurm log is the record analysis reads.
These pin that train.py's new output is parsed, and that logs from before the
change (no score, no reward line) still parse exactly as they did.

Logs are synthesised inline in train.py's exact format — nothing from logs/.
"""

from __future__ import annotations

import csv

from analysis.export_results import export
from analysis.sweep_report import parse_log, report

HEADER = """host=stud-1 job=1807500 task=0
run:      uv run python -m training.train --total-steps 15000000 --n-envs 12 --n-workers 12 --lr 3e-4 --ent-coef 0.03 --seed {seed} --wandb-run {name}
---
"""

UPDATE_NEW = (
    "update={u}/4882 steps={steps} sps=698 instant_sps=702 episodes=100 "
    "return=1.012 win_rate=0.910 score=1.213 difficulty=-1.000 pg=-0.01 "
    "vf=0.02 ent=1.90 clip_frac=0.05 lr=3.00e-04\n"
)
REWARD_LINE = (
    "  reward: objective=1.2130 alpha=0.0412 beta=-0.0031 zeta=0.0107 "
    "delta=0.0009 coverage=0.974 visit_entropy=0.912\n"
)
EVAL_NEW = (
    "  eval: win_rate={win}% score={score} turns=8.4 turns(W)=9.6 turns(L)=6.4 "
    "hideout_u=0.75 copdist=1.50 arrest=5.0% timeout=4.0%\n"
)
UPDATE_OLD = (
    "update={u}/4882 steps={steps} sps=698 instant_sps=702 episodes=100 "
    "return=0.812 win_rate=0.910 difficulty=-1.000 pg=-0.01 vf=0.02 "
    "ent=1.90 clip_frac=0.05 lr=3.00e-04\n"
)
EVAL_OLD = (
    "  eval: win_rate={win}% turns=8.4 turns(W)=9.6 turns(L)=6.4 "
    "hideout_u=0.75 copdist=1.50 arrest=5.0% timeout=4.0%\n"
)


def _new_log(tmp_path, name, seed, evals):
    text = HEADER.format(name=name, seed=seed)
    for i, (win, score) in enumerate(evals, start=1):
        for j in range(10):
            u = (i - 1) * 10 + j + 1
            text += UPDATE_NEW.format(u=u, steps=f"{u * 3072:,}")
        text += REWARD_LINE
        text += EVAL_NEW.format(win=win, score=score)
    text += "Training complete.\n"
    p = tmp_path / f"wc-train_{name}.out"
    p.write_text(text)
    return p


def test_eval_score_is_parsed(tmp_path):
    r = parse_log(_new_log(tmp_path, "a-s41", 41, [(90.0, "1.100"), (92.0, "1.250")]))
    assert r.scores == [(30720, 1.1), (61440, 1.25)]
    assert r.final_score == 1.25
    assert r.eval_rows[-1]["score"] == 1.25
    assert r.best_win == 92.0  # win-rate parsing untouched


def test_last5_score_ignores_earlier_evals(tmp_path):
    evals = [(90.0, "0.500")] + [(90.0, f"1.{i}00") for i in range(5)]
    r = parse_log(_new_log(tmp_path, "b-s41", 41, evals))
    assert abs(r.last5_score - 1.2) < 1e-9


def test_old_logs_have_no_score_and_still_parse(tmp_path):
    text = HEADER.format(name="old-s27", seed=27)
    text += UPDATE_OLD.format(u=50, steps="153,600")
    text += EVAL_OLD.format(win=40.0)
    text += "Training complete.\n"
    p = tmp_path / "wc-train_old.out"
    p.write_text(text)
    r = parse_log(p)
    assert r.best_win == 40.0
    assert not r.has_score
    assert r.eval_rows[0]["score"] is None


def test_report_prints_the_score_aggregate(tmp_path, capsys):
    for seed in (41, 42, 43):
        _new_log(tmp_path, f"arm-s{seed}", seed, [(90.0, "1.200")])
    report([str(p) for p in tmp_path.glob("*.out")])
    out = capsys.readouterr().out
    assert "last-5 eval participant score" in out
    assert "1.200" in out


def test_export_carries_score_reward_terms_and_coverage(tmp_path):
    _new_log(tmp_path, "c-s41", 41, [(90.0, "1.100"), (92.0, "1.250")])
    out = tmp_path / "out"
    export([str(p) for p in tmp_path.glob("*.out")], "w10", out, every=10)

    with (out / "w10_eval.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert [r["score"] for r in rows] == ["1.1", "1.25"]

    with (out / "w10_training.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "training rows missing"
    last = rows[-1]
    assert float(last["train_score"]) == 1.213
    assert float(last["r_objective"]) == 1.213
    assert float(last["r_beta"]) == -0.0031
    assert float(last["coverage"]) == 0.974
    assert float(last["visit_entropy"]) == 0.912
