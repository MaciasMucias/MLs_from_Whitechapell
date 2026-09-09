"""Checkpoint selection and pruning (training/checkpoints.py).

The case that motivated this module: agent_best.pt sorts after every
agent_<digits>.pt lexicographically, so the naive sorted(glob("*.pt"))[-1] that
the tools used returns "best" while claiming "latest".

Self-contained: everything is created in tmp_path, nothing is read from
data/replays/ or checkpoints/.
"""

from __future__ import annotations

import pytest

from training.checkpoints import (
    BEST_NAME,
    best_checkpoint,
    latest_checkpoint,
    periodic_checkpoints,
    prune_checkpoints,
    resolve_checkpoint,
)

STEPS = [153_600, 307_200, 1_536_000, 15_360_000]


@pytest.fixture
def run_dir(tmp_path):
    for step in STEPS:
        (tmp_path / f"agent_{step:010d}.pt").write_bytes(b"x")
    return tmp_path


def test_periodic_excludes_best(run_dir):
    (run_dir / BEST_NAME).write_bytes(b"x")
    names = [p.name for p in periodic_checkpoints(run_dir)]
    assert BEST_NAME not in names
    assert len(names) == len(STEPS)


def test_periodic_order_is_step_order(run_dir):
    names = [p.name for p in periodic_checkpoints(run_dir)]
    assert names == [f"agent_{s:010d}.pt" for s in sorted(STEPS)]


def test_latest_ignores_best(run_dir):
    """The regression this module exists for."""
    (run_dir / BEST_NAME).write_bytes(b"x")
    assert latest_checkpoint(run_dir).name == f"agent_{max(STEPS):010d}.pt"


def test_latest_on_empty_dir(tmp_path):
    assert latest_checkpoint(tmp_path) is None


def test_best_checkpoint_absent_and_present(run_dir):
    assert best_checkpoint(run_dir) is None
    (run_dir / BEST_NAME).write_bytes(b"x")
    assert best_checkpoint(run_dir).name == BEST_NAME


def test_resolve_dir_prefers_best(run_dir):
    (run_dir / BEST_NAME).write_bytes(b"x")
    assert resolve_checkpoint(run_dir).name == BEST_NAME
    assert resolve_checkpoint(run_dir, prefer_best=False).name.startswith("agent_0")


def test_resolve_dir_falls_back_to_latest(run_dir):
    assert resolve_checkpoint(run_dir).name == f"agent_{max(STEPS):010d}.pt"


def test_resolve_passes_through_a_file(run_dir):
    path = run_dir / f"agent_{STEPS[0]:010d}.pt"
    assert resolve_checkpoint(path) == path


def test_resolve_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path)


def test_prune_keeps_most_recent_n(run_dir):
    deleted = prune_checkpoints(run_dir, keep=2)
    assert len(deleted) == 2
    remaining = [p.name for p in periodic_checkpoints(run_dir)]
    assert remaining == [f"agent_{s:010d}.pt" for s in sorted(STEPS)[-2:]]


def test_prune_never_touches_best(run_dir):
    (run_dir / BEST_NAME).write_bytes(b"x")
    prune_checkpoints(run_dir, keep=1)
    assert (run_dir / BEST_NAME).is_file()


def test_prune_disabled_by_nonpositive_keep(run_dir):
    assert prune_checkpoints(run_dir, keep=0) == []
    assert len(periodic_checkpoints(run_dir)) == len(STEPS)


def test_prune_is_a_noop_when_under_the_limit(run_dir):
    assert prune_checkpoints(run_dir, keep=99) == []
    assert len(periodic_checkpoints(run_dir)) == len(STEPS)


def test_resume_target_survives_pruning(run_dir):
    """--resume takes an explicit path; the kept ones must still be there."""
    prune_checkpoints(run_dir, keep=3)
    assert latest_checkpoint(run_dir).is_file()
