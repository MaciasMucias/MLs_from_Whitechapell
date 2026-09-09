"""Checkpoint naming and selection — one place, so consumers agree.

A run directory (``checkpoints/<run-name>/``) holds two kinds of file:

``agent_0000153600.pt``
    Periodic checkpoints, named by global step, zero-padded to 10 digits so
    lexicographic order is step order. Pruned to the most recent N.

``agent_best.pt``
    A copy of whichever periodic checkpoint scored the highest
    ``eval/win_rate``. Never pruned. Only written when training runs with
    ``--eval-games > 0``; without eval there is nothing to rank by.

The distinction matters because ``agent_best.pt`` sorts *after* every
``agent_<digits>.pt`` lexicographically, so a plain ``sorted(glob("*.pt"))[-1]``
silently returns "best" while claiming "latest". Use these helpers instead.
"""

from __future__ import annotations

from pathlib import Path

BEST_NAME = "agent_best.pt"
# Periodic checkpoints only — the leading digit class excludes agent_best.pt.
PERIODIC_GLOB = "agent_[0-9]*.pt"


def periodic_checkpoints(run_dir: Path | str) -> list[Path]:
    """Periodic checkpoints in ascending step order (zero-padding makes this exact)."""
    return sorted(Path(run_dir).glob(PERIODIC_GLOB))


def latest_checkpoint(run_dir: Path | str) -> Path | None:
    """The highest-step periodic checkpoint, or None if there are none."""
    ckpts = periodic_checkpoints(run_dir)
    return ckpts[-1] if ckpts else None


def best_checkpoint(run_dir: Path | str) -> Path | None:
    """``agent_best.pt`` if the run wrote one, else None."""
    path = Path(run_dir) / BEST_NAME
    return path if path.is_file() else None


def resolve_checkpoint(path_str: str | Path, prefer_best: bool = True) -> Path:
    """Resolve a file path, or a run directory, to a single checkpoint file.

    For a directory, prefers ``agent_best.pt`` (highest eval win rate) and falls
    back to the highest-step periodic checkpoint. Pass ``prefer_best=False`` to
    always take the latest instead.
    """
    path = Path(path_str)
    if not path.is_dir():
        return path

    if prefer_best:
        best = best_checkpoint(path)
        if best is not None:
            return best

    latest = latest_checkpoint(path)
    if latest is None:
        raise FileNotFoundError(f"No agent_*.pt checkpoints in {path}")
    return latest


def prune_checkpoints(run_dir: Path | str, keep: int) -> list[Path]:
    """Delete all but the ``keep`` most recent periodic checkpoints.

    ``agent_best.pt`` is not a periodic checkpoint and is never touched.
    ``keep <= 0`` disables pruning. Returns the files actually deleted.
    """
    if keep <= 0:
        return []

    stale = periodic_checkpoints(run_dir)[:-keep]
    deleted = []
    for path in stale:
        try:
            path.unlink()
            deleted.append(path)
        except OSError:
            # A checkpoint we cannot delete is a tidiness problem, never a
            # reason to kill a run that may be hours in.
            pass
    return deleted
