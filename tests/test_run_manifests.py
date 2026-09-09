"""Pre-flight check on the cluster run manifests in docs/completion/slurm/.

`array.sbatch` runs one manifest line per array task. A typo is not caught until
the task starts, and each wave is hours of contended cluster time, so every line
is parsed here instead — on the laptop, for free.

The bug this was written for: the manifests originally passed `--gamma 0` to
disable the terminal hideout-uncertainty bonus, but `--gamma` is PPO's discount
factor. argparse accepts it happily and the run silently trains with gamma=0.
The reward coefficients are `--reward-*`; the assertion below pins that apart.

Manifests are version-controlled text, not mutable data — safe to read directly.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from training.train import parse_args

MANIFEST_DIR = Path(__file__).resolve().parents[1] / "docs/completion/slurm/manifests"
PREFIX = "uv run python -m training.train "

# Placeholders a manifest carries until an earlier stage fills them in.
PLACEHOLDERS = {"<LR>": "3e-4", "<ENT>": "0.01"}

# The submit-side cap: every array task counts against MaxSubmitJobs, pending
# included, so the `%` throttle does not reduce the submitted count.
MAX_TASKS_PER_WAVE = 20

# Fixed by 02-C3 and 03; batch size must not shift between the sweep and 04.
EXPECTED_N_ENVS = 12
EXPECTED_N_WORKERS = 12


def _manifests() -> list[Path]:
    return sorted(MANIFEST_DIR.glob("*.txt"))


def _runs(path: Path) -> list[str]:
    """The lines array.sbatch would execute: blanks and comments stripped."""
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _argv(run: str) -> list[str]:
    for placeholder, value in PLACEHOLDERS.items():
        run = run.replace(placeholder, value)
    return shlex.split(run[len(PREFIX) :])


ALL = [
    pytest.param(path, i, run, id=f"{path.stem}[{i}]")
    for path in _manifests()
    for i, run in enumerate(_runs(path))
]


def test_manifests_exist():
    assert _manifests(), f"no manifests found in {MANIFEST_DIR}"


@pytest.mark.parametrize("path", _manifests(), ids=lambda p: p.stem)
def test_wave_fits_the_submit_cap(path):
    n = len(_runs(path))
    assert 0 < n <= MAX_TASKS_PER_WAVE, (
        f"{path.name} has {n} runs; split it into waves chained with "
        f"--dependency=afterany"
    )


@pytest.mark.parametrize("path", _manifests(), ids=lambda p: p.stem)
def test_run_names_are_unique(path):
    """--wandb-run doubles as the checkpoint subdirectory; a clash overwrites."""
    names = [parse_args(_argv(r)).wandb_run for r in _runs(path)]
    assert len(set(names)) == len(names), f"duplicate --wandb-run in {path.name}"


@pytest.mark.parametrize("path,i,run", ALL)
def test_line_invokes_train(path, i, run):
    assert run.startswith(PREFIX), f"{path.name}:{i} does not invoke training.train"


@pytest.mark.parametrize("path,i,run", ALL)
def test_line_parses(path, i, run):
    """argparse rejects unknown flags by exiting; SystemExit means a typo."""
    parse_args(_argv(run))


@pytest.mark.parametrize("path,i,run", ALL)
def test_line_names_a_run(path, i, run):
    assert parse_args(_argv(run)).wandb_run, f"{path.name}:{i} has no --wandb-run"


@pytest.mark.parametrize("path,i,run", ALL)
def test_batch_shape_is_fixed(path, i, run):
    """Batch size must be identical across the sweep and the final runs."""
    args = parse_args(_argv(run))
    assert args.n_envs == EXPECTED_N_ENVS
    assert args.n_workers == EXPECTED_N_WORKERS
    assert args.n_envs % args.n_workers == 0  # AsyncVectorJackEnv asserts this


@pytest.mark.parametrize("path,i,run", ALL)
def test_ppo_discount_is_never_swept(path, i, run):
    """--gamma is PPO's discount. Disabling reward shaping is --reward-gamma."""
    assert "--gamma" not in _argv(run), (
        f"{path.name}:{i} passes --gamma (PPO discount). To change the terminal "
        f"hideout-uncertainty bonus use --reward-gamma."
    )
    assert parse_args(_argv(run)).gamma == 0.99
