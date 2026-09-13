"""The training CLI contract (training/train.py parse_args).

These are the flags every cluster manifest is written against, so a silent
change in their names, defaults or exclusivity would invalidate a wave without
failing anything. Cheap to pin, expensive to discover late.

The behavioural half of --branch-from (step continuity, fresh W&B run, CLI
curriculum flags winning over the checkpoint's, best-eval reset) needs a real
training run and is verified in docs/completion/09-director-tuning.md's session
log rather than here.
"""

from __future__ import annotations

import pytest

from training.train import parse_args


def test_defaults_match_the_documented_config():
    a = parse_args([])
    assert a.n_envs == 12
    assert a.n_steps == 256  # batch = n_steps * n_envs = 3,072
    assert a.gamma == 0.99  # PPO discount, NOT a reward coefficient
    assert a.seed == 27
    assert a.wandb_mode == "offline"


def test_n_workers_default_is_not_the_cluster_shape():
    """The default is 6; every manifest passes --n-workers 12 explicitly.

    02-C3 chose 12 workers on 8 CPUs (oversubscription measured faster), but the
    default was left at 6. That is fine — tests/test_run_manifests.py enforces
    12/12 on the manifests, which is where the batch shape has to hold. Pinned
    here so the discrepancy is deliberate rather than a lurking surprise.
    """
    assert parse_args([]).n_workers == 6
    assert parse_args(["--n-workers", "12"]).n_workers == 12


def test_reward_coefficients_are_prefixed():
    """--gamma is PPO's discount; the reward's gamma must not shadow it."""
    a = parse_args(["--reward-gamma", "0"])
    assert a.reward_gamma == 0.0
    assert a.gamma == 0.99  # untouched


def test_reward_coefficient_defaults():
    a = parse_args([])
    assert (a.reward_alpha, a.reward_beta, a.reward_delta) == (0.1, 0.05, 0.01)
    assert (a.reward_gamma, a.reward_zeta) == (0.5, 0.1)


# --- curriculum -------------------------------------------------------------


def test_curriculum_defaults():
    a = parse_args([])
    assert a.initial_difficulty == -1.0  # full suppression
    assert a.curriculum_kp == 0.1
    assert (a.curriculum_target_low, a.curriculum_target_high) == (0.4, 0.6)
    assert a.curriculum_ratchet is False
    assert a.no_curriculum is False


def test_kp_zero_is_accepted():
    """kp=0 is how a run pins difficulty — the base-run design depends on it."""
    a = parse_args(["--curriculum-kp", "0", "--initial-difficulty", "-0.5"])
    assert a.curriculum_kp == 0.0
    assert a.initial_difficulty == -0.5


def test_curriculum_max_step_default_never_binds():
    """The difficulty range is 2.0 wide, so the default must exceed it."""
    assert parse_args([]).curriculum_max_step >= 2.0


# --- resume vs branch -------------------------------------------------------


def test_resume_and_branch_from_default_to_none():
    a = parse_args([])
    assert a.resume is None and a.branch_from is None


def test_resume_and_branch_from_are_mutually_exclusive():
    """They inherit opposite things; allowing both would be ambiguous."""
    with pytest.raises(SystemExit):
        parse_args(["--resume", "a.pt", "--branch-from", "b.pt"])


@pytest.mark.parametrize(
    "flag,attr", [("--resume", "resume"), ("--branch-from", "branch_from")]
)
def test_each_checkpoint_source_parses_alone(flag, attr):
    a = parse_args([flag, "ck.pt"])
    assert getattr(a, attr) == "ck.pt"


def test_branch_from_leaves_curriculum_flags_on_the_command_line():
    """A branch must be able to override the checkpoint's difficulty."""
    a = parse_args(["--branch-from", "ck.pt", "--initial-difficulty", "0.0"])
    assert a.branch_from == "ck.pt"
    assert a.initial_difficulty == 0.0


def test_unknown_flag_is_rejected():
    """Guards manifest typos — argparse exits rather than ignoring."""
    with pytest.raises(SystemExit):
        parse_args(["--ent-cof", "0.01"])


# --- difficulty bounds ------------------------------------------------------
# The ceiling is the load-bearing one. Positive difficulty INJECTS knowledge the
# cops never earned; measured on array 1807209, adaptive runs that stopped near
# +0.19 scored 76-86% while those that ran to +0.47-0.54 scored 57-64%.


def test_difficulty_bounds_default_to_the_full_range():
    """Defaults must reproduce the old hardcoded +/-1.0 clamp exactly."""
    a = parse_args([])
    assert a.curriculum_max_difficulty == 1.0
    assert a.curriculum_min_difficulty == -1.0


def test_difficulty_ceiling_is_settable():
    a = parse_args(["--curriculum-max-difficulty", "0.15"])
    assert a.curriculum_max_difficulty == 0.15
    assert a.curriculum_min_difficulty == -1.0  # floor untouched


def test_difficulty_floor_is_settable():
    a = parse_args(["--curriculum-min-difficulty", "-0.5"])
    assert a.curriculum_min_difficulty == -0.5


def test_a_ceiling_at_zero_forbids_injection_entirely():
    """difficulty <= 0 is suppression only — cops never get unearned knowledge."""
    assert (
        parse_args(["--curriculum-max-difficulty", "0"]).curriculum_max_difficulty
        == 0.0
    )


def test_bounds_can_pin_difficulty_to_a_single_value():
    """min == max is another way to hold difficulty fixed, alongside kp=0."""
    a = parse_args(
        ["--curriculum-min-difficulty", "-0.5", "--curriculum-max-difficulty", "-0.5"]
    )
    assert a.curriculum_min_difficulty == a.curriculum_max_difficulty == -0.5
