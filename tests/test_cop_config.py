"""Pins the frozen study cop configuration.

Human participants played against `HeuristicCops()` with these exact values
(frozen 2026-06-11, commit 259ae5d). If they change, the RL agent is evaluated
against cops no human ever faced and the headline human-vs-RL comparison
silently stops meaning anything.

This test exists to make that failure loud. A retune lands as a *new* named
preset (COPS_V3) passed at the call site — never by editing COPS_STUDY_V2.

Self-contained by design: values are literals here, loaded from no fixture.
"""

from __future__ import annotations

import inspect

import pytest

from agents.heuristic_cops import COPS_PRERETUNE_V1, COPS_STUDY_V2, HeuristicCops


# The study configuration, written out independently of the source module.
EXPECTED = {
    "arrest_threshold": 0.209,
    "min_arrest_fraction": 0.357,
    "pursuit_fraction": 0.271,
    "pursuit_weight": 1.049,
    "searcher_prox_fraction": 0.339,
    "direction_certainty_threshold": 0.573,
    "arrest_discount": 0.287,
    "miss_discount_decay": 0.420,
    "hideout_blend": 0.434,
    "hideout_blend_floor": 0.076,
    "max_passes": 7,
    "cop_max_steps": 2,
}


def test_preset_values_are_pinned():
    assert dict(COPS_STUDY_V2) == EXPECTED


def test_preset_covers_every_constructor_parameter():
    """A new tunable must be added to the preset, not left as a loose default."""
    params = set(inspect.signature(HeuristicCops.__init__).parameters) - {"self"}
    assert params == set(EXPECTED)


def test_bare_constructor_uses_the_study_configuration():
    """Guards the wiring, not just the dict: every call site instantiates bare."""
    cops = HeuristicCops()
    for name, value in EXPECTED.items():
        assert getattr(cops, f"_{name}") == value, name


def test_preset_is_immutable():
    with pytest.raises(TypeError):
        COPS_STUDY_V2["arrest_threshold"] = 0.5  # type: ignore[index]


def test_overrides_still_work():
    """Retuning happens by parameterising a call site, so that path must stay open."""
    cops = HeuristicCops(arrest_threshold=0.5, max_passes=3)
    assert cops._arrest_threshold == 0.5
    assert cops._max_passes == 3
    # Unspecified parameters still fall back to the frozen study values.
    assert cops._pursuit_weight == EXPECTED["pursuit_weight"]


# --- The held-out preset ----------------------------------------------------
# COPS_PRERETUNE_V1 is the pre-2026-06-11 configuration, kept so generalisation
# can be measured: a policy scored only against the cops it trained on cannot be
# told apart from one that memorised their decision rule.

PRERETUNE_EXPECTED = {
    "arrest_threshold": 0.25,
    "min_arrest_fraction": 0.8,
    "pursuit_fraction": 0.238,
    "pursuit_weight": 0.207,
    "searcher_prox_fraction": 0.579,
    "direction_certainty_threshold": 0.499,
    "arrest_discount": 0.0,
    "miss_discount_decay": 0.7,
    "hideout_blend": 0.5,
    "hideout_blend_floor": 0.330,
    "max_passes": 5,
    "cop_max_steps": 2,
}


def test_preretune_preset_is_pinned():
    assert dict(COPS_PRERETUNE_V1) == PRERETUNE_EXPECTED


def test_preretune_preset_is_immutable():
    with pytest.raises(TypeError):
        COPS_PRERETUNE_V1["arrest_threshold"] = 0.5  # type: ignore[index]


def test_the_two_presets_have_the_same_shape():
    """Either can be splatted into HeuristicCops(**preset)."""
    assert set(COPS_PRERETUNE_V1) == set(COPS_STUDY_V2)


def test_the_two_presets_are_actually_different():
    """A held-out set identical to the training set would measure nothing."""
    differing = [k for k in COPS_STUDY_V2 if COPS_STUDY_V2[k] != COPS_PRERETUNE_V1[k]]
    assert len(differing) >= 10, differing


def test_preretune_is_not_the_default():
    """The study cops are the default; the held-out set must be opt-in."""
    cops = HeuristicCops()
    assert cops._arrest_threshold == EXPECTED["arrest_threshold"]
    assert cops._arrest_threshold != PRERETUNE_EXPECTED["arrest_threshold"]


def test_preretune_preset_constructs():
    cops = HeuristicCops(**COPS_PRERETUNE_V1)
    for name, value in PRERETUNE_EXPECTED.items():
        assert getattr(cops, f"_{name}") == value, name
