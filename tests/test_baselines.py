"""The baselines and metrics DESIGN_REQUIREMENTS §7 asks for.

Three gaps found by auditing the code against §7 on 2026-09-19:

* §7.2 "Trained Jack vs. random cops" — `RandomCops` existed but nothing outside
  tests ever played against it; `eval_agent` hardcoded `HeuristicCops`.
* §7.1 "average rounds to capture (cop-win games only)" — evaluation reported
  `mean_turns_on_loss`, which mixes arrests with running out the clock.
* The project's "human players of varying skill levels" — `gaming_habit` was
  stored and never grouped by.

Fixtures are inline or the frozen participant games; nothing reads live data.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from agents.random_agents import RandomJack
from analysis.compare import HABIT_ORDER, HumanGame, human_breakdown_by_habit
from analysis.compare import scenario_from_replay
from training.eval import eval_agent, random_cops

FIXTURES = Path(__file__).parent / "fixtures" / "participant_games.json"


# --- §7.2: trained Jack vs random cops --------------------------------------


def test_random_cops_is_a_selectable_opponent(gm):
    """eval_agent must accept a cop agent, not only cop parameters."""
    rng = random.Random(0)
    m = eval_agent(RandomJack(rng), gm, n_games=20, rng=rng, cop_factory=random_cops)
    assert m["n_games"] == 20
    assert 0.0 <= m["win_rate"] <= 1.0
    # Rates still partition every game, whoever the cops are.
    assert m["win_rate"] + m["arrest_rate"] + m["timeout_rate"] == pytest.approx(1.0)


def test_random_cops_are_easier_than_the_study_cops(gm):
    """The point of the baseline: it separates 'good policy' from 'easy board'."""
    rng_a, rng_b = random.Random(1), random.Random(1)
    vs_random = eval_agent(
        RandomJack(random.Random(7)), gm, 40, rng_a, cop_factory=random_cops
    )
    vs_heuristic = eval_agent(RandomJack(random.Random(7)), gm, 40, rng_b)
    assert vs_random["win_rate"] > vs_heuristic["win_rate"]


def test_cop_factory_takes_precedence_over_cop_params(gm):
    """Passing both is a caller error; the agent wins and nothing raises."""
    from agents.heuristic_cops import COPS_PRERETUNE_V1

    rng = random.Random(3)
    m = eval_agent(
        RandomJack(rng),
        gm,
        10,
        rng,
        cop_params=dict(COPS_PRERETUNE_V1),
        cop_factory=random_cops,
    )
    assert m["n_games"] == 10


# --- §7.1: rounds to capture ------------------------------------------------


def test_arrest_turns_are_reported_separately_from_timeouts(gm):
    """mean_turns_on_loss mixes two different failures; §7.1 wants arrests only."""
    rng = random.Random(5)
    m = eval_agent(RandomJack(rng), gm, 60, rng)
    assert "mean_turns_on_arrest" in m
    # A random Jack is arrested often, so the metric must be populated.
    assert m["arrest_rate"] > 0
    assert m["mean_turns_on_arrest"] > 0
    # Arrests end the game no later than the clock does.
    if m["timeout_rate"] > 0:
        assert m["mean_turns_on_arrest"] <= m["mean_turns"] + 1e-9


# --- skill levels -----------------------------------------------------------


def _humans():
    games = json.loads(FIXTURES.read_text())
    out = []
    for i, (row_id, entry) in enumerate(sorted(games.items())):
        replay = entry["replay"]
        out.append(
            HumanGame(
                row_id=int(row_id),
                participant=i // 3,  # the fixtures are two sessions of three
                map_name=entry["map_name"],
                gaming_habit=entry["gaming_habit"],
                outcome=entry["outcome"],
                turns_survived=entry["turns_survived"],
                initial_state=scenario_from_replay(replay),
                jack_moves=[r["jack_to"] for r in replay["rounds"]],
            )
        )
    return out


def test_habit_breakdown_covers_every_game_exactly_once():
    humans = _humans()
    scores = {h.row_id: 0.5 for h in humans}
    by_habit = human_breakdown_by_habit(humans, scores)
    assert set(by_habit) <= set(HABIT_ORDER)
    assert sum(s["games"] for s in by_habit.values()) == len(humans)
    assert sum(s["participants"] for s in by_habit.values()) == len(
        {h.participant for h in humans}
    )


def test_habit_breakdown_reports_each_groups_own_win_rate():
    humans = _humans()
    by_habit = human_breakdown_by_habit(humans, {h.row_id: 1.0 for h in humans})
    for habit, stats in by_habit.items():
        members = [h for h in humans if h.gaming_habit == habit]
        expected = sum(h.human_won for h in members) / len(members)
        assert stats["win_rate"] == pytest.approx(expected)


def test_habit_breakdown_skips_games_without_a_replayable_score():
    """A game that could not be replayed is left out of the interval, not zeroed."""
    humans = _humans()
    by_habit = human_breakdown_by_habit(humans, {})  # nothing replayable
    for stats in by_habit.values():
        assert stats["games"] > 0  # counts still describe the games
        assert stats["score"].n_items == 0  # but no score went into the interval
