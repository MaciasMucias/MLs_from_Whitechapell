"""Scenario reconstruction and paired replay (analysis/compare.py, engine E1).

Fixtures are a frozen copy of the six usable participant games, in
`tests/fixtures/participant_games.json`. They are deliberately NOT read from
`data/games.sqlite` or `data/replays/` — the replay directory is a 5-slot
rotating buffer that overwrites itself, and the database is a live snapshot that
will change when a fresher pull lands.

The load-bearing claim is E2's verification criterion: replaying a stored
scenario reproduces that game's starting position, cop placement and hideout
exactly, before any policy move is taken.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from agents.heuristic_cops import HeuristicCops
from agents.random_agents import RandomCops, RandomJack
from analysis.compare import scenario_from_replay
from engine.env import make_initial_state
from engine.game import run_game
from engine.graph import load_map

FIXTURES = Path(__file__).parent / "fixtures" / "participant_games.json"


@pytest.fixture(scope="module")
def participant_games():
    return json.loads(FIXTURES.read_text())


@pytest.fixture(scope="module")
def course_maps():
    return {
        name: load_map(Path("maps") / f"{name}.json")
        for name in ("course_1", "course_2", "course_3")
    }


def _ids(games):
    return sorted(games)


# --- E1: injecting an initial state ----------------------------------------


def test_run_game_without_initial_state_is_unchanged(gm):
    """Omitting initial_state must reproduce the previous behaviour exactly."""
    a = run_game(
        gm,
        RandomJack(random.Random(5)),
        RandomCops(random.Random(5)),
        rng=random.Random(99),
    )
    b = run_game(
        gm,
        RandomJack(random.Random(5)),
        RandomCops(random.Random(5)),
        rng=random.Random(99),
    )
    assert a.initial_state == b.initial_state
    assert a.winner == b.winner
    assert a.turns_survived == b.turns_survived


def test_run_game_starts_from_the_supplied_state(gm):
    """With a state supplied, the game must begin from precisely it."""
    pinned = make_initial_state(gm, rng=random.Random(1234))
    # A different rng would sample a different scenario if it were used at all.
    record = run_game(
        gm,
        RandomJack(random.Random(2)),
        HeuristicCops(),
        rng=random.Random(777),
        initial_state=pinned,
    )
    assert record.initial_state == pinned
    assert record.history[0].state_before == pinned


def test_supplied_state_survives_regardless_of_rng(gm):
    """The scenario must not depend on the rng once it is pinned."""
    pinned = make_initial_state(gm, rng=random.Random(31337))
    starts = {
        run_game(
            gm,
            RandomJack(random.Random(3)),
            HeuristicCops(),
            rng=random.Random(seed),
            initial_state=pinned,
        ).initial_state
        for seed in (0, 1, 2)
    }
    assert starts == {pinned}


# --- E2: scenario reconstruction from a stored replay -----------------------


def test_fixtures_cover_the_usable_sessions(participant_games):
    assert _ids(participant_games) == ["46", "47", "48", "60", "61", "62"]


@pytest.mark.parametrize("row_id", ["46", "47", "48", "60", "61", "62"])
def test_scenario_matches_the_stored_replay(participant_games, row_id):
    """THE verification criterion: the board is reproduced exactly."""
    replay = participant_games[row_id]["replay"]
    state = scenario_from_replay(replay)

    assert state.jack_pos == replay["initial_jack_pos"]
    assert list(state.cop_positions) == replay["initial_cop_positions"]
    assert state.hideout == replay["hideout"]
    assert state.hideout_zone_anchor == replay["hideout_zone_anchor"]
    assert state.hideout_zone == frozenset(replay["hideout_zone"])
    assert state.hideout in state.hideout_zone


@pytest.mark.parametrize("row_id", ["46", "47", "48", "60", "61", "62"])
def test_scenario_is_a_clean_opening_position(participant_games, row_id):
    """Turn 0, nothing searched, Jack's start public — as make_initial_state."""
    state = scenario_from_replay(participant_games[row_id]["replay"])
    start = state.jack_pos
    assert state.turn == 0
    assert state.jack_path == (start,)
    assert state.jack_trace == frozenset({start})
    assert state.cop_searched_hits == frozenset()
    assert state.cop_searched_misses == frozenset()
    assert state.cop_knowledge.jack_start == start
    assert state.cop_knowledge.visited_at == ((start, 0),)


@pytest.mark.parametrize("row_id", ["46", "47", "48", "60", "61", "62"])
def test_the_humans_first_move_is_legal_from_the_reconstruction(
    participant_games, course_maps, row_id
):
    """If the board were wrong, the human's own move would not be available."""
    from engine.env import legal_jack_edges

    entry = participant_games[row_id]
    state = scenario_from_replay(entry["replay"])
    game_map = course_maps[entry["map_name"]]
    legal = {e.destination.id for e in legal_jack_edges(state, game_map)}
    assert entry["replay"]["rounds"][0]["jack_to"] in legal


@pytest.mark.parametrize("row_id", ["46", "47", "48", "60", "61", "62"])
def test_replay_legal_move_set_matches_the_engine(
    participant_games, course_maps, row_id
):
    """The stored legal-move set and the engine's must agree at turn 0.

    A mismatch would mean the replay was produced by different rules than the
    ones the policy is about to be evaluated under.
    """
    from engine.env import legal_jack_edges

    entry = participant_games[row_id]
    state = scenario_from_replay(entry["replay"])
    game_map = course_maps[entry["map_name"]]
    engine_legal = {e.destination.id for e in legal_jack_edges(state, game_map)}
    assert engine_legal == set(entry["replay"]["rounds"][0]["jack_legal_moves"])


def test_a_policy_can_be_run_on_a_reconstructed_scenario(
    participant_games, course_maps
):
    """End-to-end: the whole point of E1 + E2."""
    entry = participant_games["47"]
    state = scenario_from_replay(entry["replay"])
    game_map = course_maps[entry["map_name"]]
    record = run_game(
        game_map,
        RandomJack(random.Random(0)),
        HeuristicCops(),
        director=None,
        rng=random.Random(0),
        initial_state=state,
    )
    assert record.initial_state == state
    assert record.winner in {"jack", "cops"}


# --- workstream 10: humans scored on the participant score --------------------


def _human(entry, row_id):
    from analysis.compare import HumanGame

    replay = entry["replay"]
    return HumanGame(
        row_id=int(row_id),
        participant=0,
        map_name=entry["map_name"],
        gaming_habit=entry["gaming_habit"],
        outcome=entry["outcome"],
        turns_survived=entry["turns_survived"],
        initial_state=scenario_from_replay(replay),
        jack_moves=[r["jack_to"] for r in replay["rounds"]],
    )


# The cops a game was played against. Rows 46-48 predate the 2026-06-11 retune
# (played 2026-05-30), rows 60-62 follow it (2026-07-04).
PRE_RETUNE = ["46", "47", "48"]
POST_RETUNE = ["60", "61", "62"]


def _cops_for(row_id):
    from agents.heuristic_cops import COPS_PRERETUNE_V1

    return COPS_PRERETUNE_V1 if row_id in PRE_RETUNE else None


@pytest.mark.parametrize("row_id", PRE_RETUNE + POST_RETUNE)
def test_human_game_replays_to_its_stored_outcome(
    participant_games, course_maps, row_id
):
    """Scores are recomputed, so the replay must reproduce the real game."""
    from analysis.compare import replay_human

    entry = participant_games[row_id]
    record = replay_human(
        course_maps[entry["map_name"]], _human(entry, row_id), _cops_for(row_id)
    )
    assert record is not None
    assert record.winner == entry["outcome"]
    assert record.turns_survived == entry["turns_survived"]


def test_a_pre_retune_game_does_not_replay_under_todays_cops(
    participant_games, course_maps
):
    """Row 46 was an arrest in 4 rounds against the old cops. Against
    COPS_STUDY_V2 the same moves are a different game, and must be rejected
    rather than scored."""
    from analysis.compare import replay_human

    entry = participant_games["46"]
    assert replay_human(course_maps[entry["map_name"]], _human(entry, "46")) is None


@pytest.mark.parametrize("row_id", PRE_RETUNE + POST_RETUNE)
def test_human_score_is_in_the_participant_score_range(
    participant_games, course_maps, row_id
):
    from analysis.compare import human_score

    entry = participant_games[row_id]
    score = human_score(
        course_maps[entry["map_name"]], _human(entry, row_id), _cops_for(row_id)
    )
    if entry["outcome"] == "jack":
        assert 1.0 <= score <= 1.5
    else:
        assert 0.0 <= score < 1.0


def test_a_move_that_never_happened_is_not_scored(participant_games, course_maps):
    from analysis.compare import human_score

    entry = participant_games["61"]
    human = _human(entry, "61")
    human.jack_moves = [-1] + human.jack_moves[1:]  # illegal first move
    assert human_score(course_maps[entry["map_name"]], human) is None
