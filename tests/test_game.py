import random

from agents.heuristic_cops import HeuristicCops
from agents.random_agents import RandomCops, RandomJack
from engine.game import run_game


def test_random_vs_random_terminates(gm):
    rng = random.Random(1)
    result = run_game(gm, RandomJack(rng), RandomCops(rng), rng=rng)
    assert result.winner in {"jack", "cops"}


def test_random_vs_heuristic_terminates(gm):
    rng = random.Random(2)
    result = run_game(gm, RandomJack(rng), HeuristicCops(), rng=rng)
    assert result.winner in {"jack", "cops"}


def test_history_nonempty(gm):
    rng = random.Random(3)
    result = run_game(gm, RandomJack(rng), RandomCops(rng), rng=rng)
    assert len(result.history) > 0


def test_round_records_consistent(gm):
    rng = random.Random(4)
    result = run_game(gm, RandomJack(rng), RandomCops(rng), rng=rng)
    for record in result.history:
        assert record.state_after_jack.jack_pos == record.jack_edge.destination.id
