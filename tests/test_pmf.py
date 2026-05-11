from agents.heuristic_cops import HeuristicCops
from engine.env import legal_jack_edges, step_jack


def test_pmf_sums_to_one(gm, initial_state):
    pmf = HeuristicCops.compute_pmf(initial_state, gm)
    assert abs(sum(pmf.values()) - 1.0) < 1e-9


def test_pmf_keys_valid(gm, initial_state):
    jack_ids = {n.id for n in gm.jack_nodes}
    pmf = HeuristicCops.compute_pmf(initial_state, gm)
    assert set(pmf.keys()).issubset(jack_ids)


def test_pmf_nonempty(gm, initial_state):
    pmf = HeuristicCops.compute_pmf(initial_state, gm)
    assert len(pmf) > 0


def test_pmf_after_one_move(gm, initial_state):
    edges = legal_jack_edges(initial_state, gm)
    state1, _, _ = step_jack(initial_state, edges[0])
    pmf = HeuristicCops.compute_pmf(state1, gm)
    jack_ids = {n.id for n in gm.jack_nodes}
    assert abs(sum(pmf.values()) - 1.0) < 1e-9
    assert set(pmf.keys()).issubset(jack_ids)


def test_pmf_all_nonnegative(gm, initial_state):
    pmf = HeuristicCops.compute_pmf(initial_state, gm)
    assert all(v >= 0.0 for v in pmf.values())
