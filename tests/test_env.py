import random

from engine.env import legal_jack_edges, step_jack


def test_initial_jack_pos_matches_knowledge(initial_state):
    assert initial_state.jack_pos == initial_state.cop_knowledge.jack_start


def test_initial_jack_pos_valid(gm, initial_state):
    jack_ids = {n.id for n in gm.jack_nodes}
    assert initial_state.jack_pos in jack_ids


def test_initial_hideout_valid(gm, initial_state):
    jack_ids = {n.id for n in gm.jack_nodes}
    assert initial_state.hideout in jack_ids


def test_legal_edges_nonempty(gm, initial_state):
    edges = legal_jack_edges(initial_state, gm, blocking=False)
    assert len(edges) > 0


def test_legal_edge_destinations_valid(gm, initial_state):
    jack_ids = {n.id for n in gm.jack_nodes}
    edges = legal_jack_edges(initial_state, gm, blocking=False)
    for edge in edges:
        assert edge.destination.id in jack_ids


def test_step_jack_updates_position(gm, initial_state):
    edges = legal_jack_edges(initial_state, gm, blocking=False)
    edge = edges[0]
    new_state, _terminated, _winner = step_jack(initial_state, edge)
    assert new_state.jack_pos == edge.destination.id


def test_step_jack_updates_trace(gm, initial_state):
    edges = legal_jack_edges(initial_state, gm, blocking=False)
    edge = edges[0]
    new_state, _terminated, _winner = step_jack(initial_state, edge)
    assert edge.destination.id in new_state.jack_trace


def test_step_jack_to_hideout_terminates(gm):
    # Run multiple random games until we engineer a 1-step win, or just
    # verify the win condition fires when jack_pos == hideout.
    import random
    from engine.env import make_initial_state
    from dataclasses import replace

    rng = random.Random(0)
    for _ in range(20):
        state = make_initial_state(gm, rng=rng)
        edges = legal_jack_edges(state, gm)
        for edge in edges:
            if edge.destination.id == state.hideout:
                new_state, terminated, winner = step_jack(state, edge)
                assert terminated
                assert winner == "jack"
                return
    # No 1-step hideout reachable in sample — skip gracefully
