def test_jack_nodes_nonempty(gm):
    assert len(gm.jack_nodes) > 0


def test_cop_nodes_nonempty(gm):
    assert len(gm.cop_nodes) > 0


def test_jack_node_ids_unique(gm):
    ids = [n.id for n in gm.jack_nodes]
    assert len(ids) == len(set(ids))


def test_cop_node_ids_unique(gm):
    ids = [n.id for n in gm.cop_nodes]
    assert len(ids) == len(set(ids))


def test_jack_edge_destinations_valid(gm):
    jack_ids = {n.id for n in gm.jack_nodes}
    for node in gm.jack_nodes:
        for edge in node.edges:
            assert edge.destination.id in jack_ids


def test_cop_jack_neighbours_valid(gm):
    jack_ids = {n.id for n in gm.jack_nodes}
    for cop in gm.cop_nodes:
        for jn in cop.jack_neighbours:
            assert jn.id in jack_ids


def test_cop_edges_valid(gm):
    cop_ids = {n.id for n in gm.cop_nodes}
    for cop in gm.cop_nodes:
        for nb in cop.edges:
            assert nb.id in cop_ids


def test_jack_starts_valid(gm):
    jack_ids = {n.id for n in gm.jack_nodes}
    for s in gm.jack_starts:
        assert s in jack_ids


def test_cop_starts_valid(gm):
    cop_ids = {n.id for n in gm.cop_nodes}
    for s in gm.cop_starts:
        assert s in cop_ids
