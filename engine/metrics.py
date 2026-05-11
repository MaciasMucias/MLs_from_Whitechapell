from __future__ import annotations


def hideout_uncertainty(hideout_zone: frozenset[int], position_pmf: dict[int, float]) -> float:
    """
    Fraction of hideout zone nodes that still carry nonzero position PMF mass
    at night end. Returns a value in [0, 1].

    1.0 = cops have no information about which zone node is the hideout.
    0.0 = cops have fully eliminated every zone node.
    Only meaningful when Jack won.
    """
    if not hideout_zone:
        return 0.0
    nonzero = sum(1 for node_id in hideout_zone if position_pmf.get(node_id, 0.0) > 0.0)
    return nonzero / len(hideout_zone)
