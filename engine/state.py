from dataclasses import dataclass


@dataclass(frozen=True)
class CopKnowledge:
    """
    The shared knowledge state that all cops act on.

    This is the Director's surface: `visited` may be suppressed or injected
    relative to ground truth. `search_misses` and `arrest_misses` are always
    accurate — the Director only manipulates positive sightings.

    Attributes:
        jack_start:      Jack's starting node (revealed at game start).
        visited:         Jack nodes cops believe Jack has visited this game.
                         Director can suppress real finds or inject false ones.
        search_misses:   (jack_node_id, turn) pairs: Jack's trace did not
                         include this node at or before this turn. Constrains
                         count[t][v] = 0 for t <= turn in the PMF.
        arrest_misses:   (jack_node_id, turn) pairs: Jack was confirmed absent
                         from this node at this exact turn. Constrains
                         count[turn][v] = 0 in the PMF.
    """
    jack_start: int
    visited: frozenset[int] = frozenset()
    search_misses: tuple[tuple[int, int], ...] = ()
    arrest_misses: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class GameState:
    """
    Complete snapshot of the game at a given point in time.

    Jack observes his own position, cop positions, hideout, turn, and can
    derive true search results from jack_trace (he knows his own walk).
    Cops observe cop_knowledge, which may differ from jack_trace due to
    Director interference.

    Attributes:
        jack_pos:       Jack's current node ID.
        cop_positions:  Cop node IDs, one per cop (index = cop index).
        hideout:        Target node Jack must reach to win.
        turn:           Current round number (0-indexed).
        jack_trace:     All jack node IDs Jack has visited this game (ground
                        truth). Used by Director and for deriving true search
                        results from Jack's perspective.
        cop_knowledge:  What cops are told — potentially Director-modified.
    """
    jack_pos: int
    cop_positions: tuple[int, ...]
    hideout: int
    hideout_zone_anchor: int
    hideout_zone: frozenset[int]
    turn: int
    jack_trace: frozenset[int]
    cop_knowledge: CopKnowledge
