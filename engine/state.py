from dataclasses import dataclass


@dataclass(frozen=True)
class CopKnowledge:
    """
    The shared knowledge state that all cops act on.

    This is the Director's surface: `visited` may be suppressed or injected
    relative to ground truth. `never_visited` and `arrest_misses` are always
    accurate — the Director only manipulates positive sightings.

    Attributes:
        jack_start:      Jack's starting node (revealed at game start).
        visited:         Jack nodes cops believe Jack has visited this game.
                         Director can suppress real finds or inject false ones.
        never_visited:   Jack nodes confirmed NOT in Jack's trace (from search
                         results returning false). Always ground truth.
        search_misses:   (jack_node_id, turn) pairs where Jack was confirmed
                         not present at, or before, that specific moment.
                         Point-in-time fact.
        arrest_misses:   (jack_node_id, turn) pairs where Jack was confirmed
                         absent at that specific moment. Point-in-time fact.
    """
    jack_start: int
    visited: frozenset[int] = frozenset()
    never_visited: frozenset[int] = frozenset()
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
    turn: int
    jack_trace: frozenset[int]
    cop_knowledge: CopKnowledge
