from dataclasses import dataclass


@dataclass(frozen=True)
class CopKnowledge:
    """
    Director's surface: `visited_at` may be suppressed or injected relative to
    ground truth. `search_misses` and `arrest_misses` are always accurate.

    Attributes:
        jack_start:    Jack's starting node (revealed at game start).
        visited_at:    (jack_node_id, depth) pairs — nodes cops believe Jack
                       visited, with depth = state.turn + 1 at observation time.
                       Director-manipulable.
        search_misses: Unique (jack_node_id, turn) pairs: Jack not seen here at
                       or before this turn.
        arrest_misses: Unique (jack_node_id, turn) pairs: Jack absent at exactly
                       this turn.
    """
    jack_start: int
    visited_at: tuple[tuple[int, int], ...] = ()
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
        jack_trace:     Frozenset of all Jack node IDs visited this game.
                        Ground truth for search hit/miss (O(1) lookup).
                        Admin endpoints may decouple it from jack_path.
        jack_path:      Ordered tuple of Jack's positions, one entry per move
                        (jack_path[0] = start). Preserves chronological order
                        for turn-depth lookups (.index()) and serialisation.
                        In normal gameplay frozenset(jack_path) == jack_trace.
        cop_knowledge:  What cops are told — potentially Director-modified.
    """
    jack_pos: int
    cop_positions: tuple[int, ...]
    hideout: int
    hideout_zone_anchor: int
    hideout_zone: frozenset[int]
    turn: int
    jack_trace: frozenset[int]
    jack_path: tuple[int, ...]  # ordered positions, one per move (jack_path[0] = start)
    cop_knowledge: CopKnowledge
