import random
from dataclasses import dataclass, field


@dataclass
class CourseSequencer:
    """Hands out a complete, counterbalanced map order to each participant.

    Each call to :meth:`next_order` returns a *full* permutation of the maps, so
    a participant's whole course is reserved atomically. Concurrent participants
    can never interleave and leave someone with a duplicated or missing map
    (the failure mode of the old drain-and-reshuffle single-draw queue).

    Ordering uses a cyclic Latin square: as the internal counter advances, each
    map appears once in every course position across each group of
    ``len(map_names)`` participants, so maps are balanced across the 1st / 2nd /
    3rd slots.
    """

    map_names: list[str]
    _counter: int = field(default=0)

    def __post_init__(self) -> None:
        # Random starting offset so ordering isn't identical across restarts.
        if self.map_names:
            self._counter = random.randrange(len(self.map_names))

    def next_order(self) -> list[tuple[str, int]]:
        """Return the next participant's course as ``(map_name, position)`` pairs."""
        n = len(self.map_names)
        r = self._counter % n
        self._counter += 1
        return [(self.map_names[(i + r) % n], i) for i in range(n)]
