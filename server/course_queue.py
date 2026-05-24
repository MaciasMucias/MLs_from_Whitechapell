import random
from dataclasses import dataclass, field


@dataclass
class CourseQueue:
    map_names: list[str]
    _queue: list[int] = field(default_factory=list)
    _cycle_position: int = 0

    def next(self) -> tuple[str, int]:
        """Return (map_name, position_in_permutation_cycle).

        Position resets to 0 each time a new Fisher-Yates permutation is
        generated, i.e. every len(map_names) calls.
        """
        if not self._queue:
            self._queue = list(range(len(self.map_names)))
            random.shuffle(self._queue)
            self._cycle_position = 0
        idx = self._queue.pop(0)
        pos = self._cycle_position
        self._cycle_position += 1
        return self.map_names[idx], pos
