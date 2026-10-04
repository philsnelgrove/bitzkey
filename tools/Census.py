"""Population census for the colour graph: which Bitzkeys were alive, sampled across the whole
run at an even spacing that widens as the run grows, so it never outgrows memory."""

from __future__ import annotations

import bisect
import colorsys
from collections.abc import Iterable

MAX_SNAPSHOTS = 1000

Snapshot = tuple[int, tuple[int, ...]]  # (turn, identifiers of the Bitzkeys alive then)


def colour_order(rgb: tuple[int, int, int]) -> tuple[int, float, float]:
    """A sort key that stacks similar colours together: greys by lightness, then hues round the
    colour wheel."""
    hue, saturation, value = colorsys.rgb_to_hsv(*(channel / 255 for channel in rgb))
    if saturation < 0.2 or value < 0.2:
        return (0, 0.0, value)
    return (1, hue, value)


class Census:
    """Snapshots every `stride` turns, plus the latest turn. When there are too many, the stride
    doubles and every other snapshot goes, keeping the spacing even."""

    def __init__(self) -> None:
        self.snapshots: list[Snapshot] = []
        self.stride = 1

    def record(self, turn: int, identifiers: Iterable[int]) -> None:
        entry = (turn, tuple(identifiers))
        if self.snapshots and self.snapshots[-1][0] % self.stride:
            self.snapshots[-1] = entry  # the last one only stood in for "latest"
        else:
            self.snapshots.append(entry)
        if len(self.snapshots) > MAX_SNAPSHOTS:
            self.stride *= 2
            latest = self.snapshots[-1]
            self.snapshots = [snapshot for snapshot in self.snapshots[:-1]
                              if snapshot[0] % self.stride == 0] + [latest]

    def forget_from(self, turn: int) -> None:
        """Drop snapshots of turns from `turn` on: a future abandoned by rewinding."""
        while self.snapshots and self.snapshots[-1][0] >= turn:
            self.snapshots.pop()

    def at(self, turn: int) -> tuple[int, ...]:
        """Who was alive at the latest snapshot on or before `turn`."""
        index = bisect.bisect_right(self.snapshots, turn, key=lambda snapshot: snapshot[0]) - 1
        return self.snapshots[index][1] if index >= 0 else ()
