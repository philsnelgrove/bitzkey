"""Colour census: how many living Bitzkeys of each colour family there are, turn by turn."""

from __future__ import annotations

import colorsys
from collections.abc import Iterable, Sequence


HUES = ("red", "orange", "yellow", "lime", "green", "spring", "cyan", "azure", "blue", "violet",
        "magenta", "rose")
FAMILIES = HUES + ("white", "grey", "black")
PARTS = ("ring", "dot")

# The colour each family is drawn in: its hue at the middle of its band, or a neutral tone.
SWATCHES = {
    **{hue: tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb(index / len(HUES), 0.75, 0.9))
       for index, hue in enumerate(HUES)},
    "white": (232, 232, 222),
    "grey": (138, 140, 136),
    "black": (14, 14, 14),
}


def family(rgb: tuple[int, int, int]) -> str:
    """Name the colour family of an RGB colour: a hue band, or white, grey or black."""
    hue, saturation, value = colorsys.rgb_to_hsv(*(channel / 255 for channel in rgb))
    if value < 0.2:
        return "black"
    if saturation < 0.2:
        return "white" if value > 0.8 else "grey"
    return HUES[int(hue * len(HUES) + 0.5) % len(HUES)]


def tally(families: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in families:
        counts[name] = counts.get(name, 0) + 1
    return counts


def bucket(census: Sequence[dict[str, dict[str, int]]], part: str,
           bars: int) -> list[tuple[int, dict[str, float]]]:
    """Split a census into at most `bars` runs of consecutive turns, each with its average
    count per family. Returns (first turn, averages) per bar."""
    bars = max(1, min(bars, len(census)))
    result = []
    for index in range(bars):
        start, end = index * len(census) // bars, (index + 1) * len(census) // bars
        totals: dict[str, float] = {}
        for entry in census[start:end]:
            for name, count in entry[part].items():
                totals[name] = totals.get(name, 0) + count
        result.append((start, {name: total / (end - start) for name, total in totals.items()}))
    return result
