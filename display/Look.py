"""How a Bitzkey is drawn: an inner dot in its mind word's colour, sized by its body, inside a
ring in its body word's colour."""

from __future__ import annotations

import pygame as pg

from tools.Genome import body_word, color_from_word, mind_word

RING_WIDTH = 2
DOT_RADIUS_RANGE = (1, 5)  # dot radius in pixels at size 0 and size 15; bigger cells raise the top


def dot_radius(size: int, largest: int = DOT_RADIUS_RANGE[1]) -> int:
    """The dot's radius shows the body's size: from the smallest radius at size 0 up to
    `largest` at size 15."""
    smallest = DOT_RADIUS_RANGE[0]
    return round(smallest + (largest - smallest) * size / 15)


def draw_bitzkey(surface: pg.Surface, dna: int, point: tuple[int, int], radius: int,
                 ring_width: int = RING_WIDTH, alive: bool = True) -> None:
    dot, ring = color_from_word(mind_word(dna)), color_from_word(body_word(dna))
    if not alive:
        dot, ring = (tuple(channel // 3 for channel in dot), tuple(channel // 3 for channel in ring))
    pg.draw.circle(surface, ring, point, radius + ring_width)
    pg.draw.circle(surface, dot, point, radius)
