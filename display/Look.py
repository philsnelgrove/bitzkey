"""How a Bitzkey is drawn: an inner dot in its mind word's colour, sized by its body, inside a
ring in its body word's colour."""

from __future__ import annotations

import pygame as pg

from tools.Genome import TRAIT_MAX, body_word, color_from_word, mind_word

RING_WIDTH = 2
CROSS = (225, 55, 55)  # the red cross through a dead Bitzkey
DOT_RADIUS_RANGE = (1, 5)  # dot radius in pixels at the smallest and biggest sizes; bigger cells raise the top


def dot_radius(size: int, largest: int = DOT_RADIUS_RANGE[1]) -> int:
    """The dot's radius shows the body's size: from the smallest radius at size 0 up to
    `largest` at the biggest size."""
    smallest = DOT_RADIUS_RANGE[0]
    return round(smallest + (largest - smallest) * size / TRAIT_MAX)


def draw_bitzkey(surface: pg.Surface, dna: int, point: tuple[int, int], radius: int,
                 ring_width: int = RING_WIDTH, alive: bool = True, opacity: float = 1.0) -> None:
    """The dot inside its ring; a dead Bitzkey is dimmed and crossed through in red. With
    `opacity` below 1, all of it is drawn that faint, as a decaying corpse fades away."""
    if opacity <= 0:
        return
    if opacity < 1:
        reach = radius + ring_width + 1
        layer = pg.Surface((2 * reach + 1, 2 * reach + 1), pg.SRCALPHA)
        draw_bitzkey(layer, dna, (reach, reach), radius, ring_width, alive)
        layer.set_alpha(round(255 * opacity))
        surface.blit(layer, (point[0] - reach, point[1] - reach))
        return
    dot, ring = color_from_word(mind_word(dna)), color_from_word(body_word(dna))
    if not alive:
        dot, ring = (tuple(channel // 3 for channel in dot), tuple(channel // 3 for channel in ring))
    pg.draw.circle(surface, ring, point, radius + ring_width)
    pg.draw.circle(surface, dot, point, radius)
    if not alive:
        x, y = point
        reach = radius + ring_width
        width = max(2, reach // 3)
        pg.draw.line(surface, CROSS, (x - reach, y - reach), (x + reach, y + reach), width)
        pg.draw.line(surface, CROSS, (x - reach, y + reach), (x + reach, y - reach), width)
