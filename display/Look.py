"""How a Bitzkey is drawn: an inner dot in its mind word's colour, sized by its body, inside a
ring in its body word's colour."""

from __future__ import annotations

import math

import pygame as pg

from display.Icons import draw_icon
from tools.Genome import TRAIT_MAX, body_word, color_from_word, decode_body, mind_word

RING_WIDTH = 2
CROSS = (225, 55, 55)  # the red cross through a dead Bitzkey
CENTRE = (0, 0, 0)  # the black centre behind a species icon
OUTLINE = (190, 190, 190)  # the thin light grey outline around a species icon
DOT_RADIUS_RANGE = (1, 5)  # dot radius in pixels at the smallest and biggest sizes; bigger cells raise the top
ICON_MIN_RADIUS = 3  # a dot this big or bigger carries its species' icon
# The world: light grey, each row tinted by its temperature now, from hot to very cold: reddish brown,
# green, white and light blue.
WORLD_GREY = (196, 198, 196)
ZONE_TINTS = ((150, 85, 55), (110, 170, 90), (255, 255, 255), (170, 210, 250))
ZONE_TINT = 0.45  # how far toward its tint a zone's grey goes
GRASS = (62, 122, 52)
GRID = (140, 146, 142)
SELECTED = (25, 32, 32)    # the ring round the selected Bitzkey and those picked on the colour graph
EXPORTING = (205, 120, 30)  # the ring round Bitzkeys on the export list


def zone_colour(temperature: int) -> tuple[int, int, int]:
    """The world's grey tinted for a temperature."""
    return tuple(round(grey + (tint - grey) * ZONE_TINT) for grey, tint in zip(WORLD_GREY, ZONE_TINTS[temperature]))


def dot_radius(size: int, largest: int = DOT_RADIUS_RANGE[1]) -> int:
    """The dot's radius shows the body's size: from the smallest radius at size 0 up to
    `largest` at the biggest size."""
    smallest = DOT_RADIUS_RANGE[0]
    return round(smallest + (largest - smallest) * size / TRAIT_MAX)


def outline_offsets(width: int) -> list[tuple[int, int]]:
    """Where a species icon is copied, in the outline colour, to outline it `width` pixels thick:
    a pixel each way, or around a circle when wider, so its corners are outlined too."""
    if width <= 1:
        return [(1, 0), (-1, 0), (0, 1), (0, -1)]
    return sorted({(round(width * math.cos(step * math.pi / 8)), round(width * math.sin(step * math.pi / 8)))
                   for step in range(16)})


def draw_bitzkey(surface: pg.Surface, dna: int, point: tuple[int, int], radius: int,
                 ring_width: int = RING_WIDTH, alive: bool = True, opacity: float = 1.0) -> None:
    """The dot inside its ring. A dot big enough to hold its species' icon is black, with the icon
    drawn in the dot's colour inside a thin light grey outline; a smaller one is simply that colour. A dead Bitzkey is dimmed and
    crossed through in red. With `opacity` below 1, all of it is drawn that faint, as a decaying
    corpse fades away."""
    if opacity <= 0:
        return
    if opacity < 1:
        reach = radius + ring_width + 1
        layer = pg.Surface((2 * reach + 1, 2 * reach + 1), pg.SRCALPHA)
        draw_bitzkey(layer, dna, (reach, reach), radius, ring_width, alive)
        layer.set_alpha(round(255 * opacity))
        surface.blit(layer, (point[0] - reach, point[1] - reach))
        return
    dot, ring, outline = color_from_word(mind_word(dna)), color_from_word(body_word(dna)), OUTLINE
    if not alive:
        dot, ring, outline = (tuple(channel // 3 for channel in colour) for colour in (dot, ring, outline))
    pg.draw.circle(surface, ring, point, radius + ring_width)
    if radius >= ICON_MIN_RADIUS:
        pg.draw.circle(surface, CENTRE, point, radius)
        species, (x, y) = str(decode_body(dna)["species"]), point
        for dx, dy in outline_offsets(max(1, ring_width // RING_WIDTH)):  # as thick as the ring is enlarged
            draw_icon(surface, species, (x + dx, y + dy), radius * 0.8, outline)
        draw_icon(surface, species, point, radius * 0.8, dot)
    else:
        pg.draw.circle(surface, dot, point, radius)
    if not alive:
        x, y = point
        reach = radius + ring_width
        width = max(2, reach // 3)
        pg.draw.line(surface, CROSS, (x - reach, y - reach), (x + reach, y + reach), width)
        pg.draw.line(surface, CROSS, (x - reach, y + reach), (x + reach, y - reach), width)
