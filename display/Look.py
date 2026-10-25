"""How a Bitzkey is drawn: an inner dot in its mind word's colour, sized by its body, inside a
ring in its body word's colour, with its species icon on its social background word's colour."""

from __future__ import annotations

import math

import pygame as pg

from display.Icons import draw_icon
from tools.Genome import TRAIT_MAX, body_word, color_from_word, decode_body, mind_word, social_word

RING_WIDTH = 2
CROSS = (225, 55, 55)  # the red cross through a dead Bitzkey
OUTLINE = (190, 190, 190)  # the thin light grey outline around a species icon
DOT_RADIUS_RANGE = (1, 5)  # dot radius in pixels at the smallest and biggest sizes; bigger cells raise the top
ICON_MIN_RADIUS = 3  # a dot this big or bigger carries its species' icon
# The world: light grey, each row tinted by its temperature now, from hot to very cold: reddish brown,
# green, white and light blue.
WORLD_GREY = (196, 198, 196)
ZONE_TINTS = ((150, 85, 55), (110, 170, 90), (255, 255, 255), (170, 210, 250))
ZONE_TINT = 0.45  # how far toward its tint a zone's grey goes
GRASS = (62, 122, 52)
# The sea: its row's temperature tint drawn this far toward deep blue, so the climate's bands show through, and its
# grass, sea grass, in kelp green.
SEA = (38, 84, 140)
SEA_TINT = 0.6
SEA_GRASS = (58, 150, 120)
NIGHT = (10, 16, 40, 90)  # the shadow over the night's half of the map
GRID = (140, 146, 142)
SELECTED = (25, 32, 32)    # the ring round the selected Bitzkey and those picked on the colour graph
EXPORTING = (205, 120, 30)  # the ring round Bitzkeys on the export list


def zone_colour(temperature: float) -> tuple[int, int, int]:
    """The world's grey tinted for a temperature, part way between two tints while the weather turns."""
    lower = min(int(temperature), len(ZONE_TINTS) - 1)
    upper = min(lower + 1, len(ZONE_TINTS) - 1)
    turned = temperature - lower
    tints = (low + (high - low) * turned for low, high in zip(ZONE_TINTS[lower], ZONE_TINTS[upper]))
    return tuple(round(grey + (tint - grey) * ZONE_TINT) for grey, tint in zip(WORLD_GREY, tints))


def sea_colour(temperature: float) -> tuple[int, int, int]:
    """The same under the sea."""
    return tuple(round(value + (sea - value) * SEA_TINT) for value, sea in zip(zone_colour(temperature), SEA))


def dot_radius(size: int, largest: int = DOT_RADIUS_RANGE[1], smallest: int = DOT_RADIUS_RANGE[0]) -> int:
    """The dot's radius shows the body's size: from `smallest` at size 0 up to `largest` at the
    biggest size."""
    return round(smallest + (largest - smallest) * size / TRAIT_MAX)


def map_dot_radii(cell: float) -> tuple[int, int]:
    """The smallest and largest dots on the map, zoomed with it: the largest body's dot, ring and
    all, fills a cell, and the smallest stays big enough for its icon, growing with the largest in
    proportion."""
    largest = max(DOT_RADIUS_RANGE[1], round(cell / 2) - RING_WIDTH)
    return max(ICON_MIN_RADIUS, round(largest * ICON_MIN_RADIUS / DOT_RADIUS_RANGE[1])), largest


def outline_offsets(width: int) -> list[tuple[int, int]]:
    """Where a species icon is copied, in the outline colour, to outline it `width` pixels thick:
    a pixel each way, or around a circle when wider, so its corners are outlined too."""
    if width <= 1:
        return [(1, 0), (-1, 0), (0, 1), (0, -1)]
    return sorted({(round(width * math.cos(step * math.pi / 8)), round(width * math.sin(step * math.pi / 8)))
                   for step in range(16)})


def draw_reticle(surface: pg.Surface, point: tuple[int, int], radius: int) -> None:
    """The ring round a selected Bitzkey, like a target's sights: a white ring inside a black one,
    each two pixels, so it shows on light ground and dark, with a pip at north, east, south and west."""
    x, y = point
    pg.draw.circle(surface, (0, 0, 0), point, radius + 2, 2)
    pg.draw.circle(surface, (255, 255, 255), point, radius, 2)
    reach = radius + 2
    length = max(3, radius // 2)
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        start = (x + dx * reach, y + dy * reach)
        end = (x + dx * (reach + length), y + dy * (reach + length))
        pg.draw.line(surface, (0, 0, 0), start, end, 4)
        pg.draw.line(surface, (255, 255, 255), start, end, 2)


def bitzkey_layer(dna: int, radius: int, ring_width: int = RING_WIDTH, alive: bool = True) -> pg.Surface:
    """A Bitzkey drawn alone, centred on a clear square, to be blitted as faintly as wanted."""
    reach = radius + ring_width + 1
    layer = pg.Surface((2 * reach + 1, 2 * reach + 1), pg.SRCALPHA)
    draw_bitzkey(layer, dna, (reach, reach), radius, ring_width, alive)
    return layer


def draw_bitzkey(surface: pg.Surface, dna: int, point: tuple[int, int], radius: int,
                 ring_width: int = RING_WIDTH, alive: bool = True, opacity: float = 1.0) -> None:
    """The dot inside its ring. A dot big enough to hold its species' icon is its social background's
    colour, with the icon drawn in the dot's colour inside a thin light grey outline; a smaller one is
    simply the dot's colour. A dead Bitzkey is dimmed and
    crossed through in red. With `opacity` below 1, all of it is drawn that faint, as a decaying
    corpse fades away."""
    if opacity <= 0:
        return
    if opacity < 1:
        layer = bitzkey_layer(dna, radius, ring_width, alive)
        layer.set_alpha(round(255 * opacity))
        reach = layer.get_width() // 2
        surface.blit(layer, (point[0] - reach, point[1] - reach))
        return
    dot, ring, outline = color_from_word(mind_word(dna)), color_from_word(body_word(dna)), OUTLINE
    social = color_from_word(social_word(dna))
    if not alive:
        dot, ring, outline, social = (tuple(channel // 3 for channel in colour) for colour in (dot, ring, outline, social))
    pg.draw.circle(surface, ring, point, radius + ring_width)
    if radius >= ICON_MIN_RADIUS:
        pg.draw.circle(surface, social, point, radius)
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
