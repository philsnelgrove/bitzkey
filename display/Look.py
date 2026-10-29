"""How a Bitzkey is drawn: an inner dot in its mind word's colour, sized by its body, inside a
ring in its body word's colour, with its species icon on its social background word's colour."""

from __future__ import annotations

import math

import numpy as np

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



# Colour-blind safe colours (F8): genome colours with the red-green differences a deuteranope or protanope loses moved
# into lightness and blue-yellow, where they can be seen (a "daltonize" correction), as the C++ version's.
COLOUR_BLIND_SAFE = False


def genome_colour(word: int) -> tuple[int, int, int]:
    """A genome word's colour, as shown: corrected for the colour-blind while COLOUR_BLIND_SAFE."""
    r, g, b = color_from_word(word)[:3]
    if COLOUR_BLIND_SAFE:
        seen_r, seen_g, seen_b = 0.625 * r + 0.375 * g, 0.7 * r + 0.3 * g, 0.3 * g + 0.7 * b  # as a deuteranope sees it
        lost_r, lost_g, lost_b = r - seen_r, g - seen_g, b - seen_b
        g, b = g + 0.7 * lost_r + lost_g, b + 0.7 * lost_r + lost_b
    return tuple(int(min(255.0, max(0.0, value)) + 0.5) for value in (r, g, b))


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


CHIP_EDGE = 0.32  # a chip's coloured edge, as a share of its radius (at least the ring's width)
CHIP_INSERTS = 6  # the lighter inserts round its edge
# Each look drawn is kept, up to LOOKS_KEPT, the least recently drawn let go first: a look is its colours to within
# LOOK_SHADES of each channel, its species, size and whether it lives, so the nearly alike share one.
LOOKS_KEPT = 20000
LOOK_SHADES = 8
TEXTURE_FROM = 6  # squares this many pixels across or more are textured, and their grass drawn as tufts
GROUND_VARIANTS = 4
GROUNDS = ("sand", "meadow", "snow", "ice")  # by the temperature now, hot to very cold; "sea" at sea
_chips: dict = {}
_tufts: dict = {}
_tiles: dict = {}


def _scaled(colour, factor):
    return tuple(max(0, min(255, round(value * factor))) for value in colour[:3])


def _mixed(start, end, share):
    return tuple(round(a + (b - a) * share) for a, b in zip(start[:3], end[:3]))


def _paint(surface: pg.Surface, colour, alpha: "np.ndarray") -> None:
    """Lay a colour over a surface by a coverage, 0 to 1, a value a pixel, as paint laid over what is there."""
    rgb = pg.surfarray.pixels3d(surface)
    below = pg.surfarray.pixels_alpha(surface)
    under = below / 255.0
    out = alpha + under * (1 - alpha)
    safe = np.where(out > 0, out, 1)
    colour = np.asarray(colour, dtype=float)
    for channel in range(3):
        paint = colour[channel] if colour.ndim == 1 else colour[:, :, channel]
        rgb[:, :, channel] = np.clip((paint * alpha + rgb[:, :, channel] * under * (1 - alpha)) / safe, 0, 255).astype(np.uint8)
    below[:, :] = np.clip(out * 255, 0, 255).astype(np.uint8)
    del rgb, below


def bitzkey_layer(dna: int, radius: int, ring_width: int = RING_WIDTH, alive: bool = True) -> pg.Surface:
    """A poker chip, lit from the top left: its edge in the ring's colour with lighter inserts, its face the social
    background's (or the dot's, too small for an icon) carrying the species icon in the dot's, a shine on the top
    left and a soft shadow below to the right, as the C++ version draws it."""
    outer = radius + ring_width
    edge = max(ring_width, round(outer * CHIP_EDGE))
    face_radius, drop = outer - edge, max(1.0, outer * 0.14)
    reach = outer + math.ceil(drop) + 2
    layer = pg.Surface((2 * reach + 1, 2 * reach + 1), pg.SRCALPHA)
    dot, ring = genome_colour(mind_word(dna)), genome_colour(body_word(dna))
    social, outline = genome_colour(social_word(dna)), OUTLINE
    if not alive:
        dot, ring, outline, social = (_scaled(colour, 1 / 3) for colour in (dot, ring, outline, social))
    icon = radius >= ICON_MIN_RADIUS
    centre = reach + 0.5
    xs, ys = np.meshgrid(np.arange(2 * reach + 1) + 0.5 - centre, np.arange(2 * reach + 1) + 0.5 - centre, indexing="ij")
    distance = np.hypot(xs, ys)
    shade = np.hypot(xs - drop, ys - drop)  # the shadow, softened at its edge
    _paint(layer, (0, 0, 0), 0.38 * np.clip((outer + 1 - shade) / (outer * 0.35 + 1), 0, 1))
    coverage = np.clip(outer + 0.5 - distance, 0, 1)
    turn = (np.arctan2(ys, xs) + math.pi) / (2 * math.pi) * CHIP_INSERTS
    inserts = (turn - np.floor(turn) < 0.4) & (distance > face_radius) & (outer >= 5)
    light = (1 + 0.2 * (-xs - ys) / (outer * math.sqrt(2))) * (1 - 0.25 * np.clip((distance - (outer - 1.5)) / 1.5, 0, 1))
    groove = (np.abs(distance - face_radius) < 0.7) & (outer >= 5)
    base = np.zeros(distance.shape + (3,))
    base[distance <= face_radius] = social if icon else dot
    base[(distance > face_radius) & ~inserts] = ring
    base[inserts] = _mixed(ring, (255, 255, 255), 0.55)
    base *= (light * np.where(groove, 0.7, 1.0))[:, :, None]  # lit from the top left; the groove round the face darker
    _paint(layer, np.clip(base, 0, 255), coverage)
    if icon:
        species = str(decode_body(dna)["species"])
        half = face_radius * 0.82
        for dx, dy in outline_offsets(max(1, int(face_radius / 6))):
            draw_icon(layer, species, (reach + dx, reach + dy), half, outline)
        draw_icon(layer, species, (reach, reach), half, dot)
    shine_x, shine_y = -outer * 0.35, -outer * 0.42
    off = np.hypot((xs - shine_x) / 1.25, ys - shine_y) / (outer * 0.5)
    _paint(layer, (255, 255, 255), np.where(off < 1, (0.42 if alive else 0.15) * (1 - off * off) * np.clip(outer - 0.5 - distance, 0, 1), 0))
    if not alive:
        width = max(2, outer // 3)
        pg.draw.line(layer, CROSS, (reach - outer, reach - outer), (reach + outer, reach + outer), width)
        pg.draw.line(layer, CROSS, (reach - outer, reach + outer), (reach + outer, reach - outer), width)
    return layer


def draw_bitzkey(surface: pg.Surface, dna: int, point: tuple[int, int], radius: int,
                 ring_width: int = RING_WIDTH, alive: bool = True, opacity: float = 1.0) -> None:
    """A Bitzkey as a poker chip (see bitzkey_layer), each look drawn once and kept (see LOOKS_KEPT). With `opacity`
    below 1, it is drawn that faint, as a decaying corpse fades away."""
    if opacity <= 0:
        return
    shades = tuple(channel // LOOK_SHADES for word in (mind_word(dna), body_word(dna), social_word(dna))
                   for channel in genome_colour(word)[:3])
    key = (decode_body(dna)["species"], shades, radius, ring_width, alive)
    layer = _chips.pop(key, None)  # taken out and put back, so the dict runs from least to most recently drawn
    if layer is None:
        if len(_chips) >= LOOKS_KEPT:
            del _chips[next(iter(_chips))]
        layer = bitzkey_layer(dna, radius, ring_width, alive)
    _chips[key] = layer
    reach = layer.get_width() // 2
    if opacity < 1:
        layer = layer.copy()
        layer.set_alpha(round(255 * opacity))
    surface.blit(layer, (point[0] - reach, point[1] - reach))


def _lines(surface: pg.Surface, colour, points, width: float = 1.0) -> None:
    if width <= 2.5:
        pg.draw.aalines(surface, colour, False, points)
    else:
        pg.draw.lines(surface, colour, False, points, round(width))


def grass_tuft(size: int, sea: bool) -> pg.Surface:
    """A tuft of grass the size of a square, or of kelp in the sea, kept once drawn."""
    key = (size, sea)
    if key not in _tufts:
        tuft = pg.Surface((size, size), pg.SRCALPHA)
        base_x, base_y, tall, thick = size * 0.5, size * 0.8, size * 0.55, max(1.0, size / 12)
        if sea:  # kelp: a few swaying strands, darker ones behind a light one
            for strand in range(3):
                x0 = base_x + (strand - 1) * size * 0.16
                points = [(x0 + math.sin(step / 8 * 5.5 + strand) * size * 0.08 * (step > 0),
                           base_y - step / 8 * tall * (1.0 if strand == 1 else 0.8)) for step in range(9)]
                _lines(tuft, SEA_GRASS if strand == 1 else _scaled(SEA_GRASS, 0.7), points, thick)
        else:  # blades fanning from a clump, dark beneath and light at their tips
            for blade, (lean, share) in enumerate(zip((-0.55, -0.25, 0.0, 0.3, 0.6), (0.7, 0.95, 1.0, 0.85, 0.65))):
                tip = (base_x + lean * size * 0.42, base_y - tall * share)
                mid = (base_x + lean * size * 0.12, base_y - tall * share * 0.55)
                _lines(tuft, _scaled(GRASS, 0.75), [(base_x + (blade - 2) * thick * 0.6, base_y), mid], thick)
                _lines(tuft, _mixed(GRASS, (170, 220, 120), 0.45), [mid, tip], thick)
        _tufts[key] = tuft
    return _tufts[key]


def ground_tile(ground: str, size: int, variant: int) -> pg.Surface:
    """A faint texture laid over a square's colour, so the season's tint still shows through: sand, meadow, snow, ice
    or sea; one of a few variants, so the pattern doesn't repeat."""
    key = (ground, size, variant)
    if key in _tiles:
        return _tiles[key]
    tile = pg.Surface((size, size), pg.SRCALPHA)
    state = (2654435761 * (variant * 7 + (GROUNDS + ("sea",)).index(ground) * 131 + 1)) & 0xFFFFFFFF or 1

    def next_() -> float:  # the same texture every time
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        return (state & 0xFFFFFF) / 0x1000000

    def speck(colour) -> None:
        tile.set_at((int(next_() * size), int(next_() * size)), colour)

    s = size
    if ground == "sand":  # ripples drifted by the wind, and grit
        for ripple in range(2):
            y0, x0 = s * (0.3 + 0.4 * ripple + 0.1 * next_()), s * 0.1 * next_()
            _lines(tile, (255, 240, 205, 75), [(x0 + step * s * 0.13, y0 + math.sin(step * 1.1 + ripple) * s * 0.05) for step in range(7)])
        for _ in range(4):
            speck((110, 75, 40, 75))
    elif ground == "meadow":  # short darker strokes, and pale flecks
        for _ in range(4):
            x0, y0 = next_() * s, next_() * s
            _lines(tile, (45, 85, 35, 75), [(x0, y0), (x0 + (next_() - 0.5) * s * 0.12, y0 - s * 0.12)])
        for _ in range(3):
            speck((215, 235, 160, 75))
    elif ground == "snow":  # a drift's blue shadow, and sparkle
        x0, y0 = s * (0.1 + 0.3 * next_()), s * (0.4 + 0.4 * next_())
        _lines(tile, (150, 175, 210, 90), [(x0, y0), (x0 + s * 0.5, y0 - s * 0.08)], max(1.0, s / 10))
        for flake in range(6):
            speck((255, 255, 255, 150) if flake % 2 else (160, 190, 225, 150))
    elif ground == "ice":  # cracks across the ice, and a glint
        points = [(next_() * s, next_() * s)]
        for _ in range(3):
            points.append((points[-1][0] + (next_() - 0.4) * s * 0.4, points[-1][1] + (next_() - 0.4) * s * 0.4))
        _lines(tile, (245, 252, 255, 100), points)
        gx, gy = int(next_() * s), int(next_() * s)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if 0 <= gx + dx < s and 0 <= gy + dy < s:
                tile.set_at((gx + dx, gy + dy), (255, 255, 255, 90))
        tile.set_at((min(gx, s - 1), min(gy, s - 1)), (255, 255, 255, 205))
    else:  # wave crests, light over the deep
        for _ in range(1 + int(next_() * 2)):
            cx, cy, span = s * (0.2 + 0.6 * next_()), s * (0.25 + 0.5 * next_()), s * 0.22
            _lines(tile, (225, 238, 255, 85), [(cx - span + 2 * span * step / 6, cy - math.sin(step / 6 * math.pi) * s * 0.08) for step in range(7)])
    _tiles[key] = tile
    return tile
