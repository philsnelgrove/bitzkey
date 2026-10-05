"""Each species' icon, drawn from simple shapes so it scales to any size: a tree for plants, a leaf
for grazers, an apple for omnivores and a drumstick for predators. Shapes are laid out on a square
from -1 to 1 either way, then scaled to `half`, half the icon's width."""

from __future__ import annotations

import pygame as pg

Colour = tuple[int, int, int]


def draw_icon(surface: pg.Surface, species: str, centre: tuple[int, int], half: float, colour: Colour) -> None:
    cx, cy = centre

    def at(x: float, y: float) -> tuple[int, int]:
        return round(cx + x * half), round(cy + y * half)

    def size(length: float) -> int:
        return max(1, round(length * half))

    if species == "plant":  # a tree: a canopy over a trunk
        pg.draw.polygon(surface, colour, [at(0, -1), at(0.85, 0.35), at(-0.85, 0.35)])
        pg.draw.rect(surface, colour, (*at(-0.18, 0.3), size(0.36), size(0.7)))
    elif species == "grazer":  # a leaf, pointing up and right, on a short stem
        pg.draw.polygon(surface, colour, [at(-0.65, 0.65), at(-0.62, -0.05), at(-0.2, -0.58), at(0.8, -0.8),
                                          at(0.58, 0.2), at(0.05, 0.62)])
        pg.draw.line(surface, colour, at(-0.6, 0.6), at(-0.95, 0.95), size(0.18))
    elif species == "omnivore":  # an apple: two lobes, a stem and a leaf
        pg.draw.circle(surface, colour, at(-0.28, 0.22), size(0.58))
        pg.draw.circle(surface, colour, at(0.28, 0.22), size(0.58))
        pg.draw.line(surface, colour, at(0, -0.3), at(0.12, -0.9), size(0.16))
        pg.draw.polygon(surface, colour, [at(0.12, -0.7), at(0.62, -0.95), at(0.4, -0.52)])
    elif species == "predator":  # a drumstick: the meat, a bone and its knuckle
        pg.draw.circle(surface, colour, at(0.25, -0.25), size(0.6))
        pg.draw.line(surface, colour, at(-0.05, 0.05), at(-0.62, 0.62), size(0.26))
        pg.draw.circle(surface, colour, at(-0.8, 0.6), size(0.2))
        pg.draw.circle(surface, colour, at(-0.6, 0.8), size(0.2))
