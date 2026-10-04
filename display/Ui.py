"""Scaling for text and controls. Screens are laid out for a 1200x820 window; on any other
screen every length is multiplied by one factor, and fonts are rendered at the scaled size so
text stays crisp rather than being stretched."""

from __future__ import annotations

import pygame as pg

DESIGN_SIZE = (1200, 820)
MIN_SCALE = 0.8


class Ui:
    def __init__(self) -> None:
        self.scale = 1.0
        self._fonts: dict[tuple[str, int, bool], pg.font.Font] = {}

    def fit(self, size: tuple[int, int]) -> None:
        """Scale to the screen: as large as fits both its width and height, never below MIN_SCALE."""
        self.scale = max(MIN_SCALE, min(size[0] / DESIGN_SIZE[0], size[1] / DESIGN_SIZE[1]))

    def __call__(self, length: float) -> int:
        """A length from the 1200x820 design, in pixels at the current scale."""
        return round(length * self.scale)

    def font(self, name: str, size: int, bold: bool = False) -> pg.font.Font:
        """A system font at `size` (design pixels), scaled; fonts are cached by their real size."""
        key = (name, max(6, round(size * self.scale)), bold)
        if key not in self._fonts:
            self._fonts[key] = pg.font.SysFont(name, key[1], bold=bold)
        return self._fonts[key]


UI = Ui()
