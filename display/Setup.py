"""The configuration panel shown before a world starts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pygame as pg

from tools.Simulation import MAX_POPULATION


@dataclass(frozen=True)
class Setting:
    """One adjustable launch setting. Its key is passed to the app as a keyword argument."""
    key: str
    label: str
    hint: str
    minimum: int
    maximum: int
    step: int
    optional: bool = False  # optional settings may be left blank, meaning "none"


SETTINGS = (
    Setting("width", "Grid width", "cells across the world", 10, 250, 10),
    Setting("height", "Grid height", "cells down the world", 10, 250, 10),
    Setting("population", "Bitzkeys", "spawned at the start, a random mix of species", 0, MAX_POPULATION, 10),
    Setting("seed", "Random seed", "the same seed replays the same world; blank for a new one",
            0, 999_999_999, 1, optional=True),
)
MAX_DIGITS = 9


class SetupForm:
    """The panel's state, independent of drawing: one text value per setting and a focused row."""

    def __init__(self, defaults: Mapping[str, int | None], settings: tuple[Setting, ...] = SETTINGS):
        self.settings = settings
        self.texts = {setting.key: "" if defaults.get(setting.key) is None else str(defaults[setting.key])
                      for setting in settings}
        self.focus = 0
        self.fresh = True  # the next digit typed replaces the value rather than extending it
        for setting in settings:
            self._settle(setting)

    @property
    def focused(self) -> Setting:
        return self.settings[self.focus]

    def value(self, setting: Setting) -> int | None:
        text = self.texts[setting.key]
        if not text:
            return None if setting.optional else setting.minimum
        return max(setting.minimum, min(setting.maximum, int(text)))

    def values(self) -> dict[str, int | None]:
        return {setting.key: self.value(setting) for setting in self.settings}

    def select(self, index: int) -> None:
        self._settle(self.focused)
        self.focus = index % len(self.settings)
        self.fresh = True

    def move(self, offset: int) -> None:
        self.select(self.focus + offset)

    def adjust(self, steps: int) -> None:
        setting = self.focused
        current = self.value(setting)
        start = setting.minimum if current is None else current + steps * setting.step
        self.texts[setting.key] = str(max(setting.minimum, min(setting.maximum, start)))
        self.fresh = True

    def type_digit(self, digit: str) -> None:
        key = self.focused.key
        text = digit if self.fresh else self.texts[key] + digit
        self.texts[key] = text.lstrip("0") or "0"
        self.texts[key] = self.texts[key][:MAX_DIGITS]
        self.fresh = False

    def backspace(self) -> None:
        key = self.focused.key
        self.texts[key] = "" if self.fresh else self.texts[key][:-1]
        self.fresh = False

    def _settle(self, setting: Setting) -> None:
        """Replace whatever was typed with the in-range value it stands for."""
        value = self.value(setting)
        self.texts[setting.key] = "" if value is None else str(value)


class SetupScreen:
    """Draws the form and handles input until the user starts a world or quits."""

    ROW_HEIGHT = 64

    def __init__(self, screen: pg.Surface, defaults: Mapping[str, int | None]):
        self.screen = screen
        self.form = SetupForm(defaults)
        self.font = pg.font.SysFont("segoeui", 15)
        self.bold = pg.font.SysFont("segoeui", 22, bold=True)
        self.mono = pg.font.SysFont("consolas", 18)
        self.small = pg.font.SysFont("segoeui", 12)
        self.clock = pg.time.Clock()

    def run(self) -> dict[str, int | None] | None:
        """Return the chosen settings, or None if the user quit."""
        while True:
            self.clock.tick(30)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return None
                if event.type == pg.VIDEORESIZE:
                    self.screen = pg.display.set_mode((max(860, event.w), max(760, event.h)), pg.RESIZABLE)
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        return None
                    if event.key in (pg.K_RETURN, pg.K_KP_ENTER):
                        return self.form.values()
                    self._key(event)
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    if self._click(event.pos):
                        return self.form.values()
            self._draw()
            pg.display.flip()

    def _key(self, event: pg.event.Event) -> None:
        form = self.form
        steps = 10 if event.mod & pg.KMOD_SHIFT else 1
        if event.key == pg.K_UP or (event.key == pg.K_TAB and event.mod & pg.KMOD_SHIFT):
            form.move(-1)
        elif event.key in (pg.K_DOWN, pg.K_TAB):
            form.move(1)
        elif event.key == pg.K_RIGHT:
            form.adjust(steps)
        elif event.key == pg.K_LEFT:
            form.adjust(-steps)
        elif event.key in (pg.K_BACKSPACE, pg.K_DELETE):
            form.backspace()
        elif event.unicode.isdigit():
            form.type_digit(event.unicode)

    def _layout(self) -> tuple[pg.Rect, list[pg.Rect], pg.Rect]:
        width, height = self.screen.get_size()
        card = pg.Rect(0, 0, 620, 110 + len(self.form.settings) * self.ROW_HEIGHT + 120)
        card.center = (width // 2, height // 2)
        rows = [pg.Rect(card.left + 20, card.top + 110 + index * self.ROW_HEIGHT, card.width - 40,
                        self.ROW_HEIGHT - 8) for index in range(len(self.form.settings))]
        start = pg.Rect(0, 0, 180, 44)
        start.midbottom = (card.centerx, card.bottom - 46)
        return card, rows, start

    @staticmethod
    def _buttons(row: pg.Rect) -> tuple[pg.Rect, pg.Rect, pg.Rect]:
        """The minus button, value box and plus button of a row."""
        plus = pg.Rect(row.right - 44, row.centery - 16, 32, 32)
        box = pg.Rect(plus.left - 152, row.centery - 18, 144, 36)
        minus = pg.Rect(box.left - 40, row.centery - 16, 32, 32)
        return minus, box, plus

    def _click(self, position: tuple[int, int]) -> bool:
        """Handle a click; return True when it starts the world."""
        _, rows, start = self._layout()
        if start.collidepoint(position):
            return True
        for index, row in enumerate(rows):
            if row.collidepoint(position):
                if index != self.form.focus:
                    self.form.select(index)
                minus, _, plus = self._buttons(row)
                if minus.collidepoint(position):
                    self.form.adjust(-1)
                elif plus.collidepoint(position):
                    self.form.adjust(1)
        return False

    def _draw(self) -> None:
        self.screen.fill((19, 26, 27))
        card, rows, start = self._layout()
        pg.draw.rect(self.screen, (27, 37, 37), card, border_radius=8)
        pg.draw.rect(self.screen, (80, 99, 85), card, 1, border_radius=8)
        self._text("BITZKEY", (card.left + 28, card.top + 24), (229, 239, 220), self.bold)
        self._text("NEW WORLD  /  CONFIGURATION", (card.left + 143, card.top + 33), (144, 170, 151), self.small)
        self._text("Set up the world, then start the simulation.", (card.left + 28, card.top + 68),
                   (157, 177, 159), self.font)

        for index, (setting, row) in enumerate(zip(self.form.settings, rows, strict=True)):
            focused = index == self.form.focus
            pg.draw.rect(self.screen, (36, 49, 47) if focused else (31, 42, 41), row, border_radius=5)
            if focused:
                pg.draw.rect(self.screen, (131, 205, 143), row, 1, border_radius=5)
            self._text(setting.label.upper(), (row.left + 16, row.top + 10), (232, 236, 219), self.font)
            self._text(setting.hint, (row.left + 16, row.top + 32), (130, 152, 136), self.small)
            minus, box, plus = self._buttons(row)
            for button, sign in ((minus, "-"), (plus, "+")):
                pg.draw.rect(self.screen, (44, 57, 52), button, border_radius=4)
                label = self.bold.render(sign, True, (220, 229, 211))
                self.screen.blit(label, label.get_rect(center=button.center))
            pg.draw.rect(self.screen, (19, 26, 27), box, border_radius=4)
            pg.draw.rect(self.screen, (131, 205, 143) if focused else (54, 68, 62), box, 1, border_radius=4)
            text = self.form.texts[setting.key]
            shown, color = (text, (238, 238, 220)) if text else ("random", (116, 139, 124))
            if focused and not self.form.fresh and pg.time.get_ticks() // 500 % 2:
                shown += "_"
            label = self.mono.render(shown, True, color)
            self.screen.blit(label, label.get_rect(midright=(box.right - 12, box.centery)))

        pg.draw.rect(self.screen, (131, 205, 143), start, border_radius=6)
        label = self.bold.render("START", True, (19, 26, 27))
        self.screen.blit(label, label.get_rect(center=start.center))
        hint = self.small.render("UP/DOWN  choose    LEFT/RIGHT  adjust (SHIFT x10)    type a number    "
                                 "ENTER  start    ESC  quit", True, (130, 152, 136))
        self.screen.blit(hint, hint.get_rect(midbottom=(card.centerx, card.bottom - 16)))

    def _text(self, text: str, position: tuple[int, int], color: tuple[int, int, int],
              font: pg.font.Font) -> None:
        self.screen.blit(font.render(text, True, color), position)
