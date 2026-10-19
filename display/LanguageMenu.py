"""The language menu: a button naming the language shown, which opens a list of every language,
each named in its own script. Choosing one switches the whole app at once and is remembered."""

from __future__ import annotations

import pygame as pg

from display.Language import LANGUAGES, language, set_language
from display.Ui import UI

ROW = 26  # design pixels per language in the open list


class LanguageMenu:
    def __init__(self) -> None:
        self.button = pg.Rect(0, 0, 0, 0)
        self.items: list[pg.Rect] = []
        self.open = False

    def draw_button(self, screen: pg.Surface, right: int, middle: int) -> int:
        """The button, its right edge at `right` and centred on `middle`; returns its left edge."""
        font = UI.font("segoeui", 12, code=language())
        name = dict(LANGUAGES)[language()]
        self.button = pg.Rect(0, 0, font.size(name)[0] + UI(34), UI(26))
        self.button.midright = (right, middle)
        pg.draw.rect(screen, (80, 99, 85) if self.open else (44, 57, 52), self.button, border_radius=UI(4))
        image = font.render(name, True, (220, 229, 211))
        screen.blit(image, image.get_rect(midleft=(self.button.left + UI(10), self.button.centery)))
        cx, cy = self.button.right - UI(13), self.button.centery  # a small arrow: the list opens below
        pg.draw.polygon(screen, (187, 199, 181), [(cx - UI(4), cy - UI(2)), (cx + UI(4), cy - UI(2)), (cx, cy + UI(3))])
        return self.button.left

    def draw_list(self, screen: pg.Surface) -> None:
        """Last, over everything else, when open."""
        self.items = []
        if not self.open:
            return
        width = max([self.button.width] + [UI.font("segoeui", 13, code=code).size(name)[0] + UI(28) for code, name in LANGUAGES])
        panel = pg.Rect(self.button.right - width, self.button.bottom + UI(4), width, len(LANGUAGES) * UI(ROW) + UI(8))
        pg.draw.rect(screen, (27, 37, 37), panel, border_radius=UI(6))
        pg.draw.rect(screen, (80, 99, 85), panel, 1, border_radius=UI(6))
        for index, (code, name) in enumerate(LANGUAGES):
            item = pg.Rect(panel.left + UI(4), panel.top + UI(4) + index * UI(ROW), panel.width - UI(8), UI(ROW))
            self.items.append(item)
            current = code == language()
            if current:
                pg.draw.rect(screen, (44, 57, 52), item, border_radius=UI(4))
            image = UI.font("segoeui", 13, bold=current, code=code).render(name, True,
                                                                         (238, 183, 110) if current else (220, 229, 211))
            screen.blit(image, image.get_rect(midleft=(item.left + UI(10), item.centery)))

    def click(self, position: tuple[int, int]) -> bool:
        """Whether the click was the menu's."""
        if self.button.collidepoint(position):
            self.open = not self.open
            return True
        if not self.open:
            return False
        for (code, _), item in zip(LANGUAGES, self.items):
            if item.collidepoint(position):
                set_language(code)
        self.open = False  # a click anywhere else only closes the list
        return True
