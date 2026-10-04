"""A small modal dialog drawn over the simulation: a title, a few lines of text and a row of
buttons. Enter picks the default button and Esc the cancelling one."""

from __future__ import annotations

from dataclasses import dataclass

import pygame as pg

from display.Ui import UI


@dataclass
class Dialog:
    title: str
    lines: list[str]
    buttons: list[tuple[str, str]]  # (choice, label), left to right; the last is the default
    cancel: str                     # the choice Esc makes
    busy: str | None = None         # a choice under way: its button greys out and nothing can be pressed
    caption: str | None = None      # shown beneath the dialog while busy

    def start(self, choice: str, label: str, caption: str) -> None:
        """Show that a choice is under way: relabel and grey out its button, and add a caption."""
        self.buttons = [(name, label if name == choice else text) for name, text in self.buttons]
        self.busy, self.caption = choice, caption

    def handle(self, event: pg.event.Event, size: tuple[int, int]) -> str | None:
        """Return the choice an event makes, if any."""
        if self.busy:
            return None
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                return self.cancel
            if event.key in (pg.K_RETURN, pg.K_KP_ENTER):
                return self.buttons[-1][0]
        elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            for choice, rect in self._layout(size)["buttons"].items():
                if rect.collidepoint(event.pos):
                    return choice
        return None

    def _layout(self, size: tuple[int, int]) -> dict:
        width, height = size
        panel = pg.Rect(0, 0, UI(560), UI(150 + 24 * len(self.lines)))
        panel.center = (width // 2, height // 2)
        buttons = {}
        right = panel.right - UI(24)
        for choice, label in reversed(self.buttons):
            button_width = max(UI(110), UI.font("segoeui", 13, bold=True).size(label)[0] + UI(32))
            buttons[choice] = pg.Rect(right - button_width, panel.bottom - UI(64), button_width, UI(40))
            right -= button_width + UI(10)
        return {"panel": panel, "buttons": buttons}

    def draw(self, screen: pg.Surface) -> None:
        layout = self._layout(screen.get_size())
        panel = layout["panel"]
        shade = pg.Surface(screen.get_size(), pg.SRCALPHA)
        shade.fill((8, 12, 12, 185))
        screen.blit(shade, (0, 0))
        pg.draw.rect(screen, (30, 40, 40), panel, border_radius=UI(8))
        pg.draw.rect(screen, (80, 99, 85), panel, 1, border_radius=UI(8))
        screen.blit(UI.font("segoeui", 22, bold=True).render(self.title, True, (229, 239, 220)),
                    (panel.left + UI(24), panel.top + UI(20)))
        body = UI.font("segoeui", 15)
        for index, line in enumerate(self.lines):
            screen.blit(body.render(line, True, (187, 199, 181)), (panel.left + UI(24), panel.top + UI(64 + 24 * index)))
        labels = dict(self.buttons)
        for choice, rect in layout["buttons"].items():
            default = choice == self.buttons[-1][0]
            if self.busy:
                fill, ink = (52, 58, 56), (128, 138, 132) if choice == self.busy else (92, 102, 96)
            else:
                fill, ink = ((131, 205, 143), (19, 26, 27)) if default else ((44, 57, 52), (220, 229, 211))
            pg.draw.rect(screen, fill, rect, border_radius=UI(6))
            text = UI.font("segoeui", 13, bold=True).render(labels[choice], True, ink)
            screen.blit(text, text.get_rect(center=rect.center))
        if self.busy and self.caption:
            caption = UI.font("segoeui", 16).render(self.caption, True, (144, 170, 151))
            screen.blit(caption, caption.get_rect(midtop=(panel.centerx, panel.bottom + UI(22))))
