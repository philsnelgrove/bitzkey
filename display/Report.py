"""A report drawn over the simulation: a title, a note, and lines of text in a fixed-width font, scrolled with
the wheel when they don't all fit. Esc, Enter or CLOSE closes it."""

from __future__ import annotations

from dataclasses import dataclass

import pygame as pg

from display.Ui import UI

LINE = 18  # design pixels to a line of the report


@dataclass
class Report:
    title: str
    note: str
    lines: list[str]
    scroll: int = 0  # lines scrolled past

    def _layout(self, size: tuple[int, int]) -> dict:
        width, height = size
        mono = UI.font("consolas", 13)
        widest = max((mono.size(line)[0] for line in self.lines), default=0)
        panel = pg.Rect(0, 0, min(width - UI(40), max(UI(560), widest + UI(48))),
                        min(height - UI(40), UI(160) + UI(LINE) * len(self.lines)))
        panel.center = (width // 2, height // 2)
        body = pg.Rect(panel.left + UI(24), panel.top + UI(84), panel.width - UI(48), panel.height - UI(84 + 70))
        close = pg.Rect(0, 0, UI(110), UI(40))
        close.bottomright = (panel.right - UI(24), panel.bottom - UI(18))
        return {"panel": panel, "body": body, "close": close}

    def _shown(self, size: tuple[int, int]) -> int:
        return max(1, self._layout(size)["body"].height // UI(LINE))

    def handle(self, event: pg.event.Event, size: tuple[int, int]) -> bool:
        """Whether an event closes it."""
        if event.type == pg.KEYDOWN:
            if event.key in (pg.K_ESCAPE, pg.K_RETURN, pg.K_KP_ENTER):
                return True
            if event.key in (pg.K_DOWN, pg.K_UP, pg.K_PAGEDOWN, pg.K_PAGEUP):
                step = {pg.K_DOWN: 1, pg.K_UP: -1, pg.K_PAGEDOWN: 10, pg.K_PAGEUP: -10}[event.key]
                self.scroll = max(0, min(len(self.lines) - self._shown(size), self.scroll + step))
        elif event.type == pg.MOUSEWHEEL:
            self.scroll = max(0, min(len(self.lines) - self._shown(size), self.scroll - 3 * event.y))
        elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            return self._layout(size)["close"].collidepoint(event.pos)
        return False

    def draw(self, screen: pg.Surface) -> None:
        layout = self._layout(screen.get_size())
        panel, body = layout["panel"], layout["body"]
        shade = pg.Surface(screen.get_size(), pg.SRCALPHA)
        shade.fill((8, 12, 12, 185))
        screen.blit(shade, (0, 0))
        pg.draw.rect(screen, (30, 40, 40), panel, border_radius=UI(8))
        pg.draw.rect(screen, (80, 99, 85), panel, 1, border_radius=UI(8))
        screen.blit(UI.font("segoeui", 22, bold=True).render(self.title, True, (229, 239, 220)),
                    (panel.left + UI(24), panel.top + UI(20)))
        screen.blit(UI.font("segoeui", 13).render(self.note, True, (144, 170, 151)), (panel.left + UI(24), panel.top + UI(54)))
        mono = UI.font("consolas", 13)
        shown = self._shown(screen.get_size())
        self.scroll = max(0, min(max(0, len(self.lines) - shown), self.scroll))
        screen.set_clip(body)
        for row, line in enumerate(self.lines[self.scroll:self.scroll + shown]):
            heading = self.scroll + row == 0
            colour = (220, 229, 211) if heading else (187, 199, 181)
            screen.blit(mono.render(line, True, colour), (body.left, body.top + row * UI(LINE)))
        screen.set_clip(None)
        if len(self.lines) > shown:
            more = f"lines {self.scroll + 1}-{min(len(self.lines), self.scroll + shown)} of {len(self.lines)}   (wheel or UP/DOWN)"
            screen.blit(UI.font("segoeui", 12).render(more, True, (116, 139, 124)), (panel.left + UI(24), panel.bottom - UI(44)))
        close = layout["close"]
        pg.draw.rect(screen, (131, 205, 143), close, border_radius=UI(6))
        text = UI.font("segoeui", 13, bold=True).render("CLOSE", True, (19, 26, 27))
        screen.blit(text, text.get_rect(center=close.center))
