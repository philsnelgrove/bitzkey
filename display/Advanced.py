"""The advanced panel: every tunable rule of the simulation, by group, opened from the
configuration panel, where applying it sets the rules the next world plays by, or from the map,
where it sets the rules of the world running there."""

from __future__ import annotations

from collections.abc import Mapping

import pygame as pg

from display import Window
from display.Ui import UI, fit_text, draw_label, wrapped_lines
from tools import Rules
from tools.Rules import RULES, Rule

MAX_CHARACTERS = 12


class AdvancedForm:
    """Every rule's value, the focused rule, and any number being typed into it."""

    def __init__(self, values: Mapping[str, float], locked: Mapping[str, str] | None = None):
        self.values = {rule.key: values.get(rule.key, Rules.DEFAULTS[rule.key]) for rule in RULES}
        self.locked = Rules.LOCKED if locked is None else locked  # rules not to be changed here, and why
        self.focus = 0
        self.typed: str | None = None  # partial input for the focused rule; None shows its value

    @property
    def focused(self) -> Rule:
        return RULES[self.focus]

    def text(self, rule: Rule) -> str:
        if rule is self.focused and self.typed is not None:
            return self.typed
        return Rules.shown(rule, self.values[rule.key])

    def changed(self) -> dict[str, float]:
        return Rules.changed(self.values)

    @property
    def problems(self) -> list[str]:
        return Rules.problems(self.values)

    def select(self, index: int) -> None:
        self._settle()
        self.focus = index % len(RULES)

    def move(self, offset: int) -> None:
        self.select(self.focus + offset)

    def adjust(self, steps: int) -> None:
        if self.focused.key in self.locked:  # not to be changed here
            return
        self._settle()
        rule = self.focused
        self.values[rule.key] = rule.clamp(round(self.values[rule.key] + steps * rule.step, 9))

    def type_text(self, text: str) -> None:
        """Digits, a decimal point for fractional rules, and a leading minus where values may be
        negative."""
        rule = self.focused
        if rule.key in self.locked:
            return
        typed = self.typed or ""
        for character in text:
            if (character.isdigit() or (character == "." and not rule.integer and "." not in typed)
                    or (character == "-" and rule.minimum < 0 and not typed)):
                typed += character
        self.typed = typed[:MAX_CHARACTERS] if typed else self.typed

    def backspace(self) -> None:
        if self.focused.key in self.locked:
            return
        self.typed = (self.text(self.focused))[:-1]

    def reset(self) -> None:
        """Put the focused rule back to its default."""
        self.typed = None
        self.values[self.focused.key] = Rules.DEFAULTS[self.focused.key]

    def reset_all(self) -> None:
        self.typed = None
        self.values = dict(Rules.DEFAULTS)

    def finish(self) -> None:
        self._settle()

    def _settle(self) -> None:
        """Replace whatever was typed with the in-range value it stands for; nothing usable
        leaves the value as it was."""
        if self.typed is not None:
            try:
                self.values[self.focused.key] = self.focused.clamp(float(self.typed))
            except ValueError:
                pass
            self.typed = None


NATURAL_LAW = ("These are the settings for what might be called 'natural law'.  They have been carefully balanced to produce a world with fair chances for all.  Messing with any of them can produce unpredictable results.  Have fun!")


class AdvancedScreen:
    """Runs the advanced panel until it is applied, cancelled or the window closes."""

    ROW_HEIGHT = 44
    HEADING_HEIGHT = 34

    def __init__(self, screen: pg.Surface, values: Mapping[str, float], running: bool = False):
        """`running`: the rules of the world running on the map, rather than of the next new world."""
        self.screen = screen
        self.running = running
        self.form = AdvancedForm(values, {**Rules.WORLD_LOCKED, **Rules.LOCKED} if running else None)
        self.scroll = 0
        self.clock = pg.time.Clock()
        self.note = ""
        self.note_until = 0
        # Each line of the list: a group heading, or the index of a rule, with its top in the list.
        self.lines: list[tuple[str | int, int]] = []
        top = 0
        for index, rule in enumerate(RULES):
            if index == 0 or rule.group != RULES[index - 1].group:
                self.lines.append((rule.group, top))
                top += self.HEADING_HEIGHT
            self.lines.append((index, top))
            top += self.ROW_HEIGHT
        self.list_height = top

    @property
    def font(self) -> pg.font.Font:
        return UI.font("segoeui", 15)

    @property
    def bold(self) -> pg.font.Font:
        return UI.font("segoeui", 22, bold=True)

    @property
    def button_font(self) -> pg.font.Font:
        return UI.font("segoeui", 13, bold=True)

    @property
    def mono(self) -> pg.font.Font:
        return UI.font("consolas", 16)

    @property
    def small(self) -> pg.font.Font:
        return UI.font("segoeui", 12)

    def run(self) -> tuple[str, dict[str, float] | None]:
        """Return ("apply", the changed rules), ("cancel", None) or ("quit", None)."""
        while True:
            self.clock.tick(30)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return "quit", None
                if event.type == pg.VIDEORESIZE:
                    self.screen = Window.resize(event)
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        return "cancel", None
                    if event.key == pg.K_F11:
                        self.screen = Window.toggle_fullscreen()
                    elif event.key in (pg.K_RETURN, pg.K_KP_ENTER):
                        if self._ready():
                            return "apply", self.form.changed()
                    else:
                        self._key(event)
                elif event.type == pg.MOUSEWHEEL:
                    self._scroll_by(-event.y * UI(self.ROW_HEIGHT))
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    action = self._click(event.pos)
                    if action == "cancel":
                        return "cancel", None
                    if action == "apply" and self._ready():
                        return "apply", self.form.changed()
            self._draw()
            pg.display.flip()

    def _ready(self) -> bool:
        self.form.finish()
        problems = self.form.problems
        if problems:
            self._tell(problems[0] + ".")
        return not problems

    def _tell(self, note: str) -> None:
        self.note = note
        self.note_until = pg.time.get_ticks() + 5000

    def _key(self, event: pg.event.Event) -> None:
        form = self.form
        steps = 10 if event.mod & pg.KMOD_SHIFT else 1
        if event.key == pg.K_UP or (event.key == pg.K_TAB and event.mod & pg.KMOD_SHIFT):
            form.move(-1)
        elif event.key in (pg.K_DOWN, pg.K_TAB):
            form.move(1)
        elif event.key == pg.K_PAGEUP:
            form.select(max(0, form.focus - 10))
        elif event.key == pg.K_PAGEDOWN:
            form.select(min(len(RULES) - 1, form.focus + 10))
        elif event.key == pg.K_RIGHT:
            form.adjust(steps)
        elif event.key == pg.K_LEFT:
            form.adjust(-steps)
        elif event.key in (pg.K_BACKSPACE, pg.K_DELETE):
            form.backspace()
        elif event.unicode and event.unicode.isprintable():
            form.type_text(event.unicode)
        else:
            return
        self._show_focus()

    def _layout(self) -> dict[str, pg.Rect]:
        UI.fit(self.screen.get_size())
        width, height = self.screen.get_size()
        card = pg.Rect(0, 0, UI(760), min(height - UI(24), UI(800)))
        card.center = (width // 2, height // 2)
        top = UI(100) + len(self._law_lines(card.width - UI(56))) * UI(self.LAW_LINE)  # beneath the word on natural law
        view = pg.Rect(card.left + UI(20), card.top + top, card.width - UI(40), card.height - top - UI(118))
        apply = pg.Rect(0, 0, UI(180), UI(44))
        apply.bottomleft = (card.centerx + UI(6), card.bottom - UI(42))
        cancel = pg.Rect(0, 0, UI(140), UI(44))
        cancel.bottomright = (card.centerx - UI(6), card.bottom - UI(42))
        reset_all = pg.Rect(0, 0, UI(100), UI(28))
        reset_all.topright = (card.right - UI(20), card.top + UI(28))
        reset = pg.Rect(0, 0, UI(70), UI(28))
        reset.topright = (reset_all.left - UI(8), reset_all.top)
        return {"card": card, "view": view, "apply": apply, "cancel": cancel, "reset": reset, "reset_all": reset_all}

    LAW_LINE = 20  # design pixels between the lines of the word on natural law

    def _law_lines(self, width: int) -> list[str]:
        """The word on natural law, wrapped to fit the width."""
        return wrapped_lines(self.font, NATURAL_LAW, width)

    def _max_scroll(self, view: pg.Rect) -> int:
        return max(0, UI(self.list_height) - view.height)

    def _scroll_by(self, amount: int) -> None:
        self.scroll = max(0, min(self._max_scroll(self._layout()["view"]), self.scroll + amount))

    def _show_focus(self) -> None:
        """Scroll just far enough to show the focused rule, and its heading when it opens a group."""
        view = self._layout()["view"]
        position = next(i for i, (item, _) in enumerate(self.lines) if item == self.form.focus)
        top = self.lines[position - 1][1] if isinstance(self.lines[position - 1][0], str) else self.lines[position][1]
        bottom = UI(self.lines[position][1] + self.ROW_HEIGHT)
        if UI(top) < self.scroll:
            self.scroll = UI(top)
        elif bottom > self.scroll + view.height:
            self.scroll = bottom - view.height
        self.scroll = max(0, min(self._max_scroll(view), self.scroll))

    def _rows(self, view: pg.Rect) -> list[tuple[str | int, pg.Rect]]:
        """The lines that show in the view, with their rectangles on screen."""
        shown = []
        for item, top in self.lines:
            height = self.HEADING_HEIGHT if isinstance(item, str) else self.ROW_HEIGHT
            rect = pg.Rect(view.left, view.top + UI(top) - self.scroll, view.width - UI(14), UI(height - 4))
            if rect.bottom > view.top and rect.top < view.bottom:
                shown.append((item, rect))
        return shown

    @staticmethod
    def _buttons(row: pg.Rect) -> tuple[pg.Rect, pg.Rect, pg.Rect]:
        """The minus button, value box and plus button of a row."""
        plus = pg.Rect(row.right - UI(40), row.centery - UI(13), UI(26), UI(26))
        box = pg.Rect(plus.left - UI(128), row.centery - UI(15), UI(120), UI(30))
        minus = pg.Rect(box.left - UI(34), row.centery - UI(13), UI(26), UI(26))
        return minus, box, plus

    def _click(self, position: tuple[int, int]) -> str | None:
        layout = self._layout()
        for name in ("apply", "cancel"):
            if layout[name].collidepoint(position):
                return name
        if layout["reset"].collidepoint(position):
            self.form.reset()
        elif layout["reset_all"].collidepoint(position):
            self.form.reset_all()
        elif layout["view"].collidepoint(position):
            for item, row in self._rows(layout["view"]):
                if isinstance(item, int) and row.collidepoint(position):
                    if item != self.form.focus:
                        self.form.select(item)
                    minus, _, plus = self._buttons(row)
                    if minus.collidepoint(position):
                        self.form.adjust(-1)
                    elif plus.collidepoint(position):
                        self.form.adjust(1)
        return None

    def _draw(self) -> None:
        self.screen.fill((19, 26, 27))
        layout = self._layout()
        card = layout["card"]
        pg.draw.rect(self.screen, (27, 37, 37), card, border_radius=UI(8))
        pg.draw.rect(self.screen, (80, 99, 85), card, 1, border_radius=UI(8))
        title = self.bold.render("BITZKEY", True, (229, 239, 220))
        self.screen.blit(title, (card.left + UI(28), card.top + UI(24)))
        self._text("THIS WORLD  /  ADVANCED RULES" if self.running else "NEW WORLD  /  ADVANCED RULES",
                   (card.left + UI(52) + title.get_width(), card.top + UI(33)),
                   (144, 170, 151), self.small)
        lines = self._law_lines(card.width - UI(56))
        for index, line in enumerate(lines):
            self._text(line, (card.left + UI(28), card.top + UI(66) + index * UI(self.LAW_LINE)), (229, 239, 220), self.font)
        changed = len(self.form.changed())
        whose = "This world plays by these from its next turn." if self.running else "The rules the world plays by."
        note = f"{whose} {changed} changed from the defaults; changed values show in amber."
        self._text(note, (card.left + UI(28), card.top + UI(66) + len(lines) * UI(self.LAW_LINE) + UI(6)), (157, 177, 159),
                   UI.fitting(self.font, note, card.width - UI(56)))
        for name, label in (("reset", "RESET"), ("reset_all", "RESET ALL")):
            button = layout[name]
            pg.draw.rect(self.screen, (44, 57, 52), button, border_radius=UI(4))
            draw_label(self.screen, self.button_font, label, (220, 229, 211), button)

        view = layout["view"]
        self.scroll = min(self.scroll, self._max_scroll(view))
        self.screen.set_clip(view)
        for item, row in self._rows(view):
            if isinstance(item, str):
                self._text(item.upper(), (row.left + UI(4), row.bottom - UI(22)), (131, 205, 143), self.button_font)
            else:
                self._draw_rule(RULES[item], row, item == self.form.focus)
        self.screen.set_clip(None)
        self._draw_scroll_bar(view)

        for name, label, color, ink in (("apply", "APPLY", (131, 205, 143), (19, 26, 27)),
                                        ("cancel", "CANCEL", (44, 57, 52), (220, 229, 211))):
            button = layout[name]
            pg.draw.rect(self.screen, color, button, border_radius=UI(6))
            draw_label(self.screen, self.bold, label, ink, button)
        warning = Rules.WARNINGS.get(RULES[self.form.focus].key)
        if pg.time.get_ticks() < self.note_until:
            hint = self.small.render(self.note, True, (238, 183, 110))
        elif warning:  # the chosen rule's warning, in full
            hint = self.small.render(self._fit(warning, card.width - UI(48)), True, (238, 183, 110))
        else:
            hint = self.small.render("UP/DOWN  choose   LEFT/RIGHT  adjust (SHIFT x10)   PGUP/PGDN  jump   "
                                     "ENTER  apply   ESC  cancel", True, (130, 152, 136))
        self.screen.blit(hint, hint.get_rect(midbottom=(card.centerx, card.bottom - UI(14))))

    def _draw_rule(self, rule: Rule, row: pg.Rect, focused: bool) -> None:
        pg.draw.rect(self.screen, (36, 49, 47) if focused else (31, 42, 41), row, border_radius=UI(5))
        if focused:
            pg.draw.rect(self.screen, (131, 205, 143), row, 1, border_radius=UI(5))
        minus, box, plus = self._buttons(row)
        self._text(rule.label.upper(), (row.left + UI(14), row.top + UI(3)),
                   (120, 134, 124) if rule.key in self.form.locked else (232, 236, 219), self.font)
        lock = self.form.locked.get(rule.key)  # not to be changed here: greyed, saying why
        warned = not lock and rule.key in Rules.WARNINGS  # a cost to heed: its hint in amber, the warning in full when chosen
        self._text(self._fit(lock or ("WARNING  " if warned else "") + rule.hint, minus.left - row.left - UI(28)),
                   (row.left + UI(14), row.top + UI(21)),
                   (96, 110, 100) if lock else (238, 183, 110) if warned else (130, 152, 136), self.small)
        for button, sign in ((minus, "-"), (plus, "+")):
            pg.draw.rect(self.screen, (34, 44, 42) if lock else (44, 57, 52), button, border_radius=UI(4))
            draw_label(self.screen, self.bold, sign, (96, 110, 100) if lock else (220, 229, 211), button)
        pg.draw.rect(self.screen, (19, 26, 27), box, border_radius=UI(4))
        pg.draw.rect(self.screen, (131, 205, 143) if focused else (54, 68, 62), box, 1, border_radius=UI(4))
        shown = self.form.text(rule)
        changed = self.form.values[rule.key] != Rules.DEFAULTS[rule.key]
        if focused and self.form.typed is not None and pg.time.get_ticks() // 500 % 2:
            shown += "_"
        label = self.mono.render(shown, True, (96, 110, 100) if rule.key in self.form.locked
                                 else (238, 183, 110) if changed else (238, 238, 220))
        self.screen.blit(label, label.get_rect(midright=(box.right - UI(10), box.centery)))

    def _draw_scroll_bar(self, view: pg.Rect) -> None:
        track = pg.Rect(view.right - UI(8), view.top, UI(6), view.height)
        pg.draw.rect(self.screen, (36, 49, 47), track, border_radius=UI(3))
        total = UI(self.list_height)
        if total <= view.height:
            return
        length = max(UI(24), track.height * view.height // total)
        top = track.top + (track.height - length) * self.scroll // self._max_scroll(view)
        pg.draw.rect(self.screen, (80, 99, 85), (track.left, top, track.width, length), border_radius=UI(3))

    def _fit(self, text: str, width: int) -> str:
        return fit_text(self.small, text, width)

    def _text(self, text: str, position: tuple[int, int], color: tuple[int, int, int],
              font: pg.font.Font) -> None:
        self.screen.blit(font.render(text, True, color), position)
