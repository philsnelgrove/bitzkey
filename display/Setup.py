"""The configuration panel shown before a world starts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pygame as pg

from config.Globals import Config
from tools.Exchange import Imported, import_bitzkeys
from tools.Genome import decode_body
from tools.Simulation import MAX_IMPORTED, MAX_POPULATION, SPAWN_GROUP

EXPORT_DIR = Config.APP_DIR / "exports"
MAX_IMPORT_BYTES = 5_000_000


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
    Setting("grass_density", "Grass density", "percent of cells with grass at the start; regrowth scales too",
            0, 100, 1),
    Setting("seed", "Random seed", "the same seed replays the same world; blank for a new one",
            0, 999_999_999, 1, optional=True),
)
MAX_DIGITS = 9


class SetupForm:
    """The panel's state, independent of drawing: one text value per setting, the import text,
    and which of them has focus (the import box comes after the settings)."""

    def __init__(self, defaults: Mapping[str, int | None], settings: tuple[Setting, ...] = SETTINGS):
        self.settings = settings
        self.texts = {setting.key: "" if defaults.get(setting.key) is None else str(defaults[setting.key])
                      for setting in settings}
        self.import_text = ""
        self._parsed: tuple[str, list[int], str | None] = ("", [], None)
        self.focus = 0
        self.fresh = True  # the next digit typed replaces the value rather than extending it
        for setting in settings:
            self._settle(setting)

    @property
    def on_import(self) -> bool:
        return self.focus == len(self.settings)

    @property
    def focused(self) -> Setting | None:
        return None if self.on_import else self.settings[self.focus]

    def value(self, setting: Setting) -> int | None:
        text = self.texts[setting.key]
        if not text:
            return None if setting.optional else setting.minimum
        return max(setting.minimum, min(setting.maximum, int(text)))

    @property
    def imports(self) -> tuple[list[int], str | None]:
        """The genomes in the import text, and an error message if it cannot all be read."""
        if self._parsed[0] != self.import_text:
            try:
                self._parsed = (self.import_text, import_bitzkeys(self.import_text), None)
            except ValueError as error:
                self._parsed = (self.import_text, [], str(error))
        return self._parsed[1], self._parsed[2]

    @property
    def can_start(self) -> bool:
        return self.imports[1] is None

    def values(self) -> dict:
        values: dict = {setting.key: self.value(setting) for setting in self.settings}
        values["imported"] = self.imports[0][:MAX_IMPORTED]
        return values

    def select(self, index: int) -> None:
        if self.focused:
            self._settle(self.focused)
        self.focus = index % (len(self.settings) + 1)
        self.fresh = True

    def move(self, offset: int) -> None:
        self.select(self.focus + offset)

    def adjust(self, steps: int) -> None:
        setting = self.focused
        if setting is None:
            return
        current = self.value(setting)
        start = setting.minimum if current is None else current + steps * setting.step
        self.texts[setting.key] = str(max(setting.minimum, min(setting.maximum, start)))
        self.fresh = True

    def type_text(self, text: str) -> None:
        """Typed characters go into the import box when it has focus; digits into a setting."""
        if self.on_import:
            self.import_text = (self.import_text + text)[:MAX_IMPORT_BYTES]
            return
        for digit in filter(str.isdigit, text):
            key = self.focused.key
            typed = digit if self.fresh else self.texts[key] + digit
            self.texts[key] = (typed.lstrip("0") or "0")[:MAX_DIGITS]
            self.fresh = False

    def backspace(self) -> None:
        if self.on_import:
            self.import_text = self.import_text[:-1]
            return
        key = self.focused.key
        self.texts[key] = "" if self.fresh else self.texts[key][:-1]
        self.fresh = False

    def paste(self, text: str) -> None:
        """Add pasted text to the import box, on a new line, and focus it."""
        self.select(len(self.settings))
        separator = "\n" if self.import_text and not self.import_text.endswith("\n") else ""
        self.type_text(separator + text)

    def load(self, text: str) -> None:
        """Replace the import text with a file's contents, and focus it."""
        self.select(len(self.settings))
        self.import_text = text[:MAX_IMPORT_BYTES]

    def clear_import(self) -> None:
        self.import_text = ""

    def _settle(self, setting: Setting) -> None:
        """Replace whatever was typed with the in-range value it stands for."""
        value = self.value(setting)
        self.texts[setting.key] = "" if value is None else str(value)


def import_summary(bitzkeys: list[Imported]) -> str:
    kept = bitzkeys[:MAX_IMPORTED]
    species = Counter(str(decode_body(bitzkey.dna)["species"]) for bitzkey in kept)
    parts = ", ".join(f"{count} {name}{'s' if count != 1 else ''}" for name, count in sorted(species.items()))
    brains = sum(bitzkey.brain is not None for bitzkey in kept)
    minds = sum(bitzkey.mind is not None for bitzkey in kept)
    smallest, largest = SPAWN_GROUP
    summary = (f"{len(kept)} Bitzkey{'s' if len(kept) != 1 else ''} ready to import ({parts}; "
               f"{brains} with saved brains, {minds} with saved temperaments), "
               f"each spawning as a group of {smallest}-{largest}")
    if len(bitzkeys) > MAX_IMPORTED:
        summary += f". Only the first {MAX_IMPORTED} of {len(bitzkeys)} are used"
    return summary


class SetupScreen:
    """Draws the form and handles input until the user starts a world or quits."""

    ROW_HEIGHT = 56
    IMPORT_HEIGHT = 170
    TEXT_LINES = 5

    def __init__(self, screen: pg.Surface, defaults: Mapping[str, int | None]):
        self.screen = screen
        self.form = SetupForm(defaults)
        self.font = pg.font.SysFont("segoeui", 15)
        self.bold = pg.font.SysFont("segoeui", 22, bold=True)
        self.button_font = pg.font.SysFont("segoeui", 13, bold=True)
        self.mono = pg.font.SysFont("consolas", 18)
        self.mono_small = pg.font.SysFont("consolas", 12)
        self.small = pg.font.SysFont("segoeui", 12)
        self.clock = pg.time.Clock()
        self.note = ""
        self.note_until = 0

    def run(self) -> dict | None:
        """Return the chosen settings, with the imported genomes under "imported", or None if
        the user quit."""
        while True:
            self.clock.tick(30)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return None
                if event.type == pg.VIDEORESIZE:
                    self.screen = pg.display.set_mode((max(860, event.w), max(760, event.h)), pg.RESIZABLE)
                elif event.type == pg.DROPFILE:
                    self._load(event.file)
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        return None
                    if event.key in (pg.K_RETURN, pg.K_KP_ENTER):
                        if self._ready():
                            return self.form.values()
                    elif event.key == pg.K_v and event.mod & pg.KMOD_CTRL:
                        self._paste()
                    else:
                        self._key(event)
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    if self._click(event.pos) and self._ready():
                        return self.form.values()
            self._draw()
            pg.display.flip()

    def _ready(self) -> bool:
        if not self.form.can_start:
            self._tell("Fix or clear the import text before starting.")
        return self.form.can_start

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
        elif event.key == pg.K_RIGHT:
            form.adjust(steps)
        elif event.key == pg.K_LEFT:
            form.adjust(-steps)
        elif event.key in (pg.K_BACKSPACE, pg.K_DELETE):
            form.backspace()
        elif event.unicode and event.unicode.isprintable():
            form.type_text(event.unicode)

    def _paste(self) -> None:
        try:
            text = pg.scrap.get_text()
        except pg.error:
            text = ""
        if text:
            self.form.paste(text)
        else:
            self._tell("The clipboard has no text to paste.")

    def _choose_file(self) -> None:
        try:
            import tkinter
            from tkinter import filedialog
        except ImportError:
            self._tell("No file picker is available; drag a file onto the window instead.")
            return
        root = tkinter.Tk()
        root.withdraw()
        try:
            path = filedialog.askopenfilename(
                title="Import Bitzkeys", initialdir=str(EXPORT_DIR) if EXPORT_DIR.exists() else None,
                filetypes=[("Bitzkey genomes", "*.json *.txt"), ("All files", "*.*")])
        finally:
            root.destroy()
        if path:
            self._load(path)

    def _load(self, path: str) -> None:
        file = Path(path)
        try:
            if file.stat().st_size > MAX_IMPORT_BYTES:
                self._tell(f"{file.name} is too large to import.")
                return
            self.form.load(file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as error:
            self._tell(f"Couldn't read {file.name}: {error}")
            return
        self._tell(f"Loaded {file.name}.")

    def _layout(self) -> dict[str, pg.Rect | list[pg.Rect]]:
        width, height = self.screen.get_size()
        settings = len(self.form.settings)
        card = pg.Rect(0, 0, 640, 110 + settings * self.ROW_HEIGHT + self.IMPORT_HEIGHT + 110)
        card.center = (width // 2, height // 2)
        rows = [pg.Rect(card.left + 20, card.top + 104 + index * self.ROW_HEIGHT, card.width - 40,
                        self.ROW_HEIGHT - 6) for index in range(settings)]
        section = pg.Rect(card.left + 20, rows[-1].bottom + 14, card.width - 40, self.IMPORT_HEIGHT - 14)
        clear = pg.Rect(0, 0, 64, 28)
        clear.topright = (section.right, section.top)
        load = pg.Rect(0, 0, 92, 28)
        load.topright = (clear.left - 8, section.top)
        paste = pg.Rect(0, 0, 70, 28)
        paste.topright = (load.left - 8, section.top)
        box = pg.Rect(section.left, section.top + 36, section.width, 16 * self.TEXT_LINES + 12)
        start = pg.Rect(0, 0, 180, 44)
        start.midbottom = (card.centerx, card.bottom - 42)
        return {"card": card, "rows": rows, "box": box, "paste": paste, "load": load, "clear": clear,
                "start": start}

    @staticmethod
    def _buttons(row: pg.Rect) -> tuple[pg.Rect, pg.Rect, pg.Rect]:
        """The minus button, value box and plus button of a row."""
        plus = pg.Rect(row.right - 44, row.centery - 15, 30, 30)
        box = pg.Rect(plus.left - 150, row.centery - 17, 142, 34)
        minus = pg.Rect(box.left - 38, row.centery - 15, 30, 30)
        return minus, box, plus

    def _click(self, position: tuple[int, int]) -> bool:
        """Handle a click; return True when it asks to start the world."""
        layout = self._layout()
        if layout["start"].collidepoint(position):
            return True
        if layout["paste"].collidepoint(position):
            self._paste()
        elif layout["load"].collidepoint(position):
            self._choose_file()
        elif layout["clear"].collidepoint(position):
            self.form.clear_import()
        elif layout["box"].collidepoint(position):
            self.form.select(len(self.form.settings))
        for index, row in enumerate(layout["rows"]):
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
        layout = self._layout()
        card = layout["card"]
        pg.draw.rect(self.screen, (27, 37, 37), card, border_radius=8)
        pg.draw.rect(self.screen, (80, 99, 85), card, 1, border_radius=8)
        self._text("BITZKEY", (card.left + 28, card.top + 24), (229, 239, 220), self.bold)
        self._text("NEW WORLD  /  CONFIGURATION", (card.left + 143, card.top + 33), (144, 170, 151), self.small)
        self._text("Set up the world, then start the simulation.", (card.left + 28, card.top + 66),
                   (157, 177, 159), self.font)

        for index, (setting, row) in enumerate(zip(self.form.settings, layout["rows"], strict=True)):
            self._draw_setting(setting, row, index == self.form.focus)
        self._draw_import(layout)

        start = layout["start"]
        ready = self.form.can_start
        pg.draw.rect(self.screen, (131, 205, 143) if ready else (70, 88, 76), start, border_radius=6)
        label = self.bold.render("START", True, (19, 26, 27))
        self.screen.blit(label, label.get_rect(center=start.center))
        if pg.time.get_ticks() < self.note_until:
            hint = self.small.render(self.note, True, (238, 183, 110))
        else:
            hint = self.small.render("UP/DOWN  choose    LEFT/RIGHT  adjust (SHIFT x10)    type a number    "
                                     "CTRL+V  paste    ENTER  start    ESC  quit", True, (130, 152, 136))
        self.screen.blit(hint, hint.get_rect(midbottom=(card.centerx, card.bottom - 14)))

    def _draw_setting(self, setting: Setting, row: pg.Rect, focused: bool) -> None:
        pg.draw.rect(self.screen, (36, 49, 47) if focused else (31, 42, 41), row, border_radius=5)
        if focused:
            pg.draw.rect(self.screen, (131, 205, 143), row, 1, border_radius=5)
        self._text(setting.label.upper(), (row.left + 16, row.top + 7), (232, 236, 219), self.font)
        self._text(setting.hint, (row.left + 16, row.top + 28), (130, 152, 136), self.small)
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

    def _draw_import(self, layout: dict) -> None:
        box = layout["box"]
        focused = self.form.on_import
        self._text("IMPORT BITZKEYS", (box.left, layout["paste"].top + 5), (232, 236, 219), self.font)
        for name, label in (("paste", "PASTE"), ("load", "LOAD FILE"), ("clear", "CLEAR")):
            button = layout[name]
            pg.draw.rect(self.screen, (44, 57, 52), button, border_radius=4)
            text = self.button_font.render(label, True, (220, 229, 211))
            self.screen.blit(text, text.get_rect(center=button.center))

        pg.draw.rect(self.screen, (19, 26, 27), box, border_radius=4)
        pg.draw.rect(self.screen, (131, 205, 143) if focused else (54, 68, 62), box, 1, border_radius=4)
        lines = self.form.import_text.split("\n")
        if not self.form.import_text and not focused:
            lines, color = ["Paste exported genomes here (CTRL+V), drop a file onto the window,",
                            "or load one. Imported Bitzkeys join the ones spawned above."], (116, 139, 124)
        else:
            lines, color = lines[-self.TEXT_LINES:], (187, 199, 181)
            if focused and pg.time.get_ticks() // 500 % 2:
                lines[-1] += "_"
        columns = (box.width - 20) // max(1, self.mono_small.size("M")[0])
        for index, line in enumerate(lines):
            shown = line if len(line) <= columns else line[:columns - 1] + "~"
            self._text(shown, (box.left + 10, box.top + 7 + index * 16), color, self.mono_small)

        genomes, error = self.form.imports
        if error:
            status, color = f"Can't import: {error}", (238, 183, 110)
        elif genomes:
            status, color = import_summary(genomes), (131, 205, 143)
        else:
            status, color = "Nothing to import.", (116, 139, 124)
        self._text(status, (box.left, box.bottom + 8), color, self.small)

    def _text(self, text: str, position: tuple[int, int], color: tuple[int, int, int],
              font: pg.font.Font) -> None:
        self.screen.blit(font.render(text, True, color), position)
