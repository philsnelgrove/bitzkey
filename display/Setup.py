"""The configuration panel shown before a world starts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pygame as pg

from config.Globals import Config
from display import Window
from display.Advanced import AdvancedScreen
from display.CatalogScreen import CatalogScreen
from display.Ui import UI
from display.Designer import Designer
from display.Dialog import Dialog
from tools.Exchange import Imported, format_records, import_bitzkeys
from tools.Genome import decode_body, genome_record
from tools.Simulation import MAX_GRID, MAX_IMPORTED, MAX_POPULATION, SPAWN_GROUP

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
    blank: str = ""         # what a blank optional setting shows


SETTINGS = (
    Setting("width", "Grid width", "cells across the world", 10, MAX_GRID, 10),
    Setting("height", "Grid height", "cells down the world", 10, MAX_GRID, 10),
    Setting("population", "Bitzkeys", "spawned at the start, a random mix of species", 0, MAX_POPULATION, 10),
    Setting("grass_density", "Grass density", "percent of cells with grass at the start; regrowth scales too",
            0, 100, 1),
    Setting("seed", "Random seed", "the same seed replays the same world; blank for a new one",
            0, 999_999_999, 1, optional=True, blank="random"),
    Setting("population_cap", "Population cap", "most animals alive at once; blank lets crowding alone decide",
            10, 99_999, 100, optional=True, blank="no limit"),
    Setting("mind_mutation", "Mind mutation", "mind mutations a birth carries: 0 none, 50 about 1.5, 100 three",
            0, 100, 5),
    Setting("body_mutation", "Body mutation", "percent chance a birth flips one body gene: 0 stable, 100 always",
            0, 100, 5),
    Setting("year_length", "Year length", "turns in a year, of four seasons; lifespans are counted in years",
            20, 3600, 20),
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

    def add_design(self, dna: int) -> None:
        """Add a designed genome to the bottom of the import text."""
        self.paste(format_records([genome_record(dna)]))

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

    ROW_HEIGHT = 46
    IMPORT_HEIGHT = 170
    TEXT_LINES = 5

    def __init__(self, screen: pg.Surface, defaults: Mapping[str, int | None]):
        self.screen = screen
        self.form = SetupForm(defaults)
        self.clock = pg.time.Clock()
        self.note = ""
        self.note_until = 0
        self.dialog: Dialog | None = None  # the quit confirmation, while it is open
        self.last_design: int | None = None  # the designer reopens on the last design applied
        self.rules: dict[str, float] = {}  # the advanced rules changed from their defaults

    # Fonts are sized for the 1200x820 design and scaled to the screen.

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
        return UI.font("consolas", 18)

    @property
    def mono_small(self) -> pg.font.Font:
        return UI.font("consolas", 12)

    @property
    def small(self) -> pg.font.Font:
        return UI.font("segoeui", 12)

    def run(self) -> dict | None:
        """Return the chosen settings, with the imported genomes under "imported" and the changed
        advanced rules under "rules", or None if the user quit. Running it again, after a
        simulation, starts from the same values."""
        self.screen = pg.display.get_surface() or self.screen
        while True:
            self.clock.tick(30)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return None
                if event.type == pg.VIDEORESIZE:
                    self.screen = Window.resize(event)
                elif self.dialog:
                    choice = self.dialog.handle(event, self.screen.get_size())
                    if choice == "quit":
                        return None
                    if choice:
                        self.dialog = None
                elif event.type == pg.DROPFILE:
                    self._load(event.file)
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        self._ask_to_quit()
                    elif event.key == pg.K_F11:
                        self.screen = Window.toggle_fullscreen()
                    elif event.key in (pg.K_RETURN, pg.K_KP_ENTER):
                        if self._ready():
                            return self._values()
                    elif event.key == pg.K_v and event.mod & pg.KMOD_CTRL:
                        self._paste()
                    else:
                        self._key(event)
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    action = self._click(event.pos)
                    if action == "quit":
                        self._ask_to_quit()
                    if action == "start" and self._ready():
                        return self._values()
            self._draw()
            pg.display.flip()

    def _values(self) -> dict:
        return {**self.form.values(), "rules": dict(self.rules)}

    def _ask_to_quit(self) -> None:
        self.dialog = Dialog("Quit Bitzkey?", ["The application will close."],
                             [("stay", "CANCEL"), ("quit", "QUIT")], cancel="stay")

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

    def _design(self) -> str | None:
        """Open the designer; an applied design joins the bottom of the import text. Returns
        "quit" if the window was closed meanwhile."""
        outcome, dna = Designer(self.screen, self.last_design).run()
        self.screen = pg.display.get_surface()
        if outcome == "quit":
            return "quit"
        if dna is not None:
            self.last_design = dna
            self.form.add_design(dna)
            self._tell(f"Added a designed {decode_body(dna)['species']} to the import list.")
        return None

    def _catalog(self) -> str | None:
        """Open the catalog; a Bitzkey picked there joins the bottom of the import text, brain and
        all. Returns "quit" if the window was closed meanwhile."""
        outcome, entry = CatalogScreen(self.screen).run()
        self.screen = pg.display.get_surface()
        if outcome == "quit":
            return "quit"
        if entry is not None:
            self.form.paste(format_records([entry]))
            self._tell("Added a Bitzkey from the catalog to the import list, with its trained brain.")
        return None

    def _advanced(self) -> str | None:
        """Open the advanced rules; applying them keeps them for the worlds started from here.
        Returns "quit" if the window was closed meanwhile."""
        outcome, rules = AdvancedScreen(self.screen, self.rules).run()
        self.screen = pg.display.get_surface()
        if outcome == "quit":
            return "quit"
        if rules is not None:
            self.rules = rules
            self._tell(f"{len(rules)} advanced rule{'s' if len(rules) != 1 else ''} changed from the defaults.")
        return None

    def start_with(self, text: str) -> dict | None:
        """Start a new world at once, with the settings as they were and these Bitzkeys alone in the
        import box, as if pasted there; if they can't be read, wait on the panel as usual."""
        self.form.clear_import()
        self.form.paste(text)
        return self._values() if self.form.can_start else self.run()

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
        UI.fit(self.screen.get_size())
        width, height = self.screen.get_size()
        settings = len(self.form.settings)
        card = pg.Rect(0, 0, UI(640), UI(110 + settings * self.ROW_HEIGHT + self.IMPORT_HEIGHT + 110))
        card.center = (width // 2, height // 2)
        rows = [pg.Rect(card.left + UI(20), card.top + UI(104 + index * self.ROW_HEIGHT), card.width - UI(40),
                        UI(self.ROW_HEIGHT - 6)) for index in range(settings)]
        section = pg.Rect(card.left + UI(20), rows[-1].bottom + UI(14), card.width - UI(40), UI(self.IMPORT_HEIGHT - 14))
        clear = pg.Rect(0, 0, UI(64), UI(28))
        clear.topright = (section.right, section.top)
        load = pg.Rect(0, 0, UI(92), UI(28))
        load.topright = (clear.left - UI(8), section.top)
        paste = pg.Rect(0, 0, UI(70), UI(28))
        paste.topright = (load.left - UI(8), section.top)
        design = pg.Rect(0, 0, UI(80), UI(28))
        design.topright = (paste.left - UI(8), section.top)
        catalog = pg.Rect(0, 0, UI(122), UI(28))
        catalog.topright = (design.left - UI(8), section.top)
        box = pg.Rect(section.left, section.top + UI(36), section.width, UI(16 * self.TEXT_LINES + 12))
        start = pg.Rect(0, 0, UI(180), UI(44))
        start.bottomleft = (card.centerx + UI(6), card.bottom - UI(42))
        quit = pg.Rect(0, 0, UI(140), UI(44))
        quit.bottomright = (card.centerx - UI(6), card.bottom - UI(42))
        advanced = pg.Rect(0, 0, UI(120), UI(28))
        advanced.topright = (card.right - UI(20), card.top + UI(28))
        return {"card": card, "rows": rows, "box": box, "catalog": catalog, "design": design, "paste": paste, "load": load,
                "clear": clear, "start": start, "quit": quit, "advanced": advanced}

    @staticmethod
    def _buttons(row: pg.Rect) -> tuple[pg.Rect, pg.Rect, pg.Rect]:
        """The minus button, value box and plus button of a row."""
        plus = pg.Rect(row.right - UI(44), row.centery - UI(15), UI(30), UI(30))
        box = pg.Rect(plus.left - UI(150), row.centery - UI(17), UI(142), UI(34))
        minus = pg.Rect(box.left - UI(38), row.centery - UI(15), UI(30), UI(30))
        return minus, box, plus

    def _click(self, position: tuple[int, int]) -> str | None:
        """Handle a click; return "start" when it asks to start the world, or "quit" if the
        window was closed from the designer or the advanced rules."""
        layout = self._layout()
        if layout["start"].collidepoint(position):
            return "start"
        if layout["quit"].collidepoint(position):
            return "quit"
        if layout["design"].collidepoint(position):
            return self._design()
        if layout["catalog"].collidepoint(position):
            return self._catalog()
        if layout["advanced"].collidepoint(position):
            return self._advanced()
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
        return None

    def _draw(self) -> None:
        self.screen.fill((19, 26, 27))
        layout = self._layout()
        card = layout["card"]
        pg.draw.rect(self.screen, (27, 37, 37), card, border_radius=UI(8))
        pg.draw.rect(self.screen, (80, 99, 85), card, 1, border_radius=UI(8))
        title = self.bold.render("BITZKEY", True, (229, 239, 220))
        self.screen.blit(title, (card.left + UI(28), card.top + UI(24)))
        self._text("NEW WORLD  /  CONFIGURATION", (card.left + UI(52) + title.get_width(), card.top + UI(33)),
                   (144, 170, 151), self.small)
        self._text("Set up the world, then start the simulation.", (card.left + UI(28), card.top + UI(66)),
                   (157, 177, 159), self.font)
        advanced = layout["advanced"]
        pg.draw.rect(self.screen, (238, 183, 110) if self.rules else (44, 57, 52), advanced, border_radius=UI(4))
        label = self.button_font.render("ADVANCED", True, (19, 26, 27) if self.rules else (220, 229, 211))
        self.screen.blit(label, label.get_rect(center=advanced.center))
        if self.rules:
            note = self.small.render(f"{len(self.rules)} rule{'s' if len(self.rules) != 1 else ''} changed", True,
                                     (238, 183, 110))
            self.screen.blit(note, note.get_rect(midtop=(advanced.centerx, advanced.bottom + UI(4))))

        for index, (setting, row) in enumerate(zip(self.form.settings, layout["rows"], strict=True)):
            self._draw_setting(setting, row, index == self.form.focus)
        self._draw_import(layout)

        start = layout["start"]
        ready = self.form.can_start
        pg.draw.rect(self.screen, (131, 205, 143) if ready else (70, 88, 76), start, border_radius=UI(6))
        label = self.bold.render("START", True, (19, 26, 27))
        self.screen.blit(label, label.get_rect(center=start.center))
        quit = layout["quit"]
        pg.draw.rect(self.screen, (44, 57, 52), quit, border_radius=UI(6))
        label = self.bold.render("QUIT", True, (220, 229, 211))
        self.screen.blit(label, label.get_rect(center=quit.center))
        if pg.time.get_ticks() < self.note_until:
            hint = self.small.render(self.note, True, (238, 183, 110))
        else:
            hint = self.small.render("UP/DOWN  choose   LEFT/RIGHT  adjust (SHIFT x10)   CTRL+V  paste   "
                                     "ENTER  start   F11  window   ESC  quit", True, (130, 152, 136))
        self.screen.blit(hint, hint.get_rect(midbottom=(card.centerx, card.bottom - UI(14))))
        if self.dialog:
            self.dialog.draw(self.screen)

    def _draw_setting(self, setting: Setting, row: pg.Rect, focused: bool) -> None:
        pg.draw.rect(self.screen, (36, 49, 47) if focused else (31, 42, 41), row, border_radius=UI(5))
        if focused:
            pg.draw.rect(self.screen, (131, 205, 143), row, 1, border_radius=UI(5))
        self._text(setting.label.upper(), (row.left + UI(16), row.top + UI(3)), (232, 236, 219), self.font)
        self._text(setting.hint, (row.left + UI(16), row.top + UI(22)), (130, 152, 136), self.small)
        minus, box, plus = self._buttons(row)
        for button, sign in ((minus, "-"), (plus, "+")):
            pg.draw.rect(self.screen, (44, 57, 52), button, border_radius=UI(4))
            label = self.bold.render(sign, True, (220, 229, 211))
            self.screen.blit(label, label.get_rect(center=button.center))
        pg.draw.rect(self.screen, (19, 26, 27), box, border_radius=UI(4))
        pg.draw.rect(self.screen, (131, 205, 143) if focused else (54, 68, 62), box, 1, border_radius=UI(4))
        text = self.form.texts[setting.key]
        shown, color = (text, (238, 238, 220)) if text else (setting.blank, (116, 139, 124))
        if focused and not self.form.fresh and pg.time.get_ticks() // 500 % 2:
            shown += "_"
        label = self.mono.render(shown, True, color)
        self.screen.blit(label, label.get_rect(midright=(box.right - UI(12), box.centery)))

    def _draw_import(self, layout: dict) -> None:
        box = layout["box"]
        focused = self.form.on_import
        self._text("IMPORT BITZKEYS", (box.left, layout["paste"].top + UI(5)), (232, 236, 219), self.font)
        for name, label in (("catalog", "FROM CATALOG"), ("design", "DESIGN"), ("paste", "PASTE"), ("load", "LOAD FILE"),
                            ("clear", "CLEAR")):
            button = layout[name]
            pg.draw.rect(self.screen, (44, 57, 52), button, border_radius=UI(4))
            text = self.button_font.render(label, True, (220, 229, 211))
            self.screen.blit(text, text.get_rect(center=button.center))

        pg.draw.rect(self.screen, (19, 26, 27), box, border_radius=UI(4))
        pg.draw.rect(self.screen, (131, 205, 143) if focused else (54, 68, 62), box, 1, border_radius=UI(4))
        lines = self.form.import_text.split("\n")
        if not self.form.import_text and not focused:
            lines, color = ["Paste exported genomes here (CTRL+V), drop a file onto the window,",
                            "load one, pick one FROM CATALOG, or DESIGN your own. Imported",
                            "Bitzkeys join the ones spawned above."], (116, 139, 124)
        else:
            lines, color = lines[-self.TEXT_LINES:], (187, 199, 181)
            if focused and pg.time.get_ticks() // 500 % 2:
                lines[-1] += "_"
        columns = (box.width - UI(20)) // max(1, self.mono_small.size("M")[0])
        for index, line in enumerate(lines):
            shown = line if len(line) <= columns else line[:columns - 1] + "~"
            self._text(shown, (box.left + UI(10), box.top + UI(7 + index * 16)), color, self.mono_small)

        genomes, error = self.form.imports
        if error:
            status, color = f"Can't import: {error}", (238, 183, 110)
        elif genomes:
            status, color = import_summary(genomes), (131, 205, 143)
        else:
            status, color = "Nothing to import.", (116, 139, 124)
        self._text(status, (box.left, box.bottom + UI(8)), color, self.small)

    def _text(self, text: str, position: tuple[int, int], color: tuple[int, int, int],
              font: pg.font.Font) -> None:
        self.screen.blit(font.render(text, True, color), position)
