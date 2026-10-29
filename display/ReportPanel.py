"""The reporting panel: a whole world's report, read from a save or from a probe port, a tab for each part of it. A
save is opened by OPEN SAVE or dropped on the window; a running world by its probe's port (the C++ app's --probe PORT,
or the daemon's host:port). A running world is read again by REFRESH, or every LIVE_SECONDS while LIVE is on, and its
herds then show how they have drifted. COPY puts the whole report on the clipboard; SAVE writes it beside the app's
other reports. As the C++ version's Bitzkey Reports."""

from __future__ import annotations

import re
import threading
from pathlib import Path

import pygame as pg

from config.Globals import Config
from display import Window
from display.Language import AsShown
from display.Ui import UI, draw_label
from display.WorldFiles import choose_world_file
from tools import WorldReport

LIVE_SECONDS = 10
REPORT_DIR = Config.APP_DIR / "reports"

BACK = (22, 30, 30)
CARD = (30, 40, 40)
EDGE = (80, 99, 85)
TEXT = (187, 199, 181)
BRIGHT = (229, 239, 220)
DIM = (144, 170, 151)
GREEN = (131, 205, 143)
AMBER = (238, 183, 110)
RED = (230, 120, 105)
INK = (19, 26, 27)
# Each species' line on the chart; the plants' against their own scale, faintly.
SERIES = (("plant", (96, 160, 86)), ("grazer", (120, 185, 235)), ("omnivore", (236, 186, 96)), ("predator", (226, 92, 80)))
LINE = 18    # design pixels to a line of the report
GAP = 18     # design pixels between a table's columns
CHART = 230  # the population chart's height, on its tab


class ReportPanel:
    def __init__(self, screen: pg.Surface) -> None:
        self.screen = screen
        self.world: WorldReport.World | None = None
        self.sections: list[tuple[str, list[str]]] = []
        self.probe: WorldReport.Probe | None = None
        self.tab = 0
        self.scrolls: list[int] = []
        self.across = 0  # pixels scrolled right, for long lines
        self.address = ""
        self.focused = False
        self.live = False
        self.last_read = 0
        # Reading happens on a thread of its own, so a large save or a slow probe never stalls the window.
        self.reader: threading.Thread | None = None
        self.result: tuple | None = None  # (world, probe or None, error)
        self.doing = ""
        self.note, self.trouble, self.note_until = "", False, 0

    # ---- reading ----

    def _tell(self, note: str, trouble: bool = False) -> None:
        self.note, self.trouble = note, trouble
        self.note_until = pg.time.get_ticks() + (12000 if trouble else 6000)

    @property
    def busy(self) -> bool:
        return self.reader is not None

    def _begin(self, read, doing: str) -> None:
        if self.busy:
            return
        self.doing = doing

        def run() -> None:
            try:
                self.result = (*read(), "")
            except Exception as error:  # whatever went wrong is shown, and the window carries on
                self.result = (None, None, str(error) or type(error).__name__)

        self.reader = threading.Thread(target=run, daemon=True)
        self.reader.start()

    def _collect(self) -> None:
        if self.reader is None or self.reader.is_alive():
            return
        self.reader = None
        world, probe, error = self.result
        self.result = None
        self.last_read = pg.time.get_ticks()
        if error:
            self._tell(error, True)
            return
        new_source = probe is not self.probe or not world.live
        self.probe = probe
        if not world.live:
            self.live = False
        self.world = world
        self.sections = WorldReport.analyse(world)
        if new_source or len(self.scrolls) != len(self.sections):
            self.scrolls = [0] * len(self.sections)
            self.across = 0
        self.tab = min(self.tab, len(self.sections) - 1)
        self._tell(f"Read {world.name} at turn {world.state['turn']}.")

    def open_save(self, path: Path) -> None:
        self._begin(lambda: (WorldReport.read_save(path), None), f"Loading {Path(path).name}...")

    def connect(self, address: str) -> None:
        self.address = address

        def read():
            probe = WorldReport.Probe(address)
            return probe.read(), probe

        self._begin(read, f"Connecting to {address}...")

    def _refresh(self) -> None:
        if self.probe is None:
            self._tell("REFRESH reads a running world again: connect to a probe first." if self.world else
                       "Open a save or connect to a probe first.")
            return
        probe = self.probe
        self._begin(lambda: (probe.read(), probe), f"Reading {probe.name}...")

    def _copy(self) -> None:
        if not self.world:
            return self._tell("There's no report to copy yet.")
        try:
            pg.scrap.init()
            pg.scrap.put_text(WorldReport.as_text(self.sections))
        except pg.error as error:
            return self._tell(f"Couldn't copy: {error}", True)
        self._tell("The whole report is on the clipboard.")

    def _save(self) -> None:
        if not self.world:
            return self._tell("There's no report to save yet.")
        stem = re.sub(r"[^A-Za-z0-9_-]", "-", self.world.name)
        path = REPORT_DIR / f"world-report-{stem}-turn-{self.world.state['turn']:05d}.txt"
        try:
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(WorldReport.as_text(self.sections), encoding="utf-8")
        except OSError as error:
            return self._tell(f"Couldn't save: {error}", True)
        self._tell(f"Saved {path}")

    # ---- layout and input ----

    def _layout(self) -> dict:
        width, height = self.screen.get_size()
        top, button = UI(16), UI(36)
        places = {}
        x = width - UI(16)
        for name, w, gap in (("save", 70, 0), ("copy", 70, 0), ("live", 70, 0), ("refresh", 90, 16), ("connect", 96, 0),
                             ("address", 170, 0), ("open", 110, 0)):
            x -= UI(w)
            places[name] = pg.Rect(x, top, UI(w), button)
            x -= UI(8) + UI(gap)
        below = UI(104)
        places["tabs"] = [pg.Rect(UI(16), below + at * UI(42), UI(176), UI(36)) for at in range(len(self.sections) if self.world else 0)]
        content = pg.Rect(UI(208), below, width - UI(224), height - below - UI(16))
        chart = bool(self.world) and self.sections[self.tab][0] == "Population"
        places["chart"] = pg.Rect(content.left, content.top, content.width, UI(CHART)) if chart else None
        places["body"] = (pg.Rect(content.left, content.top + UI(CHART + 12), content.width, content.height - UI(CHART + 12))
                          if chart else content)
        return places

    def _lines_shown(self) -> int:
        return max(1, (self._layout()["body"].height - UI(24)) // UI(LINE))

    def _scroll_by(self, lines: int) -> None:
        if not self.world:
            return
        total = len(self.sections[self.tab][1])
        self.scrolls[self.tab] = max(0, min(total - self._lines_shown(), self.scrolls[self.tab] + lines))

    def _open_dialog(self) -> None:
        path = choose_world_file()
        if path:
            self.open_save(path)

    def _click(self, pos: tuple[int, int]) -> None:
        places = self._layout()
        self.focused = places["address"].collidepoint(pos)
        if places["open"].collidepoint(pos):
            self._open_dialog()
        elif places["connect"].collidepoint(pos):
            if self.address.strip():
                self.connect(self.address.strip())
            else:
                self.focused = True
                self._tell("Type the probe's port: the C++ app's --probe PORT, or probe.json's.")
        elif places["refresh"].collidepoint(pos):
            self._refresh()
        elif places["live"].collidepoint(pos):
            if self.probe is None:
                self._tell(f"LIVE reads a running world again every {LIVE_SECONDS} seconds: connect to a probe first.")
            else:
                self.live = not self.live
                self._tell(f"LIVE: read again every {LIVE_SECONDS} seconds." if self.live else "LIVE is off.")
        elif places["copy"].collidepoint(pos):
            self._copy()
        elif places["save"].collidepoint(pos):
            self._save()
        for at, tab in enumerate(places["tabs"]):
            if tab.collidepoint(pos):
                self.tab, self.across = at, 0

    def _key(self, event: pg.event.Event) -> None:
        control = event.mod & pg.KMOD_CTRL
        if self.focused:
            if event.key == pg.K_BACKSPACE:
                self.address = self.address[:-1]
            elif event.key in (pg.K_RETURN, pg.K_KP_ENTER):
                self.focused = False
                if self.address.strip():
                    self.connect(self.address.strip())
            elif event.key == pg.K_ESCAPE:
                self.focused = False
            elif control and event.key == pg.K_v:
                try:
                    pg.scrap.init()
                    pasted = pg.scrap.get_text() or ""
                except pg.error:
                    pasted = ""
                self.address = (self.address + pasted.strip())[:80]
            return
        if event.key == pg.K_F11:
            self.screen = Window.toggle_fullscreen()
        elif event.key == pg.K_F5:
            self._refresh()
        elif control and event.key == pg.K_c:
            self._copy()
        elif control and event.key == pg.K_s:
            self._save()
        elif control and event.key == pg.K_o:
            self._open_dialog()
        elif event.key in (pg.K_DOWN, pg.K_UP):
            self._scroll_by(1 if event.key == pg.K_DOWN else -1)
        elif event.key in (pg.K_PAGEDOWN, pg.K_SPACE):
            self._scroll_by(self._lines_shown() - 2)
        elif event.key == pg.K_PAGEUP:
            self._scroll_by(2 - self._lines_shown())
        elif event.key == pg.K_HOME:
            self._scroll_by(-10 ** 9)
        elif event.key == pg.K_END:
            self._scroll_by(10 ** 9)
        elif self.world and event.key in (pg.K_TAB, pg.K_RIGHT, pg.K_LEFT):
            back = event.key == pg.K_LEFT or (event.key == pg.K_TAB and event.mod & pg.KMOD_SHIFT)
            self.tab = (self.tab + (-1 if back else 1)) % len(self.sections)
            self.across = 0

    def run(self) -> None:
        clock = pg.time.Clock()
        pg.key.start_text_input()
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return
                if event.type == pg.VIDEORESIZE:
                    self.screen = Window.resize(event)
                elif event.type == pg.KEYDOWN:
                    self._key(event)
                elif event.type == pg.TEXTINPUT and self.focused:
                    self.address = (self.address + event.text)[:80]
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    self._click(event.pos)
                elif event.type == pg.MOUSEWHEEL:
                    if pg.key.get_mods() & pg.KMOD_SHIFT:
                        self.across = max(0, self.across - UI(40) * (event.y + event.x))
                    else:
                        self._scroll_by(-3 * event.y)
                        self.across = max(0, self.across + UI(40) * event.x)
                elif event.type == pg.DROPFILE:
                    self.open_save(Path(event.file))
            self._collect()
            if self.live and not self.busy and self.probe and pg.time.get_ticks() - self.last_read > LIVE_SECONDS * 1000:
                self._refresh()
            self._draw()
            pg.display.flip()
            clock.tick(30)

    # ---- drawing ----

    def _button(self, rect: pg.Rect, label: str, primary: bool = False) -> None:
        pg.draw.rect(self.screen, GREEN if primary else (44, 57, 52), rect, border_radius=UI(6))
        draw_label(self.screen, UI.font("segoeui", 13, bold=True), label, INK if primary else (236, 238, 232), rect)

    def _text(self, font, text: str, colour, position) -> int:
        with AsShown():  # the report is in English, as it was made
            image = font.render(text, True, colour)
        self.screen.blit(image, position)
        return image.get_width()

    def _draw(self) -> None:
        screen = self.screen
        screen.fill(BACK)
        places = self._layout()
        screen.blit(UI.font("segoeui", 22, bold=True).render("BITZKEY REPORTS", True, BRIGHT), (UI(18), UI(18)))
        self._button(places["open"], "OPEN SAVE")
        box = places["address"]
        pg.draw.rect(screen, INK, box, border_radius=UI(4))
        pg.draw.rect(screen, GREEN if self.focused else (54, 68, 62), box, 1, border_radius=UI(4))
        mono = UI.font("consolas", 14)
        shown = self.address + ("|" if self.focused and pg.time.get_ticks() // 500 % 2 else "")
        screen.set_clip(box.inflate(-UI(8), 0))
        if shown:
            width = mono.size(shown)[0]
            self._text(mono, shown, (238, 238, 220), (min(box.left + UI(8), box.right - UI(8) - width), box.centery - mono.get_height() // 2))
        else:
            self._text(UI.font("segoeui", 13), "port, or host:port", (110, 125, 115), (box.left + UI(8), box.centery - UI(9)))
        screen.set_clip(None)
        self._button(places["connect"], "CONNECT")
        self._button(places["refresh"], "REFRESH", self.probe is not None)
        self._button(places["live"], "LIVE ON" if self.live else "LIVE", self.live)
        self._button(places["copy"], "COPY")
        self._button(places["save"], "SAVE")

        now = pg.time.get_ticks()
        status, tone = "", DIM
        if self.busy:
            status, tone = self.doing, AMBER
        elif now < self.note_until:
            status, tone = self.note, RED if self.trouble else DIM
        elif self.world:
            status = f"{self.world.name}  ({'running' if self.world.live else 'save'})   turn {self.world.state['turn']}"
            if self.live:
                status += f"   LIVE every {LIVE_SECONDS} s"
        font = UI.fitting(UI.font("segoeui", 14), status, screen.get_width() - UI(36))
        self._text(font, status, tone, (UI(18), UI(66)))

        if not self.world:
            card = pg.Rect(0, 0, min(screen.get_width() - UI(40), UI(720)), UI(250))
            card.center = screen.get_rect().center
            pg.draw.rect(screen, CARD, card, border_radius=UI(8))
            pg.draw.rect(screen, EDGE, card, 1, border_radius=UI(8))
            y = card.top + UI(24)
            self._text(UI.font("segoeui", 18, bold=True), "A whole world's report", BRIGHT, (card.left + UI(24), y))
            y += UI(40)
            for line in ("OPEN SAVE, or drop a .bzworld file here, to report on a saved world.",
                         "Or type a probe's port and CONNECT, to report on a running world:",
                         "the C++ app's (start it with --probe PORT, or put {\"port\": 49200} in probe.json",
                         "in the Bitzkey folder in Documents), or the daemon's host:port.",
                         "", f"REFRESH reads it again; LIVE every {LIVE_SECONDS} seconds. COPY and SAVE take the report as text."):
                self._text(UI.font("segoeui", 14), line, TEXT, (card.left + UI(24), y))
                y += UI(24)
            return

        findings = any(line.startswith("! ") for line in self.sections[0][1])
        for at, tab in enumerate(places["tabs"]):
            chosen = at == self.tab
            pg.draw.rect(screen, GREEN if chosen else CARD, tab, border_radius=UI(6))
            font = UI.font("segoeui", 14, bold=chosen)
            self._text(font, self.sections[at][0], INK if chosen else TEXT, (tab.left + UI(12), tab.centery - font.get_height() // 2))
            if at == 0 and findings:  # the summary's tab is marked while anything stands out
                pg.draw.circle(screen, (150, 80, 20) if chosen else AMBER, (tab.right - UI(14), tab.centery), UI(4))
        if places["chart"]:
            self._draw_chart(places["chart"])
        self._draw_lines(places["body"])

    def _draw_chart(self, area: pg.Rect) -> None:
        screen = self.screen
        pg.draw.rect(screen, CARD, area, border_radius=UI(8))
        pg.draw.rect(screen, EDGE, area, 1, border_radius=UI(8))
        points = (self.world.history or {}).get("history", [])
        small = UI.font("segoeui", 12)
        if len(points) < 2:
            self._text(UI.font("segoeui", 14), "No history to chart yet.", DIM, (area.left + UI(16), area.top + UI(16)))
            return
        plot = pg.Rect(area.left + UI(56), area.top + UI(36), area.width - UI(72), area.height - UI(64))
        most_animals = max([1] + [point["alive"].get(species, 0) for point in points for species, _ in SERIES[1:]])
        most_plants = max([1] + [point["alive"].get("plant", 0) for point in points])
        first = points[0]["turn"]
        last = max(first + 1, points[-1]["turn"])
        for step in range(5):
            y = plot.bottom - plot.height * step // 4
            pg.draw.line(screen, (44, 58, 56), (plot.left, y), (plot.right, y))
            label = str(most_animals * step // 4)
            self._text(small, label, DIM, (plot.left - UI(8) - small.size(label)[0], y - small.get_height() // 2))
        self._text(small, f"turn {first}", DIM, (plot.left, plot.bottom + UI(6)))
        label = f"turn {last}"
        self._text(small, label, DIM, (plot.right - small.size(label)[0], plot.bottom + UI(6)))
        for species, colour in SERIES:
            plants = species == "plant"
            most = most_plants if plants else most_animals
            line = [(plot.left + (point["turn"] - first) * plot.width // (last - first),
                     plot.bottom - point["alive"].get(species, 0) * plot.height // most) for point in points]
            shade = tuple(channel * 6 // 10 + BACK[index] * 4 // 10 for index, channel in enumerate(colour)) if plants else colour
            pg.draw.lines(screen, shade, False, line, 1 if plants else max(1, UI(2)))
        x = area.left + UI(16)
        for species, colour in SERIES:
            label = f"{species}s {self.world.state['alive'].get(species, 0)}"
            if species == "plant":
                label += f" (own scale, to {most_plants})"
            pg.draw.rect(screen, colour, pg.Rect(x, area.top + UI(14), UI(14), UI(4)))
            x += UI(20)
            x += self._text(small, label, TEXT, (x, area.top + UI(8))) + UI(22)

    def _draw_lines(self, area: pg.Rect) -> None:
        screen = self.screen
        pg.draw.rect(screen, CARD, area, border_radius=UI(8))
        pg.draw.rect(screen, EDGE, area, 1, border_radius=UI(8))
        body = pg.Rect(area.left + UI(18), area.top + UI(12), area.width - UI(36), area.height - UI(24))
        lines = self.sections[self.tab][1]
        shown = self._lines_shown()
        scroll = self.scrolls[self.tab] = max(0, min(max(0, len(lines) - shown), self.scrolls[self.tab]))
        mono, heading, finding = UI.font("consolas", 13), UI.font("segoeui", 15, bold=True), UI.font("segoeui", 14)
        # Each table's columns are as wide as its widest cells, measured over the whole table, so they hold still as it
        # scrolls.
        table_of: list[int | None] = [None] * len(lines)
        widths: dict[int, list[int]] = {}
        at = 0
        with AsShown():
            while at < len(lines):
                if "\t" not in lines[at]:
                    at += 1
                    continue
                end = at
                columns: list[int] = []
                while end < len(lines) and "\t" in lines[end]:
                    cells = lines[end].split("\t")
                    cells[0] = cells[0].lstrip("~")
                    columns += [0] * (len(cells) - len(columns))
                    for c, cell in enumerate(cells):
                        columns[c] = max(columns[c], mono.size(cell)[0])
                    table_of[end] = at
                    end += 1
                widths[at] = columns
                at = end
        screen.set_clip(body)
        left = body.left - self.across
        for row in range(min(shown, len(lines) - scroll)):
            index = scroll + row
            line, y = lines[index], body.top + row * UI(LINE)
            if table_of[index] is None:
                if line.startswith("# "):
                    self._text(heading, line[2:], BRIGHT, (left, y - UI(1)))
                elif line.startswith("! "):
                    pg.draw.circle(screen, AMBER, (left + UI(5), y + UI(LINE) // 2), UI(4))
                    self._text(finding, line[2:], AMBER, (left + UI(18), y))
                elif line:
                    self._text(mono, line, TEXT, (left, y))
                continue
            # A table's row: its first cell to the left, the rest right-aligned in their columns; a last column of words
            # (as long as it is wider than a number) to the left.
            cells = line.split("\t")
            header = cells[0].startswith("~")
            cells[0] = cells[0].lstrip("~")
            columns = widths[table_of[index]]
            x = left
            for c, cell in enumerate(cells):
                words = c == 0 or (c + 1 == len(columns) and columns[c] > UI(120))
                with AsShown():
                    width = mono.size(cell)[0]
                self._text(mono, cell, GREEN if header else BRIGHT if c == 0 else TEXT, (x if words else x + columns[c] - width, y))
                x += columns[c] + UI(GAP)
        screen.set_clip(None)
        if len(lines) > shown:
            more = f"lines {scroll + 1}-{min(len(lines), scroll + shown)} of {len(lines)}"
            small = UI.font("segoeui", 12)
            self._text(small, more, DIM, (area.right - UI(14) - small.size(more)[0], area.bottom - UI(4) - small.get_height()))
