"""The colour graph popup: every Bitzkey alive over time, stacked in its own colour.

Each bar is a moment in the run; each Bitzkey alive then is a slice of it in its exact ring or
dot colour, stacked by species and, within a species, sorted so similar colours flow together, or
by a trait chosen from the SORT menu. Hovering over a slice shows that Bitzkey's details.
Bitzkeys are picked by clicking their slice or dragging a box around slices, then copied or saved;
the pick stays selected on the map when the graph closes. The graph also travels in time: drag
the timeline beneath it, or right-click, to move the world to that turn. The mouse wheel zooms
into a stretch of the run and, as on the map, into the height around the pointer; dragging with
the right or middle button pans, and Z shows it all again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pygame as pg

from display.Ui import UI, draw_label, fit_text as shortened
from tools.Census import colour_order
from display.Icons import draw_icon
from display.Inspector import draw_tooltip
from tools.Genome import (BODY_TRAITS, MIND_TRAITS, SOCIAL_TRAITS, SPECIES, TEMPERAMENT, body_word, color_from_word,
                          decode_body, decode_climate, decode_temperament, mind_word, social_word)
from tools import Simulation as world
from tools.Simulation import Bitzkey

if TYPE_CHECKING:
    from tools.Simulation import Simulation

BAR_PIXELS = 5         # target width of one bar
DRAG_THRESHOLD = 4     # a press that moves less than this (design pixels, scaled) is a click
DIM = 0.75             # how far unselected Bitzkeys fade toward the background
FUTURE_DIM = 0.55      # how far turns after the present fade, once the world has been rewound
MIN_VIEW = 10          # the fewest turns the graph zooms in to
ZOOM_STEP = 0.8        # each wheel notch shows this share of the turns before it
HEIGHT_ZOOM_MAX = 32.0  # the closest the graph zooms into its height, times the whole
BACKGROUND = (30, 40, 40)
PARTS = ("ring", "dot", "social")
TITLES = {"ring": "RING / BODY", "dot": "DOT / MIND", "social": "SOCIAL BACKGROUND"}
# What a species' Bitzkeys are sorted by: their colour, or an inborn trait or climate gene, lowest at
# the bottom (for the home temperature, hottest at the bottom).
CLIMATE_SORTS = ("temperature", "heat tolerance", "cold tolerance")
SORTS = ("colour", *MIND_TRAITS, *SOCIAL_TRAITS, *BODY_TRAITS[1:], *CLIMATE_SORTS)
YEAR_LABEL_SPACING = 56  # design pixels the year labels on the timeline keep apart, at least
MENU_ROW = 18  # design pixels per item of the sort menu: small enough for every trait to fit


@dataclass
class Bar:
    turn: int   # the last turn the bar covers; its Bitzkeys are those alive then
    left: int
    right: int
    slices: list[tuple[int, float, float]] = field(default_factory=list)  # (identifier, top, bottom)
    future: bool = False  # after the present, when the world has been rewound


def word_of(dna: int, part: str) -> int:
    """The genome's word for a part: the ring, the dot or the social background."""
    return body_word(dna) if part == "ring" else mind_word(dna) if part == "dot" else social_word(dna)


def colour_of(dna: int, part: str) -> tuple[int, int, int]:
    return color_from_word(word_of(dna, part))


def faded(colour: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(round(channel + (back - channel) * amount) for channel, back in zip(colour, BACKGROUND))


def nice_ceiling(value: float) -> float:
    """The smallest 1, 2 or 5 times a power of ten at or above the value."""
    scale = 1
    while True:
        for step in (1, 2, 5):
            if step * scale >= value:
                return step * scale
        scale *= 10


def sort_label(sort: str) -> str:
    return "Colour value" if sort == "colour" else sort.capitalize()


def trait_of(dna: int, trait: str) -> int:
    """An inborn trait, mind, social or body, or a climate gene, from the genome."""
    if trait in CLIMATE_SORTS:
        return int(decode_climate(dna)[CLIMATE_SORTS.index(trait)])
    return decode_temperament(dna)[trait] if trait in TEMPERAMENT else int(decode_body(dna)[trait])


def build_bars(simulation: Simulation, part: str, chart: pg.Rect, first: int = 0,
               last: int | None = None, sort: str = "colour", height_zoom: float = 1.0,
               floor: float = 0.0) -> tuple[list[Bar], float]:
    """Lay out one bar per stretch of turns from `first` to `last` (by default the whole run),
    each stacking the Bitzkeys alive at its end; the bar holding the present shows it exactly.
    Within a species they are sorted by colour, or by the trait `sort` names and then colour; a
    trait sorts only the living, the dead being remembered by their colour alone, beneath them.
    Zoomed into the height, the chart shows `1 / height_zoom` of it, from `floor` (a share of the
    top) up; slices beyond it lie outside the chart. Returns the bars and the population the
    chart tops at, unzoomed."""
    last = simulation.turn if last is None else last
    turns = last - first + 1
    count = max(1, min(chart.width // BAR_PIXELS, turns))
    now = simulation.turn
    present = tuple(agent.identifier for agent in simulation.living)
    alive = set(present)
    lineage = simulation.lineage
    moments = []
    for index in range(count):
        start, end = first + index * turns // count, first + (index + 1) * turns // count - 1
        members = present if start <= now <= end else simulation.census.at(end)
        moments.append((end, members, start > now))
    peak = nice_ceiling(max(len(members) for _, members, _ in moments))
    unit = chart.height * height_zoom / peak
    base = chart.bottom + floor * height_zoom * chart.height  # where none would stand

    keys: dict[int, tuple] = {}

    def order(identifier: int) -> tuple:  # by species, plants at the bottom, then by the sort
        if identifier not in keys:
            dna = lineage[identifier]
            trait = 0 if sort == "colour" else trait_of(dna, sort) if identifier in alive else -1
            keys[identifier] = (SPECIES.index(decode_body(dna)["species"]), trait, colour_order(colour_of(dna, part)),
                                word_of(dna, part), identifier)
        return keys[identifier]

    bars = []
    for index, (end, members, future) in enumerate(moments):
        bar = Bar(end, chart.left + index * chart.width // count, chart.left + (index + 1) * chart.width // count,
                  future=future)
        bottom = base
        for identifier in sorted(members, key=order):
            bar.slices.append((identifier, bottom - unit, bottom))
            bottom -= unit
        bars.append(bar)
    return bars, peak


def death_text(simulation: Simulation, identifier: int) -> str:
    """How a Bitzkey died: its cause, and what killed it and what that was doing, when something did."""
    death = simulation.death_records.get(identifier)
    if death is None:
        return "dead"
    return f"died: {death.cause}" + ("" if death.killer is None else
                                     f" by #{death.killer:03d} {death.killer_species}, {death.activity}")


def death_note(simulation: Simulation, identifier: int) -> str:
    """How a dead Bitzkey died, as far as it is known: the turn, the cause, and what killed it, doing what."""
    death = simulation.death_records.get(identifier)
    if death is None:
        return "Dead: only its genome is known."
    killer = "" if death.killer is None else f" by #{death.killer:03d} {death.killer_species}, {death.activity}"
    return f"Died at turn {death.turn}: {death.cause}{killer}."


def fit_text(text: str, font: pg.font.Font, width: int) -> str:
    """The text, cut short with an ellipsis if it is wider than `width`."""
    return shortened(font, text, width)


def bitzkey_at(bars: list[Bar], point: tuple[int, int]) -> int | None:
    x, y = point
    for bar in bars:
        if bar.left <= x < bar.right:
            for identifier, top, bottom in bar.slices:
                if top <= y < bottom:
                    return identifier
    return None


def bitzkeys_in(bars: list[Bar], box: pg.Rect) -> set[int]:
    """Every Bitzkey with a slice touching the box, in any bar it spans."""
    found = set()
    for bar in bars:
        if bar.right > box.left and bar.left < box.right:
            found.update(identifier for identifier, top, bottom in bar.slices
                         if bottom > box.top and top < box.bottom)
    return found


def updated(selection: set[int], picked: set[int], mode: str) -> set[int]:
    if mode == "add":
        return selection | picked
    if mode == "remove":
        return selection - picked
    if mode == "toggle":
        return selection ^ picked
    return set(picked)


class ColourGraph:
    def __init__(self) -> None:
        self.part = "ring"
        self.sort = "colour"
        self.menu_open = False  # the sort menu
        self.selection: set[int] = set()
        self.view: tuple[int, int] | None = None  # the turns shown when zoomed in; None shows the run
        self.pinned = False  # the zoomed view reaches the latest turn, and keeps up with it as the run goes on
        self.height_zoom = 1.0  # how far zoomed into the height
        self.floor = 0.0  # the share of the chart's full height at its bottom, zoomed in
        self.ended = False                        # whether the run is over, when last drawn
        self._selection_version = 0
        self._size = (0, 0)
        self._mouse = (0, 0)
        self._now = 0
        self._press: tuple[tuple[int, int], int] | None = None  # where a press began, and its modifiers
        # A pan with the right or middle button: where it began, the button, the view and floor then, and
        # whether it has moved far enough to be a pan rather than a right-click.
        self._pan: tuple[tuple[int, int], int, tuple[int, int], float, bool] | None = None
        self._bars: list[Bar] = []
        self._places: dict[int, list[tuple[int, int, float, float]]] = {}  # where each Bitzkey is drawn
        self._peak = 1.0
        self._image: pg.Surface | None = None
        self._alive: set[int] = set()  # who is alive, as last drawn
        self._key: tuple | None = None

    @property
    def font(self) -> pg.font.Font:
        return UI.font("segoeui", 15)

    @property
    def bold(self) -> pg.font.Font:
        return UI.font("segoeui", 22, bold=True)

    @property
    def small(self) -> pg.font.Font:
        return UI.font("segoeui", 12)

    def switch_part(self) -> None:
        self.part = PARTS[(PARTS.index(self.part) + 1) % len(PARTS)]

    def select(self, picked: set[int], mode: str = "replace") -> None:
        self.selection = updated(self.selection, picked, mode)
        self._selection_version += 1

    def reset(self) -> None:
        """Forget the selection and zoom, for a new world."""
        self.select(set())
        self.unzoom()

    def unzoom(self) -> None:
        """The whole run, its whole height."""
        self.view, self.pinned = None, False
        self.height_zoom, self.floor = 1.0, 0.0

    # Time

    def window(self) -> tuple[int, int]:
        """The first and last turn shown."""
        if self.view is None:
            return 0, self._now
        if self.pinned and self.view[1] != self._now:  # kept at the latest turn, the living in view
            span = self.view[1] - self.view[0]
            self.view = (max(0, self._now - span), self._now)
        first, last = self.view
        return max(0, first), min(self._now, last)

    def turn_at(self, x: int) -> int:
        chart = self.layout()["chart"]
        first, last = self.window()
        turns = last - first + 1
        return max(first, min(last, first + (x - chart.left) * turns // max(1, chart.width)))

    def x_of(self, turn: int) -> int:
        chart = self.layout()["chart"]
        first, last = self.window()
        return round(chart.left + (turn - first + 0.5) * chart.width / (last - first + 1))

    def zoom(self, notches: int, x: int) -> None:
        """Zoom in (positive notches) or out around the turn at `x`, or, while the view reaches the
        latest turn, keeping it there; zooming out to the whole run returns to it."""
        first, last = self.window()
        span = last - first + 1
        whole = self._now + 1
        new_span = max(MIN_VIEW, round(span * ZOOM_STEP ** notches))
        if new_span >= whole:
            self.view = None
            return
        if self.view is None or self.pinned:  # reaching the latest turn, it stays pinned there, the living in view
            self.view, self.pinned = (whole - new_span, whole - 1), True
            return
        centre = self.turn_at(x)
        new_first = round(centre - (centre - first) * new_span / span)
        new_first = max(0, min(whole - new_span, new_first))
        self.view = (new_first, new_first + new_span - 1)
        self.pinned = self.view[1] >= self._now

    def zoom_height(self, notches: int, y: int) -> None:
        """Zoom into the height (positive notches) or out, as the map zooms, the population under `y`
        staying put."""
        chart = self.layout()["chart"]
        up = (chart.bottom - y) / max(1, chart.height)  # the pointer's height in the chart, as a share
        share = self.floor + up / self.height_zoom
        self.height_zoom = max(1.0, min(HEIGHT_ZOOM_MAX, self.height_zoom * (1 / ZOOM_STEP) ** notches))
        self._set_floor(share - up / self.height_zoom)

    def _set_floor(self, floor: float) -> None:
        self.floor = max(0.0, min(1 - 1 / self.height_zoom, floor))

    def drag(self, start: tuple[int, int], view: tuple[int, int], floor: float, to: tuple[int, int]) -> None:
        """Pan by dragging from `start` to `to`, the view then `view` and the floor `floor`: the graph
        follows the pointer, in time while zoomed into it and in height while zoomed into that."""
        chart = self.layout()["chart"]
        if self.view is not None:
            first, last = view
            span = last - first + 1
            shift = round((to[0] - start[0]) * span / max(1, chart.width))
            new_first = max(0, min(self._now + 1 - span, first - shift))
            self.view = (new_first, new_first + span - 1)
            self.pinned = self.view[1] >= self._now
        self._set_floor(floor + (to[1] - start[1]) / max(1, chart.height) / self.height_zoom)

    def pan(self, notches: int) -> None:
        """Slide the zoomed view a tenth of its width per notch, later for positive notches."""
        if self.view is None:
            return
        first, last = self.window()
        span = last - first + 1
        new_first = max(0, min(self._now + 1 - span, first + notches * max(1, span // 10)))
        self.view = (new_first, new_first + span - 1)
        self.pinned = self.view[1] >= self._now

    # Input

    def layout(self) -> dict[str, pg.Rect]:
        width, height = self._size
        panel = pg.Rect(0, 0, min(UI(940), width - UI(60)), min(UI(620), height - UI(90)))
        panel.center = (width // 2, height // 2)
        chart = pg.Rect(panel.left + UI(64), panel.top + UI(92), panel.width - UI(92), panel.height - UI(196))
        timeline = pg.Rect(chart.left, chart.bottom + UI(4), chart.width, UI(24))
        buttons = {}
        right = panel.right - UI(24)
        for name, width in (("save", 76), ("copy", 76), ("clear", 76), ("sort", 170)):
            buttons[name] = pg.Rect(right - UI(width), panel.top + UI(18), UI(width), UI(30))
            right -= UI(width + 8)
        right = panel.right - UI(24)  # beneath the timeline, at the right: what the run's end offers
        for name, width in self._bottom_buttons():
            buttons[name] = pg.Rect(right - UI(width), timeline.bottom + UI(6), UI(width), UI(28))
            right -= UI(width + 8)
        # At the bottom right, on the line of help: the catalog, to look through.
        buttons["open"] = pg.Rect(panel.right - UI(24) - UI(130), panel.bottom - UI(33), UI(130), UI(26))
        return {"panel": panel, "chart": chart, "timeline": timeline, **buttons}

    def _bottom_buttons(self) -> list[tuple[str, int]]:
        """The buttons beneath the timeline, from the right, and their widths: at the run's end, a way
        back to the map; and always, ways to catalog the Bitzkeys picked and to start a new world with them."""
        return ([("map", 140)] if self.ended else []) + [("catalog", 150), ("renew", 310)]

    def menu_items(self) -> list[tuple[str, pg.Rect]]:
        """The sort menu's choices under its button, a small gap before the mind traits, the social
        traits, the body traits and the climate genes."""
        button = self.layout()["sort"]
        items, top = [], button.bottom + UI(2)
        for index, sort in enumerate(SORTS):
            if index in (1, 1 + len(MIND_TRAITS), 1 + len(TEMPERAMENT), len(SORTS) - len(CLIMATE_SORTS)):
                top += UI(6)
            items.append((sort, pg.Rect(button.left, top, button.width, UI(MENU_ROW))))
            top += UI(MENU_ROW)
        return items

    def handle(self, event: pg.event.Event) -> str | None:
        """Handle a mouse event. Returns "copy" or "save" when a button asks for an export, "catalog"
        when it asks to catalog the Bitzkeys picked,
        and "map" when the run is over and the map is wanted."""
        layout = self.layout()
        chart, timeline = layout["chart"], layout["timeline"]
        if self.menu_open and event.type == pg.MOUSEBUTTONDOWN:  # a click picks a choice or closes the menu
            self.menu_open = False
            if event.button == 1:
                self.sort = next((sort for sort, item in self.menu_items() if item.collidepoint(event.pos)), self.sort)
            return None
        if event.type == pg.MOUSEMOTION:
            self._mouse = event.pos
            if self._pan is not None:
                start, button, view, floor, moved = self._pan
                if moved or max(abs(event.pos[0] - start[0]), abs(event.pos[1] - start[1])) >= UI(DRAG_THRESHOLD):
                    self._pan = (start, button, view, floor, True)
                    self.drag(start, view, floor, event.pos)
        elif event.type == pg.MOUSEWHEEL:
            if (chart.collidepoint(self._mouse) or timeline.collidepoint(self._mouse)):
                if event.x:
                    self.pan(event.x)
                elif pg.key.get_mods() & pg.KMOD_SHIFT:
                    self.pan(-event.y)
                else:
                    self.zoom(event.y, self._mouse[0])
                    if chart.collidepoint(self._mouse):
                        self.zoom_height(event.y, self._mouse[1])
        elif event.type == pg.MOUSEBUTTONDOWN and event.button in (2, 3):
            if chart.collidepoint(event.pos) or timeline.collidepoint(event.pos):
                self._pan = (event.pos, event.button, self.window(), self.floor, False)
        elif event.type == pg.MOUSEBUTTONUP and event.button in (2, 3) and self._pan is not None:
            self._pan = None
        elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            for action in ("copy", "save", "map", "catalog", "renew", "open"):
                if action in layout and layout[action].collidepoint(event.pos):
                    return action
            if layout["sort"].collidepoint(event.pos):
                self.menu_open = True
            elif layout["clear"].collidepoint(event.pos):
                self.select(set())
            elif chart.collidepoint(event.pos):
                self._press = (event.pos, pg.key.get_mods())
        elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
            if self._press:
                (start, modifiers), self._press = self._press, None
                shift, ctrl = modifiers & pg.KMOD_SHIFT, modifiers & pg.KMOD_CTRL
                if max(abs(event.pos[0] - start[0]), abs(event.pos[1] - start[1])) < UI(DRAG_THRESHOLD):
                    hit = bitzkey_at(self._bars, start)
                    picked = set() if hit is None else {hit} & self._alive  # only the living can be picked
                    self.select(picked, "remove" if ctrl else "toggle" if shift else "replace")
                else:
                    box = self._box(start, event.pos).clip(chart)
                    self.select(bitzkeys_in(self._bars, box) & self._alive,
                                "remove" if ctrl else "add" if shift else "replace")
        return None

    @staticmethod
    def _box(start: tuple[int, int], end: tuple[int, int]) -> pg.Rect:
        return pg.Rect(min(start[0], end[0]), min(start[1], end[1]),
                       abs(end[0] - start[0]) + 1, abs(end[1] - start[1]) + 1)

    # Drawing

    def draw(self, screen: pg.Surface, simulation: Simulation, note: str | None) -> None:
        self._size = screen.get_size()
        self._alive = {agent.identifier for agent in simulation.living}
        if self.selection - self._alive:  # the dead are forgotten but for their colour
            self.select(self.selection & self._alive)
        self._now = simulation.turn
        self.ended = simulation.ended
        layout = self.layout()
        panel, chart, timeline = layout["panel"], layout["chart"], layout["timeline"]
        shade = pg.Surface(self._size, pg.SRCALPHA)
        shade.fill((8, 12, 12, 175))
        screen.blit(shade, (0, 0))
        pg.draw.rect(screen, BACKGROUND, panel, border_radius=UI(6))
        pg.draw.rect(screen, (80, 99, 85), panel, 1, border_radius=UI(6))
        title = TITLES[self.part]
        heading = f"COLOUR GRAPH  /  {title}"  # clear of the buttons
        self._text(screen, heading, (panel.left + UI(24), panel.top + UI(18)), (229, 239, 220),
                   UI.fitting(self.bold, heading, layout["sort"].left - panel.left - UI(36)))
        closing = "new world" if self.ended else "close"
        self._text(screen, "Every Bitzkey alive over time, in its own colour.    TAB  ring / dot / social    C  copy    "
                   f"S  save    K  catalog    G or ESC  {closing}",
                   (panel.left + UI(24), panel.top + UI(54)), (144, 170, 151), self.small)
        for name, text in (("clear", "CLEAR"), ("copy", "COPY"), ("save", "SAVE")):
            pg.draw.rect(screen, (44, 57, 52), layout[name], border_radius=UI(4))
            draw_label(screen, self.font, text, (220, 229, 211), layout[name])
        self._draw_sort_button(screen, layout["sort"])
        if note:
            message = self.small.render(note, True, (238, 183, 110))
            screen.blit(message, message.get_rect(topright=(panel.right - UI(24), panel.top + UI(58))))

        first, last = self.window()
        key = (simulation.turn, len(simulation.census.snapshots), len(simulation.agents),
               simulation.deaths, self.part, chart.size, chart.topleft, self._selection_version, first, last,
               id(simulation), self.sort, self.height_zoom, self.floor)
        if key != self._key:
            self._key = key
            self._bars, self._peak = build_bars(simulation, self.part, chart, first, last, self.sort,
                                                self.height_zoom, self.floor)
            self._places = {}
            for bar in self._bars:
                for identifier, top, bottom in bar.slices:
                    self._places.setdefault(identifier, []).append((bar.left, bar.right, top, bottom))
            self._image = self._render(simulation, chart)
        screen.blit(self._image, chart.topleft)
        self._label_species(screen, simulation, chart)

        for fraction in (0, 0.5, 1):
            y = round(chart.bottom - fraction * chart.height)
            value = self._peak * (self.floor + fraction / self.height_zoom)
            label = self.small.render(f"{value:g}" if self.height_zoom == 1 else f"{value:.0f}", True, (149, 170, 150))
            screen.blit(label, (chart.left - label.get_width() - UI(8), y - label.get_height() // 2))

        self._draw_timeline(screen, chart, timeline, first, last)

        hovered = (bitzkey_at(self._bars, self._mouse)
                   if chart.collidepoint(self._mouse) and not self.menu_open else None)
        screen.set_clip(chart.inflate(0, 2))  # zoomed into the height, slices may lie beyond the chart
        for left, right, top, bottom in self._places.get(hovered, ()):
            pg.draw.rect(screen, (245, 238, 206), (left, round(top) - 1, right - left, max(1, round(bottom - top)) + 2), 1)
        screen.set_clip(None)
        if self._press:
            box = self._box(self._press[0], self._mouse).clip(chart)
            veil = pg.Surface(box.size, pg.SRCALPHA)
            veil.fill((245, 238, 206, 40))
            screen.blit(veil, box.topleft)
            pg.draw.rect(screen, (245, 238, 206), box, 1)

        room = min([layout[name].left for name, _ in self._bottom_buttons()], default=panel.right) - panel.left - UI(40)
        self._text(screen, fit_text(self._describe(simulation, hovered), self.font, room),
                   (panel.left + UI(24), timeline.bottom + UI(10)), (220, 229, 211), self.font)
        for name, text in (("map", "RETURN TO MAP"), ("catalog", "ADD TO CATALOG"),
                           ("renew", "START A NEW WORLD WITH THE SELECTED"), ("open", "OPEN CATALOG")):
            if name in layout:
                lit = name == "map"
                pg.draw.rect(screen, (238, 183, 110) if lit else (44, 57, 52), layout[name], border_radius=UI(4))
                draw_label(screen, self.font, text, (19, 26, 27) if lit else (220, 229, 211), layout[name])
        hint = ("Select: click, or drag a box (SHIFT adds, CTRL removes)    Time: drag the timeline, "
                "or right-click    Zoom: wheel    Pan: right-drag    Z: all")
        self._text(screen, hint, (panel.left + UI(24), panel.bottom - UI(28)), (144, 170, 151),
                   UI.fitting(self.small, hint, layout["open"].left - panel.left - UI(36)))
        if self.menu_open:
            self._draw_menu(screen)
        elif hovered in self._alive and not self._press:
            agent = next(item for item in simulation.agents if item.identifier == hovered)
            draw_tooltip(screen, agent, True, self._mouse, simulation.temperature(agent.y))
        elif hovered is not None and not self._press:  # the dead: what the genome fixes, and how it died
            remembered = Bitzkey(hovered, simulation.lineage[hovered], 0, 0)
            remembered.age = round(remembered.lifespan / 2)  # grown, not feeble: its genome's body
            draw_tooltip(screen, remembered, False, self._mouse, note=death_note(simulation, hovered))

    def _draw_sort_button(self, screen: pg.Surface, button: pg.Rect) -> None:
        pg.draw.rect(screen, (44, 57, 52), button, border_radius=UI(4))
        if self.menu_open or self.sort != "colour":
            pg.draw.rect(screen, (131, 205, 143), button, 1, border_radius=UI(4))
        label = self.small.render(f"SORT  {sort_label(self.sort).upper()}", True, (220, 229, 211))
        screen.blit(label, label.get_rect(midleft=(button.left + UI(10), button.centery)))
        x, y = button.right - UI(14), button.centery
        pg.draw.polygon(screen, (220, 229, 211), [(x - UI(5), y - UI(2)), (x + UI(5), y - UI(2)), (x, y + UI(4))])

    def _draw_menu(self, screen: pg.Surface) -> None:
        items = self.menu_items()
        back = items[0][1].unionall([item for _, item in items[1:]]).inflate(UI(4), UI(4))
        pg.draw.rect(screen, (19, 26, 27), back, border_radius=UI(4))
        pg.draw.rect(screen, (80, 99, 85), back, 1, border_radius=UI(4))
        for sort, item in items:
            if item.collidepoint(self._mouse):
                pg.draw.rect(screen, (44, 57, 52), item)
            color = (131, 205, 143) if sort == self.sort else (220, 229, 211)
            label = self.small.render(sort_label(sort), True, color)
            screen.blit(label, label.get_rect(midleft=(item.left + UI(10), item.centery)))

    def _label_species(self, screen: pg.Surface, simulation: Simulation, chart: pg.Rect) -> None:
        """Name each species' band, with its icon, inside the right end of the chart, where its
        latest bar is tall enough to hold the label."""
        if not self._bars:
            return
        bands: dict[str, list[float]] = {}
        for identifier, top, bottom in self._bars[-1].slices:
            species = str(decode_body(simulation.lineage[identifier])["species"])
            band = bands.setdefault(species, [top, bottom])
            band[0], band[1] = min(band[0], top), max(band[1], bottom)
        font = self.small
        for species, (top, bottom) in bands.items():
            top, bottom = max(top, chart.top), min(bottom, chart.bottom)  # the part in view
            if bottom - top < font.get_height():
                continue
            label = font.render(f"{species.upper()}S", True, (238, 238, 220))
            icon = font.get_height()
            box = pg.Rect(0, 0, label.get_width() + icon + UI(14), label.get_height() + UI(4))
            box.midright = (chart.right - UI(6), round((top + bottom) / 2))
            backing = pg.Surface(box.size, pg.SRCALPHA)
            backing.fill((19, 26, 27, 190))
            screen.blit(backing, box.topleft)
            draw_icon(screen, species, (box.left + UI(4) + icon // 2, box.centery), icon * 0.42, (238, 238, 220))
            screen.blit(label, (box.left + UI(8) + icon, box.top + UI(2)))

    def _draw_timeline(self, screen: pg.Surface, chart: pg.Rect, timeline: pg.Rect, first: int, last: int) -> None:
        """The timeline strip with the turns shown, its winters shaded and its years marked, a
        playhead at the present, and while dragging, a marker where the world will go."""
        pg.draw.rect(screen, (22, 30, 30), timeline, border_radius=UI(4))
        self._draw_seasons(screen, chart, timeline, first, last)
        pg.draw.rect(screen, (54, 68, 62), timeline, 1, border_radius=UI(4))
        if self.view is None:
            middle = f"TURN {(first + last) // 2}"
        else:
            middle = f"TURNS {first} - {last}  (ZOOM x{(self._now + 1) / (last - first + 1):.1f})"
        if self.height_zoom > 1:
            middle += f"  (HEIGHT x{self.height_zoom:.1f})"
        for text, x, align in ((f"TURN {first}", timeline.left + UI(8), 0), (middle, timeline.centerx, 0.5),
                               (f"TURN {last}", timeline.right - UI(8), 1)):
            label = self.small.render(text, True, (149, 170, 150))
            screen.blit(label, (x - round(label.get_width() * align), timeline.centery - label.get_height() // 2))

        for turn, colour in ((self._now, (245, 238, 206)),):
            if turn is None or not first <= turn <= last:
                continue
            x = self.x_of(turn)
            pg.draw.line(screen, colour, (x, chart.top), (x, timeline.bottom - 1), max(1, UI(2)))
            handle = pg.Rect(0, 0, UI(10), timeline.height)
            handle.midtop = (x, timeline.top)
            pg.draw.rect(screen, colour, handle, border_radius=UI(3))
            label = self.small.render(f"TURN {turn}", True, colour)
            tag = label.get_rect(midbottom=(x, chart.top - UI(2)))
            tag.clamp_ip(chart.inflate(0, UI(40)))
            screen.blit(label, tag)

    def _draw_seasons(self, screen: pg.Surface, chart: pg.Rect, timeline: pg.Rect, first: int, last: int) -> None:
        """Shade each winter quarter on the timeline, and mark where each year begins with a line
        up through the chart, labelled while the labels have room."""
        length = world.YEAR_LENGTH
        span = chart.width / (last - first + 1)  # pixels a turn
        left_of = lambda turn: round(chart.left + (turn - first) * span)  # noqa: E731
        quarter = first * 4 // length
        while quarter * length // 4 <= last:
            if quarter % 4 in (1, 3):
                start = max(first, -(-quarter * length // 4))
                end = min(last + 1, -(-(quarter + 1) * length // 4))
                if end > start:
                    pg.draw.rect(screen, (30, 42, 56), (left_of(start), timeline.top + 1, left_of(end) - left_of(start),
                                                        timeline.height - 2))
            quarter += 1
        every = max(1, -(-UI(YEAR_LABEL_SPACING) // max(1, round(length * span))))
        for turn in range(-(-max(1, first) // length) * length, last + 1, length):
            x = left_of(turn)
            pg.draw.line(screen, (88, 104, 112), (x, chart.top), (x, timeline.bottom - 1), 1)
            if (turn // length) % every == 0:
                label = self.small.render(f"YEAR {turn // length + 1}", True, (140, 185, 230))
                screen.blit(label, (x + UI(3), chart.top + UI(2)))

    def _describe(self, simulation: Simulation, hovered: int | None) -> str:
        living = {agent.identifier for agent in simulation.living}
        chosen = f"{len(self.selection)} selected" if self.selection else "Nothing selected"
        if self.sort != "colour":
            chosen = f"The living sorted by {self.sort}, lowest at the bottom.     " + chosen
        if simulation.end_reason:
            cause = "every animal died" if simulation.end_reason == "extinct" else "the world filled up"
            chosen = f"Ended at turn {simulation.turn}: {cause}.     " + chosen
        if hovered is None:
            return chosen
        dna = simulation.lineage[hovered]
        if hovered not in living:  # remembered by its colour, and how it died
            return (f"{chosen}     |     #{hovered:03d}  {decode_body(dna)['species']}  {death_text(simulation, hovered)}   "
                    f"ring {body_word(dna):08X}   dot {mind_word(dna):08X}   social {social_word(dna):08X}")
        return (f"{chosen}     |     #{hovered:03d}  {decode_body(dna)['species']}  alive   "
                f"ring {body_word(dna):08X}   dot {mind_word(dna):08X}   social {social_word(dna):08X}")

    def _render(self, simulation: Simulation, chart: pg.Rect) -> pg.Surface:
        """Draw the bars once into an image; slices too thin to see merge with their neighbours.
        Turns after the present, when rewound, are faded."""
        image = pg.Surface(chart.size)
        image.fill(BACKGROUND)
        for fraction in (0.5, 1):
            y = round(chart.height - fraction * chart.height)
            pg.draw.line(image, (54, 68, 62), (0, y), (chart.width, y), 1)
        pg.draw.line(image, (54, 68, 62), (0, chart.height - 1), (chart.width, chart.height - 1), 1)
        lineage, choosing = simulation.lineage, bool(self.selection)
        for bar in self._bars:
            left, width = bar.left - chart.left, max(1, bar.right - bar.left - (1 if bar.right - bar.left > 3 else 0))
            run_colour, run_bottom = None, None
            for identifier, top, bottom in bar.slices:
                colour = colour_of(lineage[identifier], self.part)
                if choosing and identifier not in self.selection:
                    colour = faded(colour, DIM)
                if bar.future:
                    colour = faded(colour, FUTURE_DIM)
                if colour != run_colour:
                    if run_colour is not None:
                        self._fill(image, run_colour, left, width, run_top, run_bottom, chart.top)
                    run_colour, run_bottom = colour, bottom
                run_top = top
            if run_colour is not None:
                self._fill(image, run_colour, left, width, run_top, run_bottom, chart.top)
        return image

    @staticmethod
    def _fill(image: pg.Surface, colour: tuple[int, int, int], left: int, width: int,
              top: float, bottom: float, offset: int) -> None:
        top, bottom = round(top) - offset, round(bottom) - offset
        if bottom > top:
            pg.draw.rect(image, colour, (left, top, width, bottom - top))

    @staticmethod
    def _text(screen: pg.Surface, text: str, position: tuple[int, int], colour: tuple[int, int, int],
              font: pg.font.Font) -> None:
        screen.blit(font.render(text, True, colour), position)
