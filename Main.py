"""PyGame interface for the Bitzkey ecology experiment."""

import argparse
import math
import os
import sys
from collections import Counter
from collections.abc import Sequence

import numpy as np
import pygame as pg

from config.Globals import Config
from display import Window
from display.Dialog import Dialog
from display.Report import Report
from display.Inspector import draw_details
from display.Look import (DOT_RADIUS_RANGE, EXPORTING, GRASS, GRID, GROUND_VARIANTS, GROUNDS, NIGHT, RING_WIDTH, SEA_GRASS, TEXTURE_FROM,
                          grass_tuft, ground_tile, bitzkey_layer, dot_radius, draw_reticle,
                          map_dot_radii, draw_bitzkey, sea_colour, zone_colour)
from display.Setup import EXPORT_DIR, SetupScreen
from display.Language import start_language
from display.LanguageMenu import LanguageMenu
from display.Ui import UI, fit_text, draw_label, tr
from tools.Brain import ACTIONS
from display.Advanced import AdvancedScreen
from display.CatalogScreen import CatalogScreen
from display.ColourGraph import ColourGraph
from display.WorldFiles import choose_save_path
from tools.Exchange import Imported, bitzkey_record, export_bitzkeys, format_records
from tools.Genome import genome_record
from tools.Genome import TEMPERAMENT, TRAIT_MAX, decode_climate
from tools.Insights import ANIMALS, brain_matrix_lines, brain_report, death_report_lines
from tools import Catalog, Rules
from tools import Simulation as world
from tools.Simulation import DEPLOY_GROUP, SEASONS, SPECIES, Bitzkey, Simulation, season, year
from tools.WorldFile import WORLD_SUFFIX, save_world

REPORT_DIR = Config.APP_DIR / "reports"


def about() -> str:
    """What every screen shows at its top: the version running, and the processor cores it thinks on."""
    return f"v{Config.VERSION}  /  {Config.CORES} CORE{'' if Config.CORES == 1 else 'S'}"
INSIGHTS_TOP = 290  # the insights panel's top, in design pixels: just below the population counts
MAP_ZOOM_STEP = 1.25  # each wheel notch zooms the map in (or out) by this much
MAP_ZOOM_MAX = 12.0   # the closest the map zooms in, times its whole-map size
DRAG_THRESHOLD = 5    # design pixels a press must move on the map to draw a marquee rather than click
RETICLE_PULSE_MS = 2000  # a selected Bitzkey's sights swell to twice their size and back this often, paused or not
TRAIL_OPACITY = 0.5   # how faintly a Bitzkey shows on each cell it passed through this turn
# What an exemplar card can show: the living animal that stands out most by a statistic, highest, or
# lowest where `lowest`. Each is (title, kind, name, lowest): kind "mind" is an inborn temperament
# trait, of the mind or social, "body" a body trait, "home" its home temperature, and "state" what it is now.
EXEMPLAR_STATS = (
    ("SMARTEST", "mind", "intellect", False), ("MOST AGGRESSIVE", "mind", "aggression", False),
    ("MOST AMOROUS", "mind", "libido", False), ("MOST SOCIABLE", "mind", "sociability", False),
    ("MOST CURIOUS", "mind", "curiosity", False), ("MOST FEARFUL", "mind", "fear", False),
    ("GREATEST APPETITE", "mind", "appetite", False), ("MOST PATIENT", "mind", "patience", False),
    ("MOST EMPATHETIC", "mind", "empathy", False), ("MOST TOLERANT", "mind", "tolerance", False),
    ("BEST MEMORY", "mind", "memory", False), ("MOST RESILIENT", "mind", "resilience", False),
    ("MOODIEST", "mind", "moodiness", False), ("MOST LOYAL", "mind", "loyalty", False),
    ("MOST DOMINANT", "mind", "dominance", False), ("MOST TRUSTING", "mind", "trust", False),
    ("MOST CLANNISH", "mind", "kinship", False),
    ("SIMPLEST", "mind", "intellect", True), ("GENTLEST", "mind", "aggression", True),
    ("BRAVEST", "mind", "fear", True), ("MOST SOLITARY", "mind", "sociability", True),
    ("LARGEST", "body", "size", False), ("SMALLEST", "body", "size", True),
    ("STRONGEST", "body", "strength", False), ("BEST ARMOURED", "body", "armour", False),
    ("BEST ARMED", "body", "weapons", False), ("FASTEST", "body", "speed", False),
    ("SLOWEST", "body", "speed", True), ("SHARPEST-EYED", "body", "sight", False),
    ("MOST FERTILE", "body", "fertility", False), ("BIGGEST LITTERS", "body", "litter", False),
    ("BEST SCAVENGER", "body", "carrion", False),
    ("HOTTEST HOME", "home", "temperature", True), ("COLDEST HOME", "home", "temperature", False),
    ("OLDEST", "state", "age", False), ("YOUNGEST", "state", "age", True),
    ("MOST CHILDREN", "state", "children", False), ("LATEST GENERATION", "state", "generation", False),
    ("BEST FED", "state", "hunger", True), ("HUNGRIEST", "state", "hunger", False),
    ("HEALTHIEST", "state", "health", False), ("MOST WOUNDED", "state", "health", True),
    ("LONGEST LIFESPAN", "state", "lifespan", False), ("SHORTEST LIFESPAN", "state", "lifespan", True),
    ("FASTEST METABOLISM", "state", "burn", False), ("SLOWEST METABOLISM", "state", "burn", True),
    ("BIGGEST STOMACH", "state", "stomach", False), ("MOST NEURONS", "state", "neurons", False),
    ("MOST HERD-MINDED", "state", "herd", False), ("MOST CHANGED BY EXPERIENCE", "state", "changed", False),
)
# The statistic each card shows, left column then right, top to bottom; changed from a card's title,
# and kept for the rest of the session.
EXEMPLAR_CHOICES = [[stat[0] for stat in EXEMPLAR_STATS].index(title) for title in (
    "SMARTEST", "LARGEST", "OLDEST", "MOST CHILDREN", "LATEST GENERATION",
    "STRONGEST", "FASTEST", "BEST ARMOURED", "MOST FERTILE", "BEST FED")]
HOMES = ("hot", "medium", "cold", "very cold")


def exemplar_value(agent: Bitzkey, kind: str, name: str) -> float:
    """An animal's statistic, as an exemplar card judges it."""
    if kind == "mind":
        return agent.inborn[name]
    if kind == "body":
        return int(agent.body[name])
    if kind == "home":
        return int(decode_climate(agent.dna)[0])
    if name == "health":
        return agent.health / agent.max_health
    if name == "neurons":  # in its hidden layers
        return agent.brain.depth * agent.brain.width if agent.brain else 0
    if name == "herd":
        return agent.herd_instinct
    if name == "changed":  # how far experience has moved its temperament from what it was born with
        return sum(abs(agent.mind[trait] - agent.inborn[trait]) for trait in TEMPERAMENT)
    return getattr(agent, name)


def exemplar_text(agent: Bitzkey, kind: str, name: str) -> str:
    """What an exemplar card says of an animal's statistic."""
    value = exemplar_value(agent, kind, name)
    if kind in ("mind", "body"):
        return f"{name} {int(value)}"
    if kind == "home":
        return f"home {HOMES[int(value)]}"
    return {"age": lambda: f"age {int(value)}", "children": lambda: f"{int(value)} children",
            "generation": lambda: f"generation {int(value)}", "hunger": lambda: f"hunger {int(value)}%",
            "health": lambda: f"health {int(value * 100)}%", "lifespan": lambda: f"lifespan {int(value)} turns",
            "burn": lambda: f"burn {value:.2f}", "stomach": lambda: f"stomach {value:.2f}",
            "neurons": lambda: f"{int(value)} neurons", "herd": lambda: f"herd instinct {value:.2f}",
            "changed": lambda: f"changed {value:.1f}"}[name]()


EXEMPLAR_WIDTH = 170  # each column of exemplars, in design pixels
MENU_ROW, MENU_COLUMN = 20, 200  # an exemplar menu's rows and columns, in design pixels
# The globe: its radius this share of the view's smaller side at no zoom, zooming in to GLOBE_ZOOM_MAX times; the map
# drawn onto it from a picture about GLOBE_TEXTURE pixels across; and a stack's Bitzkeys each STACK_SPACING of the one
# below's radius above it, so its face shows.
GLOBE_SHARE = 0.46
GLOBE_ZOOM_MAX = 8.0
GLOBE_TEXTURE = 1600
STACK_SPACING = 0.7
# The level of detail: zoomed out all the way, only the largest Bitzkeys (size 7) are drawn, and each step in shows
# the next size down, until from ALL_SIZES_AT of the way in (on the wheel's scale) every size shows. The one chosen,
# those picked and those to export are drawn whatever their size.
ALL_SIZES_AT = 0.85
TOO_FAR = "Zoomed out too far to see anybody"


def smallest_shown(zoom: float, least: float, most: float) -> int:
    """The smallest size drawn at `zoom`, between `least` (all the way out) and `most` (all the way in)."""
    inward = math.log(zoom / least) / math.log(most / least)  # 0 all the way out, 1 all the way in
    return max(0, min(TRAIT_MAX, math.ceil(TRAIT_MAX * (1 - inward / ALL_SIZES_AT) - 1e-9)))


class BitzkeyApp:
    def __init__(self, width: int, height: int, population: int, grass_density: int, ocean: float, seed: int | None,
                 population_cap: int | None = None, mind_mutation: int = Config.MIND_MUTATION,
                 body_mutation: int = Config.BODY_MUTATION, year_length: int = Config.YEAR_LENGTH,
                 imported: Sequence[Imported] = (), simulation: Simulation | None = None):
        """Start a world by these settings, or carry on `simulation`, a world loaded; restarting starts afresh by
        the settings either way."""
        self.screen = Window.open_window()
        self.language_menu = LanguageMenu()
        self.width = width
        self.height = height
        self.population = population
        self.grass_density = grass_density
        self.ocean = ocean
        self.population_cap = population_cap
        self.mind_mutation = mind_mutation
        self.body_mutation = body_mutation
        self.year_length = year_length
        self.seed = seed
        self.imported = list(imported)
        self.simulation = simulation if simulation is not None else self._new_world()
        width, height = self.simulation.width, self.simulation.height
        self.selected_id: int | None = None
        self.show_insights = False
        self.show_census = False
        # The map's zoom (1 shows it all) and the cell at the middle of the view, while zoomed in;
        # while it is dragged, where the drag began and the middle then.
        self.map_zoom = 1.0
        # The globe: shown rather than the map, turned about its axis and tipped, and zoomed; H hides the rest.
        self.globe = False
        self.blind = False  # neither map nor globe drawn, to spare the time
        self.hud_hidden = False
        self.globe_yaw, self.globe_tilt, self.globe_zoom = 0.0, 0.35, 1.0
        self._turn_from: tuple | None = None  # a drag on the globe: where it began, and its turn then
        self._globe_look: dict | None = None  # where each pixel of its disc lies on the map, kept while its size and tilt stay
        self._view_buttons: dict[str, pg.Rect] = {}
        self.map_centre: tuple[float, float] = (width / 2, height / 2)
        self._pan_from: tuple[tuple[int, int], tuple[float, float]] | None = None
        self.graph = ColourGraph()
        # Bitzkeys added to export, by identifier: their genome and their record when added.
        self.export_list: dict[int, tuple[int, dict]] = {}
        # The insights panel scrolls: how far down it is, and while its scroll bar is dragged,
        # where on the thumb it was taken.
        self.insights_scroll = 0
        self._insights_height = 0
        self._scroll_grip: int | None = None
        self.dialog: Dialog | None = None
        self.report: Report | None = None  # the death report or a brain inspected, while open
        self._deaths_button: pg.Rect | None = None  # on the insights panel
        self._brains_button: pg.Rect | None = None  # beneath the selected Bitzkey's details
        self._dialog_kind = ""
        self._add_button: pg.Rect | None = None
        self._catalog_button: pg.Rect | None = None
        self._exemplar_cards: list[tuple[pg.Rect, int]] = []  # each card beside the map, and whom it shows
        # A left press on the map, while it is held: where it began and whether CTRL was down.
        self._marquee_from: tuple[tuple[int, int], bool] | None = None
        self._exemplar_titles: list[pg.Rect] = []  # each card's title, a menu of statistics, card by card
        self._exemplar_menu: int | None = None  # the card whose menu is open
        self._square: tuple[int, int] | None = None  # the map square under the pointer, listed in a popup
        self._square_panel: pg.Rect | None = None
        self._square_rows: list[tuple[pg.Rect, int]] = []
        self._menu_items: list[tuple[pg.Rect, int]] = []  # the open menu's choices, and which statistic each is
        self.renewal = ""  # the Bitzkeys to start a new world with, on leaving for one
        self._export_button = pg.Rect(0, 0, 0, 0)
        self._open_button = pg.Rect(0, 0, 0, 0)  # the catalog, to look through
        self._advanced_button = pg.Rect(0, 0, 0, 0)  # this world's rules
        self._save_button = pg.Rect(0, 0, 0, 0)  # the whole world, to a file
        # Deploying from the catalog: the Bitzkeys to paint onto the map, and the cell last painted while the button is
        # held; none, when not deploying.
        self.deploying: list[Imported] = []
        self._painted: tuple[int, int] | None = None
        self.message = ""
        self.message_until = 0
        self.playing = True
        self.clock = pg.time.Clock()
        self.elapsed = 0.0

    # Fonts are sized for the 1200x820 design and scaled to the screen.

    @property
    def font(self) -> pg.font.Font:
        return UI.font("segoeui", 15)

    @property
    def bold(self) -> pg.font.Font:
        return UI.font("segoeui", 22, bold=True)

    @property
    def mono(self) -> pg.font.Font:
        return UI.font("consolas", 14)

    @property
    def mono_small(self) -> pg.font.Font:
        return UI.font("consolas", 12)

    @property
    def small(self) -> pg.font.Font:
        return UI.font("segoeui", 12)

    def run(self) -> str:
        """Run until the window closes ("quit") or the user leaves for the configuration panel
        ("configure")."""
        while True:
            self.elapsed += self.clock.tick(30) / 1000
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return "quit"
                if event.type == pg.VIDEORESIZE:
                    self.screen = Window.resize(event)
                elif self.report:
                    if self.report.handle(event, self.screen.get_size()):
                        self.report = None
                elif self.dialog:
                    dialog = self.dialog
                    choice = dialog.handle(event, self.screen.get_size())
                    if choice and self._answer(choice) == "leave":
                        self._leave(dialog)
                        return "configure"
                elif self.deploying and not self.show_census and self._deploy_event(event):
                    pass
                elif not self.show_census and self._square_event(event):
                    pass
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE and self._exemplar_menu is not None:
                        self._exemplar_menu = None  # just closes the menu
                    elif event.key == pg.K_ESCAPE:
                        if self.show_census:
                            if self._leave_graph():
                                return "configure"
                        else:
                            self._ask_to_leave()
                    elif event.key == pg.K_F11:
                        self.screen = Window.toggle_fullscreen()
                    elif event.key == pg.K_SPACE:
                        if self.simulation.ended:
                            self._say(self._end_note())
                        else:
                            self.playing = not self.playing
                    elif event.key == pg.K_RIGHT:
                        self.playing = False
                        self.simulation.advance()
                    elif event.key == pg.K_r:
                        self.simulation = self._new_world()
                        self.selected_id = None
                        self.graph.reset()
                    elif event.key == pg.K_i:
                        self.show_insights = not self.show_insights
                    elif event.key == pg.K_b:
                        self._save_report()
                    elif event.key == pg.K_g:
                        if self.show_census:
                            if self._leave_graph():
                                return "configure"
                        else:
                            self.show_census = True
                    elif event.key == pg.K_TAB and self.show_census:
                        self.graph.switch_part()
                    elif event.key == pg.K_c:
                        if self.show_census:
                            self._export_graph(to_file=False)
                        else:
                            self._copy_selected()
                    elif event.key == pg.K_s and self.show_census:
                        self._export_graph(to_file=True)
                    elif event.key == pg.K_a and not self.show_census:
                        self._toggle_export()
                    elif event.key == pg.K_k:
                        self._catalog_picked() if self.show_census else self._catalog_selected()
                    elif event.key == pg.K_e:
                        self._open_export_dialog()
                    elif event.key == pg.K_z and self.show_census:
                        self.graph.unzoom()  # the whole run, its whole height
                    elif event.key == pg.K_h:
                        self.hud_hidden = not self.hud_hidden  # the map or globe alone, filling the window, or everything
                    elif event.key == pg.K_z:
                        self.map_zoom, self.map_centre, self.globe_zoom = 1.0, (self.width / 2, self.height / 2), 1.0
                elif self.show_insights and not self.show_census and self._scroll_event(event):
                    pass
                elif not self.show_census and self._map_event(event):
                    pass
                elif self.show_census and event.type in (pg.MOUSEBUTTONDOWN, pg.MOUSEBUTTONUP, pg.MOUSEMOTION,
                                                         pg.MOUSEWHEEL):
                    action = self.graph.handle(event)
                    if action == "map":
                        self._close_graph()  # at the run's end, to look over the map before leaving
                    elif action == "catalog":
                        self._catalog_picked()
                    elif action == "open":
                        self._browse_catalog()
                    elif action == "renew":
                        if self._renew():
                            return "renew"
                    elif action:
                        self._export_graph(to_file=action == "save")
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    if self.language_menu.click(event.pos):
                        pass  # the language menu opened, closed or chose
                    elif not self.hud_hidden and self._view_buttons.get("map", pg.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                        self.globe = self.blind = False
                    elif not self.hud_hidden and self._view_buttons.get("globe", pg.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                        self.globe, self.blind = True, False
                    elif not self.hud_hidden and self._view_buttons.get("blind", pg.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                        self.blind = True
                    elif self.globe and self._map_view()[0].collidepoint(event.pos):
                        self._pick_on_globe(event.pos)  # a click picks the Bitzkey there; the right button turns it
                    elif self._exemplar_menu is None and self._map_view()[0].collidepoint(event.pos):
                        self._marquee_from = (event.pos, bool(pg.key.get_mods() & pg.KMOD_CTRL))
                    else:
                        self._click(event.pos)
                elif event.type == pg.MOUSEBUTTONUP and event.button == 1 and self._marquee_from:
                    (start, adding), self._marquee_from = self._marquee_from, None
                    self._pick_on_map(start, event.pos, adding)

            self._tick()
            self._draw()
            pg.display.flip()

    def _tick(self) -> None:
        """While playing, advance a turn every 0.24 s, and stop to show the graph once the
        simulation has ended."""
        if not self.playing or self.dialog or self.report:
            return
        if self.simulation.ended:
            self._finish()
        elif self.elapsed >= 0.24:
            self.simulation.advance()
            self.elapsed = 0
            if self.simulation.ended:
                self._finish()

    def _end_note(self) -> str:
        cause = ("every animal has died" if self.simulation.end_reason == "extinct"
                 else "the world is full, with no cell left free")
        return f"The simulation ended at turn {self.simulation.turn}: {cause}."

    def _finish(self) -> None:
        """The run is over: stop, and show how it went on the colour graph."""
        self.playing = False
        self.show_census = True
        self._say(self._end_note())

    def _close_graph(self) -> None:
        """Close the colour graph. The Bitzkeys picked on it stay selected on the map; the sidebar
        then shows no one Bitzkey, unless the pick is just the one it was showing."""
        self.show_census = False
        self.graph.menu_open = False
        if self.graph.selection and self.graph.selection != {self.selected_id}:
            self.selected_id = None

    def _leave_graph(self) -> bool:
        """Close the graph with G or ESC. Before the run's end, that returns to the map; after it, it
        leaves for the configuration panel, asking first only if Bitzkeys wait on the export list.
        Returns whether to leave now."""
        if not self.simulation.ended:
            self._close_graph()
            return False
        if self.export_list:
            self._ask_to_leave()
            return False
        self._leave(Dialog("Leaving this simulation", [], [("leave", "LEAVE")], cancel="leave"))
        return True

    def _click(self, position: tuple[int, int]) -> None:
        if self._exemplar_menu is not None:  # a click picks a statistic or closes the menu
            picked = [stat for item, stat in self._menu_items if item.collidepoint(position)]
            if picked:
                EXEMPLAR_CHOICES[self._exemplar_menu] = picked[0]
            self._exemplar_menu = None
            return
        if titled := [slot for slot, title in enumerate(self._exemplar_titles) if title.collidepoint(position)]:
            self._exemplar_menu = titled[0]
            return
        if self._deaths_button and self._deaths_button.collidepoint(position):
            self._open_death_report()
        elif self._brains_button and self._brains_button.collidepoint(position):
            self._inspect_brains()
        elif self._add_button and self._add_button.collidepoint(position):
            self._toggle_export()
        elif self._catalog_button and self._catalog_button.collidepoint(position):
            self._catalog_selected()
        elif self._export_button.collidepoint(position):
            self._open_export_dialog()
        elif self._open_button.collidepoint(position):
            self._browse_catalog()
        elif self._advanced_button.collidepoint(position):
            self._edit_rules()
        elif self._save_button.collidepoint(position):
            if self._save_world() and not self.simulation.ended:
                self.playing = True  # saved: it plays on
        elif picked := [identifier for card, identifier in self._exemplar_cards if card.collidepoint(position)]:
            self._select(picked[0])
        elif self._map_view()[0].collidepoint(position):  # a click on the map, pressed and let go at once
            self._pick_on_map(position, position, bool(pg.key.get_mods() & pg.KMOD_CTRL))

    def _edit_rules(self) -> None:
        """Pause, and change the rules this world plays by: applied, they hold from its next turn, and
        for its restarts. Closing the window from there closes the program."""
        self.playing = False
        outcome, changed = AdvancedScreen(self.screen, Rules.current(), running=True).run()
        self.screen = pg.display.get_surface()
        if outcome == "quit":
            pg.event.post(pg.event.Event(pg.QUIT))
        elif outcome == "apply":
            Rules.apply(changed)
            self.simulation.rules_changed()
            self._say(f"This world now plays by {len(changed)} rule{'s' if len(changed) != 1 else ''} changed from the "
                      "defaults. Paused: SPACE plays on.")

    def _browse_catalog(self) -> None:
        """Look through the catalog, importing and exporting nothing; the world waits meanwhile. DEPLOY
        closes it to paint the Bitzkeys chosen onto the map. Closing the window from there closes the program."""
        outcome, entries = CatalogScreen(self.screen, browsing=True).run()
        self.screen = pg.display.get_surface()
        if outcome == "quit":
            pg.event.post(pg.event.Event(pg.QUIT))
        elif outcome == "deploy" and entries:
            self._start_deploying([Catalog.imported(entry) for entry in entries])

    def _start_deploying(self, bitzkeys: list[Imported]) -> None:
        """Deploy mode: the cursor becomes a target, and each click on the map, or each step of a drag, puts
        DEPLOY_GROUP of each Bitzkey there."""
        self.show_census = False
        self.deploying, self._painted = bitzkeys, None
        pg.mouse.set_cursor(pg.SYSTEM_CURSOR_CROSSHAIR)
        what = "it" if len(bitzkeys) == 1 else f"each of the {len(bitzkeys)}"
        self._say(f"DEPLOY: click or drag on the map to place {DEPLOY_GROUP} of {what}.  RIGHT-CLICK or ESC  stop")

    def _stop_deploying(self) -> None:
        self.deploying, self._painted = [], None
        pg.mouse.set_cursor(pg.SYSTEM_CURSOR_ARROW)
        self._say("Deploying stopped.")

    def _deploy_event(self, event: pg.event.Event) -> bool:
        """While deploying: a left press on the map places a group, and holding it, the drag paints another every
        DEPLOY_GROUP cells; a right-click or ESC stops. Returns whether the event was used."""
        if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE or event.type == pg.MOUSEBUTTONDOWN and event.button == 3:
            self._stop_deploying()
            return True
        if event.type == pg.MOUSEBUTTONUP and event.button == 1:
            self._painted = None
            return True
        painting = event.type == pg.MOUSEMOTION and self._painted is not None
        if not (painting or event.type == pg.MOUSEBUTTONDOWN and event.button == 1):
            return False
        cell = self._cell_at(event.pos)
        if cell is None:
            return event.type != pg.MOUSEBUTTONDOWN  # a press off the map works the rest of the screen
        if painting and max(abs(cell[0] - self._painted[0]), abs(cell[1] - self._painted[1])) < DEPLOY_GROUP:
            return True
        self._painted = cell
        placed = self.simulation.deploy(self.deploying, *cell)
        self._say(f"Deployed {len(placed)} at ({cell[0]}, {cell[1]}).  RIGHT-CLICK or ESC  stop")
        return True

    def _square_event(self, event: pg.event.Event) -> bool:
        """The square popup: moving the pointer over a map square lists the Bitzkeys stacked there, and
        the list stays while the pointer is over it; clicking a name selects that Bitzkey. Returns whether
        the event was used."""
        if event.type == pg.MOUSEMOTION:
            on_panel = self._square_panel is not None and self._square_panel.collidepoint(event.pos)
            if not on_panel:
                quiet = not any(event.buttons) and self._exemplar_menu is None and not self.deploying
                self._square = self._cell_at(event.pos) if quiet else None
            return False
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1 and self._square_panel is not None                 and self._square_panel.collidepoint(event.pos):
            for row, identifier in self._square_rows:
                if row.collidepoint(event.pos):
                    self._select(identifier)
            return True
        return False

    def _draw_square(self) -> None:
        """The popup listing the Bitzkeys in the square under the pointer, beside the square."""
        self._square_panel, self._square_rows = None, []
        if self._square is None or self.show_census or self.dialog or self.report:
            return
        here = [agent for agent in self.simulation.living if (agent.x, agent.y) == self._square]
        if not here:
            return
        board, cell_width, cell_height = self._board_geometry()
        row = UI(MENU_ROW)
        width = UI(230)
        title = f"SQUARE ({self._square[0]}, {self._square[1]})   {sum(agent.space for agent in here)}/{world.CELL_SPACE} SPACES"
        panel = pg.Rect(0, 0, width, (len(here) + 1) * row + UI(8))
        panel.left = int(board.left + (self._square[0] + 1) * cell_width)  # touching the square, to reach it
        panel.top = int(board.top + self._square[1] * cell_height)
        if panel.right > self.screen.get_width() - UI(4):
            panel.right = int(board.left + self._square[0] * cell_width)
        panel.top = max(UI(4), min(self.screen.get_height() - panel.height - UI(4), panel.top))
        pg.draw.rect(self.screen, (27, 37, 37), panel, border_radius=UI(5))
        pg.draw.rect(self.screen, (131, 205, 143), panel, 1, border_radius=UI(5))
        self._text(title, (panel.left + UI(10), panel.top + UI(6)), (144, 170, 151), self.small)
        for index, agent in enumerate(here):
            item = pg.Rect(panel.left + UI(4), panel.top + UI(4) + (index + 1) * row, width - UI(8), row)
            if item.collidepoint(pg.mouse.get_pos()):
                pg.draw.rect(self.screen, (44, 57, 52), item, border_radius=UI(3))
            chosen = agent.identifier == self.selected_id
            self._text(f"#{agent.identifier:03d}  {agent.species.upper()}   size {agent.body['size']}  ·  {agent.space} spaces",
                       (item.left + UI(6), item.top + UI(2)), (131, 205, 143) if chosen else (220, 229, 211), self.small)
            self._square_rows.append((item, agent.identifier))
        self._square_panel = panel

    def _cell_at(self, position: tuple[int, int]) -> tuple[int, int] | None:
        """The world's cell under a point on the map, or None off it."""
        if not self._map_view()[0].collidepoint(position):
            return None
        board, cell_width, cell_height = self._board_geometry()
        x, y = int((position[0] - board.left) // cell_width), int((position[1] - board.top) // cell_height)
        return (x, y) if 0 <= x < self.width and 0 <= y < self.height else None

    # Dialogs

    def _open(self, kind: str, dialog: Dialog) -> None:
        self._dialog_kind, self.dialog = kind, dialog

    def _answer(self, choice: str) -> str | None:
        """Act on a dialog's choice; returns "leave" when the user leaves the simulation."""
        kind, self.dialog = self._dialog_kind, None
        if kind == "leave":
            if choice == "save":  # saved first; if that's cancelled or fails, it stays
                return "leave" if self._save_world() else None
            return "leave" if choice == "leave" else None
        if choice in ("save", "copy"):
            self._export_chosen(to_file=choice == "save")
        elif choice == "catalog":
            self._catalog_export_list()
        elif choice == "clear":
            count = len(self.export_list)
            self.export_list.clear()
            self._say(f"Cleared {count} Bitzkey{'s' if count != 1 else ''} from the export list.")
        return None

    def _renew(self) -> bool:
        """Copy the living Bitzkeys picked on the graph to the clipboard and leave, to start a new
        world with them alone imported. Returns whether it left."""
        living = {agent.identifier: agent for agent in self.simulation.living}
        chosen = [living[identifier] for identifier in sorted(self.graph.selection) if identifier in living]
        if not chosen:
            self._say("Pick living Bitzkeys on the graph first, then start a new world with them.")
            return False
        self.renewal = export_bitzkeys(chosen)
        self._copy(self.renewal, f"{len(chosen)} Bitzkey{'s' if len(chosen) != 1 else ''}")
        self._leave(Dialog("Starting a new world", [], [("leave", "LEAVE")], cancel="leave"))
        return True

    def _leave(self, dialog: Dialog) -> None:
        """Say that it's leaving, then delete the world's history from the disk and let it go."""
        dialog.start("leave", "Leaving, please be patient", "Cleaning up messy neural networks...")
        self.dialog = dialog
        self.playing = False
        self._draw()
        pg.display.flip()
        self.simulation = None

    def _ask_to_leave(self) -> None:
        lines = ["You'll return to the configuration panel to set up a new world.",
                 "This world won't be kept unless you save it."]
        if self.export_list:
            count = len(self.export_list)
            lines.append(f"{count} Bitzkey{'s' if count != 1 else ''} added to export will be lost unless exported first.")
        self._open("leave", Dialog("Leave this simulation?", lines, [("stay", "STAY"), ("save", "SAVE AND LEAVE"),
                                                                     ("leave", "LEAVE")], cancel="stay"))

    def _save_world(self) -> bool:
        """Pause, and save the whole world to a file of the user's choosing, to load from the configuration panel
        and carry on exactly as it would have. Returns whether it saved."""
        self.playing = False
        path = choose_save_path(f"world-turn-{self.simulation.turn:05d}{WORLD_SUFFIX}")
        if path is None:
            return False
        self._say("Saving the world, please be patient...")
        self._draw()
        pg.display.flip()
        try:
            save_world(self.simulation, path)
        except OSError as error:
            self._say(f"Couldn't save the world to {path.name}: {error}")
            return False
        self._say(f"Saved the world at turn {self.simulation.turn} to {path.name}")
        return True

    def _open_export_dialog(self) -> None:
        count = len(self.export_list)
        if not count:
            self._open("export", Dialog("Export all selected", [
                "Nothing has been added to export yet.",
                "Select a Bitzkey and press ADD TO EXPORT (or A)."], [("close", "CLOSE")], cancel="close"))
            return
        _, gone = self._chosen_records()
        lines = [f"{count} Bitzkey{'s' if count != 1 else ''} added to export.",
                 "The living export as they are now, with what they have learned since."]
        if gone:
            lines.append(f"{gone} {'have' if gone != 1 else 'has'} died; they export as they were when added.")
        lines.append("Save them to a file in exports/, copy them to the clipboard, or add them to the catalog.")
        self._open("export", Dialog("Export all selected", lines, [
            ("clear", "CLEAR LIST"), ("cancel", "CANCEL"), ("catalog", "CATALOG"), ("copy", "COPY"),
            ("save", "SAVE FILE")], cancel="cancel"))

    # The export list

    def _toggle_export(self) -> None:
        agent = next((item for item in self.simulation.agents if item.identifier == self.selected_id), None)
        if agent is None:
            self._say("Select a Bitzkey first, then add it to export.")
            return
        if agent.identifier in self.export_list:
            del self.export_list[agent.identifier]
            self._say(f"Removed #{agent.identifier:03d} from the export list ({len(self.export_list)} on it).")
        else:
            self.export_list[agent.identifier] = (agent.dna, bitzkey_record(agent))
            self._say(f"Added #{agent.identifier:03d} to the export list ({len(self.export_list)} on it).")

    def _chosen_records(self) -> tuple[list[dict], int]:
        """Records for the export list: the living as they are now, the dead as they were when
        added (their brain is gone). Also returns how many have died."""
        agents = {agent.identifier: agent for agent in self.simulation.agents}
        records, gone = [], 0
        for identifier, (dna, saved) in self.export_list.items():
            agent = agents.get(identifier)
            if agent is not None and agent.alive and agent.dna == dna:
                records.append(bitzkey_record(agent))
            else:
                records.append(saved)
                gone += 1
        return records, gone

    def _export_chosen(self, to_file: bool) -> None:
        records, gone = self._chosen_records()
        count = len(records)
        what = f"{count} Bitzkey{'s' if count != 1 else ''} from the export list"
        if gone:
            what += f" ({gone} as they were when added)"
        text = format_records(records)
        if not to_file:
            self._copy(text, what)
            return
        EXPORT_DIR.mkdir(exist_ok=True)
        path = EXPORT_DIR / f"bitzkeys-turn-{self.simulation.turn:05d}-chosen-{count}.json"
        path.write_text(text + "\n", encoding="utf-8")
        self._say(f"Saved {what} to exports/{path.name}")

    # The catalog

    def _add_to_catalog(self, entries: list[dict], what: str, skipped: str = "") -> None:
        try:
            added = Catalog.add(entries)
        except (ValueError, OSError) as error:
            self._say(f"Couldn't add to the catalog: {error}")
            return
        already = len(entries) - added
        note = f"Added {what} to the catalog" if added else f"{what[0].upper()}{what[1:]} already in the catalog"
        if added and already:
            note += f" ({already} already there)"
        self._say(note + "." + (f" {skipped}" if skipped else ""))

    def _catalog_selected(self) -> None:
        """Catalog the Bitzkey selected on the map, as it is now, brain and all."""
        agent = next((item for item in self.simulation.agents if item.identifier == self.selected_id), None)
        if agent is None or not agent.alive:
            self._say("Select a living Bitzkey first, then add it to the catalog.")
            return
        self._add_to_catalog([Catalog.entry_for(agent, self.simulation.turn)], f"#{agent.identifier:03d}")

    def _catalog_picked(self) -> None:
        """Catalog the living Bitzkeys picked on the colour graph; the dead have lost their brains."""
        if not self.graph.selection:
            self._say("Pick Bitzkeys on the graph first, then add them to the catalog.")
            return
        living = {agent.identifier: agent for agent in self.simulation.living}
        chosen = [living[identifier] for identifier in sorted(self.graph.selection) if identifier in living]
        dead = len(self.graph.selection) - len(chosen)
        skipped = f"{dead} dead left out: their brains are gone." if dead else ""
        if not chosen:
            self._say("Only living Bitzkeys can be catalogued, with their brains; " + skipped)
            return
        self._add_to_catalog([Catalog.entry_for(agent, self.simulation.turn) for agent in chosen],
                             f"{len(chosen)} Bitzkey{'s' if len(chosen) != 1 else ''}", skipped)

    def _catalog_export_list(self) -> None:
        """Catalog the export list: the living as they are now, the dead as they were when added to it."""
        agents = {agent.identifier: agent for agent in self.simulation.agents}
        entries = []
        for identifier, (dna, saved) in self.export_list.items():
            agent = agents.get(identifier)
            if agent is not None and agent.alive and agent.dna == dna:
                entries.append(Catalog.entry_for(agent, self.simulation.turn))
            else:
                entries.append({**saved, "catalog": {"identifier": identifier}})
        self._add_to_catalog(entries, f"{len(entries)} Bitzkey{'s' if len(entries) != 1 else ''} from the export list")

    def _pick_on_map(self, start: tuple[int, int], end: tuple[int, int], adding: bool) -> None:
        """Finish a left press on the map. A click picks the living Bitzkey under the pointer, or
        clears the selection on empty ground; a drag picks every living Bitzkey in its marquee. With
        CTRL (`adding`), a click adds the Bitzkey to the selection or takes it out, and a marquee adds
        to it."""
        if max(abs(end[0] - start[0]), abs(end[1] - start[1])) < UI(DRAG_THRESHOLD):
            hit = self._living_at(start)
            if adding:
                picked = self.graph.selection ^ {hit} if hit is not None else set(self.graph.selection)
            else:
                picked = {hit} if hit is not None else set()
        else:
            board, cell_width, cell_height = self._board_geometry()
            box = pg.Rect(min(start[0], end[0]), min(start[1], end[1]), abs(end[0] - start[0]) + 1,
                          abs(end[1] - start[1]) + 1)
            inside = {agent.identifier for agent in self.simulation.living
                      if box.collidepoint(board.left + (agent.x + 0.5) * cell_width,
                                          board.top + (agent.y + 0.5) * cell_height)}
            picked = (self.graph.selection | inside) if adding else inside
        self.graph.select(picked)  # the map and the graph share one selection
        self.selected_id = next(iter(picked)) if len(picked) == 1 else None  # one picked shows in the inspector
        if self.selected_id is not None:
            self.show_insights = False

    def _living_at(self, position: tuple[int, int]) -> int | None:
        """The living Bitzkey whose dot is nearest the pointer, if near enough to have been clicked."""
        board, cell_width, cell_height = self._board_geometry()
        nearest, nearest_distance = None, float("inf")
        for agent in self.simulation.living:
            point = (board.left + (agent.x + 0.5) * cell_width, board.top + (agent.y + 0.5) * cell_height)
            distance = pg.Vector2(position).distance_to(point)
            if distance < nearest_distance:
                nearest, nearest_distance = agent, distance
        if nearest and board.collidepoint(position) and nearest_distance < max(UI(9), min(cell_width, cell_height) * 1.6):
            return nearest.identifier
        return None

    def _select(self, identifier: int) -> None:
        self.selected_id = identifier
        self.graph.select({identifier})  # the map and the graph share one selection
        self.show_insights = False  # selecting shows the inspector

    def _exemplars(self) -> list[tuple[str, Bitzkey | None, str]]:
        """Each card's title, the living animal that stands out most by its statistic, if any, and what
        the card says. Ties go to the oldest, then the first born."""
        animals = [agent for agent in self.simulation.living if agent.species != "plant"]
        found = []
        for choice in EXEMPLAR_CHOICES:
            title, kind, name, lowest = EXEMPLAR_STATS[choice]
            sign = -1 if lowest else 1
            best = max(animals, key=lambda agent: (sign * exemplar_value(agent, kind, name), agent.age, -agent.identifier),
                       default=None)
            found.append((title, best, exemplar_text(best, kind, name) if best else "none alive"))
        return found

    def _exemplar_columns(self) -> tuple[pg.Rect, pg.Rect]:
        """The columns either side of the map where the exemplars show."""
        area = self._board_area()
        width = UI(EXEMPLAR_WIDTH)
        return (pg.Rect(area.left + UI(16), area.top + UI(21), width, area.height - UI(42)),
                pg.Rect(area.right - UI(16) - width, area.top + UI(21), width, area.height - UI(42)))

    def _sidebar_width(self) -> int:
        return min(UI(340), max(UI(290), self.screen.get_width() // 3))

    def _map_view(self) -> tuple[pg.Rect, float]:
        """Where the map shows, and the scale at which the whole map fits it: between the exemplars, or with the HUD
        hidden, the whole window."""
        if self.hud_hidden:
            view = self.screen.get_rect().inflate(-UI(16), -UI(16))
        else:
            view = self._board_area().inflate(-UI(42) - 2 * UI(EXEMPLAR_WIDTH + 16), -UI(42))
        return view, min(view.width / self.width, view.height / self.height)

    def _board_area(self) -> pg.Rect:
        """The space left of the sidebar, between the header and the footer."""
        UI.fit(self.screen.get_size())
        screen_width, screen_height = self.screen.get_size()
        return pg.Rect(0, UI(62), screen_width - self._sidebar_width(), screen_height - UI(140))

    def _clamped_centre(self, centre: tuple[float, float], view: pg.Rect, scale: float) -> tuple[float, float]:
        """The view's middle kept so the map fills the view where it can, or centred where it can't."""
        clamped = []
        for middle, cells, pixels in zip(centre, (self.width, self.height), view.size):
            half = pixels / 2 / scale  # half the view, in cells
            clamped.append(cells / 2 if cells <= 2 * half else min(cells - half, max(half, middle)))
        return clamped[0], clamped[1]

    def _board_geometry(self) -> tuple[pg.Rect, float, float]:
        """The whole map's rectangle on screen, at the current zoom (larger than the view when zoomed
        in), and the size of a cell."""
        view, fit = self._map_view()
        scale = fit * self.map_zoom
        centre_x, centre_y = self._clamped_centre(self.map_centre, view, scale)
        board = pg.Rect(round(view.centerx - centre_x * scale), round(view.centery - centre_y * scale),
                        round(scale * self.width), round(scale * self.height))
        return board, scale, scale

    def _map_event(self, event: pg.event.Event) -> bool:
        """Zoom the map with the wheel around the pointer, and pan it by dragging with the right or
        middle button; the world carries on meanwhile. Returns whether the event was used."""
        view, fit = self._map_view()
        if self.globe:
            return self._globe_event(event, view)
        if event.type == pg.MOUSEWHEEL and view.collidepoint(pg.mouse.get_pos()):
            board, scale, _ = self._board_geometry()
            x, y = pg.mouse.get_pos()
            cell = ((x - board.left) / scale, (y - board.top) / scale)  # the cell under the pointer stays put
            self.map_zoom = max(1.0, min(MAP_ZOOM_MAX, self.map_zoom * MAP_ZOOM_STEP ** event.y))
            scale = fit * self.map_zoom
            self.map_centre = self._clamped_centre(
                (cell[0] - (x - view.centerx) / scale, cell[1] - (y - view.centery) / scale), view, scale)
            return True
        if event.type == pg.MOUSEBUTTONDOWN and event.button in (2, 3) and view.collidepoint(event.pos):
            self._pan_from = (event.pos, self._clamped_centre(self.map_centre, view, fit * self.map_zoom))
            return True
        if event.type == pg.MOUSEBUTTONUP and event.button in (2, 3) and self._pan_from:
            self._pan_from = None
            return True
        if event.type == pg.MOUSEMOTION and self._pan_from:
            (start_x, start_y), (centre_x, centre_y) = self._pan_from
            scale = fit * self.map_zoom
            self.map_centre = self._clamped_centre(
                (centre_x - (event.pos[0] - start_x) / scale, centre_y - (event.pos[1] - start_y) / scale), view, scale)
            return True
        return False

    def _globe_event(self, event: pg.event.Event, view: pg.Rect) -> bool:
        """The wheel zooms the globe; a drag turns it about its axis, or tips it up or down."""
        if event.type == pg.MOUSEWHEEL and view.collidepoint(pg.mouse.get_pos()):
            self.globe_zoom = max(0.5, min(GLOBE_ZOOM_MAX, self.globe_zoom * MAP_ZOOM_STEP ** event.y))
            return True
        if event.type == pg.MOUSEBUTTONDOWN and event.button in (2, 3) and view.collidepoint(event.pos):
            self._turn_from = (event.pos, self.globe_yaw, self.globe_tilt)
            return True
        if event.type == pg.MOUSEBUTTONUP and self._turn_from:
            self._turn_from = None
            return True
        if event.type == pg.MOUSEMOTION and self._turn_from:
            (x, y), yaw, tilt = self._turn_from
            radius = min(view.width, view.height) * GLOBE_SHARE * self.globe_zoom
            self.globe_yaw = yaw - (event.pos[0] - x) / radius
            self.globe_tilt = max(-1.3, min(1.3, tilt + (event.pos[1] - y) / radius))
            return True
        return False

    def _pick_on_globe(self, position: tuple[int, int]) -> None:
        """Select the topmost Bitzkey drawn under the pointer on the globe, or none."""
        for (hx, hy), reach, identifier in reversed(getattr(self, "_globe_hits", [])):
            if (position[0] - hx) ** 2 + (position[1] - hy) ** 2 <= (reach + 2) ** 2:
                self.selected_id = identifier
                return
        self.selected_id = None

    def _draw_view_buttons(self) -> None:
        """MAP and GLOBE, above the map."""
        view = self._map_view()[0]
        font = UI.font("segoeui", 12, bold=True)
        left = view.left
        for key, label, on in (("map", "MAP", not self.globe and not self.blind), ("globe", "GLOBE", self.globe and not self.blind),
                               ("blind", "BLIND", self.blind)):
            button = pg.Rect(left, view.top - UI(21), UI(56 if key == "map" else 64), UI(18))
            pg.draw.rect(self.screen, (131, 205, 143) if on else (44, 57, 52), button, border_radius=UI(4))
            draw_label(self.screen, font, label, (19, 26, 27) if on else (220, 229, 211), button)
            self._view_buttons[key] = button
            left = button.right + UI(6)

    def _draw_globe(self) -> None:
        """The map wrapped round a sphere, turned and tipped, lit from the upper left, with each square's Bitzkeys
        stacked on it, the largest at the bottom, each above the one below so its face shows, as the C++ version draws it."""
        view = self._map_view()[0]
        self.screen.set_clip(view)
        width, height = self.width, self.height
        per = max(2, min(12, GLOBE_TEXTURE // max(width, height)))  # the map flat, a few pixels a square, to wrap round it
        texture = pg.Surface((width * per, height * per))
        self._draw_ground(texture, texture.get_rect(), per, per, texture.get_rect())
        radius = min(view.width, view.height) * GLOBE_SHARE * self.globe_zoom
        size = 2 * math.ceil(radius) + 1
        look = self._globe_look
        if look is None or look["size"] != size or look["tilt"] != self.globe_tilt:
            half = size / 2
            px, py = np.meshgrid(np.arange(size) + 0.5 - half, np.arange(size) + 0.5 - half, indexing="ij")  # x first, as surfarray
            nx, ny = px / radius, -py / radius
            disc = nx * nx + ny * ny < 1
            nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
            c, sn = math.cos(self.globe_tilt), math.sin(self.globe_tilt)
            y1, z1 = ny * c + nz * sn, -ny * sn + nz * c  # tipped back upright
            look = self._globe_look = {
                "size": size, "tilt": self.globe_tilt, "disc": disc, "longitude": np.arctan2(nx, z1),
                "row": 0.5 - np.arcsin(np.clip(y1, -1, 1)) / math.pi,
                "light": 0.35 + 0.65 * np.clip(-0.35 * nx + 0.45 * ny + 0.82 * nz, 0, 1)}
        u = look["longitude"] / (2 * math.pi) + 0.5 + self.globe_yaw / (2 * math.pi)
        u -= np.floor(u)
        ground = pg.surfarray.array3d(texture)
        tx = np.minimum(texture.get_width() - 1, (u * texture.get_width()).astype(int))
        ty = np.minimum(texture.get_height() - 1, (look["row"] * texture.get_height()).astype(int))
        colours = (ground[tx, ty] * look["light"][:, :, None]).clip(0, 255).astype(np.uint8)
        disc_surface = pg.Surface((size, size), pg.SRCALPHA)
        pg.surfarray.pixels3d(disc_surface)[:, :, :] = colours
        pg.surfarray.pixels_alpha(disc_surface)[:, :] = np.where(look["disc"], 255, 0).astype(np.uint8)
        cx, cy = view.centerx, view.centery
        self.screen.blit(disc_surface, (cx - size // 2, cy - size // 2))
        squares: dict[tuple[int, int], list] = {}
        for agent in self.simulation.agents:
            if self._corpse_opacity(agent) > 0:
                squares.setdefault((agent.x, agent.y), []).append(agent)
        c, sn = math.cos(self.globe_tilt), math.sin(self.globe_tilt)
        stacks = []
        for (x, y), members in squares.items():
            latitude = (0.5 - (y + 0.5) / height) * math.pi
            longitude = ((x + 0.5) / width - 0.5) * 2 * math.pi - self.globe_yaw
            x1, y1, z1 = math.cos(latitude) * math.sin(longitude), math.sin(latitude), math.cos(latitude) * math.cos(longitude)
            up, depth = y1 * c - z1 * sn, y1 * sn + z1 * c
            if depth >= 0.08:  # not round the back, nor too near the edge to see
                stacks.append((depth, cx + x1 * radius, cy - up * radius, members))
        stacks.sort(key=lambda stack: stack[0])
        self._globe_hits = []
        smallest, largest = map_dot_radii(math.pi * radius / height)
        least, drawn, hidden = smallest_shown(self.globe_zoom, 0.5, GLOBE_ZOOM_MAX), 0, 0
        for depth, sx, sy, members in stacks:
            shrink = 0.55 + 0.45 * depth  # foreshortened toward the edge
            lift = 0
            shown = [agent for agent in members if int(agent.body["size"]) >= least or self._marked(agent)]
            drawn, hidden = drawn + len(shown), hidden + len(members) - len(shown)  # the rest too small to see from this far out
            for agent in sorted(shown, key=lambda agent: -int(agent.body["size"])):
                dot = max(2, round(dot_radius(int(agent.body["size"]), largest, smallest) * shrink))
                point = (round(sx), round(sy - lift))
                draw_bitzkey(self.screen, agent.dna, point, dot, alive=agent.alive, opacity=self._corpse_opacity(agent))
                if agent.alive:
                    self._globe_hits.append((point, dot + RING_WIDTH, agent.identifier))  # drawn last, on top
                if agent.identifier == self.selected_id:
                    draw_reticle(self.screen, point, round((dot + RING_WIDTH + UI(3)) * reticle_pulse()))
                lift += max(2, round((dot + RING_WIDTH) * STACK_SPACING))  # the next above, its face clear
        if not drawn and hidden:
            self._say_too_far(view)
        self.screen.set_clip(None)

    def _marked(self, agent: Bitzkey) -> bool:
        """Whether a Bitzkey is drawn whatever its size: the one chosen, one picked, or one to export."""
        return agent.identifier == self.selected_id or agent.identifier in self.graph.selection or agent.identifier in self.export_list

    def _say_too_far(self, view: pg.Rect) -> None:
        """Every Bitzkey in view too small to draw from this far out: say so, in the middle of the view."""
        font = UI.font("segoeui", 15)
        label = font.render(TOO_FAR, True, (238, 183, 110))
        box = label.get_rect(center=view.center).inflate(UI(28), UI(14))
        veil = pg.Surface(box.size, pg.SRCALPHA)
        veil.fill((19, 26, 27, 220))
        self.screen.blit(veil, box.topleft)
        self.screen.blit(label, label.get_rect(center=box.center))

    def _forget_the_dead(self) -> None:
        """Let go of any Bitzkey selected or picked that has died, so the sidebar returns to its
        neutral state rather than speaking of a selection no longer there."""
        if self.simulation is None:
            return
        living = {agent.identifier for agent in self.simulation.living}
        if self.selected_id is not None and self.selected_id not in living:
            self.selected_id = None
        if self.graph.selection - living:
            self.graph.select(self.graph.selection & living)

    def _draw(self) -> None:
        UI.fit(self.screen.get_size())
        self._forget_the_dead()
        self.screen.fill((19, 26, 27))
        if self.hud_hidden:  # the map or the globe alone, filling the window (H brings the rest back)
            self._draw_world()
            if self.dialog:
                self.dialog.draw(self.screen)
            if self.report:
                self.report.draw(self.screen)
            return
        self._draw_header()
        self._draw_world()
        self._draw_view_buttons()
        self._draw_exemplars()
        self._draw_sidebar()
        self._draw_footer()
        self._draw_square()
        self._draw_exemplar_menu()  # over everything but the graph and dialogs
        self.language_menu.draw_list(self.screen)
        if self.show_census:
            note = self.message if pg.time.get_ticks() < self.message_until else None
            self.graph.draw(self.screen, self.simulation, note)
        if self.dialog:
            self.dialog.draw(self.screen)
        if self.report:
            self.report.draw(self.screen)

    def _open_death_report(self) -> None:
        """Every death so far, by cause and species."""
        total = self.simulation.deaths
        self.report = Report("Death report", f"Every death in this world up to turn {self.simulation.turn}: {total} in all, "
                             "by cause and species, who killed whom and what the killers were doing, and the deadliest "
                             "killers.", death_report_lines(self.simulation.death_tally, tr))

    def _inspect_brains(self) -> None:
        """Pause, and show the selected Bitzkey's chance of choosing each action in each standard situation."""
        agent = next((item for item in self.simulation.agents if item.identifier == self.selected_id), None)
        if agent is None or agent.brain is None:
            self._say("Only a living animal has a brain to inspect.")
            return
        self.playing = False
        self.report = Report(f"Brains of #{agent.identifier:03d}, {agent.species}",
                             "Its chance of choosing each action open to it, in each standard situation, with its temperament "
                             "now. * marks the likeliest; · an action not open.", brain_matrix_lines(agent, tr))

    def _draw_header(self) -> None:
        pg.draw.rect(self.screen, (27, 37, 37), (0, 0, self.screen.get_width(), UI(62)))
        title = self.bold.render("BITZKEY", True, (229, 239, 220))
        self.screen.blit(title, (UI(24), UI(11)))
        subtitle = self.small.render("ARTIFICIAL ECOLOGY  /  RESEARCH BUILD", True, (144, 170, 151))
        self.screen.blit(subtitle, (UI(48) + title.get_width(), UI(20)))
        self._text(f"/  {about()}", (UI(56) + title.get_width() + subtitle.get_width(), UI(20)), (144, 170, 151),
                   self.small)  # the version, and the cores
        status = "RUNNING" if self.playing else "ENDED" if self.simulation.ended else "PAUSED"
        status_color = (131, 205, 143) if self.playing else (238, 183, 110)
        menu_left = self.language_menu.draw_button(self.screen, self.screen.get_width() - UI(24), UI(31))
        label = self.font.render(status, True, status_color)
        self.screen.blit(label, (menu_left - label.get_width() - UI(20), UI(20)))

    def _draw_ground(self, surface: pg.Surface, board: pg.Rect, cell_width: float, cell_height: float, view: pg.Rect) -> None:
        """The ground: each row in its temperature's colour, the sea, their textures, the grass and the night."""
        reach = view.inflate(2 * cell_width + UI(20), 2 * cell_height + UI(20))  # what may show, near enough
        for y in range(self.height):  # each row in its temperature's colour
            top, bottom = round(board.top + y * cell_height), round(board.top + (y + 1) * cell_height)
            pg.draw.rect(surface, zone_colour(self.simulation.temperature(y)),
                         (board.left, top, board.width, bottom - top))
            if not self.simulation.has_sea:
                continue
            sea, x = sea_colour(self.simulation.temperature(y)), 0
            while x < self.width:  # each stretch of sea along the row at once
                if not self.simulation.sea(x, y):
                    x += 1
                    continue
                end = x
                while end < self.width and self.simulation.sea(end, y):
                    end += 1
                left, right = round(board.left + x * cell_width), round(board.left + end * cell_width)
                pg.draw.rect(surface, sea, (left, top, right - left, bottom - top))
                x = end
        size = round(min(cell_width, cell_height))
        detailed = size >= TEXTURE_FROM
        if detailed:  # the ground's texture by the temperature now (sand to ice), or waves at sea
            x0, x1 = max(0, int((view.left - board.left) // cell_width) - 1), min(self.width, int((view.right - board.left) // cell_width) + 2)
            y0, y1 = max(0, int((view.top - board.top) // cell_height) - 1), min(self.height, int((view.bottom - board.top) // cell_height) + 2)
            for y in range(y0, y1):
                land = GROUNDS[max(0, min(3, round(self.simulation.temperature(y))))]
                for x in range(x0, x1):
                    variant = ((x * 73856093) ^ (y * 19349663)) % GROUND_VARIANTS
                    surface.blit(ground_tile("sea" if self.simulation.sea(x, y) else land, size, variant),
                                     (round(board.left + x * cell_width), round(board.top + y * cell_height)))
        for x, y in self.simulation.grass:  # grass: tufts, or kelp at sea, where squares are large enough; else dots
            point = (round(board.left + (x + 0.5) * cell_width),
                     round(board.top + (y + 0.5) * cell_height))
            if not reach.collidepoint(point):
                continue
            if detailed:
                surface.blit(grass_tuft(size, self.simulation.sea(x, y)),
                                 (round(board.left + x * cell_width), round(board.top + y * cell_height)))
            else:
                pg.draw.circle(surface, SEA_GRASS if self.simulation.sea(x, y) else GRASS, point,
                               max(1, round(min(cell_width, cell_height) * 0.18)))
        night = pg.Surface((max(1, round(self.simulation.width // 2 * cell_width)), max(1, round(board.height))), pg.SRCALPHA)
        night.fill(NIGHT)  # the night, a shadow over half the map's columns, wrapping round east to west
        for start in (self.simulation.night_start(), self.simulation.night_start() - self.simulation.width):
            surface.blit(night, (round(board.left + start * cell_width), board.top))

    def _draw_world(self) -> None:
        if self.blind:  # neither map nor globe drawn, to spare the time it takes
            view = self._map_view()[0]
            text = self.small.render("BLIND: the map isn't drawn", True, (116, 139, 124))
            self.screen.blit(text, text.get_rect(center=view.center))
            return
        if self.globe:
            self._draw_globe()
            return
        board, cell_width, cell_height = self._board_geometry()
        view = self._map_view()[0].inflate(2, 2)
        self.screen.set_clip(view)  # zoomed in, only part of the map shows
        reach = view.inflate(2 * cell_width + UI(20), 2 * cell_height + UI(20))  # what may show, near enough
        self._draw_ground(self.screen, board, cell_width, cell_height, view)
        smallest, largest = map_dot_radii(min(cell_width, cell_height))  # zoomed with the map
        least = smallest_shown(self.map_zoom, 1.0, MAP_ZOOM_MAX)
        shown = lambda agent: int(agent.body["size"]) >= least or self._marked(agent)  # noqa: E731
        for agent in self.simulation.agents:  # the way each came this turn, faint, beneath them all
            if not shown(agent):
                continue
            for x, y in agent.trail:
                point = (round(board.left + (x + 0.5) * cell_width), round(board.top + (y + 0.5) * cell_height))
                if (x, y) == (agent.x, agent.y) or not reach.collidepoint(point):
                    continue
                draw_bitzkey(self.screen, agent.dna, point, self._dot_radius(agent, largest, smallest), alive=agent.alive,
                             opacity=TRAIL_OPACITY)
        drawn = hidden = 0
        for agent in self.simulation.agents:
            point = (round(board.left + (agent.x + 0.5) * cell_width),
                     round(board.top + (agent.y + 0.5) * cell_height))
            if not reach.collidepoint(point):
                continue
            if not shown(agent):  # too small to see from this far out
                hidden += 1
                continue
            drawn += 1
            radius = self._dot_radius(agent, largest, smallest)
            self._draw_bitzkey(agent, point, radius, self._corpse_opacity(agent))
            if agent.identifier == self.selected_id or (agent.alive and agent.identifier in self.graph.selection):
                draw_reticle(self.screen, point, round((radius + RING_WIDTH + UI(3)) * reticle_pulse()))
            if agent.identifier in self.export_list:
                pg.draw.circle(self.screen, EXPORTING, point, radius + RING_WIDTH + UI(6), max(1, UI(1)))
        if self._marquee_from:  # the marquee being drawn
            start, mouse = self._marquee_from[0], pg.mouse.get_pos()
            if max(abs(mouse[0] - start[0]), abs(mouse[1] - start[1])) >= UI(DRAG_THRESHOLD):
                box = pg.Rect(min(start[0], mouse[0]), min(start[1], mouse[1]), abs(mouse[0] - start[0]) + 1,
                              abs(mouse[1] - start[1]) + 1)
                veil = pg.Surface(box.size, pg.SRCALPHA)
                veil.fill((245, 238, 206, 40))
                self.screen.blit(veil, box.topleft)
                pg.draw.rect(self.screen, (245, 238, 206), box, 1)
        for x in range(0, self.width + 1, 10):
            px = round(board.left + x * cell_width)
            pg.draw.line(self.screen, GRID, (px, board.top), (px, board.bottom), 1)
        for y in range(0, self.height + 1, 10):
            py = round(board.top + y * cell_height)
            pg.draw.line(self.screen, GRID, (board.left, py), (board.right, py), 1)
        pg.draw.rect(self.screen, (80, 99, 85), board, 1)
        if not drawn and hidden:
            self._say_too_far(view)
        self.screen.set_clip(None)

    def _draw_exemplars(self) -> None:
        """Five cards a side, evenly spaced, each the living animal that stands out in its way; a
        click selects it."""
        self._exemplar_cards, self._exemplar_titles = [], []
        exemplars = self._exemplars()
        gap = UI(8)
        for column, picks in zip(self._exemplar_columns(), (exemplars[:5], exemplars[5:])):
            height = (column.height - 4 * gap) // 5
            for index, (title, agent, said) in enumerate(picks):
                card = pg.Rect(column.left, column.top + index * (height + gap), column.width, height)
                chosen = agent is not None and agent.identifier == self.selected_id
                pg.draw.rect(self.screen, (27, 37, 37), card, border_radius=UI(5))
                pg.draw.rect(self.screen, (238, 183, 110) if chosen else (54, 68, 62), card, 1, border_radius=UI(5))  # the selected one lit
                heading = pg.Rect(card.left + UI(5), card.top + UI(5), card.width - UI(10), UI(22))  # a menu of statistics
                pg.draw.rect(self.screen, (44, 57, 52), heading, border_radius=UI(4))
                if self._exemplar_menu == len(self._exemplar_titles):
                    pg.draw.rect(self.screen, (131, 205, 143), heading, 1, border_radius=UI(4))
                self._text(self._fit(title, heading.width - UI(26)), (heading.left + UI(5), heading.top + UI(3)),
                           (220, 229, 211), self.small)
                x, y = heading.right - UI(10), heading.centery
                pg.draw.polygon(self.screen, (220, 229, 211), [(x - UI(4), y - UI(2)), (x + UI(4), y - UI(2)), (x, y + UI(3))])
                self._exemplar_titles.append(heading)
                if agent is None:
                    self._text(said, (card.left + UI(10), card.top + UI(28)), (116, 139, 124), self.small)
                    continue
                radius = max(1, min(UI(16), height // 5))
                point = (card.left + UI(10) + radius + RING_WIDTH, card.top + (height + UI(20)) // 2)
                draw_bitzkey(self.screen, agent.dna, point, radius)
                x = point[0] + radius + UI(12)
                self._text(f"#{agent.identifier:03d}  {agent.species}", (x, point[1] - UI(16)), (231, 236, 222),
                           self.small)
                self._text(said, (x, point[1] + UI(2)), (149, 170, 150), self.small)
                self._exemplar_cards.append((card, agent.identifier))

    def _draw_exemplar_menu(self) -> None:
        """The open card's statistics, beneath its title, in as many columns as the screen needs."""
        self._menu_items = []
        if self._exemplar_menu is None or self._exemplar_menu >= len(self._exemplar_titles):
            return
        heading = self._exemplar_titles[self._exemplar_menu]
        row, column = UI(MENU_ROW), UI(MENU_COLUMN)
        top = heading.bottom + UI(2)
        rows = max(1, min(len(EXEMPLAR_STATS), (self.screen.get_height() - top - UI(12)) // row))
        columns = -(-len(EXEMPLAR_STATS) // rows)
        rows = -(-len(EXEMPLAR_STATS) // columns)  # as even as the columns allow
        width = columns * column + UI(8)
        left = heading.left if self._exemplar_menu < 5 else heading.right - width  # toward the map
        left = max(UI(4), min(self.screen.get_width() - width - UI(4), left))
        panel = pg.Rect(left, top, width, rows * row + UI(8))
        pg.draw.rect(self.screen, (27, 37, 37), panel, border_radius=UI(5))
        pg.draw.rect(self.screen, (131, 205, 143), panel, 1, border_radius=UI(5))
        current = EXEMPLAR_CHOICES[self._exemplar_menu]
        for stat, (title, _, _, _) in enumerate(EXEMPLAR_STATS):
            item = pg.Rect(panel.left + UI(4) + stat // rows * column, panel.top + UI(4) + stat % rows * row,
                           column, row)
            if item.collidepoint(pg.mouse.get_pos()):
                pg.draw.rect(self.screen, (44, 57, 52), item, border_radius=UI(3))
            self._text(title, (item.left + UI(6), item.top + UI(2)),
                       (131, 205, 143) if stat == current else (220, 229, 211), self.small)
            self._menu_items.append((item, stat))

    def _draw_sidebar(self) -> None:
        sidebar_x = self.screen.get_width() - self._sidebar_width()
        pg.draw.line(self.screen, (54, 68, 62), (sidebar_x, UI(62)), (sidebar_x, self.screen.get_height() - UI(58)), 1)
        x = sidebar_x + UI(20)
        width = self.screen.get_width() - x - UI(20)
        counts = Counter(agent.species for agent in self.simulation.living)
        self._text(f"TURN  {self.simulation.turn:05d}", (x, UI(78)), (232, 236, 219), self.bold)
        self._text(f"{len(self.simulation.living)} ALIVE    {self.simulation.deaths} LOST",
                   (x, UI(108)), (157, 177, 159), self.small)
        turn = self.simulation.turn
        label = self.font.render(f"YEAR {year(turn)}", True, (232, 236, 219))
        self.screen.blit(label, (x + width - label.get_width(), UI(84)))
        label = self.small.render(SEASONS[season(turn)].upper(), True,
                                  (140, 185, 230) if season(turn) % 2 else (157, 177, 159))
        self.screen.blit(label, (x + width - label.get_width(), UI(108)))
        pg.draw.line(self.screen, (54, 68, 62), (x, UI(138)), (x + width, UI(138)), 1)
        self._text("POPULATION", (x, UI(153)), (220, 229, 211), self.font)
        for index, species in enumerate(SPECIES):
            y = UI(184 + index * 23)
            self._text(species.upper(), (x, y), (149, 170, 150), self.small)
            self._text(str(counts.get(species, 0)), (x + width - UI(32), y), (231, 236, 222), self.font)

        pg.draw.line(self.screen, (54, 68, 62), (x, UI(286)), (x + width, UI(286)), 1)
        self._add_button = self._catalog_button = self._deaths_button = self._brains_button = None
        if self.show_insights:
            self._draw_insights(x, width)
            return
        self._text("SELECTED BITZKEY", (x, UI(301)), (220, 229, 211), UI.fitting(self.font, "SELECTED BITZKEY", width - UI(140)))
        agent = next((item for item in self.simulation.agents if item.identifier == self.selected_id), None)
        if agent is None:
            self._text("Click a dot to inspect its genome.", (x, UI(332)), (146, 165, 149), self.small)
            self._text("Living: click its current position.", (x, UI(352)), (116, 139, 124), self.small)
            if self.graph.selection:
                count = len(self.graph.selection)
                self._text(f"{count} picked on the colour graph, ringed on the map.", (x, UI(380)),
                           (238, 183, 110), self.small)
            return

        draw_details(self.screen, agent, x, lambda offset: UI(332 + offset), width,
                     here=self.simulation.temperature(agent.y))
        self._text("C  copy as JSON      A  export      K  catalog", (x, UI(719)), (116, 139, 124), self.small)
        chosen = agent.identifier in self.export_list
        half = (width - UI(8)) // 2
        self._add_button = pg.Rect(x, UI(737), half, UI(28))
        self._catalog_button = pg.Rect(x + half + UI(8), UI(737), width - half - UI(8), UI(28))
        pg.draw.rect(self.screen, (44, 57, 52), self._catalog_button, border_radius=UI(5))
        draw_label(self.screen, UI.font("segoeui", 13, bold=True), "ADD TO CATALOG", (220, 229, 211), self._catalog_button)
        pg.draw.rect(self.screen, (44, 57, 52), self._add_button, border_radius=UI(5))
        if chosen:
            pg.draw.rect(self.screen, (238, 183, 110), self._add_button, 1, border_radius=UI(5))
        draw_label(self.screen, UI.font("segoeui", 13, bold=True), "REMOVE FROM EXPORT" if chosen else "ADD TO EXPORT",
              (238, 183, 110) if chosen else (220, 229, 211), self._add_button)
        if agent.brain is not None:  # a living animal: its brain can be inspected, beside the heading
            self._brains_button = pg.Rect(x + width - UI(130), UI(299), UI(130), UI(26))
            pg.draw.rect(self.screen, (44, 57, 52), self._brains_button, border_radius=UI(5))
            draw_label(self.screen, UI.font("segoeui", 13, bold=True), "INSPECT BRAINS", (220, 229, 211), self._brains_button)

    @staticmethod
    def _dot_radius(agent: Bitzkey, largest: int = DOT_RADIUS_RANGE[1], smallest: int = DOT_RADIUS_RANGE[0]) -> int:
        return dot_radius(int(agent.body["size"]), largest, smallest)

    def _draw_bitzkey(self, agent: Bitzkey, point: tuple[int, int], radius: int, opacity: float = 1.0) -> None:
        draw_bitzkey(self.screen, agent.dna, point, radius, alive=agent.alive, opacity=opacity)

    def _corpse_opacity(self, agent: Bitzkey) -> float:
        """How visible a Bitzkey is: fully while alive, then fading as its corpse decays, gone
        when the corpse is."""
        if agent.died_turn is None:
            return 1.0
        return max(0.0, 1 - (self.simulation.turn - agent.died_turn) / world.CORPSE_TURNS)  # read live: it is a tunable rule

    # The insights panel scrolls within the sidebar, below the population counts.

    def _insights_view(self) -> pg.Rect:
        sidebar_x = self.screen.get_width() - self._sidebar_width()
        top = UI(INSIGHTS_TOP)
        return pg.Rect(sidebar_x + 1, top, self._sidebar_width() - 1, self.screen.get_height() - UI(58) - top)

    def _scroll_track(self) -> pg.Rect:
        view = self._insights_view()
        return pg.Rect(self.screen.get_width() - UI(11), view.top + UI(4), UI(6), view.height - UI(8))

    def _max_scroll(self) -> int:
        return max(0, self._insights_height - self._insights_view().height)

    def _scroll_thumb(self) -> pg.Rect | None:
        """The scroll bar's thumb, sized to the share of the panel in view; none when it all fits."""
        track, most = self._scroll_track(), self._max_scroll()
        if not most:
            return None
        height = max(UI(24), track.height * self._insights_view().height // self._insights_height)
        return pg.Rect(track.left, track.top + (track.height - height) * self.insights_scroll // most, track.width, height)

    def _scroll_to(self, scroll: int) -> None:
        self.insights_scroll = max(0, min(self._max_scroll(), scroll))

    def _scroll_event(self, event: pg.event.Event) -> bool:
        """Scroll the insights panel with the wheel over the sidebar, or by dragging its thumb.
        Returns whether the event was used."""
        view = self._insights_view()
        if event.type == pg.MOUSEWHEEL and view.collidepoint(pg.mouse.get_pos()):
            self._scroll_to(self.insights_scroll - event.y * UI(48))
            return True
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            track, thumb = self._scroll_track().inflate(UI(8), 0), self._scroll_thumb()
            if thumb is None or not track.collidepoint(event.pos):
                return False
            if not thumb.collidepoint(event.pos):  # a click on the track jumps the thumb there
                self._scroll_to((event.pos[1] - track.top - thumb.height // 2) * self._max_scroll()
                                // max(1, self._scroll_track().height - thumb.height))
                thumb = self._scroll_thumb()
            self._scroll_grip = event.pos[1] - thumb.top
            return True
        if event.type == pg.MOUSEMOTION and self._scroll_grip is not None:
            track, thumb = self._scroll_track(), self._scroll_thumb()
            if thumb is not None:
                self._scroll_to((event.pos[1] - self._scroll_grip - track.top) * self._max_scroll()
                                // max(1, track.height - thumb.height))
            return True
        if event.type == pg.MOUSEBUTTONUP and event.button == 1 and self._scroll_grip is not None:
            self._scroll_grip = None
            return True
        return False

    def _draw_insights(self, x: int, width: int) -> None:
        """What the animals are doing and what they are like, by species, scrolled within the
        sidebar. Lengths are in design pixels from the top of the sidebar."""
        insights = self.simulation.insights
        view = self._insights_view()
        self._scroll_to(self.insights_scroll)
        offset = self.insights_scroll
        self.screen.set_clip(view)

        def at(y: float) -> int:
            return UI(y) - offset

        columns = [x + UI(96) + index * (width - UI(96)) // len(ANIMALS) for index in range(len(ANIMALS))]

        def heading(title: str, note: str, y: float) -> float:
            self._text(title, (x, at(y)), (220, 229, 211), self.font)
            self._text(note, (x, at(y + 22)), (116, 139, 124), self.small)
            for column, species in zip(columns, ANIMALS):
                self._text(species[:4].upper(), (column, at(y + 46)), (149, 170, 150), self.small)
            return y + 66

        button = pg.Rect(x + width - UI(130), at(INSIGHTS_TOP + 9), UI(130), UI(26))  # every death so far, opened
        if view.colliderect(button):
            self._deaths_button = button.clip(view)
            pg.draw.rect(self.screen, (44, 57, 52), button, border_radius=UI(5))
            draw_label(self.screen, UI.font("segoeui", 13, bold=True), "DEATH REPORT", (220, 229, 211), button)
        y = heading("BEHAVIOUR", "% of chances taken, recent turns", INSIGHTS_TOP + 11)
        for action in ACTIONS:
            self._text(action.upper(), (x, at(y)), (149, 170, 150), self.small)
            for column, species in zip(columns, ANIMALS):
                share, _ = insights.rate(action, species)
                self._text("  -" if share is None else f"{share:>4.0%}", (column, at(y)), (231, 236, 222), self.mono)
            y += 17

        blocks = (
            ("LEAST KEEN TO BREED", insights.extremes("breed", 3, lowest=True)),
            ("KEENEST TO BREED", insights.extremes("breed", 2, lowest=False)),
            ("LEAST KEEN TO FIGHT", insights.extremes("attack", 2, lowest=True)),
        )
        y += 6
        for title, groups in blocks:
            self._text(title, (x, at(y)), (220, 229, 211), self.small)
            y += 19
            if not groups:
                self._text("not enough chances yet", (x, at(y)), (116, 139, 124), self.small)
                y += 18
            for label, share, _ in groups:
                self._text(label, (x, at(y)), (175, 192, 172), self.small)
                self._text(f"{share:>4.0%}", (x + width - UI(40), at(y)), (231, 236, 222), self.mono)
                y += 18
            y += 8

        members = {species: [agent for agent in self.simulation.living if agent.species == species]
                   for species in ANIMALS}

        def average(species: str, value) -> float | None:
            group = members[species]
            return sum(value(agent) for agent in group) / len(group) if group else None

        y = heading("LIFE", "averages of the living", y + 10)
        rows = (
            ("ALIVE", lambda species: str(len(members[species]))),
            ("SIZE", lambda species: _shown(average(species, lambda agent: agent.body["size"]), "{:.1f}")),
            ("AGE", lambda species: _shown(average(species, lambda agent: agent.age), "{:.0f}")),
            ("GENERATION", lambda species: _shown(average(species, lambda agent: agent.generation), "{:.1f}")),
            ("HUNGER", lambda species: _shown(average(species, lambda agent: min(100.0, agent.hunger) / 100), "{:.0%}")),
            ("HEALTH", lambda species: _shown(average(species, lambda agent: max(0.0, agent.health) / agent.max_health),
                                              "{:.0%}")),
            ("CHILDREN", lambda species: _shown(average(species, lambda agent: agent.children), "{:.1f}")),
            ("HERDING", lambda species: _shown(average(species, lambda agent: agent.herd_instinct), "{:.0%}")),
            ("RIVALRY", lambda species: "-" if species == "grazer"
             else _shown(average(species, lambda agent: agent.rivalry), "{:.0%}")),
        )
        for label, value in rows:
            self._text(label, (x, at(y)), (149, 170, 150), self.small)
            for column, species in zip(columns, ANIMALS):
                self._text(f"{value(species):>4}", (column, at(y)), (231, 236, 222), self.mono)
            y += 17

        y = heading("TEMPERAMENT", "average now; green above inborn, amber below", y + 14)
        for trait in TEMPERAMENT:
            self._text(trait.upper(), (x, at(y)), (149, 170, 150), self.small)
            for column, species in zip(columns, ANIMALS):
                now = average(species, lambda agent: agent.mind[trait])
                if now is None:
                    self._text("   -", (column, at(y)), (231, 236, 222), self.mono)
                    continue
                shift = now - average(species, lambda agent: agent.inborn[trait])
                color = ((231, 236, 222) if abs(shift) < 0.25 else (131, 205, 143) if shift > 0 else (238, 183, 110))
                self._text(f"{now:>4.1f}", (column, at(y)), color, self.mono)
            y += 17

        self._text("KEENEST BY ACTION", (x, at(y + 14)), (220, 229, 211), self.font)
        self._text("the group most likely to take each chance", (x, at(y + 36)), (116, 139, 124), self.small)
        y += 60
        shown = False
        for action in ACTIONS:
            groups = insights.extremes(action, 1, lowest=False)
            if not groups:
                continue
            label, share, _ = groups[0]
            self._text(action.upper(), (x, at(y)), (149, 170, 150), self.small)
            self._text(self._fit(label, width - UI(124)), (x + UI(76), at(y)), (175, 192, 172), self.small)
            self._text(f"{share:>4.0%}", (x + width - UI(40), at(y)), (231, 236, 222), self.mono)
            y += 18
            shown = True
        if not shown:
            self._text("not enough chances yet", (x, at(y)), (116, 139, 124), self.small)
            y += 18

        self._insights_height = UI(y + 12) - UI(INSIGHTS_TOP)
        self.screen.set_clip(None)
        thumb = self._scroll_thumb()
        if thumb is not None:
            pg.draw.rect(self.screen, (36, 49, 47), self._scroll_track(), border_radius=UI(3))
            pg.draw.rect(self.screen, (238, 183, 110) if self._scroll_grip is not None else (96, 120, 104), thumb,
                         border_radius=UI(3))

    def _save_report(self) -> None:
        report = brain_report(self.simulation.turn, self.simulation.living, self.simulation.insights)
        REPORT_DIR.mkdir(exist_ok=True)
        path = REPORT_DIR / f"brain-report-turn-{self.simulation.turn:05d}.txt"
        path.write_text(report, encoding="utf-8")
        print(report)
        self._say(f"Brain report saved to reports/{path.name}")

    def _new_world(self) -> Simulation:
        return Simulation(self.width, self.height, self.population, self.seed, self.grass_density, self.imported,
                          self.population_cap, self.mind_mutation, self.body_mutation, self.year_length, self.ocean)

    def _say(self, message: str) -> None:
        self.message = message
        self.message_until = pg.time.get_ticks() + 5000

    def _copy(self, text: str, what: str) -> None:
        try:
            pg.scrap.put_text(text)
        except pg.error as error:
            self._say(f"Couldn't copy {what} to the clipboard: {error}")
        else:
            self._say(f"Copied {what} to the clipboard.")

    def _copy_selected(self) -> None:
        agent = next((item for item in self.simulation.agents if item.identifier == self.selected_id), None)
        if agent is None:
            self._say("Click a Bitzkey first, then press C to copy its genome.")
        else:
            what = "genome and brain" if agent.brain else "genome"
            self._copy(export_bitzkeys([agent]), f"#{agent.identifier:03d}'s {what}")

    def _export_graph(self, to_file: bool) -> None:
        """Export the Bitzkeys selected in the colour graph, or everyone alive if none are.
        The living go with brain and temperament; the dead keep no brain, and the long dead
        are remembered only by their genome."""
        chosen = sorted(self.graph.selection)
        agents = {agent.identifier: agent for agent in self.simulation.agents}
        if chosen:
            chosen = [identifier for identifier in chosen if identifier in agents or identifier in self.simulation.lineage]
            records = [bitzkey_record(agents[identifier]) if identifier in agents
                       else genome_record(self.simulation.lineage[identifier]) for identifier in chosen]
            dead = sum(identifier not in agents or not agents[identifier].alive for identifier in chosen)
        else:
            records = [bitzkey_record(agent) for agent in self.simulation.living]
            dead = 0
        if not records:
            self._say("Nobody is alive to export.")
            return
        count = len(records)
        what = f"{count} {'selected ' if chosen else ''}Bitzkey{'s' if count != 1 else ''}"
        if dead:
            what += f" ({dead} dead, so without a brain)"
        text = format_records(records)
        if not to_file:
            self._copy(text, what)
            return
        EXPORT_DIR.mkdir(exist_ok=True)
        path = EXPORT_DIR / f"bitzkeys-turn-{self.simulation.turn:05d}-{'selected' if chosen else 'alive'}-{count}.json"
        path.write_text(text + "\n", encoding="utf-8")
        self._say(f"Saved {what} to exports/{path.name}")

    def _draw_footer(self) -> None:
        height = self.screen.get_height()
        pg.draw.rect(self.screen, (27, 37, 37), (0, height - UI(52), self.screen.get_width(), UI(52)))
        turn_label = f"TURN  {self.simulation.turn}"
        self._text(turn_label, (UI(22), height - UI(36)), (158, 181, 160), self.small)
        label = UI.font("segoeui", 13, bold=True).render(f"EXPORT ALL SELECTED ({len(self.export_list)})", True,
                                                         (19, 26, 27) if self.export_list else (220, 229, 211))
        self._export_button = pg.Rect(0, 0, label.get_width() + UI(32), UI(34))
        self._export_button.midright = (self.screen.get_width() - UI(20), height - UI(26))
        pg.draw.rect(self.screen, (238, 183, 110) if self.export_list else (44, 57, 52), self._export_button,
                     border_radius=UI(6))
        self.screen.blit(label, label.get_rect(center=self._export_button.center))
        label = UI.font("segoeui", 13, bold=True).render("OPEN CATALOG", True, (220, 229, 211))
        self._open_button = pg.Rect(0, 0, label.get_width() + UI(32), UI(34))
        self._open_button.midright = (self._export_button.left - UI(10), height - UI(26))
        pg.draw.rect(self.screen, (44, 57, 52), self._open_button, border_radius=UI(6))
        self.screen.blit(label, label.get_rect(center=self._open_button.center))
        changed = bool(Rules.changed(Rules.current()))  # amber while this world plays by changed rules
        label = UI.font("segoeui", 13, bold=True).render("ADVANCED", True, (19, 26, 27) if changed else (220, 229, 211))
        self._advanced_button = pg.Rect(0, 0, label.get_width() + UI(32), UI(34))
        self._advanced_button.midright = (self._open_button.left - UI(10), height - UI(26))
        pg.draw.rect(self.screen, (238, 183, 110) if changed else (44, 57, 52), self._advanced_button, border_radius=UI(6))
        self.screen.blit(label, label.get_rect(center=self._advanced_button.center))
        label = UI.font("segoeui", 13, bold=True).render("SAVE WORLD", True, (220, 229, 211))
        self._save_button = pg.Rect(0, 0, label.get_width() + UI(32), UI(34))
        self._save_button.midright = (self._advanced_button.left - UI(10), height - UI(26))
        pg.draw.rect(self.screen, (44, 57, 52), self._save_button, border_radius=UI(6))
        self.screen.blit(label, label.get_rect(center=self._save_button.center))
        left = max(UI(185), UI(22) + self.small.size(turn_label)[0] + UI(24))  # clear of the turn, however long
        room = self._save_button.left - left - UI(16)
        if pg.time.get_ticks() < self.message_until:
            self._footer_lines(self.message, left, room, (238, 183, 110))
            return
        self._footer_lines("SPACE  pause   RIGHT  step   WHEEL / Z  zoom   I  insights   B  report   "
                           "G  graph   E  export   R  restart   F11  window   ESC  menu", left, room, (187, 199, 181))

    def _footer_lines(self, text: str, left: int, width: int, color: tuple[int, int, int]) -> None:
        """A footer note or the keys, on one line if it fits, else on two, broken between groups of keys (or words),
        the second shortened with an ellipsis only if it still doesn't fit."""
        height = self.screen.get_height()
        lines = self._wrapped(text, width)
        tops = [height - UI(36)] if len(lines) == 1 else [height - UI(44), height - UI(28)]
        for line, top in zip(lines, tops):
            self._text(line, (left, top), color, self.small)

    def _wrapped(self, text: str, width: int) -> list[str]:
        """Text in the small font in at most two lines of the width, broken between groups (three spaces apart) where it
        has them, else between words."""
        text = tr(text)  # wrapped in the language it is shown in
        if self.small.size(text)[0] <= width:
            return [text]
        gap = "   " if "   " in text else " "
        parts = text.split(gap)
        first = parts[0]
        taken = 1
        while taken < len(parts) and self.small.size(first + gap + parts[taken])[0] <= width:
            first += gap + parts[taken]
            taken += 1
        return [self._fit(first, width), self._fit(gap.join(parts[taken:]), width)] if taken < len(parts) else [self._fit(first, width)]

    def _fit(self, text: str, width: int) -> str:
        """Shorten text in the small font, with an ellipsis, to fit the width: footer notes, and
        group names in the insights panel."""
        return fit_text(self.small, text, width)

    def _text(self, text: str, position: tuple[int, int], color: tuple[int, int, int],
              font: pg.font.Font) -> None:
        self.screen.blit(font.render(text, True, color), position)


def reticle_pulse() -> float:
    """How large the selected Bitzkeys' sights are now: from their size to twice it and back, every RETICLE_PULSE_MS."""
    return 1.5 - 0.5 * math.cos(2 * math.pi * pg.time.get_ticks() / RETICLE_PULSE_MS)


def _shown(value: float | None, pattern: str) -> str:
    """A number for the insights tables, or a dash when there is none."""
    return "-" if value is None else pattern.format(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Bitzkey artificial ecology.")
    parser.add_argument("--width", type=int, default=Config.GRID_WIDTH, help="world width in cells (default: 360)")
    parser.add_argument("--height", type=int, default=Config.GRID_HEIGHT, help="world height in cells (default: 180)")
    parser.add_argument("--population", type=int, default=Config.INITIAL_POPULATION)
    parser.add_argument("--grass-density", type=int, default=Config.GRASS_DENSITY,
                        help="percent of cells with grass at the start (default: 5)")
    parser.add_argument("--ocean", type=int, default=Config.OCEAN,
                        help="percent of the map under the sea, laid out by the seed (default: 67)")
    parser.add_argument("--seed", type=int, default=None, help="seed the world for repeatable experiments")
    parser.add_argument("--population-cap", type=int, default=None,
                        help="most animals alive at once (default: no limit; crowding decides)")
    parser.add_argument("--mind-mutation", type=int, default=Config.MIND_MUTATION,
                        help="percent chance a birth mutates the child's mind, 0 to 100 (default: 50)")
    parser.add_argument("--body-mutation", type=int, default=Config.BODY_MUTATION,
                        help="percent chance a birth mutates the child's body, 0 to 100 (default: 50)")
    parser.add_argument("--year-length", type=int, default=Config.YEAR_LENGTH,
                        help="turns in a year, of four seasons (default: 360)")
    parser.add_argument("--windowed", action="store_true", help="start in a window instead of fullscreen")
    return parser.parse_args()


def main() -> None:
    """Configure a world, run it, and come back to the configuration panel until the user quits."""
    options = parse_args()
    world.remove_stale_histories()  # left behind by a run that ended without closing its world
    screen = Window.open_window(fullscreen=not options.windowed)
    start_language()  # the language chosen last time, else the system's
    setup = SetupScreen(screen, vars(options))
    settings = setup.run()
    while settings is not None:
        rules = settings.pop("rules", {})
        if "simulation" not in settings:  # a world loaded brought its own rules
            Rules.apply(rules)  # the advanced rules hold for this world and its restarts
        app = BitzkeyApp(**settings)
        outcome = app.run()
        if outcome == "quit":
            break
        # A new world with the Bitzkeys picked on the graph starts at once; otherwise, back to the panel.
        settings = setup.start_with(app.renewal) if outcome == "renew" else setup.run()
    pg.quit()


if __name__ == "__main__":
    main()
    # Quitting ends the process at once, rather than waiting on Python to tidy away every last
    # brain the session made.
    if sys.stdout:
        sys.stdout.flush()
    os._exit(0)
