"""PyGame interface for the Bitzkey ecology experiment."""

import argparse
import os
import sys
from collections import Counter
from collections.abc import Sequence

import pygame as pg

from config.Globals import Config
from display import Window
from display.Dialog import Dialog
from display.Inspector import draw_details
from display.Look import (DOT_RADIUS_RANGE, EXPORTING, GRASS, GRID, RING_WIDTH, SELECTED, bitzkey_layer, dot_radius,
                          draw_bitzkey, zone_colour)
from display.Setup import EXPORT_DIR, SetupScreen
from display.Ui import UI
from tools.Brain import ACTIONS
from display.ColourGraph import ColourGraph
from tools.Exchange import Imported, bitzkey_record, export_bitzkeys, format_records
from tools.Genome import genome_record
from tools.Genome import MIND_TRAITS
from tools.Insights import ANIMALS, brain_report
from tools import Catalog, Rules
from tools import Simulation as world
from tools.Simulation import SEASONS, SPECIES, Bitzkey, Simulation, season, year

REPORT_DIR = Config.APP_DIR / "reports"
INSIGHTS_TOP = 290  # the insights panel's top, in design pixels: just below the population counts
MAP_ZOOM_STEP = 1.25  # each wheel notch zooms the map in (or out) by this much
MAP_ZOOM_MAX = 12.0   # the closest the map zooms in, times its whole-map size
TRAIL_OPACITY = 0.5   # how faintly a Bitzkey shows on each cell it passed through this turn


class BitzkeyApp:
    def __init__(self, width: int, height: int, population: int, grass_density: int, seed: int | None,
                 population_cap: int | None = None, mind_mutation: int = Config.MIND_MUTATION,
                 body_mutation: int = Config.BODY_MUTATION, year_length: int = Config.YEAR_LENGTH,
                 imported: Sequence[Imported] = ()):
        self.screen = Window.open_window()
        self.width = width
        self.height = height
        self.population = population
        self.grass_density = grass_density
        self.population_cap = population_cap
        self.mind_mutation = mind_mutation
        self.body_mutation = body_mutation
        self.year_length = year_length
        self.seed = seed
        self.imported = list(imported)
        self.simulation = self._new_world()
        self.selected_id: int | None = None
        self.show_insights = False
        self.show_census = False
        # The map's zoom (1 shows it all) and the cell at the middle of the view, while zoomed in;
        # while it is dragged, where the drag began and the middle then.
        self.map_zoom = 1.0
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
        self._dialog_kind = ""
        self._add_button: pg.Rect | None = None
        self._catalog_button: pg.Rect | None = None
        self._export_button = pg.Rect(0, 0, 0, 0)
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
                elif self.dialog:
                    dialog = self.dialog
                    choice = dialog.handle(event, self.screen.get_size())
                    if choice and self._answer(choice) == "leave":
                        self._leave(dialog)
                        return "configure"
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
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
                    elif event.key == pg.K_LEFT:
                        self.playing = False
                        self.simulation.rewind()
                    elif event.key == pg.K_r:
                        self.simulation.close()
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
                    elif event.key == pg.K_z and not self.show_census:
                        self.map_zoom, self.map_centre = 1.0, (self.width / 2, self.height / 2)
                elif self.show_insights and not self.show_census and self._scroll_event(event):
                    pass
                elif not self.show_census and self._map_event(event):
                    pass
                elif self.show_census and event.type in (pg.MOUSEBUTTONDOWN, pg.MOUSEBUTTONUP, pg.MOUSEMOTION,
                                                         pg.MOUSEWHEEL):
                    action = self.graph.handle(event)
                    if action == "seek":
                        self._seek(self.graph.seek_turn)
                    elif action == "map":
                        self._close_graph()  # at the run's end, to look over the map before leaving
                    elif action == "catalog":
                        self._catalog_picked()
                    elif action:
                        self._export_graph(to_file=action == "save")
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    self._click(event.pos)

            self._tick()
            self._draw()
            pg.display.flip()

    def _tick(self) -> None:
        """While playing, advance a turn every 0.24 s, and stop to show the graph once the
        simulation has ended."""
        if not self.playing or self.dialog:
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

    def _seek(self, turn: int) -> None:
        """Pause and move the world to a turn already simulated. A long replay says so first."""
        self.playing = False
        if self.simulation.replay_cost(turn) > 30:
            self._say(f"Replaying to turn {turn}...")
            self._draw()
            pg.display.flip()
        self.simulation.seek(turn)
        self._say(f"Moved to turn {turn}.")

    def _click(self, position: tuple[int, int]) -> None:
        if self._add_button and self._add_button.collidepoint(position):
            self._toggle_export()
        elif self._catalog_button and self._catalog_button.collidepoint(position):
            self._catalog_selected()
        elif self._export_button.collidepoint(position):
            self._open_export_dialog()
        else:
            self._select_at(position)

    # Dialogs

    def _open(self, kind: str, dialog: Dialog) -> None:
        self._dialog_kind, self.dialog = kind, dialog

    def _answer(self, choice: str) -> str | None:
        """Act on a dialog's choice; returns "leave" when the user leaves the simulation."""
        kind, self.dialog = self._dialog_kind, None
        if kind == "leave":
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

    def _leave(self, dialog: Dialog) -> None:
        """Say that it's leaving, then delete the world's history from the disk and let it go."""
        dialog.start("leave", "Leaving, please be patient", "Cleaning up messy neural networks...")
        self.dialog = dialog
        self.playing = False
        self._draw()
        pg.display.flip()
        self.simulation.close()
        self.simulation = None

    def _ask_to_leave(self) -> None:
        lines = ["You'll return to the configuration panel to set up a new world.",
                 "This world won't be kept."]
        if self.export_list:
            count = len(self.export_list)
            lines.append(f"{count} Bitzkey{'s' if count != 1 else ''} added to export will be lost unless exported first.")
        self._open("leave", Dialog("Leave this simulation?", lines, [("stay", "STAY"), ("leave", "LEAVE")],
                                   cancel="stay"))

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

    def _select_at(self, position: tuple[int, int]) -> None:
        board, cell_width, cell_height = self._board_geometry()
        nearest = None
        nearest_distance = float("inf")
        for agent in self.simulation.agents:
            point = (board.left + (agent.x + 0.5) * cell_width,
                     board.top + (agent.y + 0.5) * cell_height)
            distance = pg.Vector2(position).distance_to(point)
            if distance < nearest_distance:
                nearest = agent
                nearest_distance = distance
        on_view = self._map_view()[0].collidepoint(position)  # zoomed in, the board reaches under the sidebar
        if nearest and on_view and board.collidepoint(position) and nearest_distance < max(UI(9), min(cell_width, cell_height) * 1.6):
            self.selected_id = nearest.identifier
            self.graph.select({nearest.identifier})  # the map and the graph share one selection
            self.show_insights = False  # selecting shows the inspector

    def _sidebar_width(self) -> int:
        return min(UI(340), max(UI(290), self.screen.get_width() // 3))

    def _map_view(self) -> tuple[pg.Rect, float]:
        """Where the map shows, and the scale at which the whole map fits it."""
        UI.fit(self.screen.get_size())
        screen_width, screen_height = self.screen.get_size()
        board_area = pg.Rect(0, UI(62), screen_width - self._sidebar_width(), screen_height - UI(140))
        view = board_area.inflate(-UI(42), -UI(42))
        return view, min(view.width / self.width, view.height / self.height)

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

    def _draw(self) -> None:
        UI.fit(self.screen.get_size())
        self.screen.fill((19, 26, 27))
        self._draw_header()
        self._draw_world()
        self._draw_sidebar()
        self._draw_footer()
        if self.show_census:
            note = self.message if pg.time.get_ticks() < self.message_until else None
            self.graph.draw(self.screen, self.simulation, note)
        if self.dialog:
            self.dialog.draw(self.screen)

    def _draw_header(self) -> None:
        pg.draw.rect(self.screen, (27, 37, 37), (0, 0, self.screen.get_width(), UI(62)))
        title = self.bold.render("BITZKEY", True, (229, 239, 220))
        self.screen.blit(title, (UI(24), UI(11)))
        self._text("ARTIFICIAL ECOLOGY  /  RESEARCH BUILD", (UI(48) + title.get_width(), UI(20)),
                   (144, 170, 151), self.small)
        status = "RUNNING" if self.playing else "ENDED" if self.simulation.ended else "PAUSED"
        status_color = (131, 205, 143) if self.playing else (238, 183, 110)
        label = self.font.render(status, True, status_color)
        self.screen.blit(label, (self.screen.get_width() - label.get_width() - UI(24), UI(20)))

    def _draw_world(self) -> None:
        board, cell_width, cell_height = self._board_geometry()
        view = self._map_view()[0].inflate(2, 2)
        self.screen.set_clip(view)  # zoomed in, only part of the map shows
        reach = view.inflate(2 * cell_width + UI(20), 2 * cell_height + UI(20))  # what may show, near enough
        for y in range(self.height):  # each row in its temperature's colour
            top, bottom = round(board.top + y * cell_height), round(board.top + (y + 1) * cell_height)
            pg.draw.rect(self.screen, zone_colour(self.simulation.temperature(y)),
                         (board.left, top, board.width, bottom - top))
        for x, y in self.simulation.grass:
            point = (round(board.left + (x + 0.5) * cell_width),
                     round(board.top + (y + 0.5) * cell_height))
            if not reach.collidepoint(point):
                continue
            pg.draw.circle(self.screen, GRASS, point, max(1, round(min(cell_width, cell_height) * 0.18)))
        largest = max(DOT_RADIUS_RANGE[1], round(min(cell_width, cell_height) * 0.45))
        for agent in self.simulation.agents:  # the way each came this turn, faint, beneath them all
            layer = None
            for x, y in agent.trail:
                point = (round(board.left + (x + 0.5) * cell_width), round(board.top + (y + 0.5) * cell_height))
                if (x, y) == (agent.x, agent.y) or not reach.collidepoint(point):
                    continue
                if layer is None:
                    layer = bitzkey_layer(agent.dna, self._dot_radius(agent, largest), alive=agent.alive)
                    layer.set_alpha(round(255 * TRAIL_OPACITY))
                half = layer.get_width() // 2
                self.screen.blit(layer, (point[0] - half, point[1] - half))
        for agent in self.simulation.agents:
            point = (round(board.left + (agent.x + 0.5) * cell_width),
                     round(board.top + (agent.y + 0.5) * cell_height))
            if not reach.collidepoint(point):
                continue
            radius = self._dot_radius(agent, largest)
            self._draw_bitzkey(agent, point, radius, self._corpse_opacity(agent))
            if agent.identifier == self.selected_id or agent.identifier in self.graph.selection:
                pg.draw.circle(self.screen, SELECTED, point, radius + RING_WIDTH + UI(3), max(1, UI(1)))
            if agent.identifier in self.export_list:
                pg.draw.circle(self.screen, EXPORTING, point, radius + RING_WIDTH + UI(6), max(1, UI(1)))
        for x in range(0, self.width + 1, 10):
            px = round(board.left + x * cell_width)
            pg.draw.line(self.screen, GRID, (px, board.top), (px, board.bottom), 1)
        for y in range(0, self.height + 1, 10):
            py = round(board.top + y * cell_height)
            pg.draw.line(self.screen, GRID, (board.left, py), (board.right, py), 1)
        pg.draw.rect(self.screen, (80, 99, 85), board, 1)
        self.screen.set_clip(None)

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
        self._add_button = self._catalog_button = None
        if self.show_insights:
            self._draw_insights(x, width)
            return
        self._text("SELECTED BITZKEY", (x, UI(301)), (220, 229, 211), self.font)
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
        label = UI.font("segoeui", 13, bold=True).render("ADD TO CATALOG", True, (220, 229, 211))
        self.screen.blit(label, label.get_rect(center=self._catalog_button.center))
        pg.draw.rect(self.screen, (44, 57, 52), self._add_button, border_radius=UI(5))
        if chosen:
            pg.draw.rect(self.screen, (238, 183, 110), self._add_button, 1, border_radius=UI(5))
        label = UI.font("segoeui", 13, bold=True).render("REMOVE FROM EXPORT" if chosen else "ADD TO EXPORT", True,
                                                         (238, 183, 110) if chosen else (220, 229, 211))
        self.screen.blit(label, label.get_rect(center=self._add_button.center))

    @staticmethod
    def _dot_radius(agent: Bitzkey, largest: int = DOT_RADIUS_RANGE[1]) -> int:
        return dot_radius(int(agent.body["size"]), largest)

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
        for trait in MIND_TRAITS:
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
                          self.population_cap, self.mind_mutation, self.body_mutation, self.year_length)

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
        history = f"HISTORY  {self.simulation.turn}/{self.simulation.frontier}"
        self._text(history, (UI(22), height - UI(36)), (158, 181, 160), self.small)
        label = UI.font("segoeui", 13, bold=True).render(f"EXPORT ALL SELECTED ({len(self.export_list)})", True,
                                                         (19, 26, 27) if self.export_list else (220, 229, 211))
        self._export_button = pg.Rect(0, 0, label.get_width() + UI(32), UI(34))
        self._export_button.midright = (self.screen.get_width() - UI(20), height - UI(26))
        pg.draw.rect(self.screen, (238, 183, 110) if self.export_list else (44, 57, 52), self._export_button,
                     border_radius=UI(6))
        self.screen.blit(label, label.get_rect(center=self._export_button.center))
        room = self._export_button.left - UI(185) - UI(16)
        if pg.time.get_ticks() < self.message_until:
            self._text(self._fit(self.message, room), (UI(185), height - UI(36)), (238, 183, 110), self.small)
            return
        self._text(self._fit("SPACE  pause   LEFT / RIGHT  rewind / step   WHEEL / Z  zoom   I  insights   B  report   "
                             "G  graph   E  export   R  restart   F11  window   ESC  menu", room),
                   (UI(185), height - UI(36)), (187, 199, 181), self.small)

    def _fit(self, text: str, width: int) -> str:
        """Shorten text in the small font, with an ellipsis, to fit the width: footer notes, and
        group names in the insights panel."""
        if self.small.size(text)[0] <= width:
            return text
        while text and self.small.size(text + "...")[0] > width:
            text = text[:-1]
        return text.rstrip() + "..."

    def _text(self, text: str, position: tuple[int, int], color: tuple[int, int, int],
              font: pg.font.Font) -> None:
        self.screen.blit(font.render(text, True, color), position)


def _shown(value: float | None, pattern: str) -> str:
    """A number for the insights tables, or a dash when there is none."""
    return "-" if value is None else pattern.format(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Bitzkey artificial ecology.")
    parser.add_argument("--width", type=int, default=Config.GRID_WIDTH, help="world width in cells (default: 100)")
    parser.add_argument("--height", type=int, default=Config.GRID_HEIGHT, help="world height in cells (default: 100)")
    parser.add_argument("--population", type=int, default=Config.INITIAL_POPULATION)
    parser.add_argument("--grass-density", type=int, default=Config.GRASS_DENSITY,
                        help="percent of cells with grass at the start (default: 5)")
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
    setup = SetupScreen(Window.open_window(fullscreen=not options.windowed), vars(options))
    while (settings := setup.run()) is not None:
        Rules.apply(settings.pop("rules", {}))  # the advanced rules hold for this world and its restarts
        app = BitzkeyApp(**settings)
        outcome = app.run()
        if app.simulation is not None:
            app.simulation.close()  # its history's files go with it
        if outcome == "quit":
            break
    pg.quit()


if __name__ == "__main__":
    main()
    # Quitting ends the process at once, rather than waiting on Python to tidy away every last
    # brain the session made.
    if sys.stdout:
        sys.stdout.flush()
    os._exit(0)
