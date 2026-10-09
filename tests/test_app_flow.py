import os
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # no real window during tests

import pygame as pg

from display.Dialog import Dialog
from Main import BitzkeyApp
from tools.Exchange import import_bitzkeys


def key(code: int) -> pg.event.Event:
    return pg.event.Event(pg.KEYDOWN, key=code, unicode="", mod=0)


class DialogTests(unittest.TestCase):
    dialog = Dialog("Title", ["A line."], [("stay", "STAY"), ("leave", "LEAVE")], cancel="stay")

    def setUp(self):
        pg.font.init()

    def test_enter_picks_the_default_and_escape_cancels(self):
        self.assertEqual(self.dialog.handle(key(pg.K_RETURN), (1200, 820)), "leave")
        self.assertEqual(self.dialog.handle(key(pg.K_ESCAPE), (1200, 820)), "stay")
        self.assertIsNone(self.dialog.handle(key(pg.K_a), (1200, 820)))

    def test_buttons_can_be_clicked(self):
        for choice, rect in self.dialog._layout((1200, 820))["buttons"].items():
            click = pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=rect.center)
            self.assertEqual(self.dialog.handle(click, (1200, 820)), choice)


class SimulationFlowTests(unittest.TestCase):
    def setUp(self):
        self.app = BitzkeyApp(30, 30, 30, 5, 3)
        pg.event.clear()
        for _ in range(5):
            self.app.simulation.advance()
        self.animal = next(agent for agent in self.app.simulation.living if agent.brain)
        self.app.selected_id = self.animal.identifier

    def tearDown(self):
        pg.event.clear()
        pg.display.quit()  # leave no window behind for other tests

    def test_selecting_a_bitzkey_closes_the_insights_panel(self):
        self.app.show_insights = True
        board, cell_width, cell_height = self.app._board_geometry()
        point = (round(board.left + (self.animal.x + 0.5) * cell_width), round(board.top + (self.animal.y + 0.5) * cell_height))
        self.app.selected_id = None
        self.app._click(point)
        self.assertIsNotNone(self.app.selected_id)
        self.assertFalse(self.app.show_insights)

    def test_the_map_zooms_round_the_pointer_and_pans_while_the_world_runs_on(self):
        view, _ = self.app._map_view()
        board, scale, _ = self.app._board_geometry()
        pointer = (view.left + view.width // 3, view.top + view.height // 3)
        cell = ((pointer[0] - board.left) / scale, (pointer[1] - board.top) / scale)
        with patch.object(pg.mouse, "get_pos", return_value=pointer):
            self.assertTrue(self.app._map_event(pg.event.Event(pg.MOUSEWHEEL, x=0, y=3)))
        board, zoomed, _ = self.app._board_geometry()
        self.assertAlmostEqual(zoomed, scale * 1.25 ** 3)
        self.assertAlmostEqual((pointer[0] - board.left) / zoomed, cell[0], delta=0.05)  # the cell under the pointer stays put
        self.assertAlmostEqual((pointer[1] - board.top) / zoomed, cell[1], delta=0.05)
        self.assertTrue(self.app.playing)
        start = board.topleft
        self.app._map_event(pg.event.Event(pg.MOUSEBUTTONDOWN, button=3, pos=view.center))
        self.app._map_event(pg.event.Event(pg.MOUSEMOTION, pos=(view.centerx + 30, view.centery), buttons=(0, 0, 1)))
        self.app._map_event(pg.event.Event(pg.MOUSEBUTTONUP, button=3, pos=(view.centerx + 30, view.centery)))
        board, _, _ = self.app._board_geometry()
        self.assertEqual(board.left - start[0], 30)  # dragged right, the map follows
        for _ in range(40):  # it never pans past the map's edge
            self.app._map_event(pg.event.Event(pg.MOUSEBUTTONDOWN, button=3, pos=view.center))
            self.app._map_event(pg.event.Event(pg.MOUSEMOTION, pos=(view.centerx + 300, view.centery), buttons=(0, 0, 1)))
            self.app._map_event(pg.event.Event(pg.MOUSEBUTTONUP, button=3, pos=view.center))
        board, _, _ = self.app._board_geometry()
        self.assertLessEqual(abs(board.left - view.left), 1)
        board, zoomed, _ = self.app._board_geometry()  # picking still finds the Bitzkey under the pointer
        point = (round(board.left + (self.animal.x + 0.5) * zoomed), round(board.top + (self.animal.y + 0.5) * zoomed))
        if view.collidepoint(point):
            self.app.selected_id = None
            self.app._click(point)
            self.assertEqual(self.app.selected_id, self.animal.identifier)
        self.app._draw()  # zoomed in, it draws only what shows
        with patch.object(pg.mouse, "get_pos", return_value=view.center):
            self.app._map_event(pg.event.Event(pg.MOUSEWHEEL, x=0, y=-30))
        self.assertEqual(self.app.map_zoom, 1.0)  # no further out than the whole map
        self.app.map_zoom = 4.0
        pg.event.post(key(pg.K_z))
        with patch.object(BitzkeyApp, "_draw"), patch.object(pg.display, "flip"),                 patch.object(BitzkeyApp, "_tick", side_effect=[None, KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                self.app.run()
        self.assertEqual(self.app.map_zoom, 1.0)  # Z shows the whole map again

    def test_a_pick_on_the_graph_stays_selected_on_the_map_with_no_one_inspected(self):
        others = [agent.identifier for agent in self.app.simulation.living][:3]
        self.app.show_census = True
        self.app.graph.select(set(others))
        pg.event.post(key(pg.K_g))
        self.app.playing = False
        with patch.object(self.app, "_tick", side_effect=[None, SystemExit]):
            with self.assertRaises(SystemExit):
                self.app.run()
        self.assertFalse(self.app.show_census)
        self.assertIsNone(self.app.selected_id)  # the sidebar shows no one Bitzkey
        self.assertEqual(self.app.graph.selection, set(others))  # but the pick stays, ringed on the map
        self.app._draw()

        # clicking a dot selects just it, on the map and the graph alike; closing the graph keeps it
        board, cell_width, cell_height = self.app._board_geometry()
        self.app._click((round(board.left + (self.animal.x + 0.5) * cell_width),
                         round(board.top + (self.animal.y + 0.5) * cell_height)))
        chosen = self.app.selected_id
        self.assertEqual(self.app.graph.selection, {chosen})
        self.app.show_census = True
        self.app._close_graph()
        self.assertEqual(self.app.selected_id, chosen)

    def test_the_insights_panel_scrolls_by_wheel_and_scroll_bar_within_its_length(self):
        self.app.show_insights = True
        self.app._draw()
        self.assertGreater(self.app._insights_height, self.app._insights_view().height)  # too long to fit
        view = self.app._insights_view()
        with patch.object(pg.mouse, "get_pos", return_value=view.center):
            self.assertTrue(self.app._scroll_event(pg.event.Event(pg.MOUSEWHEEL, x=0, y=-2)))
        self.assertGreater(self.app.insights_scroll, 0)
        with patch.object(pg.mouse, "get_pos", return_value=view.center):
            self.app._scroll_event(pg.event.Event(pg.MOUSEWHEEL, x=0, y=-1000))
        self.assertEqual(self.app.insights_scroll, self.app._max_scroll())  # no further than the end

        track = self.app._scroll_track()
        self.app._scroll_event(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=(track.centerx, track.top)))
        self.app._scroll_event(pg.event.Event(pg.MOUSEBUTTONUP, button=1, pos=(track.centerx, track.top)))
        self.assertEqual(self.app.insights_scroll, 0)  # a click at the top of the track goes back there
        self.app._draw()  # and the panel still draws at every position

    def test_a_selected_bitzkey_can_be_added_to_and_removed_from_export(self):
        self.app._toggle_export()
        self.assertIn(self.animal.identifier, self.app.export_list)

        self.app._toggle_export()
        self.assertNotIn(self.animal.identifier, self.app.export_list)

    def test_the_add_button_appears_under_the_selected_bitzkey(self):
        self.app._draw()
        self.app._click(self.app._add_button.center)

        self.assertIn(self.animal.identifier, self.app.export_list)

    def test_the_living_export_as_they_are_now_and_the_dead_as_when_added(self):
        self.app._toggle_export()
        self.animal.mind["fear"] = 14.25
        records, gone = self.app._chosen_records()
        self.assertEqual((records[0]["mind"]["fear"], gone), (14.25, 0))

        self.animal.health = 0
        self.app.simulation._die(self.animal, "killed")
        records, gone = self.app._chosen_records()
        self.assertEqual(gone, 1)
        self.assertIn("brain", records[0])
        self.assertEqual(import_bitzkeys(str(records[0]).replace("'", '"'))[0].dna, self.animal.dna)

    def test_the_footer_button_opens_the_export_dialog(self):
        self.app._toggle_export()
        self.app._draw()

        self.app._click(self.app._export_button.center)

        self.assertEqual(self.app._dialog_kind, "export")
        self.assertIn("1 Bitzkey added to export.", self.app.dialog.lines)
        self.assertEqual([choice for choice, _ in self.app.dialog.buttons], ["clear", "cancel", "catalog", "copy", "save"])

    def test_the_death_report_and_inspecting_brains_open_their_reports(self):
        self.app.show_insights = True
        self.app._draw()
        self.app._click(self.app._deaths_button.center)
        self.assertEqual(self.app.report.title, "Death report")
        self.assertTrue(self.app.report.lines[0].startswith("CAUSE"))
        self.app.report.handle(key(pg.K_ESCAPE), self.app.screen.get_size())
        self.app.report = None
        self.app.show_insights = False
        self.app.playing = True
        self.app._draw()
        self.app._click(self.app._brains_button.center)
        self.assertFalse(self.app.playing)  # paused to look
        self.assertTrue(self.app.report.title.startswith("Brains of #"))
        self.assertTrue(self.app.report.lines[0].startswith("SITUATION"))

    def test_escape_then_leave_returns_to_the_configuration_panel(self):
        for event in (key(pg.K_ESCAPE), key(pg.K_RETURN)):
            pg.event.post(event)

        self.assertEqual(self.app.run(), "configure")

    def test_leaving_greys_out_the_button_and_deletes_the_history(self):
        for _ in range(20):
            self.app.simulation.advance()
        world = self.app.simulation
        self.assertTrue(world.history)  # turns kept to rewind to
        for event in (key(pg.K_ESCAPE), key(pg.K_RETURN)):
            pg.event.post(event)

        self.assertEqual(self.app.run(), "configure")
        dialog = self.app.dialog
        self.assertEqual(dialog.busy, "leave")
        self.assertIn(("leave", "Leaving, please be patient"), dialog.buttons)
        self.assertEqual(dialog.caption, "Cleaning up messy neural networks...")
        self.assertIsNone(dialog.handle(key(pg.K_RETURN), self.app.screen.get_size()))  # nothing more to press
        self.assertFalse(world.history)  # let go

    def test_escape_then_stay_keeps_the_simulation_running(self):
        pg.event.post(key(pg.K_ESCAPE))
        pg.event.post(key(pg.K_ESCAPE))
        pg.event.post(pg.event.Event(pg.QUIT))

        self.assertEqual(self.app.run(), "quit")
        self.assertIsNone(self.app.dialog)

    def test_leaving_warns_about_bitzkeys_not_yet_exported(self):
        self.app._toggle_export()
        self.app._ask_to_leave()

        self.assertTrue(any("will be lost" in line for line in self.app.dialog.lines))

    def test_when_the_simulation_ends_it_stops_and_shows_the_graph(self):
        for agent in self.app.simulation.living:
            if agent.species != "plant":
                agent.health = 0
                self.app.simulation._die(agent, "killed")
        self.app.elapsed = 10

        self.app._tick()

        self.assertTrue(self.app.simulation.ended)
        self.assertFalse(self.app.playing)
        self.assertTrue(self.app.show_census)
        self.assertIn("every animal has died", self.app.message)

    def _end_the_run(self):
        for agent in self.app.simulation.living:
            if agent.species != "plant":
                agent.health = 0
                self.app.simulation._die(agent, "killed")
        self.app.elapsed = 10
        self.app._tick()
        self.app._draw()  # the graph learns the run is over

    def test_at_the_run_end_the_graph_offers_the_map_and_closing_it_leaves(self):
        self.app._draw()
        self.assertNotIn("map", self.app.graph.layout())  # not before the end
        self._end_the_run()
        button = self.app.graph.layout()["map"]
        self.assertEqual(self.app.graph.handle(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=button.center)), "map")
        pg.event.post(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=button.center))
        pg.event.post(pg.event.Event(pg.QUIT))
        self.assertEqual(self.app.run(), "quit")
        self.assertFalse(self.app.show_census)  # RETURN TO MAP shows the map at the end

        self.app.show_census = True
        pg.event.post(key(pg.K_ESCAPE))  # closing the graph at the end leaves for the configuration panel
        self.assertEqual(self.app.run(), "configure")

    def test_at_the_run_end_closing_the_graph_asks_first_if_bitzkeys_await_export(self):
        self.app._toggle_export()
        self._end_the_run()
        pg.event.post(key(pg.K_g))
        pg.event.post(pg.event.Event(pg.QUIT))
        self.assertEqual(self.app.run(), "quit")
        self.assertEqual(self.app.dialog.title, "Leave this simulation?")

    def test_before_the_run_end_closing_the_graph_returns_to_the_map(self):
        self.app.show_census = True
        pg.event.post(key(pg.K_ESCAPE))
        pg.event.post(pg.event.Event(pg.QUIT))
        self.assertEqual(self.app.run(), "quit")
        self.assertFalse(self.app.show_census)

    def test_the_world_pauses_while_a_dialog_is_open(self):
        turn = self.app.simulation.turn
        self.app._ask_to_leave()
        self.app.elapsed = 10
        pg.event.post(pg.event.Event(pg.QUIT))

        self.app.run()

        self.assertEqual(self.app.simulation.turn, turn)


class CatalogFlowTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self.folder = tempfile.TemporaryDirectory()
        patcher = patch("tools.Catalog.CATALOG_PATH", Path(self.folder.name) / "catalog.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.folder.cleanup)
        self.app = BitzkeyApp(30, 30, 30, 5, 3)
        pg.event.clear()
        for _ in range(5):
            self.app.simulation.advance()
        self.animals = [agent for agent in self.app.simulation.living if agent.brain]

    def tearDown(self):
        pg.event.clear()
        pg.display.quit()

    def test_exemplars_beside_the_map_show_the_living_animals_that_stand_out_and_a_click_selects_one(self):
        from Main import EXEMPLAR_CHOICES, EXEMPLAR_STATS
        oldest = max(self.animals, key=lambda agent: (agent.age, -agent.identifier))
        oldest.age += 1000
        self.app._draw()
        self.assertEqual(len(self.app._exemplar_cards), 10)  # five a side
        exemplars = self.app._exemplars()
        self.assertEqual([title for title, _, _ in exemplars], [EXEMPLAR_STATS[choice][0] for choice in EXEMPLAR_CHOICES])
        found = dict((title, agent) for title, agent, _ in exemplars)
        self.assertIs(found["OLDEST"], oldest)
        smartest = max(agent.inborn["intellect"] for agent in self.animals)
        self.assertEqual(found["SMARTEST"].inborn["intellect"], smartest)
        self.assertNotEqual(found["BEST FED"].species, "plant")
        left, right = self.app._exemplar_columns()
        view = self.app._map_view()[0]
        self.assertTrue(left.right < view.left and view.right < right.left)  # either side of the map
        card, identifier = self.app._exemplar_cards[2]
        self.app._click(card.center)
        self.assertEqual((self.app.selected_id, self.app.graph.selection), (oldest.identifier, {oldest.identifier}))

    def test_each_exemplar_cards_title_is_a_menu_of_several_dozen_statistics(self):
        import Main
        from Main import EXEMPLAR_CHOICES, EXEMPLAR_STATS, exemplar_text, exemplar_value
        self.assertGreaterEqual(len(EXEMPLAR_STATS), 36)
        self.assertEqual(len({title for title, *_ in EXEMPLAR_STATS}), len(EXEMPLAR_STATS))
        for agent in self.animals:  # every statistic reads, and says something, for every animal
            for title, kind, name, _ in EXEMPLAR_STATS:
                self.assertIsInstance(exemplar_value(agent, kind, name), (int, float))
                self.assertTrue(exemplar_text(agent, kind, name))
        saved = list(EXEMPLAR_CHOICES)
        self.addCleanup(lambda: Main.EXEMPLAR_CHOICES.__setitem__(slice(None), saved))
        self.app._draw()
        self.app._click(self.app._exemplar_titles[1].center)  # opens the second card's menu
        self.assertEqual(self.app._exemplar_menu, 1)
        self.app._draw()
        self.assertEqual(len(self.app._menu_items), len(EXEMPLAR_STATS))
        youngest = [title for title, *_ in EXEMPLAR_STATS].index("YOUNGEST")
        item = next(rect for rect, stat in self.app._menu_items if stat == youngest)
        self.app._click(item.center)
        self.assertIsNone(self.app._exemplar_menu)
        self.assertEqual(EXEMPLAR_CHOICES[1], youngest)
        title, agent, said = self.app._exemplars()[1]
        self.assertEqual(title, "YOUNGEST")
        self.assertEqual(agent.age, min(animal.age for animal in self.app.simulation.living if animal.species != "plant"))
        self.assertEqual(said, f"age {agent.age}")
        self.app._click(self.app._exemplar_titles[0].center)  # a click elsewhere just closes it
        self.app._click((1, 1))
        self.assertEqual((self.app._exemplar_menu, EXEMPLAR_CHOICES[0]), (None, saved[0]))

    def test_a_new_world_starts_with_the_living_bitzkeys_picked_on_the_graph(self):
        from display import Window
        from display.Setup import SetupScreen
        chosen = self.animals[:2]
        self.app.show_census = True
        self.app.graph.select({agent.identifier for agent in chosen})
        self.app._draw()
        button = self.app.graph.layout()["renew"]
        self.assertEqual(self.app.graph.handle(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=button.center)), "renew")
        self.assertTrue(self.app._renew())
        self.assertIsNone(self.app.simulation)  # it left
        self.assertEqual([imported.dna for imported in import_bitzkeys(self.app.renewal)], [agent.dna for agent in chosen])
        self.assertEqual(pg.scrap.get_text(), self.app.renewal)  # copied to the clipboard too
        setup = SetupScreen(Window.open_window(fullscreen=False),
                            {"width": 30, "height": 30, "population": 20, "grass_density": 5, "seed": 1})
        setup.form.import_text = "something old"
        settings = setup.start_with(self.app.renewal)
        self.assertEqual(setup.form.import_text, self.app.renewal)  # the import box cleared, then pasted into
        self.assertEqual([imported.dna for imported in settings["imported"]], [agent.dna for agent in chosen])
        self.assertEqual((settings["width"], settings["seed"]), (30, 1))

    def test_a_selected_bitzkey_that_dies_is_let_go_and_the_sidebar_returns_to_neutral(self):
        chosen = self.animals[0]
        self.app._select(chosen.identifier)
        chosen.health = 0
        self.app.simulation._die(chosen, "killed")
        self.app._draw()
        self.assertIsNone(self.app.selected_id)
        self.assertEqual(self.app.graph.selection, set())  # nothing left to speak of as picked

    def test_on_the_map_a_click_picks_ctrl_adds_or_removes_a_marquee_picks_many_and_empty_ground_deselects(self):
        board, cell, _ = self.app._board_geometry()
        def at(agent):
            return round(board.left + (agent.x + 0.5) * cell), round(board.top + (agent.y + 0.5) * cell)
        first, second = self.animals[0], next(a for a in self.animals[1:] if (a.x, a.y) != (self.animals[0].x, self.animals[0].y))
        self.app._pick_on_map(at(first), at(first), False)
        self.assertEqual((self.app.selected_id, self.app.graph.selection), (first.identifier, {first.identifier}))
        self.app._pick_on_map(at(second), at(second), True)  # CTRL: added
        self.assertEqual((self.app.selected_id, self.app.graph.selection), (None, {first.identifier, second.identifier}))
        self.app._pick_on_map(at(first), at(first), True)  # CTRL again: taken out
        self.assertEqual((self.app.selected_id, self.app.graph.selection), (second.identifier, {second.identifier}))
        empty = (board.left + 1, board.top + 1)
        while self.app._living_at(empty) is not None:
            empty = (empty[0] + 7, empty[1])
        self.app._pick_on_map(empty, empty, False)  # nothing there: deselected
        self.assertEqual((self.app.selected_id, self.app.graph.selection), (None, set()))
        self.app._pick_on_map(board.topleft, board.bottomright, False)  # a marquee over the whole map
        self.assertEqual(self.app.graph.selection, {agent.identifier for agent in self.app.simulation.living})
        self.app._draw()  # the picked show their sights

    def test_a_new_world_needs_living_bitzkeys_picked(self):
        self.app.graph.select(set())
        self.assertFalse(self.app._renew())
        self.assertIsNotNone(self.app.simulation)
        self.assertIn("Pick living Bitzkeys", self.app.message)

    def test_a_selected_bitzkey_goes_into_the_catalog_from_the_sidebar_button_or_k(self):
        from tools import Catalog
        self.app.selected_id = self.animals[0].identifier
        self.app._draw()  # lays out the sidebar's buttons
        self.app._click(self.app._catalog_button.center)
        self.assertEqual([Catalog.inspected(entry).identifier for entry in Catalog.load()], [self.animals[0].identifier])
        self.assertIn("Added", self.app.message)
        self.app.selected_id = self.animals[1].identifier
        pg.event.post(key(pg.K_k))
        pg.event.post(pg.event.Event(pg.QUIT))
        self.app.run()
        self.assertEqual(len(Catalog.load()), 2)

    def test_the_graph_catalogs_the_living_it_picked_and_the_export_list_catalogs_its_own(self):
        from tools import Catalog
        dead = self.animals[2]
        dead.health = 0
        self.app.simulation._die(dead, "killed")
        self.app.show_census = True
        self.app.graph.select({self.animals[0].identifier, dead.identifier})
        self.app._draw()
        self.assertEqual(self.app.graph.selection, {self.animals[0].identifier})  # the dead are let go
        button = self.app.graph.layout()["catalog"]
        self.assertEqual(self.app.graph.handle(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=button.center)), "catalog")
        self.app._catalog_picked()
        self.assertEqual([Catalog.inspected(entry).identifier for entry in Catalog.load()], [self.animals[0].identifier])
        self.app.show_census = False
        self.app.selected_id = self.animals[1].identifier
        self.app._toggle_export()
        self.app._open_export_dialog()
        self.app._answer("catalog")
        self.assertEqual(len(Catalog.load()), 2)

    def test_from_catalog_on_the_setup_screen_adds_a_bitzkey_with_its_brain_to_the_import(self):
        from display import Window
        from display.CatalogScreen import CatalogScreen
        from display.Setup import SetupScreen
        from tools import Catalog
        entry = Catalog.entry_for(self.animals[0], self.app.simulation.turn)
        setup = SetupScreen(Window.open_window(fullscreen=False),
                            {"width": 30, "height": 30, "population": 20, "grass_density": 5, "seed": 1})
        with patch.object(CatalogScreen, "run", return_value=("import", [entry])):
            setup._click(setup._layout()["catalog"].center)
        [imported] = setup.form.imports[0]
        self.assertEqual(imported.dna, self.animals[0].dna)
        self.assertIsNotNone(imported.brain)


class ConfigPanelTests(unittest.TestCase):
    def tearDown(self):
        pg.display.quit()

    def test_quit_button_leaves_and_the_population_cap_reaches_the_world(self):
        from display import Window
        from display.Setup import SETTINGS, SetupScreen
        setup = SetupScreen(Window.open_window(fullscreen=False),
                            {"width": 30, "height": 30, "population": 20, "grass_density": 5, "seed": 1})
        self.assertEqual(setup._click(setup._layout()["quit"].center), "quit")
        self.assertIsNone(setup.form.values()["population_cap"])  # no limit by default

        setup.form.select([setting.key for setting in SETTINGS].index("population_cap"))
        setup.form.type_text("250")
        app = BitzkeyApp(**setup.form.values())
        self.assertEqual(app.simulation.population_cap, 250)

    def test_the_advanced_button_keeps_the_rules_applied_for_the_worlds_started(self):
        from display import Window
        from display.Advanced import AdvancedScreen
        from display.Setup import SetupScreen
        setup = SetupScreen(Window.open_window(fullscreen=False),
                            {"width": 30, "height": 30, "population": 20, "grass_density": 5, "seed": 1})
        self.assertEqual(setup._values()["rules"], {})
        advanced = setup._layout()["advanced"].center
        with patch.object(AdvancedScreen, "run", return_value=("apply", {"CORPSE_TURNS": 60})):
            self.assertIsNone(setup._click(advanced))
        self.assertEqual(setup._values()["rules"], {"CORPSE_TURNS": 60})
        with patch.object(AdvancedScreen, "run", return_value=("cancel", None)):
            setup._click(advanced)
        self.assertEqual(setup._values()["rules"], {"CORPSE_TURNS": 60})
        with patch.object(AdvancedScreen, "run", return_value=("quit", None)):
            self.assertEqual(setup._click(advanced), "quit")

    def test_quitting_the_configuration_panel_asks_first(self):
        from display import Window
        from display.Setup import SetupScreen
        setup = SetupScreen(Window.open_window(fullscreen=False),
                            {"width": 30, "height": 30, "population": 20, "grass_density": 5, "seed": 1})
        quit_button = setup._layout()["quit"].center
        # Esc asks, Esc again cancels; the QUIT button asks, and QUIT (Enter) confirms
        for event in (key(pg.K_ESCAPE), key(pg.K_ESCAPE),
                      pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=quit_button), key(pg.K_RETURN)):
            pg.event.post(event)
        with patch.object(SetupScreen, "_ask_to_quit", wraps=setup._ask_to_quit) as asked:
            self.assertIsNone(setup.run())
        self.assertEqual(asked.call_count, 2)
        self.assertEqual(setup.dialog.title, "Quit Bitzkey?")


class MainLoopTests(unittest.TestCase):
    def test_leaving_a_simulation_returns_to_the_configuration_panel_until_quit(self):
        import Main
        from unittest.mock import MagicMock, patch

        setup = MagicMock()
        # start, start again with an advanced rule changed, then quit
        setup.run.side_effect = [{"width": 10, "rules": {}}, {"width": 20, "rules": {"CORPSE_TURNS": 60}}, None]
        app = MagicMock()
        app.return_value.run.return_value = "configure"
        with patch.object(Main, "parse_args", return_value=MagicMock(windowed=True)),                 patch.object(Main.Window, "open_window"), patch.object(Main, "SetupScreen", return_value=setup),                 patch.object(Main, "BitzkeyApp", app), patch.object(Main.pg, "quit"),                 patch.object(Main.Rules, "apply") as apply:
            Main.main()

        self.assertEqual([call.kwargs for call in app.call_args_list], [{"width": 10}, {"width": 20}])
        self.assertEqual(setup.run.call_count, 3)
        self.assertEqual([call.args for call in apply.call_args_list], [({},), ({"CORPSE_TURNS": 60},)])


if __name__ == "__main__":
    unittest.main()
