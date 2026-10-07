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
        self.assertEqual([choice for choice, _ in self.app.dialog.buttons], ["clear", "cancel", "copy", "save"])

    def test_escape_then_leave_returns_to_the_configuration_panel(self):
        for event in (key(pg.K_ESCAPE), key(pg.K_RETURN)):
            pg.event.post(event)

        self.assertEqual(self.app.run(), "configure")

    def test_leaving_greys_out_the_button_and_frees_the_history(self):
        for _ in range(5):
            self.app.simulation.advance()
        history = self.app.simulation.history
        for event in (key(pg.K_ESCAPE), key(pg.K_RETURN)):
            pg.event.post(event)

        self.assertEqual(self.app.run(), "configure")
        dialog = self.app.dialog
        self.assertEqual(dialog.busy, "leave")
        self.assertIn(("leave", "Leaving, please be patient"), dialog.buttons)
        self.assertEqual(dialog.caption, "Cleaning up messy neural networks...")
        self.assertIsNone(dialog.handle(key(pg.K_RETURN), self.app.screen.get_size()))  # nothing more to press
        self.assertEqual(history, [])

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

    def test_the_world_pauses_while_a_dialog_is_open(self):
        turn = self.app.simulation.turn
        self.app._ask_to_leave()
        self.app.elapsed = 10
        pg.event.post(pg.event.Event(pg.QUIT))

        self.app.run()

        self.assertEqual(self.app.simulation.turn, turn)


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
