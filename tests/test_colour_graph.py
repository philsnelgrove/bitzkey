import unittest
import unittest.mock

import pygame as pg

from display.ColourGraph import (BAR_PIXELS, MIN_VIEW, SORTS, Bar, ColourGraph, bitzkey_at, bitzkeys_in, build_bars,
                                 nice_ceiling, trait_of, updated)
from display.Inspector import draw_tooltip
from display.Ui import UI
from tools.Genome import decode_body
from tools.Simulation import Simulation
from unittest.mock import patch


def setUpModule():  # grown bodies: a Bitzkey made at age 0 isn't feeble here, unless a test says so
    global _grown
    _grown = patch("tools.Simulation.FEEBLE_YOUTH", 0.0)
    _grown.start()


def tearDownModule():
    _grown.stop()


CHART = pg.Rect(100, 50, 400, 200)


class BuildBarsTests(unittest.TestCase):
    def setUp(self):
        self.simulation = Simulation(width=30, height=30, population=40, seed=6)
        for _ in range(60):
            self.simulation.advance()

    def test_last_bar_is_exactly_the_present(self):
        bars, _ = build_bars(self.simulation, "ring", CHART)

        self.assertEqual(sorted(identifier for identifier, _, _ in bars[-1].slices),
                         sorted(agent.identifier for agent in self.simulation.living))
        self.assertEqual(bars[-1].turn, self.simulation.turn)

    def test_bars_span_the_chart_and_stack_one_slice_per_bitzkey(self):
        bars, peak = build_bars(self.simulation, "dot", CHART)

        self.assertEqual((bars[0].left, bars[-1].right), (CHART.left, CHART.right))
        self.assertLessEqual(len(bars), CHART.width // BAR_PIXELS)
        tallest = max(len(bar.slices) for bar in bars)
        self.assertGreaterEqual(peak, tallest)
        for bar in bars:
            if bar.slices:
                self.assertAlmostEqual(bar.slices[0][2], CHART.bottom)
                self.assertGreaterEqual(bar.slices[-1][1], CHART.top - 1e-9)

    def test_a_trait_sorts_each_species_lowest_at_the_bottom(self):
        for sort in ("intellect", "weapons", "temperature", "cold tolerance"):
            bars, _ = build_bars(self.simulation, "ring", CHART, sort=sort)
            living = {agent.identifier for agent in self.simulation.living}
            for bar in bars:  # the living sorted by the trait; the dead, remembered by colour alone, beneath them
                stacked = [(decode_body(self.simulation.lineage[identifier])["species"],
                            trait_of(self.simulation.lineage[identifier], sort) if identifier in living else -1)
                           for identifier, _, _ in bar.slices]
                for (species, value), (next_species, next_value) in zip(stacked, stacked[1:]):
                    if species == next_species:
                        self.assertLessEqual(value, next_value)
            # the same Bitzkeys, only in another order
            colour, _ = build_bars(self.simulation, "ring", CHART)
            self.assertEqual(sorted(identifier for identifier, _, _ in bars[-1].slices),
                             sorted(identifier for identifier, _, _ in colour[-1].slices))

    def test_one_bar_per_turn_early_in_a_run(self):
        simulation = Simulation(width=30, height=30, population=10, seed=1)
        for _ in range(4):
            simulation.advance()

        bars, _ = build_bars(simulation, "ring", CHART)

        self.assertEqual([bar.turn for bar in bars], [0, 1, 2, 3, 4])


class PickingTests(unittest.TestCase):
    bars = [Bar(0, 0, 10, [(1, 90.0, 100.0), (2, 80.0, 90.0)]),
            Bar(1, 10, 20, [(2, 90.0, 100.0), (3, 80.0, 90.0), (4, 70.0, 80.0)])]

    def test_a_click_finds_the_slice_under_it(self):
        self.assertEqual(bitzkey_at(self.bars, (5, 95)), 1)
        self.assertEqual(bitzkey_at(self.bars, (15, 75)), 4)
        self.assertIsNone(bitzkey_at(self.bars, (5, 50)))

    def test_a_box_takes_every_slice_it_touches_across_bars(self):
        self.assertEqual(bitzkeys_in(self.bars, pg.Rect(5, 85, 10, 3)), {2, 3})
        self.assertEqual(bitzkeys_in(self.bars, pg.Rect(0, 0, 30, 200)), {1, 2, 3, 4})
        self.assertEqual(bitzkeys_in(self.bars, pg.Rect(25, 0, 5, 200)), set())

    def test_selection_modes(self):
        self.assertEqual(updated({1, 2}, {3}, "replace"), {3})
        self.assertEqual(updated({1, 2}, {3}, "add"), {1, 2, 3})
        self.assertEqual(updated({1, 2}, {2}, "remove"), {1})
        self.assertEqual(updated({1, 2}, {2, 3}, "toggle"), {1, 3})

    def test_chart_tops_at_a_round_number(self):
        self.assertEqual([nice_ceiling(value) for value in (0, 1, 3, 7, 12, 260)], [1, 1, 5, 10, 20, 500])


class TimeTravelTests(unittest.TestCase):
    def setUp(self):
        pg.font.init()
        UI.fit((1200, 820))
        mild = unittest.mock.patch("tools.Simulation.CLIMATE_DAMAGE", 0)  # a world that lasts the hundred turns
        mild.start()
        self.addCleanup(mild.stop)
        self.simulation = Simulation(width=30, height=30, population=30, seed=2)
        for _ in range(100):
            self.simulation.advance()
        self.graph = ColourGraph()
        self.graph._size = (1200, 820)
        self.graph._now, self.graph._frontier = self.simulation.turn, self.simulation.frontier
        layout = self.graph.layout()
        self.chart, self.timeline = layout["chart"], layout["timeline"]

    def click(self, kind, x, y, button=1):
        return self.graph.handle(pg.event.Event(kind, pos=(x, y), button=button))

    def test_the_sort_menu_opens_picks_a_trait_and_a_click_elsewhere_just_closes_it(self):
        self.assertEqual(self.graph.sort, "colour")
        self.assertEqual(SORTS[0], "colour")
        self.click(pg.MOUSEBUTTONDOWN, *self.graph.layout()["sort"].center)
        self.assertTrue(self.graph.menu_open)
        sort, item = next((sort, item) for sort, item in self.graph.menu_items() if sort == "intellect")
        self.click(pg.MOUSEBUTTONDOWN, *item.center)
        self.assertEqual((self.graph.sort, self.graph.menu_open), ("intellect", False))

        self.click(pg.MOUSEBUTTONDOWN, *self.graph.layout()["sort"].center)
        self.click(pg.MOUSEBUTTONDOWN, self.chart.left + 2, self.chart.bottom - 2)  # closes, picks no one
        self.assertEqual((self.graph.sort, self.graph.menu_open, self.graph.selection), ("intellect", False, set()))

    def test_hovering_a_slice_shows_its_details_in_a_tooltip_on_the_screen(self):
        surface = pg.Surface((1200, 820))
        living = next(agent for agent in self.simulation.living)
        for agent, known in ((living, True), (living, False)):
            for mouse in ((10, 10), (1190, 810)):
                box = draw_tooltip(surface, agent, known, mouse)
                self.assertTrue(surface.get_rect().contains(box))
        self.assertLess(draw_tooltip(surface, living, False, (10, 10)).height,
                        draw_tooltip(surface, living, True, (10, 10)).height)

        self.graph.draw(surface, self.simulation, None)
        identifier, top, bottom = self.graph._bars[-1].slices[0]
        self.graph._mouse = (self.graph._bars[-1].left + 1, round((top + bottom) / 2))
        with unittest.mock.patch("display.ColourGraph.draw_tooltip") as tooltip:
            self.graph.draw(surface, self.simulation, None)
        self.assertEqual(tooltip.call_args.args[1].identifier, identifier)

        living = {agent.identifier for agent in self.simulation.living}  # and a dead one: what is known of it
        dead = next((bar, slice_) for bar in self.graph._bars for slice_ in bar.slices if slice_[0] not in living)
        bar, (identifier, top, bottom) = dead
        self.graph._mouse = (bar.left + 1, round((top + bottom) / 2))
        with unittest.mock.patch("display.ColourGraph.draw_tooltip") as tooltip:
            self.graph.draw(surface, self.simulation, None)
        shown, known = tooltip.call_args.args[1], tooltip.call_args.args[2]
        self.assertEqual((shown.identifier, shown.dna, known), (identifier, self.simulation.lineage[identifier], False))
        self.assertIn(tooltip.call_args.kwargs["note"].split(":")[0], ("Died at turn " + str(self.simulation.death_records[identifier].turn)
                                                                        if identifier in self.simulation.death_records else "Dead"))
        self.graph.draw(surface, self.simulation, None)  # and it draws

    def test_dragging_the_timeline_seeks_where_it_is_let_go(self):
        y = self.timeline.centery
        self.click(pg.MOUSEBUTTONDOWN, self.chart.left + 5, y)
        self.graph.handle(pg.event.Event(pg.MOUSEMOTION, pos=(self.chart.centerx, y), rel=(0, 0), buttons=(1, 0, 0)))
        action = self.click(pg.MOUSEBUTTONUP, self.chart.centerx, y)

        self.assertEqual(action, "seek")
        self.assertEqual(self.graph.seek_turn, self.graph.turn_at(self.chart.centerx))
        self.assertAlmostEqual(self.graph.seek_turn, 50, delta=1)

    def test_right_clicking_the_graph_seeks_that_turn(self):
        self.assertIsNone(self.click(pg.MOUSEBUTTONDOWN, self.chart.left + 1, self.chart.centery, button=3))
        self.assertEqual(self.click(pg.MOUSEBUTTONUP, self.chart.left + 1, self.chart.centery, button=3), "seek")
        self.assertEqual(self.graph.seek_turn, 0)

    def test_the_wheel_zooms_into_the_height_around_the_pointer_and_a_right_drag_pans(self):
        y = self.chart.bottom - self.chart.height // 4
        up = (self.chart.bottom - y) / self.chart.height
        self.graph.zoom_height(3, y)
        self.assertAlmostEqual(self.graph.height_zoom, 1.25 ** 3)
        self.assertAlmostEqual(self.graph.floor + up / self.graph.height_zoom, up)  # what was under it stays put
        floor = self.graph.floor
        start = (self.chart.centerx, self.chart.centery)
        self.click(pg.MOUSEBUTTONDOWN, *start, button=3)
        self.graph.handle(pg.event.Event(pg.MOUSEMOTION, pos=(start[0], start[1] + 40), rel=(0, 40), buttons=(0, 0, 1)))
        self.assertIsNone(self.click(pg.MOUSEBUTTONUP, start[0], start[1] + 40, button=3))  # a pan, not a seek
        self.assertAlmostEqual(self.graph.floor, floor + 40 / self.chart.height / self.graph.height_zoom)
        self.graph.zoom_height(-40, y)
        self.assertEqual((self.graph.height_zoom, self.graph.floor), (1.0, 0.0))  # no further out than the whole
        self.graph.zoom_height(3, y)
        self.graph.unzoom()
        self.assertEqual((self.graph.view, self.graph.height_zoom, self.graph.floor), (None, 1.0, 0.0))

    def test_zoomed_into_the_height_the_bars_stand_taller_from_the_floor(self):
        bars, peak = build_bars(self.simulation, "ring", CHART)
        tall, same = build_bars(self.simulation, "ring", CHART, height_zoom=4.0, floor=0.25)
        self.assertEqual(same, peak)
        unit = CHART.height / peak
        self.assertEqual(tall[-1].slices[0][0], bars[-1].slices[0][0])
        self.assertAlmostEqual(tall[-1].slices[0][2], CHART.bottom + 0.25 * 4 * CHART.height)
        self.assertAlmostEqual(tall[-1].slices[0][2] - tall[-1].slices[0][1], 4 * unit)

    def test_the_wheel_zooms_pinned_to_the_present_or_around_the_cursor_and_back_out_to_the_whole_run(self):
        x = self.graph.x_of(80)
        self.graph.zoom(3, x)  # from the whole run: pinned to the latest turn, the living in view
        first, last = self.graph.window()
        self.assertLess(last - first + 1, 101)
        self.assertEqual(last, 100)
        self.graph.pan(-3)  # back into the past: now the wheel zooms around the cursor
        x = self.graph.x_of(self.graph.window()[0] + 5)
        under = self.graph.turn_at(x)
        self.graph.zoom(1, x)
        self.assertAlmostEqual(self.graph.turn_at(x), under, delta=1)

        self.graph.zoom(40, x)
        self.assertEqual(self.graph.window()[1] - self.graph.window()[0] + 1, MIN_VIEW)
        self.graph.zoom(-60, x)
        self.assertIsNone(self.graph.view)
        self.assertEqual(self.graph.window(), (0, 100))

    def test_panning_slides_a_zoomed_view_within_the_run(self):
        self.graph.zoom(5, self.graph.x_of(50))
        first, last = self.graph.window()

        self.graph.pan(-1)
        self.assertLess(self.graph.window()[0], first)
        self.graph.pan(1)
        self.assertEqual(self.graph.window()[0], first)
        self.graph.pan(-1000)
        self.assertEqual(self.graph.window()[0], 0)
        self.assertEqual(self.graph.window()[1] - self.graph.window()[0], last - first)

    def test_after_rewinding_the_later_turns_show_as_the_future(self):
        self.simulation.seek(40)

        bars, _ = build_bars(self.simulation, "ring", CHART)

        present = next(bar for bar in bars if bar.turn >= 40)
        self.assertEqual(sorted(identifier for identifier, _, _ in present.slices),
                         sorted(agent.identifier for agent in self.simulation.living))
        self.assertTrue(all(bar.future for bar in bars if bar.turn > present.turn))
        self.assertFalse(any(bar.future for bar in bars if bar.turn <= present.turn))
        self.assertEqual(bars[-1].turn, 100)

    def test_a_zoomed_graph_draws_only_the_turns_in_view(self):
        bars, _ = build_bars(self.simulation, "ring", CHART, 20, 29)

        self.assertEqual([bar.turn for bar in bars], list(range(20, 30)))


class SpeciesBandTests(unittest.TestCase):
    def test_a_zoomed_view_reaching_the_latest_turn_keeps_up_with_it(self):
        graph = ColourGraph()
        graph._frontier = 400
        graph.zoom(4, 10_000)  # zoomed in at the right-hand end
        first, last = graph.window()
        self.assertTrue(graph.pinned)
        self.assertEqual(last, 400)
        graph._frontier = 450  # the run goes on
        self.assertEqual(graph.window(), (first + 50, 450))
        graph.pan(-3)  # panned back into the past, it stays put
        self.assertFalse(graph.pinned)
        held = graph.window()
        graph._frontier = 500
        self.assertEqual(graph.window(), held)

    def test_bars_stack_by_species_and_the_dead_drop_out_of_the_selection(self):
        from tools.Genome import encode_dna
        simulation = Simulation(width=20, height=20, population=0, seed=1)
        order = ["predator", "plant", "grazer", "omnivore", "grazer"]
        agents = [simulation._add_agent(encode_dna(species), index, 1) for index, species in enumerate(order)]
        chart = pg.Rect(0, 0, 200, 100)
        bars, _ = build_bars(simulation, "ring", chart)
        stacked = [decode_body(simulation.lineage[identifier])["species"] for identifier, _, _ in bars[-1].slices]
        self.assertEqual(stacked, ["plant", "grazer", "grazer", "omnivore", "predator"])  # from the bottom up

        graph = ColourGraph()
        graph.select({agent.identifier for agent in agents})
        agents[0].health = 0  # the predator dies
        pg.init()
        surface = pg.Surface((1200, 820))
        graph.draw(surface, simulation, None)
        self.assertEqual(graph.selection, {agent.identifier for agent in agents[1:]})
        graph._bars = [Bar(0, 0, 10, [(agents[0].identifier, 0.0, 10.0), (agents[1].identifier, 10.0, 20.0)])]
        for point in ((5, 5), (5, 15)):  # the dead can't be picked; the living can
            graph.handle(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=point))
            graph._press = (point, 0)
            graph.handle(pg.event.Event(pg.MOUSEBUTTONUP, button=1, pos=point))
        self.assertEqual(graph.selection, {agents[1].identifier})


if __name__ == "__main__":
    unittest.main()
