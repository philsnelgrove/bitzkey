import unittest

import pygame as pg

from display.ColourGraph import (BAR_PIXELS, MIN_VIEW, Bar, ColourGraph, bitzkey_at, bitzkeys_in, build_bars,
                                 nice_ceiling, updated)
from display.Ui import UI
from tools.Genome import decode_body
from tools.Simulation import Simulation


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
        self.simulation = Simulation(width=30, height=30, population=30, seed=6)
        for _ in range(100):
            self.simulation.advance()
        self.graph = ColourGraph()
        self.graph._size = (1200, 820)
        self.graph._now, self.graph._frontier = self.simulation.turn, self.simulation.frontier
        layout = self.graph.layout()
        self.chart, self.timeline = layout["chart"], layout["timeline"]

    def click(self, kind, x, y, button=1):
        return self.graph.handle(pg.event.Event(kind, pos=(x, y), button=button))

    def test_dragging_the_timeline_seeks_where_it_is_let_go(self):
        y = self.timeline.centery
        self.click(pg.MOUSEBUTTONDOWN, self.chart.left + 5, y)
        self.graph.handle(pg.event.Event(pg.MOUSEMOTION, pos=(self.chart.centerx, y), rel=(0, 0), buttons=(1, 0, 0)))
        action = self.click(pg.MOUSEBUTTONUP, self.chart.centerx, y)

        self.assertEqual(action, "seek")
        self.assertEqual(self.graph.seek_turn, self.graph.turn_at(self.chart.centerx))
        self.assertAlmostEqual(self.graph.seek_turn, 50, delta=1)

    def test_right_clicking_the_graph_seeks_that_turn(self):
        self.assertEqual(self.click(pg.MOUSEBUTTONDOWN, self.chart.left + 1, self.chart.centery, button=3), "seek")
        self.assertEqual(self.graph.seek_turn, 0)

    def test_the_wheel_zooms_around_the_cursor_and_back_out_to_the_whole_run(self):
        x = self.graph.x_of(80)
        self.graph.zoom(3, x)
        first, last = self.graph.window()
        self.assertLess(last - first + 1, 101)
        self.assertLessEqual(first, 80)
        self.assertGreaterEqual(last, 80)
        self.assertAlmostEqual(self.graph.turn_at(x), 80, delta=1)

        self.graph.zoom(40, x)
        self.assertEqual(self.graph.window()[1] - self.graph.window()[0] + 1, MIN_VIEW)
        self.graph.zoom(-60, x)
        self.assertIsNone(self.graph.view)
        self.assertEqual(self.graph.window(), (0, 100))

    def test_panning_slides_a_zoomed_view_within_the_run(self):
        self.graph.zoom(5, self.graph.x_of(50))
        first, last = self.graph.window()

        self.graph.pan(1)
        self.assertGreater(self.graph.window()[0], first)
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
    def test_bars_stack_by_species_and_living_only_drops_the_dead(self):
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
        graph.handle(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=graph.layout()["living"].center))
        self.assertEqual(graph.selection, {agent.identifier for agent in agents[1:]})


if __name__ == "__main__":
    unittest.main()
