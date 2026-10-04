import unittest

import pygame as pg

from display.ColourGraph import BAR_PIXELS, Bar, bitzkey_at, bitzkeys_in, build_bars, nice_ceiling, updated
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


if __name__ == "__main__":
    unittest.main()
