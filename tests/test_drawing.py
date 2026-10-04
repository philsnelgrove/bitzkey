import unittest
from unittest.mock import MagicMock

import pygame as pg

from display.Look import CROSS, draw_bitzkey
from Main import DOT_RADIUS_RANGE, BitzkeyApp
from tools.Genome import encode_dna
from tools.Simulation import CORPSE_TURNS, Bitzkey


def bitzkey(size: int) -> Bitzkey:
    return Bitzkey(0, encode_dna("grazer", size=size), 0, 0)


class DotSizeTests(unittest.TestCase):
    def test_dot_radius_spans_the_range_by_body_size(self):
        self.assertEqual(BitzkeyApp._dot_radius(bitzkey(0)), DOT_RADIUS_RANGE[0])
        self.assertEqual(BitzkeyApp._dot_radius(bitzkey(7)), DOT_RADIUS_RANGE[1])

    def test_bigger_bodies_never_have_smaller_dots(self):
        radii = [BitzkeyApp._dot_radius(bitzkey(size)) for size in range(8)]

        self.assertEqual(radii, sorted(radii))
        self.assertEqual(len(set(radii)), DOT_RADIUS_RANGE[1] - DOT_RADIUS_RANGE[0] + 1)

    def test_bigger_cells_raise_the_largest_dot(self):
        self.assertEqual(BitzkeyApp._dot_radius(bitzkey(7), largest=14), 14)
        self.assertEqual(BitzkeyApp._dot_radius(bitzkey(0), largest=14), DOT_RADIUS_RANGE[0])



class DeadCrossTests(unittest.TestCase):
    def test_the_dead_are_crossed_through_in_red_and_the_living_are_not(self):
        dna = encode_dna("grazer", size=7)
        for alive in (True, False):
            surface = pg.Surface((40, 40))
            surface.fill((0, 0, 0))
            draw_bitzkey(surface, dna, (20, 20), 10, alive=alive)
            corner = surface.get_at((14, 14))[:3]
            self.assertEqual(corner == CROSS, not alive)


class CorpseFadeTests(unittest.TestCase):
    def test_a_corpse_fades_as_it_decays_until_it_vanishes(self):
        dna = encode_dna("grazer", size=7)
        seen = []
        for opacity in (1.0, 0.5, 0.0):
            surface = pg.Surface((40, 40))
            surface.fill((0, 0, 0))
            draw_bitzkey(surface, dna, (20, 20), 10, alive=False, opacity=opacity)
            seen.append(surface.get_at((14, 14))[0])  # the red of the cross
        self.assertEqual(seen[0], CROSS[0])
        self.assertAlmostEqual(seen[1], CROSS[0] / 2, delta=3)
        self.assertEqual(seen[2], 0)

    def test_corpse_opacity_falls_from_death_to_the_corpse_fading(self):
        app = MagicMock(simulation=MagicMock(turn=115))
        living, corpse = bitzkey(3), bitzkey(3)
        corpse.died_turn = 100
        self.assertEqual(BitzkeyApp._corpse_opacity(app, living), 1)
        self.assertAlmostEqual(BitzkeyApp._corpse_opacity(app, corpse), 1 - 15 / CORPSE_TURNS)
        app.simulation.turn = 100 + CORPSE_TURNS
        self.assertEqual(BitzkeyApp._corpse_opacity(app, corpse), 0)


if __name__ == "__main__":
    unittest.main()
