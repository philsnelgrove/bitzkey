import unittest

import pygame as pg

from display.Look import CROSS, draw_bitzkey
from Main import DOT_RADIUS_RANGE, BitzkeyApp
from tools.Genome import encode_dna
from tools.Simulation import Bitzkey


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


if __name__ == "__main__":
    unittest.main()
