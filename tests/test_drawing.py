import unittest
from unittest.mock import MagicMock

import pygame as pg

from display.Look import CROSS, ICON_MIN_RADIUS, OUTLINE, draw_bitzkey, outline_offsets
from Main import DOT_RADIUS_RANGE, BitzkeyApp
from tools.Genome import color_from_word, encode_dna, mind_word, social_word
from tools.Simulation import CORPSE_TURNS, Bitzkey
from unittest.mock import patch


def setUpModule():  # grown bodies: a Bitzkey made at age 0 isn't feeble here, unless a test says so
    global _grown
    _grown = patch("tools.Simulation.FEEBLE_YOUTH", 0.0)
    _grown.start()


def tearDownModule():
    _grown.stop()


def bitzkey(size: int) -> Bitzkey:
    return Bitzkey(0, encode_dna("grazer", size=size), 0, 0)


class DotSizeTests(unittest.TestCase):
    def test_map_dots_zoom_with_the_map_the_largest_filling_a_cell_the_smallest_showing_its_icon(self):
        from display.Look import ICON_MIN_RADIUS, RING_WIDTH, dot_radius, map_dot_radii
        self.assertEqual(map_dot_radii(6.7), (ICON_MIN_RADIUS, 5))  # the whole map: as before, every dot with its icon
        for cell in (20, 40, 80):  # zoomed in: the largest, ring and all, as wide as a cell
            smallest, largest = map_dot_radii(cell)
            self.assertEqual(2 * (largest + RING_WIDTH), cell)
            self.assertGreaterEqual(smallest, ICON_MIN_RADIUS)
            self.assertAlmostEqual(smallest / largest, ICON_MIN_RADIUS / 5, delta=0.05)  # in proportion
            self.assertEqual((dot_radius(0, largest, smallest), dot_radius(7, largest, smallest)), (smallest, largest))

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



class CentreTests(unittest.TestCase):
    def test_a_big_dot_is_its_social_backgrounds_colour_with_its_icon_in_the_dots_colour(self):
        dna = encode_dna("grazer", size=7, loyalty=5, kinship=3, neurodivergence=11)
        surface = pg.Surface((40, 40))
        surface.fill((90, 90, 90))
        draw_bitzkey(surface, dna, (20, 20), 10)
        self.assertEqual(surface.get_at((20, 20))[:3], color_from_word(mind_word(dna)))  # on the leaf
        self.assertEqual(surface.get_at((26, 26))[:3], color_from_word(social_word(dna)))  # beside it
        outlined = [(x, y) for x in range(40) for y in range(40) if surface.get_at((x, y))[:3] == OUTLINE]
        self.assertTrue(outlined)
        self.assertTrue(all(abs(x - 20) < 10 and abs(y - 20) < 10 for x, y in outlined))  # inside the dot

    def test_the_outline_is_a_pixel_thick_in_the_world_and_rounds_corners_when_wider(self):
        self.assertEqual(sorted(outline_offsets(1)), [(-1, 0), (0, -1), (0, 1), (1, 0)])
        wide = outline_offsets(16)
        self.assertIn((11, 11), wide)  # the diagonals too
        self.assertTrue(all(15 <= (x * x + y * y) ** 0.5 <= 17 for x, y in wide))

    def test_a_dot_too_small_for_an_icon_is_its_colour(self):
        dna = encode_dna("grazer", size=0)
        surface = pg.Surface((10, 10))
        draw_bitzkey(surface, dna, (5, 5), ICON_MIN_RADIUS - 1)
        self.assertEqual(surface.get_at((5, 5))[:3], color_from_word(mind_word(dna)))


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
