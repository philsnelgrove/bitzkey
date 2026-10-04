import unittest

import pygame as pg

from display.Ui import DESIGN_SIZE, MIN_SCALE, Ui


class UiScaleTests(unittest.TestCase):
    def setUp(self):
        pg.font.init()
        self.ui = Ui()

    def test_the_design_size_is_scale_one(self):
        self.ui.fit(DESIGN_SIZE)

        self.assertEqual(self.ui.scale, 1.0)
        self.assertEqual(self.ui(340), 340)

    def test_scale_fits_the_tighter_of_width_and_height(self):
        self.ui.fit((1920, 1080))
        self.assertAlmostEqual(self.ui.scale, 1080 / 820)
        self.assertEqual(self.ui(62), round(62 * 1080 / 820))

        self.ui.fit((3840, 2160))
        self.assertAlmostEqual(self.ui.scale, 2160 / 820)

    def test_small_windows_keep_text_readable(self):
        self.ui.fit((860, 760))

        self.assertEqual(self.ui.scale, MIN_SCALE)

    def test_fonts_are_rendered_at_the_scaled_size(self):
        self.ui.fit(DESIGN_SIZE)
        normal = self.ui.font("segoeui", 15).get_height()
        self.ui.fit((DESIGN_SIZE[0] * 2, DESIGN_SIZE[1] * 2))
        doubled = self.ui.font("segoeui", 15).get_height()

        self.assertGreater(doubled, normal * 1.7)
        self.assertIs(self.ui.font("segoeui", 15), self.ui.font("segoeui", 15))


if __name__ == "__main__":
    unittest.main()
