import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # no real window during tests

import pygame as pg

from config.Globals import Config
from display import Window


class WindowTests(unittest.TestCase):
    def tearDown(self):
        pg.display.quit()

    def test_opens_fullscreen_at_the_desktop_size(self):
        screen = Window.open_window(fullscreen=True)

        self.assertTrue(Window.is_fullscreen())
        self.assertEqual(screen.get_size(), pg.display.get_desktop_sizes()[0])

    def test_f11_switches_to_a_window_and_back(self):
        Window.open_window(fullscreen=True)

        windowed = Window.toggle_fullscreen()
        self.assertFalse(Window.is_fullscreen())
        self.assertEqual(windowed.get_size(), (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))

        full = Window.toggle_fullscreen()
        self.assertTrue(Window.is_fullscreen())
        self.assertEqual(full.get_size(), pg.display.get_desktop_sizes()[0])

    def test_a_window_resizes_but_not_below_the_minimum(self):
        Window.open_window(fullscreen=False)

        bigger = Window.resize(pg.event.Event(pg.VIDEORESIZE, w=1400, h=900))
        self.assertEqual(bigger.get_size(), (1400, 900))
        smaller = Window.resize(pg.event.Event(pg.VIDEORESIZE, w=300, h=200))
        self.assertEqual(smaller.get_size(), Window.MIN_SIZE)

    def test_fullscreen_ignores_resize_requests(self):
        screen = Window.open_window(fullscreen=True)

        self.assertEqual(Window.resize(pg.event.Event(pg.VIDEORESIZE, w=900, h=800)).get_size(), screen.get_size())


if __name__ == "__main__":
    unittest.main()
