"""The level of detail: zoomed out, only the largest Bitzkeys are drawn, every size from 85% of the way in."""

import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # no real window during tests

from Main import ALL_SIZES_AT, GLOBE_ZOOM_MAX, MAP_ZOOM_MAX, smallest_shown


class LevelOfDetailTests(unittest.TestCase):
    def test_zoomed_out_only_size_seven_shows_and_from_85_percent_in_every_size(self):
        self.assertEqual(smallest_shown(1.0, 1.0, MAP_ZOOM_MAX), 7)
        self.assertEqual(smallest_shown(MAP_ZOOM_MAX ** ALL_SIZES_AT, 1.0, MAP_ZOOM_MAX), 0)
        self.assertEqual(smallest_shown(MAP_ZOOM_MAX, 1.0, MAP_ZOOM_MAX), 0)
        sizes = [smallest_shown(1.25 ** notch, 1.0, MAP_ZOOM_MAX) for notch in range(12)]
        self.assertEqual(sizes, sorted(sizes, reverse=True))  # more show with every step in
        self.assertEqual(smallest_shown(0.5, 0.5, GLOBE_ZOOM_MAX), 7)  # the globe, over its own range


if __name__ == "__main__":
    unittest.main()
