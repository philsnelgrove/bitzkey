import unittest

from display.Sound import LENGTH, RATE, click_sound


class SoundTests(unittest.TestCase):
    def test_the_click_lasts_a_quarter_second_starts_sharp_and_ends_silent(self):
        samples = click_sound(RATE)
        self.assertEqual(len(samples), int(LENGTH * RATE))
        self.assertLessEqual(max(abs(sample) for sample in samples), 1.0)
        self.assertGreater(max(abs(sample) for sample in samples[:200]), 0.1)  # the click, at once
        self.assertLess(abs(samples[-1]), 1e-6)  # faded out, no pop


if __name__ == "__main__":
    unittest.main()
