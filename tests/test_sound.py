import unittest

from display.Sound import GAP, LENGTH, RATE, call_sequence, call_sound, click_sound


class SoundTests(unittest.TestCase):
    def test_the_click_lasts_a_quarter_second_starts_sharp_and_ends_silent(self):
        samples = click_sound(RATE)
        self.assertEqual(len(samples), int(LENGTH * RATE))
        self.assertLessEqual(max(abs(sample) for sample in samples), 1.0)
        self.assertGreater(max(abs(sample) for sample in samples[:200]), 0.1)  # the click, at once
        self.assertLess(abs(samples[-1]), 1e-6)  # faded out, no pop

    def test_a_calls_sounds_are_a_sixth_as_long_and_play_one_after_another(self):
        self.assertEqual(len(call_sound(3, RATE)), int(0.32 / 6 * RATE))  # the trill, a sixth of its 0.32 seconds
        self.assertEqual(len(call_sound(1, RATE)), int(0.26 * RATE))  # the alert, sounded alone, as it was
        two = call_sequence((2, 3, 4, 2, 6, 5), RATE)  # two words of three sounds
        self.assertEqual(len(two), sum(len(call_sound(sound, RATE)) + int(GAP * RATE) for sound in (2, 3, 4, 2, 6, 5)))
        self.assertLess(len(two) / RATE, 0.5)  # both words within half a second


if __name__ == "__main__":
    unittest.main()
