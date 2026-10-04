import inspect
import unittest

from display.Setup import SETTINGS, SetupForm
from Main import BitzkeyApp
from tools.Simulation import MAX_POPULATION


DEFAULTS = {"width": 100, "height": 100, "population": 90, "seed": None}


class SetupFormTests(unittest.TestCase):
    def test_defaults_become_values_and_blank_seed_means_random(self):
        self.assertEqual(SetupForm(DEFAULTS).values(), DEFAULTS)

    def test_out_of_range_defaults_are_brought_into_range(self):
        form = SetupForm({**DEFAULTS, "width": 5, "population": 99999})

        self.assertEqual(form.values()["width"], 10)
        self.assertEqual(form.values()["population"], MAX_POPULATION)

    def test_adjusting_steps_and_stays_in_range(self):
        form = SetupForm(DEFAULTS)

        form.adjust(1)
        self.assertEqual(form.values()["width"], 110)
        form.adjust(-100)
        self.assertEqual(form.values()["width"], 10)

    def test_first_digit_replaces_then_digits_extend(self):
        form = SetupForm(DEFAULTS)

        for digit in "64":
            form.type_digit(digit)
        self.assertEqual(form.values()["width"], 64)
        form.backspace()
        self.assertEqual(form.values()["width"], 10)

    def test_typed_value_out_of_range_settles_when_leaving(self):
        form = SetupForm(DEFAULTS)
        for digit in "999":
            form.type_digit(digit)

        form.move(1)

        self.assertEqual(form.texts["width"], "250")
        self.assertEqual(form.focused.key, "height")

    def test_seed_can_be_set_and_cleared(self):
        form = SetupForm(DEFAULTS)
        form.select(3)

        form.adjust(1)
        self.assertEqual(form.values()["seed"], 0)
        for digit in "42":
            form.type_digit(digit)
        self.assertEqual(form.values()["seed"], 42)
        form.select(3)
        form.backspace()
        self.assertIsNone(form.values()["seed"])

    def test_moving_wraps_around(self):
        form = SetupForm(DEFAULTS)

        form.move(-1)

        self.assertEqual(form.focused.key, "seed")

    def test_every_setting_is_a_parameter_of_the_app(self):
        parameters = list(inspect.signature(BitzkeyApp.__init__).parameters)[1:]

        self.assertEqual([setting.key for setting in SETTINGS], parameters)


if __name__ == "__main__":
    unittest.main()
