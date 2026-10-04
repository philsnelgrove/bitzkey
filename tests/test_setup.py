import inspect
import unittest

from display.Setup import SETTINGS, SetupForm, import_summary
from Main import BitzkeyApp
from tools.Exchange import Imported, export_bitzkeys
from tools.Genome import encode_dna
from tools.Simulation import MAX_IMPORTED, MAX_POPULATION, Bitzkey


def exported(*dnas: int) -> str:
    return export_bitzkeys(Bitzkey(0, dna, 0, 0) for dna in dnas)


DEFAULTS = {"width": 100, "height": 100, "population": 90, "grass_density": 5, "seed": None}
SEED = [setting.key for setting in SETTINGS].index("seed")


class SetupFormTests(unittest.TestCase):
    def test_defaults_become_values_and_blank_seed_means_random(self):
        self.assertEqual(SetupForm(DEFAULTS).values(), {**DEFAULTS, "imported": []})

    def test_out_of_range_defaults_are_brought_into_range(self):
        form = SetupForm({**DEFAULTS, "width": 5, "population": 99999, "grass_density": 300})

        self.assertEqual(form.values()["width"], 10)
        self.assertEqual(form.values()["population"], MAX_POPULATION)
        self.assertEqual(form.values()["grass_density"], 100)

    def test_adjusting_steps_and_stays_in_range(self):
        form = SetupForm(DEFAULTS)

        form.adjust(1)
        self.assertEqual(form.values()["width"], 110)
        form.adjust(-100)
        self.assertEqual(form.values()["width"], 10)

    def test_first_digit_replaces_then_digits_extend(self):
        form = SetupForm(DEFAULTS)

        form.type_text("6")
        form.type_text("4")
        self.assertEqual(form.values()["width"], 64)
        form.backspace()
        self.assertEqual(form.values()["width"], 10)

    def test_typed_value_out_of_range_settles_when_leaving(self):
        form = SetupForm(DEFAULTS)
        form.type_text("999")

        form.move(1)

        self.assertEqual(form.texts["width"], "250")
        self.assertEqual(form.focused.key, "height")

    def test_seed_can_be_set_and_cleared(self):
        form = SetupForm(DEFAULTS)
        form.select(SEED)

        form.adjust(1)
        self.assertEqual(form.values()["seed"], 0)
        form.type_text("42")
        self.assertEqual(form.values()["seed"], 42)
        form.select(SEED)
        form.backspace()
        self.assertIsNone(form.values()["seed"])

    def test_moving_wraps_around_through_the_import_box(self):
        form = SetupForm(DEFAULTS)

        form.move(-1)
        self.assertTrue(form.on_import)
        form.move(1)
        self.assertEqual(form.focused.key, "width")

    def test_pasted_genomes_are_imported(self):
        form = SetupForm(DEFAULTS)
        grazer, predator = encode_dna("grazer", size=4), encode_dna("predator", fear=2)

        form.paste(exported(grazer))
        form.paste(exported(predator, predator))

        self.assertTrue(form.on_import)
        self.assertEqual([item.dna for item in form.values()["imported"]], [grazer, predator, predator])
        self.assertEqual(import_summary(form.imports[0]),
                         "3 Bitzkeys ready to import (1 grazer, 2 predators; 0 with saved brains, "
                         "3 with saved temperaments), each spawning as a group of 3-5")

    def test_typing_in_the_import_box_edits_its_text(self):
        form = SetupForm(DEFAULTS)
        form.select(len(SETTINGS))

        form.type_text('{"ring": "00000001", "dot": "00000000"}x')
        form.backspace()

        self.assertEqual(form.values()["imported"], [Imported(1 << 32)])
        self.assertEqual(form.values()["width"], 100)

    def test_loading_a_file_replaces_the_import_text(self):
        form = SetupForm(DEFAULTS)
        form.paste("not json")

        form.load(exported(encode_dna("grazer")))

        self.assertEqual([item.dna for item in form.values()["imported"]], [encode_dna("grazer")])

    def test_unreadable_import_blocks_the_start_until_cleared(self):
        form = SetupForm(DEFAULTS)
        form.paste('{"ring": "12", "dot": "00000000"}')

        self.assertFalse(form.can_start)
        self.assertIn("eight hex digits", form.imports[1])
        form.clear_import()
        self.assertTrue(form.can_start)

    def test_imports_whose_groups_could_pass_the_population_cap_are_dropped(self):
        form = SetupForm(DEFAULTS)
        form.paste(exported(*[encode_dna("grazer")] * (MAX_IMPORTED + 5)))

        self.assertEqual(len(form.values()["imported"]), MAX_IMPORTED)
        self.assertLessEqual(MAX_IMPORTED * 5, MAX_POPULATION)
        self.assertIn(f"Only the first {MAX_IMPORTED}", import_summary(form.imports[0]))

    def test_every_setting_is_a_parameter_of_the_app(self):
        parameters = list(inspect.signature(BitzkeyApp.__init__).parameters)[1:]

        self.assertEqual([setting.key for setting in SETTINGS] + ["imported"], parameters)


if __name__ == "__main__":
    unittest.main()
