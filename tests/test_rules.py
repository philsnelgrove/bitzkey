import unittest

from display.Advanced import AdvancedForm
from tools import Brain as brain
from tools import Rules
from tools import Simulation as world
from tools.Rules import BY_KEY, DEFAULTS, RULES
from tools.Simulation import Simulation
from unittest.mock import patch


def setUpModule():  # grown bodies: a Bitzkey made at age 0 isn't feeble here, unless a test says so
    global _grown
    _grown = patch.multiple("tools.Simulation", FEEBLE_YOUTH=0.0, DEFAULT_YEAR_LENGTH=360, SLEEP=0)  # and 360-turn years, no sleeping
    _grown.start()


def tearDownModule():
    _grown.stop()


def focus(form: AdvancedForm, key: str) -> None:
    form.select([rule.key for rule in RULES].index(key))


class RulesTests(unittest.TestCase):
    def tearDown(self):
        Rules.apply({})

    def test_think_on_the_gpu_is_greyed_out_in_the_python_version_which_has_no_gpu(self):
        self.assertIn("USE_GPU", Rules.LOCKED)
        form = AdvancedForm(DEFAULTS)
        form.select([rule.key for rule in RULES].index("USE_GPU"))
        form.adjust(1)
        form.type_text("1")
        form.finish()
        self.assertEqual(form.values["USE_GPU"], 0)  # it stays as it is

    def test_every_default_is_within_its_rule_and_of_its_kind(self):
        self.assertEqual(len(BY_KEY), len(RULES))
        for rule in RULES:
            with self.subTest(rule.key):
                value = DEFAULTS[rule.key]
                self.assertLessEqual(rule.minimum, value)
                self.assertLessEqual(value, rule.maximum)
                self.assertEqual(rule.integer, isinstance(value, int))
                self.assertEqual(rule.clamp(value), value)

    def test_applying_sets_constants_tuples_and_tables_and_resets_the_rest(self):
        Rules.apply({"CORPSE_TURNS": 60, "LITTER_RANGE.1": 9, "TERRITORY.omnivore.most": 7,
                     "GRAZING.grazer": 2.0, "EXPERIENCES.kill.fear": -0.1, "MUTATION": 0.5})
        self.assertEqual(world.CORPSE_TURNS, 60)
        self.assertEqual(world.LITTER_RANGE, (1, 9))
        self.assertEqual(world.TERRITORY["omnivore"], ((2, 7), 0.5))
        self.assertEqual(world.GRAZING["grazer"], 2.0)
        self.assertEqual(world.EXPERIENCES["kill"]["fear"], -0.1)
        self.assertEqual(brain.MUTATION, 0.5)

        Rules.apply({"LITTER_RANGE.1": 9})
        self.assertEqual(world.CORPSE_TURNS, DEFAULTS["CORPSE_TURNS"])
        self.assertEqual(Rules.changed(Rules.current()), {"LITTER_RANGE.1": 9})

    def test_values_are_clamped_and_seed_reach_updates_where_seeds_land(self):
        Rules.apply({"CORPSE_TURNS": 10_000, "AGEING": -3.0, "SEED_REACH": 1})
        self.assertEqual(world.CORPSE_TURNS, BY_KEY["CORPSE_TURNS"].maximum)
        self.assertEqual(world.AGEING, 0.0)
        self.assertEqual(world.SEED_OFFSETS, 4)

    def test_problems_name_pairs_out_of_order_or_alike(self):
        values = {**DEFAULTS, "LITTER_RANGE.0": 5, "LITTER_RANGE.1": 2}
        self.assertEqual(Rules.problems(values), ["Smallest litter must not exceed largest litter"])
        values = {**DEFAULTS, "METABOLISM.0": 1.0, "METABOLISM.1": 1.0}
        self.assertEqual(Rules.problems(values), ["Metabolism, smallest must differ from metabolism, largest"])
        values = {**DEFAULTS, "ARCTIC_SHARE": 0.4, "EQUATOR_SHARE": 0.3}
        self.assertEqual(Rules.problems(values), ["The two arctic zones and the equator must fit in the map together"])
        self.assertEqual(Rules.problems({**values, "EQUATOR_SHARE": 0.2}), [])
        self.assertEqual(Rules.problems(DEFAULTS), [])

    def test_a_world_plays_by_the_rules_set_when_it_runs(self):
        def corpses_kept(turns: int) -> int:
            simulation = Simulation(30, 30, 40, seed=3, grass_density=0)
            for _ in range(turns):
                simulation.advance()
            return sum(not agent.alive for agent in simulation.agents)

        usual = corpses_kept(40)
        Rules.apply({"CORPSE_TURNS": 1})
        self.assertLess(corpses_kept(40), usual)

    def test_a_running_world_works_out_afresh_what_it_settled_from_the_rules_as_it_began(self):
        simulation = Simulation(30, 30, 20, seed=3, grass_density=10)  # 90 tufts to start with
        simulation.advance()
        bands = len(set(simulation._band_of_row))
        Rules.apply({"GRASS_REGROWTH": 0.05, "ARCTIC_SHARE": 0.0})  # no arctic: two fewer climate bands
        simulation.rules_changed()
        self.assertAlmostEqual(simulation.grass_regrowth, 90 * 0.05)
        simulation.advance()
        self.assertEqual(len(set(simulation._band_of_row)), bands - 2)
        self.assertEqual(Rules.WORLD_LOCKED.keys(), {"USE_GPU"})  # where a world thinks is set as it starts


class AdvancedFormTests(unittest.TestCase):
    def test_adjusting_steps_by_the_rule_and_stays_in_range(self):
        form = AdvancedForm({})
        focus(form, "GRASS_REGROWTH")
        form.adjust(1)
        self.assertEqual(form.values["GRASS_REGROWTH"], 0.033)
        form.adjust(-1000)
        self.assertEqual(form.values["GRASS_REGROWTH"], 0.0)
        self.assertEqual(form.changed(), {"GRASS_REGROWTH": 0.0})

    def test_typing_replaces_the_value_and_settles_on_moving(self):
        form = AdvancedForm({})
        focus(form, "CORPSE_TURNS")
        form.type_text("4.5x")  # whole-number rules take no decimal point
        self.assertEqual(form.text(form.focused), "45")
        form.move(1)
        self.assertEqual(form.values["CORPSE_TURNS"], 45)

        focus(form, "GRASS_REGROWTH")
        form.type_text("0.25")
        form.finish()
        self.assertEqual(form.values["GRASS_REGROWTH"], 0.1)  # brought into range

    def test_minus_only_where_values_may_be_negative(self):
        form = AdvancedForm({})
        focus(form, "AGEING")
        form.type_text("-0.5")
        form.finish()
        self.assertEqual(form.values["AGEING"], 0.5)
        focus(form, "EXPERIENCES.kill.fear")
        form.type_text("-0.1")
        form.finish()
        self.assertEqual(form.values["EXPERIENCES.kill.fear"], -0.1)

    def test_backspace_edits_and_an_empty_entry_keeps_the_value(self):
        form = AdvancedForm({})
        focus(form, "CORPSE_TURNS")
        form.backspace()
        self.assertEqual(form.text(form.focused), "3")
        form.backspace()
        form.finish()
        self.assertEqual(form.values["CORPSE_TURNS"], DEFAULTS["CORPSE_TURNS"])

    def test_reset_and_reset_all(self):
        form = AdvancedForm({"CORPSE_TURNS": 60, "AGEING": 0.5})
        self.assertEqual(form.changed(), {"CORPSE_TURNS": 60, "AGEING": 0.5})
        focus(form, "AGEING")
        form.reset()
        self.assertEqual(form.changed(), {"CORPSE_TURNS": 60})
        form.reset_all()
        self.assertEqual(form.changed(), {})



class AdvancedSearchTests(unittest.TestCase):
    def test_a_search_finds_the_rules_holding_every_word_of_it_in_any_case(self):
        from display.Advanced import rule_matches
        stamina = [rule.key for rule in RULES if rule_matches(rule, "STAMINA")]
        self.assertIn("REST_RECOVERY", stamina)  # in its description
        self.assertTrue(all(rule_matches(rule, "") for rule in RULES))  # nothing typed: every rule
        self.assertTrue(rule_matches(BY_KEY["FORCED_LESSONS"], "forced_lessons"))  # by its key
        self.assertTrue(rule_matches(BY_KEY["FORCED_LESSONS"], "mind forced"))  # its group and its name
        self.assertFalse(rule_matches(BY_KEY["FORCED_LESSONS"], "forced unicorns"))  # every word must be there

    def test_the_panel_lists_only_what_the_search_finds_and_esc_clears_it_first(self):
        import os
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame as pg
        from display.Advanced import AdvancedScreen
        pg.init()
        panel = AdvancedScreen(pg.display.set_mode((1200, 820)), {})
        everything = len(panel._listed())
        self.assertEqual(everything, len(RULES))
        panel.searching = True
        for letter in "fatigue":
            panel._key(pg.event.Event(pg.KEYDOWN, key=pg.K_a, mod=0, unicode=letter))
        found = panel._listed()
        self.assertTrue(0 < len(found) < everything)
        self.assertIn(panel.form.focus, found)  # the focus kept on a rule found
        panel._step(len(found))  # round the found rules, back where it started
        self.assertIn(panel.form.focus, found)
        panel._draw()
        panel.search = "unicorns"
        panel._find()
        self.assertEqual(panel._listed(), [])
        panel._draw()  # says nothing matches


if __name__ == "__main__":
    unittest.main()
