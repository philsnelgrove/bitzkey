import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # no real window during tests

import pygame as pg

from display.CatalogScreen import CatalogScreen
from tools import Catalog
from tools.Exchange import import_bitzkeys
from tools.Simulation import Simulation


def setUpModule():  # grown bodies: a Bitzkey made at age 0 isn't feeble here, unless a test says so
    global _grown
    _grown = patch.multiple("tools.Simulation", FEEBLE_YOUTH=0.0, SMALL_HIDING=0.0, DEFAULT_YEAR_LENGTH=360, SLEEP=0)  # and 360-turn years, no sleeping
    _grown.start()


def tearDownModule():
    _grown.stop()


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "catalog.json"
        patcher = patch("tools.Catalog.CATALOG_PATH", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.folder.cleanup)
        self.simulation = Simulation(width=30, height=30, population=40, seed=3)
        for _ in range(5):
            self.simulation.advance()
        self.animal = next(agent for agent in self.simulation.living if agent.brain)

    def test_an_entry_keeps_the_brain_temperament_and_what_the_inspector_shows(self):
        entry = Catalog.entry_for(self.animal, self.simulation.turn)
        self.assertEqual(Catalog.add([entry]), 1)
        [saved] = Catalog.load()
        rebuilt = Catalog.inspected(saved)
        self.assertEqual(rebuilt.dna, self.animal.dna)
        self.assertIsNotNone(rebuilt.brain)
        senses = [0.3] * len(self.animal.brain.layers[0][0][0])
        for mine, theirs in zip(rebuilt.brain.values(senses), self.animal.brain.values(senses), strict=True):
            self.assertAlmostEqual(mine, theirs, delta=1e-5 * max(1.0, abs(theirs)))  # its weights saved to six places
        for trait, value in self.animal.mind.items():
            self.assertAlmostEqual(rebuilt.mind[trait], value, places=3)
        self.assertEqual((rebuilt.identifier, rebuilt.age, rebuilt.generation, rebuilt.last_action),
                         (self.animal.identifier, self.animal.age, self.animal.generation, self.animal.last_action))
        self.assertAlmostEqual(rebuilt.health, self.animal.health, places=3)
        self.assertEqual(set(rebuilt.odds), set(self.animal.odds))
        self.assertIn(f"#{self.animal.identifier:03d}", Catalog.describe(saved))
        [imported] = import_bitzkeys(self.path.read_text(encoding="utf-8"))  # the catalog is itself an export
        self.assertIsNotNone(imported.brain)

    def test_the_same_bitzkey_is_kept_once_and_entries_can_be_removed(self):
        entry = Catalog.entry_for(self.animal, self.simulation.turn)
        other = next(agent for agent in self.simulation.living if agent.brain and agent is not self.animal)
        self.assertEqual(Catalog.add([entry, entry]), 1)
        self.assertEqual(Catalog.add([entry, Catalog.entry_for(other, self.simulation.turn)]), 1)
        self.assertEqual(len(Catalog.load()), 2)
        Catalog.remove(0)
        self.assertEqual(Catalog.inspected(Catalog.load()[0]).identifier, other.identifier)
        self.assertEqual(Catalog.load(Path(self.folder.name) / "none.json"), [])  # no catalog yet: empty
        self.path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            Catalog.load()

    def test_the_catalog_screen_shows_the_entries_and_hands_one_back_to_import(self):
        others = [agent for agent in self.simulation.living if agent.brain][:3]
        Catalog.add([Catalog.entry_for(agent, self.simulation.turn) for agent in others])
        pg.init()
        screen = pg.display.set_mode((1200, 820))
        catalog = CatalogScreen(screen)
        catalog._draw()  # draws the list and the picked one's inspector
        catalog._click(catalog._row(1).center)
        self.assertEqual(catalog.picked, 1)
        pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_DOWN, unicode="", mod=0))
        pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_RETURN, unicode="", mod=0))
        outcome, entries = catalog.run()
        self.assertEqual(outcome, "import")
        self.assertEqual([Catalog.inspected(entry).identifier for entry in entries], [others[2].identifier])
        catalog._choose(0)
        pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_DELETE, unicode="", mod=0))  # asks first
        pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_RETURN, unicode="", mod=0))  # REMOVE
        pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_ESCAPE, unicode="", mod=0))
        self.assertEqual(catalog.run(), ("close", None))
        self.assertEqual([Catalog.inspected(entry).identifier for entry in Catalog.load()],
                         [others[1].identifier, others[2].identifier])
        pg.display.quit()

    def test_browsing_from_a_running_world_deploy_takes_the_place_of_import(self):
        others = [agent for agent in self.simulation.living if agent.brain][:3]
        Catalog.add([Catalog.entry_for(agent, self.simulation.turn) for agent in others])
        pg.init()
        screen = pg.display.set_mode((1200, 820))
        catalog = CatalogScreen(screen, browsing=True)
        catalog._draw()
        catalog._choose(1)
        self.assertEqual(catalog._click(catalog._layout()["import"].center), "deploy")
        pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_RETURN, unicode="", mod=0))
        outcome, entries = catalog.run()
        self.assertEqual(outcome, "deploy")
        self.assertEqual([Catalog.inspected(entry).identifier for entry in entries], [others[1].identifier])
        pg.display.quit()

    def test_deploying_puts_five_of_each_around_a_cell_with_their_brains_and_minds(self):
        from tools.Simulation import BIRTH_REACH, DEPLOY_GROUP, SPAWN_SPREAD
        entry = Catalog.entry_for(self.animal, self.simulation.turn)
        bitzkey = Catalog.imported(entry)
        turn, before = self.simulation.turn, len(self.simulation.agents)
        deployed = self.simulation.deploy([bitzkey, bitzkey], 10, 1)  # near the top: kept on the map
        self.assertEqual((DEPLOY_GROUP, len(deployed), len(self.simulation.agents)), (5, 10, before + 10))
        for agent in deployed:
            self.assertEqual(agent.dna, self.animal.dna)
            self.assertLessEqual(min(abs(agent.x - 10), 30 - abs(agent.x - 10)), SPAWN_SPREAD + BIRTH_REACH)  # the nearest room to its spot
            self.assertTrue(0 <= agent.y <= 1 + SPAWN_SPREAD + BIRTH_REACH)
            self.assertEqual(agent.brain, bitzkey.brain.resized(agent.brain.depth, agent.brain.width))  # its catalogued brain
            self.assertIsNot(agent.brain, deployed[0].brain if agent is not deployed[0] else None)
            self.assertAlmostEqual(agent.mind["fear"], self.animal.mind["fear"], places=3)
        self.assertEqual(self.simulation.turn, turn)  # between turns
        living = set(self.simulation.census.at(turn))
        self.assertTrue({agent.identifier for agent in deployed} <= living)

        self.simulation.advance()
        self.assertEqual(self.simulation.turn, turn + 1)

    def test_several_are_chosen_as_in_windows_with_ctrl_and_shift(self):
        from unittest.mock import patch
        others = [agent for agent in self.simulation.living if agent.brain][:5]
        Catalog.add([Catalog.entry_for(agent, self.simulation.turn) for agent in others])
        pg.init()
        screen = pg.display.set_mode((1200, 820))
        catalog = CatalogScreen(screen)
        catalog._draw()

        def click(index, mods=0):
            with patch("pygame.key.get_mods", return_value=mods):
                catalog._click(catalog._row(index).center)

        click(1)
        self.assertEqual(catalog.chosen, {1})
        click(3, pg.KMOD_CTRL)  # added
        self.assertEqual(catalog.chosen, {1, 3})
        click(1, pg.KMOD_CTRL)  # taken out
        self.assertEqual(catalog.chosen, {3})
        click(0, pg.KMOD_SHIFT)  # the run from the last one clicked, entry 1
        self.assertEqual(catalog.chosen, {0, 1})
        click(4, pg.KMOD_CTRL | pg.KMOD_SHIFT)  # the run from there to 4, added
        self.assertEqual(catalog.chosen, {0, 1, 2, 3, 4})
        pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_RETURN, unicode="", mod=0))
        outcome, entries = catalog.run()
        self.assertEqual([Catalog.inspected(entry).identifier for entry in entries], [agent.identifier for agent in others])
        click(0)
        pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_DOWN, unicode="", mod=pg.KMOD_SHIFT))  # SHIFT+DOWN extends
        pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_DELETE, unicode="", mod=0))  # asks first
        pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_RETURN, unicode="", mod=0))  # REMOVE both
        pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_ESCAPE, unicode="", mod=0))
        self.assertEqual(catalog.run(), ("close", None))
        self.assertEqual([Catalog.inspected(entry).identifier for entry in Catalog.load()],
                         [agent.identifier for agent in others[2:]])
        pg.display.quit()

    def test_the_designer_button_adds_a_design_to_the_catalog_as_a_genome_alone(self):
        from tools.Genome import encode_dna
        Catalog.add([Catalog.entry_for(self.animal, self.simulation.turn)])
        designed = encode_dna("grazer", size=5, speed=7)
        screen = pg.display.set_mode((1200, 820))
        catalog = CatalogScreen(screen, browsing=True)
        catalog._draw()
        button = catalog._layout()["design"]
        with patch("display.CatalogScreen.Designer") as designer:
            designer.return_value.run.return_value = ("apply", designed)
            pg.event.post(pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=button.center))
            pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_ESCAPE, unicode="", mod=0))
            self.assertEqual(catalog.run(), ("close", None))
        self.assertEqual(designer.call_args.args[1], self.animal.dna)  # opened on the entry picked
        entries = Catalog.load()
        self.assertEqual(len(entries), 2)
        self.assertEqual(Catalog.imported(entries[1]).dna, designed)
        self.assertIsNone(Catalog.imported(entries[1]).brain)  # its genome alone
        self.assertTrue(Catalog.describe(entries[1]).startswith("designed grazer"))
        self.assertEqual(catalog.picked, 1)  # the design, picked
        with patch("display.CatalogScreen.Designer") as designer:
            designer.return_value.run.return_value = ("cancel", None)
            self.assertTrue(catalog._design())
        self.assertEqual(len(Catalog.load()), 2)  # cancelled: nothing added

    def test_a_dead_bitzkey_on_the_graph_shows_how_it_died(self):
        from display.ColourGraph import death_note
        while not self.simulation.death_records:  # until something has died
            self.simulation.advance()
        dead = next(identifier for identifier in self.simulation.death_records)
        death = self.simulation.death_records[dead]
        self.assertTrue(death_note(self.simulation, dead).startswith(f"Died at turn {death.turn}: {death.cause}"))
        self.assertEqual(death_note(self.simulation, 10_000), "Dead: only its genome is known.")


if __name__ == "__main__":
    unittest.main()
