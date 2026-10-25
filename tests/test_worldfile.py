import unittest
from unittest.mock import patch

from tools import Rules
from tools.Simulation import Simulation
from tools.WorldFile import WORLD_MAGIC, world_bytes, world_from_bytes


class WorldFileTests(unittest.TestCase):
    def tearDown(self):
        Rules.apply({})

    def test_a_world_loaded_carries_on_exactly_as_it_would_have(self):
        world = Simulation(30, 30, population=40, seed=5)
        for _ in range(40):
            world.advance()
        data = world_bytes(world)
        loaded = world_from_bytes(data)
        self.assertEqual(world_bytes(loaded), data)  # every Bitzkey, brain, seed and tuft as it was
        self.assertEqual((loaded.turn, loaded.frontier, loaded.earliest), (world.turn, world.turn, world.turn))
        for _ in range(30):
            world.advance()
            loaded.advance()
        self.assertEqual(world_bytes(loaded), world_bytes(world))

    def test_a_world_that_has_forgotten_its_oldest_genomes_saves_and_loads_them_from_where_they_start(self):
        with patch("tools.Census.GRAPH_TURNS", 5):
            world = Simulation(20, 20, population=30, seed=3, year_length=8)  # short lives
            for _ in range(200):
                world.advance()
        data = world_bytes(world)
        loaded = world_from_bytes(data)
        self.assertGreater(loaded.lineage_first, 0)
        self.assertEqual((loaded.lineage_first, loaded.lineage), (world.lineage_first, world.lineage))
        self.assertEqual(world_bytes(loaded), data)

    def test_a_world_brings_its_rules_but_not_how_this_computer_runs_it(self):
        Rules.apply({"CORPSE_TURNS": 60, "HISTORY_TURNS": 50})
        data = world_bytes(Simulation(20, 20, population=10, seed=2))
        Rules.apply({"HISTORY_TURNS": 80})
        world_from_bytes(data)
        self.assertEqual((Rules.get("CORPSE_TURNS"), Rules.get("HISTORY_TURNS")), (60, 80))

    def test_what_isnt_a_world_says_so(self):
        with self.assertRaisesRegex(ValueError, "isn't a Bitzkey world"):
            world_from_bytes(b"not a world at all")
        data = world_bytes(Simulation(20, 20, population=10, seed=2))
        with self.assertRaisesRegex(ValueError, "cut short"):
            world_from_bytes(data[:-4])
        self.assertTrue(data.startswith(WORLD_MAGIC))


if __name__ == "__main__":
    unittest.main()
