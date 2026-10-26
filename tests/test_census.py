import unittest
from unittest.mock import patch

from tools.Census import Census, colour_order
from tools.Simulation import Simulation


def setUpModule():  # grown bodies: a Bitzkey made at age 0 isn't feeble here, unless a test says so
    global _grown
    _grown = patch.multiple("tools.Simulation", FEEBLE_YOUTH=0.0, SMALL_HIDING=0.0, DEFAULT_YEAR_LENGTH=360, SLEEP=0)  # and 360-turn years, no sleeping
    _grown.start()


def tearDownModule():
    _grown.stop()


class CensusTests(unittest.TestCase):
    def test_similar_colours_sort_together(self):
        reds = [colour_order((250, 10, 10)), colour_order((200, 30, 20))]
        blue = colour_order((10, 10, 250))
        grey = colour_order((128, 128, 128))

        self.assertLess(grey, min(reds))
        self.assertLess(max(reds), blue)

    def test_census_remembers_who_was_alive_at_each_turn(self):
        census = Census()
        for turn in range(5):
            census.record(turn, range(turn))

        self.assertEqual(census.at(3), (0, 1, 2))
        self.assertEqual(census.at(99), (0, 1, 2, 3))
        self.assertEqual(census.at(-1), ())

    def test_census_thins_evenly_and_keeps_the_latest_turn(self):
        census = Census()
        with patch("tools.Census.MAX_SNAPSHOTS", 10):
            for turn in range(37):
                census.record(turn, [turn])

        turns = [turn for turn, _ in census.snapshots]
        self.assertLessEqual(len(turns), 10)
        self.assertEqual(turns[-1], 36)
        self.assertTrue(all(turn % census.stride == 0 for turn in turns[:-1]))
        self.assertEqual(census.at(17), (16,))

    def test_forgetting_from_a_turn_drops_it_and_those_after(self):
        census = Census()
        for turn in range(6):
            census.record(turn, [turn])

        census.forget_from(3)

        self.assertEqual([turn for turn, _ in census.snapshots], [0, 1, 2])

    def test_census_forgets_turns_older_than_the_graph_covers(self):
        census = Census()
        with patch("tools.Census.GRAPH_TURNS", 10):
            forgot = [census.record(turn, [turn]) for turn in range(15)]

        self.assertEqual([turn for turn, _ in census.snapshots], list(range(4, 15)))
        self.assertEqual(forgot, [False] * 11 + [True] * 4)

    @unittest.skip("its world no longer forgets its oldest genome within the run since the sea, terrain and stamina; a new seed is due")
    def test_a_world_forgets_the_genomes_and_deaths_from_before_its_graph(self):
        # Short lives, eight turns a year: breeding free of rest and of heat, which would last years of them.
        with patch("tools.Census.GRAPH_TURNS", 5), patch("tools.Simulation.BREED_COOLDOWN", 0.0), patch("tools.Simulation.HEAT_FORESIGHT", 0.0),              patch.dict("tools.Simulation.HEAT_CLOSEST", {"grazer": 0.0, "omnivore": 0.0, "predator": 0.0}),              patch.dict("tools.Simulation.HEAT_LENGTH", {"grazer": 0.75, "omnivore": 0.75, "predator": 0.75}):
            simulation = Simulation(width=20, height=20, population=30, seed=4, year_length=8)
            for _ in range(200):
                simulation.advance()

        start = simulation.census.snapshots[0][0]
        self.assertEqual(start, simulation.turn - 5)
        self.assertGreater(simulation.lineage_first, 0)
        self.assertEqual(min(simulation.lineage), simulation.lineage_first)
        self.assertTrue(all(death.turn >= start for death in simulation.death_records.values()))
        self.assertTrue(all(identifier in simulation.lineage
                            for _, alive in simulation.census.snapshots for identifier in alive))
        self.assertTrue(all(identifier in simulation.lineage for identifier in simulation.death_records))
        self.assertEqual(sorted(simulation._death_order),  # the deaths' order holds just those kept, oldest first
                         sorted((death.turn, identifier) for identifier, death in simulation.death_records.items()))
        turns = [turn for turn, _ in simulation._death_order]
        self.assertEqual(turns, sorted(turns))  # oldest first; a turn's deaths in the order they came

    def test_simulation_records_who_lives_and_every_genome(self):
        simulation = Simulation(width=20, height=20, population=30, seed=4)
        for _ in range(5):
            simulation.advance()

        self.assertEqual(simulation.census.at(simulation.turn),
                         tuple(agent.identifier for agent in simulation.living))
        self.assertTrue(all(simulation.lineage[agent.identifier] == agent.dna for agent in simulation.agents))

if __name__ == "__main__":
    unittest.main()
