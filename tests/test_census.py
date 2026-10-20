import unittest
from unittest.mock import patch

from tools.Census import Census, colour_order
from tools.Simulation import Simulation


def setUpModule():  # grown bodies: a Bitzkey made at age 0 isn't feeble here, unless a test says so
    global _grown
    _grown = patch("tools.Simulation.FEEBLE_YOUTH", 0.0)
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

    def test_rewinding_forgets_the_abandoned_future(self):
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

    def test_a_world_forgets_the_genomes_and_deaths_from_before_its_graph(self):
        with patch("tools.Census.GRAPH_TURNS", 5):
            simulation = Simulation(width=20, height=20, population=30, seed=4, year_length=8)  # short lives
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

    def test_simulation_records_who_lives_and_every_genome(self):
        simulation = Simulation(width=20, height=20, population=30, seed=4)
        for _ in range(5):
            simulation.advance()

        self.assertEqual(simulation.census.at(simulation.turn),
                         tuple(agent.identifier for agent in simulation.living))
        self.assertTrue(all(simulation.lineage[agent.identifier] == agent.dna for agent in simulation.agents))

    def test_seeking_back_and_forth_leaves_the_census_as_recorded(self):
        simulation = Simulation(width=20, height=20, population=30, seed=4)
        for _ in range(12):
            simulation.advance()
        recorded = list(simulation.census.snapshots)

        simulation.seek(3)
        simulation.seek(9)
        for _ in range(3):
            simulation.advance()

        self.assertEqual(simulation.census.snapshots, recorded)

if __name__ == "__main__":
    unittest.main()
