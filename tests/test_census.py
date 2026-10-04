import unittest
from unittest.mock import patch

from tools.Census import Census, colour_order
from tools.Simulation import Simulation


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

    def test_simulation_records_who_lives_and_every_genome(self):
        simulation = Simulation(width=20, height=20, population=30, seed=4)
        for _ in range(5):
            simulation.advance()

        self.assertEqual(simulation.census.at(simulation.turn),
                         tuple(agent.identifier for agent in simulation.living))
        self.assertTrue(all(simulation.lineage[agent.identifier] == agent.dna for agent in simulation.agents))

    def test_a_new_turn_after_rewinding_replaces_the_census_future(self):
        simulation = Simulation(width=20, height=20, population=30, seed=4)
        for _ in range(6):
            simulation.advance()
        for _ in range(3):
            simulation.rewind()

        simulation.history = simulation.history[:simulation.history_index + 1]
        simulation.advance()

        self.assertEqual(simulation.census.snapshots[-1][0], simulation.turn)
        self.assertEqual(simulation.census.at(simulation.turn),
                         tuple(agent.identifier for agent in simulation.living))


if __name__ == "__main__":
    unittest.main()
