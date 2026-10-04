import unittest

from tools.Census import FAMILIES, SWATCHES, bucket, family, tally
from tools.Simulation import Simulation


class CensusTests(unittest.TestCase):
    def test_colours_fall_into_families(self):
        self.assertEqual(family((255, 0, 0)), "red")
        self.assertEqual(family((250, 20, 40)), "red")
        self.assertEqual(family((0, 0, 255)), "blue")
        self.assertEqual(family((255, 255, 0)), "yellow")
        self.assertEqual(family((250, 250, 245)), "white")
        self.assertEqual(family((120, 122, 118)), "grey")
        self.assertEqual(family((30, 0, 0)), "black")

    def test_every_family_has_a_swatch_in_its_own_family(self):
        self.assertEqual(set(SWATCHES), set(FAMILIES))
        for name, swatch in SWATCHES.items():
            self.assertEqual(family(swatch), name)

    def test_tally_counts_names(self):
        self.assertEqual(tally(["red", "blue", "red"]), {"red": 2, "blue": 1})

    def test_bucket_averages_consecutive_turns(self):
        census = [{"ring": {"red": count}, "dot": {}} for count in (2, 4, 6, 8)]

        self.assertEqual(bucket(census, "ring", 2), [(0, {"red": 3.0}), (2, {"red": 7.0})])
        self.assertEqual(len(bucket(census, "ring", 10)), 4)

    def test_simulation_counts_living_bitzkeys_each_turn(self):
        simulation = Simulation(width=20, height=20, population=30, seed=4)
        for _ in range(5):
            simulation.advance()

        self.assertEqual(len(simulation.census), 6)
        for part in ("ring", "dot"):
            self.assertEqual(sum(simulation.census[-1][part].values()), len(simulation.living))

    def test_new_turn_after_rewind_replaces_the_abandoned_future(self):
        simulation = Simulation(width=20, height=20, population=30, seed=4)
        for _ in range(6):
            simulation.advance()
        recorded = list(simulation.census)

        for _ in range(3):
            simulation.rewind()
        simulation.advance()
        self.assertEqual(simulation.census, recorded)

        simulation.history = simulation.history[:simulation.history_index + 1]
        simulation.advance()
        self.assertEqual(len(simulation.census), simulation.turn + 1)
        self.assertEqual(simulation.census[:simulation.turn], recorded[:simulation.turn])


if __name__ == "__main__":
    unittest.main()
