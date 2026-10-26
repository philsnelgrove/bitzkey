"""Days and nights: night over half the map, moving round it once a day, and animals resting through their own, as in
the C++ version."""

import unittest
from unittest.mock import patch

from tools.Brain import ACTIONS
from tools.Genome import NOCTURNAL, encode_dna
from tools.Simulation import DAY_LENGTH, DAYS_IN_YEAR, FULL_STAMINA, LIFESPAN_YEARS, Bitzkey, Simulation


def animal(nocturnal: bool, terrain: str = "terrestrial") -> int:
    return encode_dna("grazer", terrain=terrain) | (NOCTURNAL if nocturnal else 0)


class DayTests(unittest.TestCase):
    def test_a_new_worlds_year_is_360_days_of_eight_turns(self):
        self.assertEqual((DAY_LENGTH, 8 * DAYS_IN_YEAR), (8, 2880))
        Simulation(20, 20, 0, 1)
        smallest = Bitzkey(0, encode_dna("grazer", size=0), 0, 0)
        self.assertAlmostEqual(smallest.lifespan, LIFESPAN_YEARS[0] * 2880)  # three years of 360 days

    def test_night_covers_half_the_map_and_moves_west_round_it_once_a_day(self):
        simulation = Simulation(40, 20, 0, 1)
        dark_turns = [0] * 40
        for _ in range(DAY_LENGTH):
            dark = [x for x in range(40) if simulation.night(x)]
            self.assertEqual(len(dark), 20)
            for x in dark:
                dark_turns[x] += 1
            simulation.turn += 1
        self.assertEqual(set(dark_turns), {DAY_LENGTH // 2})  # every column dark half of each day
        self.assertEqual(simulation.night_start(), Simulation(40, 20, 0, 1).night_start())  # round again in a day

    @patch("tools.Simulation.SLEEP", 1)
    def test_an_animal_rests_through_its_night(self):
        simulation = Simulation(40, 20, 0, 1)
        simulation.grass = {}
        dark = next(x for x in range(40) if simulation.night(x))
        light = next(x for x in range(40) if not simulation.night(x))
        day_owl = simulation._add_agent(animal(False), dark, 10)
        night_owl = simulation._add_agent(animal(True), dark, 12)
        late = simulation._add_agent(animal(True), light, 10)
        simulation._index_cells()
        self.assertEqual([simulation.resting_time(agent) for agent in (day_owl, night_owl, late)], [True, False, True])
        for _ in range(3):
            for agent in (day_owl, night_owl, late):
                agent.stamina = FULL_STAMINA
            simulation._think_all([day_owl, night_owl, late])
            self.assertEqual((day_owl.last_action, late.last_action), ("rested", "rested"))
        senses = dict(zip(__import__("tools.Brain", fromlist=["SENSES"]).SENSES, simulation._sense(night_owl)[0]))
        self.assertEqual(senses["night"], float(simulation.night(night_owl.x)))
        self.assertEqual(senses["nocturnal"], 1.0)
        self.assertTrue(set(day_owl.odds) <= {"rest", "flee", "defend"})

    @patch("tools.Simulation.FEEBLE_YOUTH", 0.5)  # infants for half their lives, in this test
    def test_an_infant_beside_a_parent_is_sheltered_from_the_climate(self):
        simulation = Simulation(20, 20, 0, 1)
        mother = simulation._add_agent(animal(False), 5, 5)
        near = simulation._add_agent(animal(False), 5, 6)
        apart = simulation._add_agent(animal(False), 9, 9)
        near.mother = apart.mother = mother.identifier
        mother.age = int(mother.lifespan * 0.6)  # grown
        self.assertTrue(simulation.sheltered(near))
        self.assertFalse(simulation.sheltered(apart))  # too far from its mother
        self.assertFalse(simulation.sheltered(mother))  # grown, and an orphan besides
        mother.health = 0
        self.assertFalse(simulation.sheltered(near))  # no shelter from a dead parent


if __name__ == "__main__":
    unittest.main()
