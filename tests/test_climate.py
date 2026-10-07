import unittest
from unittest.mock import patch

from tools.Brain import ACTIONS, SENSES
from tools.Genome import decode_climate, encode_dna
from tools.Simulation import (COLD, HOT, MEDIUM, PLANT_MEAL, VERY_COLD, Simulation, season, temperature_at, year,
                              zone_temperature)


def world(turn: int = 50, size: int = 100) -> Simulation:
    """An empty world at a turn, its grass cleared."""
    simulation = Simulation(width=size, height=size, population=0, seed=3)
    simulation.grass.clear()
    simulation.turn = turn
    simulation._update_climate()
    return simulation


def take_turn(simulation: Simulation) -> None:
    """One turn in which animals neither think nor die of age."""
    with patch.object(Simulation, "_think"), patch.object(Simulation, "_dies_of_age", return_value=False):
        simulation._take_turn()


class ZoneTests(unittest.TestCase):
    def test_zones_band_the_map_arctic_temperate_equator_temperate_arctic(self):
        rows = [zone_temperature(y, 100) for y in range(100)]
        self.assertEqual(rows, [COLD] * 20 + [MEDIUM] * 20 + [HOT] * 20 + [MEDIUM] * 20 + [COLD] * 20)

    def test_a_year_is_180_turns_in_four_quarters(self):
        self.assertEqual([season(turn) for turn in (0, 44, 45, 89, 90, 134, 135, 179, 180)],
                         [0, 0, 1, 1, 2, 2, 3, 3, 0])
        self.assertEqual([year(turn) for turn in (0, 179, 180, 400)], [1, 1, 2, 3])

    def test_each_hemisphere_and_the_equator_are_a_step_colder_in_its_winter(self):
        normal = [zone_temperature(y, 100) for y in range(100)]
        north = [temperature_at(y, 100, 10) for y in range(100)]  # first quarter
        south = [temperature_at(y, 100, 100) for y in range(100)]  # third quarter
        self.assertEqual(north, [VERY_COLD] * 20 + [COLD] * 20 + [MEDIUM] * 20 + normal[60:])
        self.assertEqual(south, normal[:40] + [MEDIUM] * 20 + [COLD] * 20 + [VERY_COLD] * 20)
        for turn in (50, 150, 185 + 45):  # second and fourth quarters, and next year's second
            self.assertEqual([temperature_at(y, 100, turn) for y in range(100)], normal)
        self.assertEqual([temperature_at(y, 100, 190) for y in range(100)], north)  # the next year repeats


class FeelingTests(unittest.TestCase):
    def test_tolerance_spares_an_animal_one_direction_and_plants_none(self):
        simulation = world()
        tender = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM), 1, 1)
        hardy = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM, cold_tolerance=True), 1, 1)
        warm = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM, heat_tolerance=True), 1, 1)
        plant = simulation._add_agent(encode_dna("plant", temperature=MEDIUM, cold_tolerance=True,
                                                 heat_tolerance=True), 1, 1)
        self.assertEqual([tender.climate_steps(here) for here in (HOT, MEDIUM, COLD, VERY_COLD)], [1, 0, 1, 2])
        self.assertEqual([hardy.climate_steps(here) for here in (HOT, MEDIUM, COLD, VERY_COLD)], [1, 0, 0, 0])
        self.assertEqual([warm.climate_steps(here) for here in (HOT, MEDIUM, COLD, VERY_COLD)], [0, 0, 1, 2])
        self.assertEqual([plant.climate_steps(here) for here in (HOT, MEDIUM, COLD, VERY_COLD)], [1, 0, 1, 2])


class EffectTests(unittest.TestCase):
    def test_a_step_from_home_doubles_an_animals_food_and_two_steps_also_hurt(self):
        simulation = world(turn=60)  # the second quarter: no winter
        at_home = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM), 10, 30)
        one_away = simulation._add_agent(encode_dna("grazer", temperature=COLD), 20, 30)
        two_away = simulation._add_agent(encode_dna("grazer", temperature=VERY_COLD), 30, 30)
        simulation._add_agent(encode_dna("predator"), 90, 90)
        take_turn(simulation)
        self.assertAlmostEqual(at_home.hunger, at_home.burn)
        self.assertAlmostEqual(one_away.hunger, 2 * one_away.burn)
        self.assertAlmostEqual(two_away.hunger, 2 * two_away.burn)
        self.assertEqual((at_home.health, one_away.health), (at_home.max_health, one_away.max_health))
        self.assertAlmostEqual(two_away.health, 0.9 * two_away.max_health)

    def test_winter_turns_a_home_zone_hostile_and_the_cold_can_kill(self):
        simulation = world(turn=0)  # turn 1, next, is the north's winter
        grazer = simulation._add_agent(encode_dna("grazer", temperature=HOT), 50, 10)  # arctic: very cold now
        grazer.health = 5
        simulation._add_agent(encode_dna("predator"), 90, 90)
        take_turn(simulation)
        self.assertFalse(grazer.alive)
        self.assertEqual(grazer.last_action, "froze")

    def test_a_plant_out_of_its_temperature_withers_without_growing_or_seeding_and_two_steps_hurt(self):
        simulation = world(turn=60)
        one_away = simulation._add_agent(encode_dna("plant", temperature=HOT, fertility=7, cold_tolerance=True), 50, 30)
        two_away = simulation._add_agent(encode_dna("plant", temperature=VERY_COLD, fertility=7), 20, 30)
        for plant in (one_away, two_away):
            plant.health = plant.max_health / 2
            plant.hunger = 70  # too hungry to heal
        simulation._add_agent(encode_dna("predator"), 90, 90)
        with patch.object(simulation.random, "random", return_value=0.0):  # they would always seed
            take_turn(simulation)
        self.assertEqual((one_away.last_action, two_away.last_action), ("withering", "withering"))
        self.assertAlmostEqual(one_away.health, 0.5 * one_away.max_health)  # a step away doesn't hurt
        self.assertAlmostEqual(two_away.health, 0.4 * two_away.max_health)
        self.assertEqual((len(simulation.agents), simulation.seeds), (3, []))

    def test_a_plant_out_of_its_temperature_gives_half_the_food(self):
        simulation = world(turn=60)
        at_home = simulation._add_agent(encode_dna("plant", temperature=MEDIUM), 50, 30)
        away = simulation._add_agent(encode_dna("plant", temperature=COLD), 60, 30)
        self.assertEqual((simulation._plant_yield(at_home), simulation._plant_yield(away)), (1.0, 0.5))
        for plant in (at_home, away):
            grazer = simulation._add_agent(encode_dna("grazer", size=7), plant.x, plant.y + 1)  # a big stomach
            grazer.hunger = 90
            simulation._browse(grazer, plant)
            meal = PLANT_MEAL * grazer.browsing * simulation._plant_yield(plant) / grazer.stomach
            self.assertAlmostEqual(grazer.hunger, 90 - meal)

    def test_spawned_plants_are_at_home_in_their_zone_and_imports_keep_theirs(self):
        simulation = Simulation(width=60, height=60, population=300, seed=4)
        plants = [agent for agent in simulation.agents if agent.species == "plant"]
        self.assertTrue(plants)
        for plant in plants:  # the group's centre is within two rows
            self.assertIn(plant.climate[0], {zone_temperature(plant.y + dy, 60) for dy in range(-2, 3)
                                             if 0 <= plant.y + dy < 60})
        self.assertEqual(len({plant.climate[0] for plant in plants}), 3)  # hot, medium and cold zones
        from tools.Exchange import Imported
        dna = encode_dna("plant", temperature=VERY_COLD)
        imported = Simulation(width=30, height=30, population=0, seed=4, imported=[Imported(dna)])
        self.assertTrue(all(decode_climate(agent.dna)[0] == VERY_COLD for agent in imported.agents))

    def test_grass_grows_only_where_it_is_medium_now(self):
        simulation = world(turn=10)  # the north's winter: its temperate zone is cold, the equator medium
        simulation._sow(3000)
        rows = {y for _, y in simulation.grass}
        self.assertEqual(rows, set(range(40, 80)))


class SenseTests(unittest.TestCase):
    def test_an_animal_senses_how_far_it_is_from_home_and_how_close_home_weather_is(self):
        simulation = world(turn=60)
        grazer = simulation._add_agent(encode_dna("grazer", temperature=COLD, sight=7), 50, 23)
        self.assertEqual(simulation._comfort_row(grazer, grazer.sight), (50, 19))  # the arctic, four rows north
        senses = simulation._sense(grazer)[0]
        self.assertEqual(senses[SENSES.index("climate_discomfort")], 0.5)  # temperate: a step too warm
        self.assertAlmostEqual(senses[SENSES.index("comfort_closeness")], 1 - 3 / 9)

        at_home = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM), 50, 30)
        senses = simulation._sense(at_home)[0]
        self.assertEqual((senses[SENSES.index("climate_discomfort")], senses[SENSES.index("comfort_closeness")]),
                         (0.0, 1.0))

        lost = simulation._add_agent(encode_dna("grazer", temperature=VERY_COLD, sight=0), 50, 50)  # none in sight
        senses = simulation._sense(lost)[0]
        self.assertEqual((senses[SENSES.index("climate_discomfort")], senses[SENSES.index("comfort_closeness")]),
                         (1.0, 0.0))
        self.assertNotIn("migrate", ACTIONS)  # moving toward home is left for brains to learn


class SeedTests(unittest.TestCase):
    def test_a_seed_that_falls_where_it_cant_grow_waits_for_its_temperature(self):
        simulation = world(turn=60)
        simulation.body_mutation = 0
        plant = simulation._add_agent(encode_dna("plant", temperature=COLD, fertility=7), 50, 30)
        simulation._index_cells()
        with patch.object(simulation.random, "random", return_value=0.0):
            simulation._seed(plant)
        self.assertEqual(len(simulation.agents), 1)
        [(x, y, dna, generation, sown)] = simulation.seeds
        self.assertEqual((generation, sown, plant.children), (1, 60, 1))

        simulation.turn = 190  # next year's winter in the north makes its temperate zone cold
        simulation._update_climate()
        simulation._index_cells()
        simulation._germinate()
        self.assertEqual(simulation.seeds, [])
        seedling = simulation.agents[-1]
        self.assertEqual((seedling.x, seedling.y, seedling.dna, seedling.generation), (x, y, dna, 1))

    def test_a_seed_stays_viable_for_three_quarters_of_a_year(self):
        simulation = world(turn=0)
        seed = (50, 30, encode_dna("plant", temperature=VERY_COLD), 1, 0)  # never very cold there
        simulation.seeds = [seed]
        for turn, waiting in ((134, [seed]), (135, [])):
            simulation.turn = turn
            simulation._update_climate()
            simulation._germinate()
            self.assertEqual(simulation.seeds, waiting)

    def test_waiting_seeds_rewind_and_replay_with_the_world(self):
        simulation = Simulation(width=40, height=40, population=60, seed=1)
        for _ in range(90):
            simulation.advance()
        seeds = list(simulation.seeds)
        self.assertTrue(seeds)
        simulation.seek(30)
        simulation.seek(90)
        self.assertEqual(simulation.seeds, seeds)


if __name__ == "__main__":
    unittest.main()
