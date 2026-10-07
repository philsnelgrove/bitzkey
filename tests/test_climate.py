import unittest
from unittest.mock import patch

from tools.Brain import ACTIONS, SENSES
from tools.Genome import decode_climate, encode_dna
from tools.Simulation import (COLD, CURIOSITY_JOY, DEATH_PENALTY, HOT, KILL_MEAL, KILL_MEAL_FOR_SIZE, LEARNING_RATE, MEDIUM, PLANT_MEAL,
                              SOCIAL_MEMORY, SOCIAL_MOOD, VERY_COLD, Simulation, season, temperature_at, year,
                              zone_temperature)


def world(turn: int = 200, size: int = 100) -> Simulation:
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

    def test_a_year_is_360_turns_in_four_quarters_by_default(self):
        self.assertEqual([season(turn) for turn in (0, 89, 90, 179, 180, 269, 270, 359, 360)],
                         [0, 0, 1, 1, 2, 2, 3, 3, 0])
        self.assertEqual([year(turn) for turn in (0, 359, 360, 800)], [1, 1, 2, 3])

    def test_a_world_sets_the_year_length_for_seasons_and_lifespans(self):
        short = Simulation(width=20, height=20, population=0, seed=1, year_length=100)
        self.assertEqual((short.year_length, season(25), season(50), year(100)), (100, 1, 2, 2))
        animal = short._add_agent(encode_dna("grazer", size=0), 1, 1)
        self.assertEqual(animal.lifespan, 100)  # the smallest live a year
        with self.assertRaisesRegex(ValueError, "at least 4"):
            Simulation(width=20, height=20, population=0, seed=1, year_length=3)
        Simulation(width=20, height=20, population=0, seed=1)
        self.assertEqual(year(360), 2)  # back to the default

    def test_each_hemisphere_and_the_equator_are_a_step_colder_in_its_winter(self):
        normal = [zone_temperature(y, 100) for y in range(100)]
        north = [temperature_at(y, 100, 100) for y in range(100)]  # second quarter
        south = [temperature_at(y, 100, 300) for y in range(100)]  # fourth quarter
        self.assertEqual(north, [VERY_COLD] * 20 + [COLD] * 20 + [MEDIUM] * 20 + normal[60:])
        self.assertEqual(south, normal[:40] + [MEDIUM] * 20 + [COLD] * 20 + [VERY_COLD] * 20)
        for turn in (10, 200, 370):  # first and third quarters, and next year's first: mild
            self.assertEqual([temperature_at(y, 100, turn) for y in range(100)], normal)
        self.assertEqual([temperature_at(y, 100, 460) for y in range(100)], north)  # the next year repeats


class FeelingTests(unittest.TestCase):
    def test_hardiness_halves_an_animals_exposure_in_one_direction_and_plants_have_none(self):
        simulation = world()
        tender = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM), 1, 1)
        hardy = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM, cold_tolerance=True), 1, 1)
        warm = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM, heat_tolerance=True), 1, 1)
        plant = simulation._add_agent(encode_dna("plant", temperature=MEDIUM, cold_tolerance=True,
                                                 heat_tolerance=True), 1, 1)
        heres = (HOT, MEDIUM, COLD, VERY_COLD)
        for animal in (tender, hardy, warm, plant):  # every step is felt
            self.assertEqual([animal.climate_steps(here) for here in heres], [1, 0, 1, 2])
        self.assertEqual([tender.exposure_share(here) for here in heres], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual([hardy.exposure_share(here) for here in heres], [1.0, 1.0, 0.5, 0.5])
        self.assertEqual([warm.exposure_share(here) for here in heres], [0.5, 1.0, 1.0, 1.0])
        self.assertEqual([plant.exposure_share(here) for here in heres], [1.0, 1.0, 1.0, 1.0])

    def test_a_hardy_animal_suffers_half_the_extra_hunger_and_damage(self):
        simulation = world()  # mid-year, mild
        tender = simulation._add_agent(encode_dna("grazer", temperature=HOT), 20, 10)  # the arctic: two steps too cold
        hardy = simulation._add_agent(encode_dna("grazer", temperature=HOT, cold_tolerance=True), 40, 10)
        simulation._add_agent(encode_dna("predator"), 90, 90)
        take_turn(simulation)
        self.assertAlmostEqual(tender.hunger, 2 * tender.burn)
        self.assertAlmostEqual(hardy.hunger, 1.5 * hardy.burn)
        self.assertAlmostEqual(tender.health, 0.8 * tender.max_health)
        self.assertAlmostEqual(hardy.health, 0.9 * hardy.max_health)


class EffectTests(unittest.TestCase):
    def test_each_step_from_home_doubles_an_animals_food_and_hurts_a_tenth_more(self):
        simulation = world(turn=200)  # the second quarter: no winter
        at_home = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM), 10, 30)
        one_away = simulation._add_agent(encode_dna("grazer", temperature=COLD), 20, 30)
        two_away = simulation._add_agent(encode_dna("grazer", temperature=VERY_COLD), 30, 30)
        simulation._add_agent(encode_dna("predator"), 90, 90)
        take_turn(simulation)
        self.assertAlmostEqual(at_home.hunger, at_home.burn)
        self.assertAlmostEqual(one_away.hunger, 2 * one_away.burn)
        self.assertAlmostEqual(two_away.hunger, 2 * two_away.burn)
        self.assertEqual(at_home.health, at_home.max_health)
        self.assertAlmostEqual(one_away.health, 0.9 * one_away.max_health)
        self.assertAlmostEqual(two_away.health, 0.8 * two_away.max_health)

    def test_starving_out_of_its_temperature_is_dying_of_exposure(self):
        simulation = world(turn=200)
        away = simulation._add_agent(encode_dna("grazer", temperature=COLD), 20, 50)  # on the equator: too hot
        at_home = simulation._add_agent(encode_dna("grazer", temperature=HOT), 40, 50)
        for animal in (away, at_home):
            animal.hunger = 99.9
        simulation._add_agent(encode_dna("predator"), 90, 90)
        take_turn(simulation)
        self.assertEqual((away.last_action, at_home.last_action), ("overheated", "starved"))

    def test_tolerance_is_rare_in_the_starting_genomes(self):
        simulation = Simulation(width=100, height=100, population=600, seed=5)
        flags = [flag for agent in simulation.agents for flag in agent.climate[1:]]
        self.assertAlmostEqual(sum(flags) / len(flags), 0.1, delta=0.05)

    def test_winter_turns_a_home_zone_hostile_and_the_cold_can_kill(self):
        simulation = world(turn=90)  # turn 91, next, is in the north's winter
        grazer = simulation._add_agent(encode_dna("grazer", temperature=HOT), 50, 10)  # arctic: very cold now
        grazer.health = 5
        simulation._add_agent(encode_dna("predator"), 90, 90)
        take_turn(simulation)
        self.assertFalse(grazer.alive)
        self.assertEqual(grazer.last_action, "froze")

    def test_a_plant_out_of_its_temperature_withers_without_growing_or_seeding_and_two_steps_hurt(self):
        simulation = world(turn=200)
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
        self.assertAlmostEqual(two_away.health, 0.3 * two_away.max_health)  # two steps, a fifth
        self.assertEqual((len(simulation.agents), simulation.seeds), (3, []))

    def test_a_plant_out_of_its_temperature_gives_no_food(self):
        simulation = world(turn=200)
        at_home = simulation._add_agent(encode_dna("plant", temperature=MEDIUM), 50, 30)
        away = simulation._add_agent(encode_dna("plant", temperature=COLD), 60, 30)
        self.assertEqual((simulation._plant_yield(at_home), simulation._plant_yield(away)), (1.0, 0.0))
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

    def test_a_plant_too_hot_needs_twice_the_food_and_too_cold_doesnt(self):
        simulation = world()
        too_hot = simulation._add_agent(encode_dna("plant", temperature=COLD), 20, 30)  # temperate: a step too warm
        too_cold = simulation._add_agent(encode_dna("plant", temperature=HOT), 40, 30)
        for plant in (too_hot, too_cold):
            plant.hunger = 50
        simulation._add_agent(encode_dna("predator"), 90, 90)
        take_turn(simulation)
        self.assertAlmostEqual(too_hot.hunger, 50 + 2 * too_hot.burn - 1.5)
        self.assertAlmostEqual(too_cold.hunger, 50 + too_cold.burn - 1.5)

    def test_the_year_starts_mild(self):
        self.assertEqual([season(turn) for turn in (0, 90, 180, 270)], [0, 1, 2, 3])
        self.assertEqual([temperature_at(y, 100, 0) for y in range(100)], [zone_temperature(y, 100) for y in range(100)])

    def test_grass_grows_only_where_it_is_medium_now(self):
        simulation = world(turn=100)  # the north's winter: its temperate zone is cold, the equator medium
        simulation._sow(3000)
        rows = {y for _, y in simulation.grass}
        self.assertEqual(rows, set(range(40, 80)))


class SenseTests(unittest.TestCase):
    def test_an_animal_senses_how_far_it_is_from_home_and_how_close_home_weather_is(self):
        simulation = world(turn=200)
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

    def test_an_animal_senses_where_it_is_and_how_warm_it_is_here_north_and_south(self):
        simulation = world(turn=0)  # mild: arctic rows 0-19, temperate 20-39, equator 40-59
        grazer = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM, sight=7), 50, 35)  # sees 9 rows
        senses = simulation._sense(grazer)[0]
        sensed = {name: senses[SENSES.index(name)] for name in ("latitude", "warmth", "warmth_north", "warmth_south")}
        self.assertEqual(sensed, {"latitude": 2 * 35.5 / 100 - 1, "warmth": 0.0, "warmth_north": 0.0,
                                  "warmth_south": 0.5})  # row 26 temperate, row 44 the equator
        cold = simulation._add_agent(encode_dna("grazer", temperature=HOT, sight=7), 50, 3)
        senses = simulation._sense(cold)[0]
        self.assertEqual([senses[SENSES.index(name)] for name in ("warmth", "warmth_north", "warmth_south")],
                         [-1.0, -1.0, -1.0])  # two steps too cold, north to the map's edge too

    def test_an_animal_can_head_north_or_south_but_not_off_the_map(self):
        simulation = world(turn=0)
        grazer = simulation._add_agent(encode_dna("grazer", speed=0), 50, 50)  # two rows a step
        options, situation = simulation._sense(grazer)[1:]
        self.assertTrue({ACTIONS.index("go_north"), ACTIONS.index("go_south")} <= set(options))
        simulation._act(grazer, "go_north", situation)
        self.assertEqual((grazer.x, grazer.y, grazer.last_action), (50, 48, "went north"))
        simulation._act(grazer, "go_south", situation)
        self.assertEqual((grazer.x, grazer.y, grazer.last_action), (50, 50, "went south"))
        top = simulation._add_agent(encode_dna("grazer"), 50, 0)
        bottom = simulation._add_agent(encode_dna("grazer", speed=7), 50, 98)
        self.assertNotIn(ACTIONS.index("go_north"), simulation._sense(top)[1])
        options, situation = simulation._sense(bottom)[1:]
        self.assertIn(ACTIONS.index("go_south"), options)
        simulation._act(bottom, "go_south", situation)
        self.assertEqual(bottom.y, 99)  # stopped by the edge
        self.assertNotIn(ACTIONS.index("go_south"), simulation._sense(bottom)[1])

    def test_an_animal_can_always_head_east_or_west_round_the_map_and_enjoys_setting_off(self):
        simulation = world(turn=0)
        grazer = simulation._add_agent(encode_dna("grazer", speed=0, curiosity=7), 1, 50)  # two cells a step
        options, situation = simulation._sense(grazer)[1:]
        self.assertTrue({ACTIONS.index("go_east"), ACTIONS.index("go_west"), ACTIONS.index("rest")} <= set(options))
        self.assertNotIn("explore", ACTIONS)
        simulation._act(grazer, "go_west", situation)
        self.assertEqual((grazer.x, grazer.y, grazer.last_action, grazer.reward), (99, 50, "went west", CURIOSITY_JOY))
        simulation._act(grazer, "go_east", situation)
        self.assertEqual((grazer.x, grazer.y, grazer.last_action), (1, 50, "went east"))

    def test_being_out_of_its_temperature_is_uncomfortable_less_so_for_the_hardy(self):
        simulation = world(turn=0)
        at_home = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM), 50, 30)
        one_step = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM), 52, 10)
        two_steps = simulation._add_agent(encode_dna("grazer", temperature=HOT), 54, 10)
        hardy = simulation._add_agent(encode_dna("grazer", temperature=HOT, cold_tolerance=True), 56, 10)
        exposure = [simulation._sense(agent)[2]["exposure"] for agent in (at_home, one_step, two_steps, hardy)]
        self.assertEqual(exposure, [0.0, 0.5, 1.0, 0.5])
        with patch("tools.Simulation.CLIMATE_UNEASE", 0.4):
            situation = simulation._sense(two_steps)[2]
            cold = simulation._comfort(two_steps, situation)
            situation["exposure"] = 0.0
            self.assertAlmostEqual(simulation._comfort(two_steps, situation) - cold, 0.4)

    def test_foraging_and_herding_never_lead_further_from_home(self):
        simulation = world(turn=0)  # row 40 is the hot equator, rows 20-39 temperate
        grazer = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM, sight=7), 50, 38)
        simulation.grass.update({(50, 41): 1, (50, 33): 1})
        self.assertEqual(simulation._nearest_grass(grazer, grazer.sight), (50, 33))  # past the nearer, hot one
        hot = simulation._add_agent(encode_dna("grazer", temperature=HOT, sight=7), 50, 42)
        self.assertEqual(simulation._nearest_grass(hot, hot.sight), (50, 41))  # it is at home there

        herder = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM, empathy=7, patience=7), 70, 36)
        mates = [simulation._add_agent(herder.dna, 70 + dx, 40) for dx in (-1, 0, 1)]  # on the equator
        simulation._index_cells()
        options, situation = simulation._sense(herder)[1:]
        self.assertEqual(situation["herd_middle"], (70, 40))
        self.assertNotIn(ACTIONS.index("herd"), options)
        for mate in mates:
            simulation._move(mate, mate.x, 32)  # into the temperate zone
        self.assertIn(ACTIONS.index("herd"), simulation._sense(herder)[1])


class TimeTests(unittest.TestCase):
    def test_animals_sense_the_time_of_year_as_two_clock_hands_a_quarter_apart(self):
        simulation = world(turn=0)
        grazer = simulation._add_agent(encode_dna("grazer"), 50, 50)
        hands = []
        for turn in (0, 90, 180, 270, 360):
            simulation.turn = turn
            senses = simulation._sense(grazer)[0]
            hands.append((senses[SENSES.index("time_of_year_a")], senses[SENSES.index("time_of_year_b")]))
        self.assertEqual(hands, [(1.0, 0.5), (0.5, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 0.5)])  # round the year

    def test_the_clock_follows_the_year_length(self):
        simulation = Simulation(width=20, height=20, population=0, seed=1, year_length=100)
        self.assertEqual([tuple(simulation._time_of_year().values()) for simulation.turn in (25, 50)],
                         [(0.5, 0.0), (0.0, 0.5)])
        Simulation(width=20, height=20, population=0, seed=1)  # back to the default year


class SocietyTests(unittest.TestCase):
    def setUp(self):
        self.simulation = world()
        self.simulation._add_agent(encode_dna("predator"), 90, 90)  # far off

    def decided(self, animal):
        """An animal that has made a decision, which a witness can learn from."""
        self.simulation._index_cells()
        senses, options, _ = self.simulation._sense(animal)
        animal.last_senses, animal.last_choice = senses, options[0]
        return animal

    def test_herd_mates_in_sight_learn_from_a_death_and_feel_it_by_empathy_and_intellect(self):
        bright = encode_dna("grazer", empathy=7, intellect=7, sight=3)
        dying = self.decided(self.simulation._add_agent(bright, 50, 50))
        witness = self.simulation._add_agent(bright, 50, 53)  # three cells off: in sight
        dull = self.simulation._add_agent(encode_dna("grazer", empathy=7, intellect=0, sight=3), 51, 50)
        far = self.simulation._add_agent(bright, 50, 59)  # out of sight
        stranger = self.simulation._add_agent(encode_dna("omnivore", empathy=7, intellect=7, sight=3), 49, 50)
        self.simulation._index_cells()
        with patch.object(witness.brain, "learn") as learns:
            dying.health = 0
            self.simulation._die(dying, "froze")
        learns.assert_called_once_with(dying.last_senses, dying.last_choice, -DEATH_PENALTY, LEARNING_RATE)
        self.assertEqual((witness.reward, witness.seen_failure), (SOCIAL_MOOD * -DEATH_PENALTY, 1.0))
        for unmoved in (dull, far, stranger):  # no intellect, out of sight, not of its herd
            self.assertEqual((unmoved.reward, unmoved.seen_failure), (0.0, 0.0))

    def test_old_age_teaches_nothing(self):
        bright = encode_dna("grazer", empathy=7, intellect=7, sight=3)
        old = self.decided(self.simulation._add_agent(bright, 50, 50))
        witness = self.simulation._add_agent(bright, 50, 52)
        self.simulation._index_cells()
        old.health = 0
        self.simulation._die(old, "old age")
        self.assertEqual(witness.seen_failure, 0.0)

    def test_a_pack_learns_from_a_kill_and_enjoys_it(self):
        pack = encode_dna("predator", empathy=7, intellect=7, sight=3, strength=7, weapons=7)
        hunter = self.decided(self.simulation._add_agent(pack, 50, 50))
        mate = self.simulation._add_agent(pack, 52, 50)
        prey = self.simulation._add_agent(encode_dna("grazer"), 50, 51)
        prey.health = 0.1
        self.simulation._index_cells()
        with patch.object(mate.brain, "learn") as learns:
            self.simulation._attack(hunter, prey)
        self.assertFalse(prey.alive)
        gained = (KILL_MEAL + KILL_MEAL_FOR_SIZE * prey.physique("size")) / hunter.stomach / 100
        learns.assert_called_once_with(hunter.last_senses, hunter.last_choice, gained, LEARNING_RATE)
        self.assertEqual(mate.seen_success, 1.0)
        self.assertGreater(mate.reward, SOCIAL_MOOD * gained - 2 * 0.15 * 0.4 - 1e-9)  # less its pity for the prey

    def test_what_it_saw_is_an_input_that_fades(self):
        grazer = self.simulation._add_agent(encode_dna("grazer"), 50, 50)
        grazer.seen_success, grazer.seen_failure = 1.0, 1.0
        senses = self.simulation._sense(grazer)[0]
        self.assertEqual((senses[SENSES.index("herd_success")], senses[SENSES.index("herd_failure")]), (1.0, 1.0))
        take_turn(self.simulation)
        self.assertAlmostEqual(grazer.seen_failure, SOCIAL_MEMORY)


class SeedTests(unittest.TestCase):
    def test_a_seed_that_falls_where_it_cant_grow_waits_for_its_temperature(self):
        simulation = world(turn=200)
        simulation.body_mutation = 0
        plant = simulation._add_agent(encode_dna("plant", temperature=COLD, fertility=7), 50, 30)
        simulation._index_cells()
        with patch.object(simulation.random, "random", return_value=0.0):
            simulation._seed(plant)
        self.assertEqual(len(simulation.agents), 1)
        [(x, y, dna, generation, sown)] = simulation.seeds
        self.assertEqual((generation, sown, plant.children), (1, 200, 1))

        simulation.turn = 460  # next year's winter in the north makes its temperate zone cold
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
        for turn, waiting in ((269, [seed]), (270, [])):
            simulation.turn = turn
            simulation._update_climate()
            simulation._germinate()
            self.assertEqual(simulation.seeds, waiting)

    def test_waiting_seeds_rewind_and_replay_with_the_world(self):
        simulation = Simulation(width=40, height=40, population=60, seed=4)
        while not simulation.seeds and simulation.turn < 300 and not simulation.ended:  # until some seeds wait
            simulation.advance()
        seeds, turn = list(simulation.seeds), simulation.turn
        self.assertTrue(seeds)
        simulation.seek(turn // 2)
        simulation.seek(turn)
        self.assertEqual(simulation.seeds, seeds)


if __name__ == "__main__":
    unittest.main()
