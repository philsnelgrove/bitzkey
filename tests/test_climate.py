import math
import unittest
from unittest.mock import patch

from tools.Brain import ACTIONS, SENSES
from tools.Genome import decode_climate, encode_dna
from tools.Simulation import (COLD, HOMEWARD_UNEASE, PLANT_BITE, CURIOSITY_JOY, DEATH_PENALTY, GO_EFFORT, WALK_REST, GRASS_DIEBACK, dieback_threshold, season_progress, season_temperature, wave_band, HOT, KILL_MEAL, KILL_MEAL_FOR_SIZE, LEARNING_RATE, MEDIUM, PLANT_MEAL,
                              SOCIAL_MEMORY, SOCIAL_MOOD, VERY_COLD, Simulation, season, temperature_at, year,
                              zone_temperature)


from tools.Simulation import EIGHT_TURN_DAYS

def setUpModule():  # grown bodies: a Bitzkey made at age 0 isn't feeble here, unless a test says so
    global _grown
    _grown = patch.multiple("tools.Simulation", FEEBLE_YOUTH=0.0, SMALL_HIDING=0.0, DEFAULT_YEAR_LENGTH=360, SLEEP=0, **EIGHT_TURN_DAYS)  # and 360-turn years, no sleeping, spans in eight-turn days
    _grown.start()


def tearDownModule():
    _grown.stop()


def world(turn: int = 240, size: int = 100) -> Simulation:
    """An empty world at a turn, its grass cleared: by default mid-year, mild, the weather settled."""
    simulation = Simulation(width=size, height=size, population=0, seed=3)
    simulation.grass.clear()
    simulation.turn = turn
    simulation._update_climate()
    return simulation


def take_turn(simulation: Simulation) -> None:
    """One turn in which animals neither think nor die of age."""
    with patch.object(Simulation, "_think_all"), patch.object(Simulation, "_dies_of_age", return_value=False):
        simulation._take_turn()


class ZoneTests(unittest.TestCase):
    @patch("tools.Simulation.POLAR_SHARE", 0.0)  # the zones as they first were: no polar caps,
    @patch("tools.Simulation.WINTER_EQUATOR_SHARE", 0.0)  # and the whole equator turning in either winter
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
        self.assertEqual(animal.lifespan, 300)  # the smallest live three years
        with self.assertRaisesRegex(ValueError, "at least 4"):
            Simulation(width=20, height=20, population=0, seed=1, year_length=3)
        Simulation(width=20, height=20, population=0, seed=1)
        self.assertEqual(year(360), 2)  # back to the default

    @patch("tools.Simulation.POLAR_SHARE", 0.0)  # the zones as they first were: no polar caps,
    @patch("tools.Simulation.WINTER_EQUATOR_SHARE", 0.0)  # and the whole equator turning in either winter
    def test_each_hemisphere_and_the_equator_are_a_step_colder_in_its_winter(self):
        normal = [zone_temperature(y, 100) for y in range(100)]
        north = [temperature_at(y, 100, 135) for y in range(100)]  # second quarter, once the wave has passed
        south = [temperature_at(y, 100, 315) for y in range(100)]  # fourth quarter
        self.assertEqual(north, [VERY_COLD] * 20 + [COLD] * 20 + [MEDIUM] * 20 + normal[60:])
        self.assertEqual(south, normal[:40] + [MEDIUM] * 20 + [COLD] * 20 + [VERY_COLD] * 20)
        for turn in (10, 240, 420):  # first and third quarters, and next year's first: mild
            self.assertEqual([temperature_at(y, 100, turn) for y in range(100)], normal)
        self.assertEqual([temperature_at(y, 100, 495) for y in range(100)], north)  # the next year repeats

    @patch("tools.Simulation.SEASON_TURNING", 0.5)  # the pace the wave first had
    @patch("tools.Simulation.CLIMATE_DAMAGE", 0.01)
    @patch("tools.Simulation.POLAR_SHARE", 0.0)  # the zones as they first were: no polar caps,
    @patch("tools.Simulation.WINTER_EQUATOR_SHARE", 0.0)  # and the whole equator turning in either winter
    def test_every_zones_winter_washes_in_at_once_in_a_wave_and_ebbs_back_the_way_it_came(self):
        # The north's winter covers the arctic (rows 0-19), the temperate zone (20-39) and the equator
        # (40-59), each 10 bands of two with a wave of its own, all setting off together. A wave takes
        # half a season, 45 turns, and has a front four bands wide. 27 turns in, each front has come
        # 0.6 of 14 bands.
        self.assertEqual((season_progress(90), season_progress(135)), (0.0, 1.0))
        self.assertAlmostEqual(season_progress(117), 0.6)
        # each zone's wave sets off at once: the arctic, the temperate zone and the equator each its own
        self.assertEqual((wave_band(0, 100, 1), wave_band(37, 100, 1), wave_band(59, 100, 1)), ((0, 10), (8, 10), (9, 10)))
        north = [temperature_at(y, 100, 117) for y in range(60)]
        for start in (0, 20, 40):  # in every zone alike: winter come behind the front, a staircase, not yet ahead
            for band, turned in enumerate((1, 1, 1, 1, 1, 0.85, 0.6, 0.35, 0.1, 0)):
                for row in (start + 2 * band, start + 2 * band + 1):
                    self.assertAlmostEqual(north[row], season_temperature(row, 100, 0) + turned)
        self.assertEqual([temperature_at(y, 100, 117) for y in range(60, 100)], [MEDIUM] * 20 + [COLD] * 20)  # the south: no change

        ebbing = [temperature_at(y, 100, 162) for y in range(60)]  # 27 turns into the ebb, back from the equator
        self.assertEqual((ebbing[59], ebbing[0]), (HOT, VERY_COLD))
        self.assertAlmostEqual(ebbing[24], COLD - 0.35)  # the temperate zone's eighth band from the equator
        south = [temperature_at(y, 100, 297) for y in range(40, 100)]  # the south's winter, up from the bottom
        self.assertEqual((south[-1], south[0]), (VERY_COLD, HOT))
        self.assertEqual(temperature_at(99, 100, 342), VERY_COLD)  # and ebbing back down to it
        self.assertEqual(temperature_at(41, 100, 342), HOT)
        self.assertEqual(temperature_at(70, 100, 0), MEDIUM)  # the world's first season is settled
        with patch("tools.Simulation.SEASON_TURNING", 0):
            self.assertEqual(temperature_at(50, 100, 90), MEDIUM)  # all at once


class FeelingTests(unittest.TestCase):
    def setUp(self):  # exposure's harm measured at the harsher rate climate began with
        harsh = patch("tools.Simulation.CLIMATE_DAMAGE", 0.1)
        harsh.start()
        self.addCleanup(harsh.stop)

    def test_hardiness_halves_an_animals_exposure_one_way_and_doubles_it_the_other_and_plants_have_none(self):
        simulation = world()
        tender = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM), 1, 1)
        hardy = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM, cold_tolerance=True), 1, 1)
        warm = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM, heat_tolerance=True), 1, 1)
        both = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM, heat_tolerance=True,
                                                cold_tolerance=True), 1, 1)  # the flags cancel out
        plant = simulation._add_agent(encode_dna("plant", temperature=MEDIUM, cold_tolerance=True,
                                                 heat_tolerance=True), 1, 1)
        heres = (HOT, MEDIUM, COLD, VERY_COLD)
        for animal in (tender, hardy, warm, plant):  # every step is felt
            self.assertEqual([animal.climate_steps(here) for here in heres], [1, 0, 1, 2])
        self.assertEqual([tender.exposure_share(here) for here in heres], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual([hardy.exposure_share(here) for here in heres], [2.0, 1.0, 0.5, 0.5])
        self.assertEqual([warm.exposure_share(here) for here in heres], [0.5, 1.0, 2.0, 2.0])
        self.assertEqual(both.climate, (MEDIUM, False, False))
        self.assertEqual([both.exposure_share(here) for here in heres], [1.0, 1.0, 1.0, 1.0])
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

    def test_an_animal_tolerant_of_heat_suffers_twice_the_extra_hunger_and_damage_in_the_cold(self):
        simulation = world()  # mid-year, mild
        tender = simulation._add_agent(encode_dna("grazer", temperature=HOT), 20, 10)  # the arctic: two steps too cold
        frail = simulation._add_agent(encode_dna("grazer", temperature=HOT, heat_tolerance=True), 40, 10)
        simulation._add_agent(encode_dna("predator"), 90, 90)
        take_turn(simulation)
        self.assertAlmostEqual(frail.hunger, 3 * frail.burn)
        self.assertAlmostEqual(tender.health, 0.8 * tender.max_health)
        self.assertAlmostEqual(frail.health, 0.6 * frail.max_health)


class EffectTests(unittest.TestCase):
    def setUp(self):  # exposure's harm measured at the harsher rate climate began with
        harsh = patch("tools.Simulation.CLIMATE_DAMAGE", 0.1)
        harsh.start()
        self.addCleanup(harsh.stop)

    def test_each_step_from_home_doubles_an_animals_food_and_hurts_a_tenth_more(self):
        simulation = world(turn=240)  # the second quarter: no winter
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

    def test_an_animal_endures_the_weather_unharmed_through_its_grace_and_regains_it_at_home(self):
        simulation = world(turn=240)
        frail = simulation._add_agent(encode_dna("grazer", temperature=COLD), 20, 30)
        stoic = simulation._add_agent(encode_dna("grazer", temperature=COLD), 40, 30)
        frail.mind["resilience"], stoic.mind["resilience"] = 0, 7
        frail.weathered = stoic.weathered = 12  # its grace spent, for the least resilient
        simulation._add_agent(encode_dna("predator"), 90, 90)
        with patch("tools.Simulation.WEATHER_GRACE", 1.0):  # a day: 12 turns for the least resilient, 36 for the most
            take_turn(simulation)
            self.assertAlmostEqual(frail.health, 0.9 * frail.max_health)  # past its grace: harmed
            self.assertEqual(stoic.health, stoic.max_health)  # uneasy, but unharmed
            self.assertEqual(stoic.weathered, 13)
            stoic.climate = (MEDIUM, *stoic.climate[1:])  # home again
            take_turn(simulation)
            self.assertEqual(stoic.weathered, 12)  # a turn of grace regained
            hardy = simulation._add_agent(encode_dna("grazer", temperature=COLD, heat_tolerance=True), 60, 30)  # the medium row is hotter than its home
            hardy.mind["resilience"] = 0
            hardy.weathered = 12  # past the grace of the least resilient, but twice that hardy to the heat
            frail.climate, frail.health, frail.weathered = (COLD, False, True), frail.max_health, 12  # hardy to the cold, not the heat it is out in
            take_turn(simulation)
            self.assertEqual(hardy.health, hardy.max_health)
            self.assertLess(frail.health, frail.max_health)

    def test_an_animal_that_chose_to_rest_out_of_its_temperature_takes_the_weathers_harm_twice_over(self):
        simulation = world(turn=240)
        chose = simulation._add_agent(encode_dna("grazer", temperature=COLD), 20, 30)
        made = simulation._add_agent(encode_dna("grazer", temperature=COLD), 40, 30)
        for agent in (chose, made):
            agent.last_senses, agent.last_choice = [0.0] * len(SENSES), ACTIONS.index("rest")
        made.last_forced = True  # asleep, or spent
        simulation._add_agent(encode_dna("predator"), 90, 90)
        take_turn(simulation)
        self.assertAlmostEqual(chose.max_health - chose.health, 2 * (made.max_health - made.health))
        self.assertLess(made.health, made.max_health)

    def test_starving_out_of_its_temperature_is_dying_of_exposure(self):
        simulation = world(turn=240)
        away = simulation._add_agent(encode_dna("grazer", temperature=COLD), 20, 50)  # on the equator: too hot
        at_home = simulation._add_agent(encode_dna("grazer", temperature=HOT), 40, 50)
        for animal in (away, at_home):
            animal.hunger = 99.9
            animal.health = 0.05 * animal.max_health  # wasting, it has a turn left
        simulation._add_agent(encode_dna("predator"), 90, 90)
        take_turn(simulation)
        self.assertEqual((away.last_action, at_home.last_action), ("overheated", "starved"))

    def test_tolerance_is_rare_in_the_starting_genomes(self):
        simulation = Simulation(width=100, height=100, population=600, seed=5)
        flags = [flag for agent in simulation.agents for flag in agent.climate[1:]]
        self.assertAlmostEqual(sum(flags) / len(flags), 0.1, delta=0.05)

    def test_winter_turns_a_home_zone_hostile_and_the_cold_can_kill(self):
        simulation = world(turn=134)  # by turn 135, next, the north's winter has set in
        grazer = simulation._add_agent(encode_dna("grazer", temperature=HOT), 50, 10)  # arctic: very cold now
        grazer.health = 5
        simulation._add_agent(encode_dna("predator"), 90, 90)
        take_turn(simulation)
        self.assertFalse(grazer.alive)
        self.assertEqual(grazer.last_action, "froze")

    def test_a_plant_out_of_its_temperature_withers_without_growing_or_seeding_and_two_steps_hurt(self):
        simulation = world(turn=240)
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
        simulation = world(turn=240)
        at_home = simulation._add_agent(encode_dna("plant", temperature=MEDIUM), 50, 30)
        away = simulation._add_agent(encode_dna("plant", temperature=COLD), 60, 30)
        self.assertEqual((simulation._plant_yield(at_home), simulation._plant_yield(away)), (1.0, 0.0))
        for plant in (at_home, away):
            grazer = simulation._add_agent(encode_dna("grazer", size=7), plant.x, plant.y + 1)  # a big stomach
            grazer.hunger = 90
            simulation._browse(grazer, plant)
            meal = PLANT_MEAL * grazer.browsing * simulation._plant_yield(plant) / grazer.stomach
            self.assertAlmostEqual(grazer.hunger, 90 - meal)

    def test_a_plant_is_eaten_in_portions_of_the_same_meal_more_of_them_the_larger_it_is(self):
        simulation = world(turn=240)
        small = simulation._add_agent(encode_dna("plant", temperature=MEDIUM, size=0), 50, 30)
        large = simulation._add_agent(encode_dna("plant", temperature=MEDIUM, size=7), 60, 30)
        for plant in (small, large):
            grazer = simulation._add_agent(encode_dna("grazer", size=7), plant.x, plant.y + 1)
            grazer.hunger = 90
            simulation._browse(grazer, plant)
            self.assertAlmostEqual((90 - grazer.hunger) * grazer.stomach / grazer.browsing, PLANT_MEAL)  # the same bite
        self.assertEqual((math.ceil(small.max_health / PLANT_BITE), math.ceil(large.max_health / PLANT_BITE)), (4, 10))

    def test_spawned_plants_are_at_home_in_their_zone_and_imports_keep_theirs(self):
        simulation = Simulation(width=60, height=60, population=300, seed=4)
        plants = [agent for agent in simulation.agents if agent.species == "plant"]
        self.assertTrue(plants)
        for plant in plants:  # the group's centre is within two rows
            self.assertIn(plant.climate[0], {zone_temperature(plant.y + dy, 60) for dy in range(-2, 3)
                                             if 0 <= plant.y + dy < 60})
        self.assertEqual(len({plant.climate[0] for plant in plants}), 4)  # hot, medium and cold zones, and the polar caps' biomes very cold
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

    @patch("tools.Simulation.POLAR_SHARE", 0.0)  # the zones as they first were: no polar caps,
    @patch("tools.Simulation.WINTER_EQUATOR_SHARE", 0.0)  # and the whole equator turning in either winter
    def test_grass_grows_only_where_it_is_medium_now(self):
        simulation = world(turn=135)  # the north's winter: its temperate zone is cold, the equator medium
        simulation._sow(3000)
        rows = {y for _, y in simulation.grass}
        self.assertEqual(rows, set(range(40, 80)))

    @patch("tools.Simulation.SEASON_TURNING", 0.5)  # the pace the wave first had
    def test_while_the_weather_turns_its_effects_turn_with_it(self):
        simulation = world(turn=116)  # at turn 117, next, the temperate zone's front has turned row 32 0.6 of the way to cold
        grazer = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM), 50, 32)
        simulation._add_agent(encode_dna("predator"), 90, 90)
        take_turn(simulation)
        self.assertAlmostEqual(grazer.hunger, 1.6 * grazer.burn)  # 0.6 of the extra food
        self.assertAlmostEqual(grazer.health, 0.94 * grazer.max_health)  # 0.6 of a step's harm
        plant = simulation._add_agent(encode_dna("plant", temperature=MEDIUM), 60, 32)
        self.assertAlmostEqual(simulation._plant_yield(plant), 0.4)

    @patch("tools.Simulation.SEASON_TURNING", 0.5)  # the pace the wave first had
    def test_grass_grows_and_dies_back_by_how_medium_a_turning_row_is(self):
        simulation = world(turn=117)  # rows 32 and 33 are 0.6 of the way to cold; the south is medium
        simulation._sow(20000)
        turning = sum(tufts for (_, y), tufts in simulation.grass.items() if y in (32, 33))
        south = sum(tufts for (_, y), tufts in simulation.grass.items() if y in (70, 71))
        self.assertAlmostEqual(turning / south, 0.4, delta=0.08)
        self.assertAlmostEqual(simulation._growth[32], 0.4)
        simulation.grass = {(x, y): 3 for x in range(100) for y in (32, 70)}
        for _ in range(EIGHT_TURN_DAYS["GRASS_DIEBACK"]):
            simulation._wither_grass()
            simulation.turn += 1
        withered = {x for x in range(100) if simulation.grass[(x, 32)] < 3}
        self.assertEqual(withered, {x for x in range(100) if dieback_threshold(x, 32) < 0.6})
        self.assertTrue(all(simulation.grass[(x, 70)] == 3 for x in range(100)))

    def test_the_zones_colour_turns_with_the_weather(self):
        from display.Look import zone_colour
        halfway = zone_colour(MEDIUM + 0.5)
        for channel, low, high in zip(halfway, zone_colour(MEDIUM), zone_colour(COLD)):
            self.assertLessEqual(abs(channel - (low + high) / 2), 1)
        self.assertEqual(zone_colour(float(VERY_COLD)), zone_colour(VERY_COLD))


class SenseTests(unittest.TestCase):
    def test_an_animal_senses_how_far_it_is_from_home_and_how_close_home_weather_is(self):
        simulation = world(turn=240)
        grazer = simulation._add_agent(encode_dna("grazer", temperature=COLD, sight=2), 50, 23)  # sees 8 rows
        self.assertEqual(simulation._comfort_row(grazer, grazer.sight), (50, 19))  # the arctic, four rows north
        senses = simulation._sense(grazer)[0]
        self.assertEqual(senses[SENSES.index("climate_discomfort")], 0.5)  # temperate: a step too warm
        self.assertAlmostEqual(senses[SENSES.index("comfort_closeness")], 1 - 3 / 8)

        at_home = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM), 50, 30)
        senses = simulation._sense(at_home)[0]
        self.assertEqual((senses[SENSES.index("climate_discomfort")], senses[SENSES.index("comfort_closeness")]),
                         (0.0, 1.0))

        lost = simulation._add_agent(encode_dna("grazer", temperature=VERY_COLD, sight=0), 50, 50)  # none in sight
        senses = simulation._sense(lost)[0]
        self.assertEqual((senses[SENSES.index("climate_discomfort")], senses[SENSES.index("comfort_closeness")]),
                         (1.0, 0.0))
        self.assertNotIn("migrate", ACTIONS)  # moving toward home is left for brains to learn

    @patch("tools.Simulation.POLAR_SHARE", 0.0)  # very cold nowhere this season
    def test_an_animal_senses_which_way_home_lies_and_how_far_beyond_its_sight_and_minds_the_distance(self):
        simulation = world(turn=240)  # mild: arctic rows 0 to 19 and 80 to 99, the equator 40 to 59
        grazer = simulation._add_agent(encode_dna("grazer", temperature=COLD, sight=0), 50, 35)
        senses, _, situation = simulation._sense(grazer)
        self.assertEqual((senses[SENSES.index("comfort_closeness")], senses[SENSES.index("home_direction")]), (0.0, -1.0))
        self.assertAlmostEqual(senses[SENSES.index("home_distance")], 16 / 50)  # out of sight, to the north
        far = simulation._comfort(grazer, situation)

        nearer = simulation._add_agent(encode_dna("grazer", temperature=COLD, sight=0), 50, 70)
        senses, _, situation = simulation._sense(nearer)
        self.assertEqual(senses[SENSES.index("home_direction")], 1.0)
        self.assertAlmostEqual(senses[SENSES.index("home_distance")], 10 / 50)
        self.assertAlmostEqual(simulation._comfort(nearer, situation) - far, HOMEWARD_UNEASE * 6 / 50)  # each row eases it

        at_home = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM), 50, 30)
        senses = simulation._sense(at_home)[0]
        self.assertEqual((senses[SENSES.index("home_direction")], senses[SENSES.index("home_distance")]), (0.0, 0.0))

        lost = simulation._add_agent(encode_dna("grazer", temperature=VERY_COLD), 50, 50)
        senses = simulation._sense(lost)[0]
        self.assertEqual((senses[SENSES.index("home_direction")], senses[SENSES.index("home_distance")]), (0.0, 1.0))

    def test_the_polar_caps_are_very_cold_all_year_and_the_equators_heart_stays_hot_through_both_winters(self):
        for quarter in range(4):
            for y in (0, 1, 98, 99):
                self.assertEqual(season_temperature(y, 100, quarter), VERY_COLD)
            self.assertEqual(season_temperature(2, 100, quarter), VERY_COLD if quarter == 1 else COLD)  # the arctic turns
            self.assertEqual(season_temperature(97, 100, quarter), VERY_COLD if quarter == 3 else COLD)
            for y in (48, 49, 50, 51):
                self.assertEqual(season_temperature(y, 100, quarter), HOT)
            self.assertEqual(season_temperature(45, 100, quarter), MEDIUM if quarter % 2 else HOT)  # the rest turns

    @patch("tools.Simulation.SEASON_TURNING", 0.5)  # the pace the wave first had
    @patch("tools.Simulation.CLIMATE_DAMAGE", 0.01)
    def test_an_animal_sees_the_wave_coming_as_a_staircase_of_warmth(self):
        simulation = world(turn=117)  # the temperate zone's winter front at rows 30 to 37
        grazer = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM, sight=2), 50, 32)  # sees 8 rows
        senses = simulation._sense(grazer)[0]
        warmth = [senses[SENSES.index(name)] for name in ("warmth_north", "warmth", "warmth_south")]
        for sensed, expected in zip(warmth, (-0.5, -0.3, 0.0)):  # cold behind it, part way here, the equator's turned mild
            self.assertAlmostEqual(sensed, expected)
        self.assertAlmostEqual(senses[SENSES.index("climate_discomfort")], 0.3)

    def test_an_animal_senses_where_it_is_and_how_warm_it_is_here_north_and_south(self):
        simulation = world(turn=0)  # mild: arctic rows 0-19, temperate 20-39, equator 40-59
        grazer = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM, sight=2), 50, 35)  # sees 8 rows
        senses = simulation._sense(grazer)[0]
        sensed = {name: senses[SENSES.index(name)] for name in ("latitude", "warmth", "warmth_north", "warmth_south")}
        self.assertEqual(sensed, {"latitude": 2 * 35.5 / 100 - 1, "warmth": 0.0, "warmth_north": 0.0,
                                  "warmth_south": 0.5})  # row 27 temperate, row 43 the equator
        cold = simulation._add_agent(encode_dna("grazer", temperature=HOT, sight=2), 50, 3)
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

    def test_a_walk_moves_one_cell_for_half_a_turns_food_and_going_further_costs_more(self):
        simulation = world(turn=0)
        grazer = simulation._add_agent(encode_dna("grazer", speed=7, curiosity=7), 50, 50)  # six cells a step
        options, situation = simulation._sense(grazer)[1:]
        walks = {ACTIONS.index(name) for name in ("walk_north", "walk_south", "walk_east", "walk_west")}
        self.assertTrue(walks <= set(options))
        grazer.hunger = 50
        simulation._act(grazer, "walk_north", situation)
        self.assertEqual((grazer.x, grazer.y, grazer.last_action), (50, 49, "walked north"))
        self.assertAlmostEqual(grazer.hunger, 50 - grazer.burn * WALK_REST)  # saves half the turn's burn, as resting does
        self.assertAlmostEqual(grazer.reward, CURIOSITY_JOY)  # walking explores too, but eases no hunger
        grazer.hunger = 50
        simulation._act(grazer, "go_south", situation)
        self.assertEqual(grazer.y, 55)
        self.assertAlmostEqual(grazer.hunger, 50 + grazer.burn * GO_EFFORT * 5)  # a quarter of a turn's burn for each cell past one
        simulation._act(grazer, "walk_west", situation)
        self.assertEqual((grazer.x, grazer.last_action), (49, "walked west"))
        edge = simulation._add_agent(encode_dna("grazer", speed=7), 50, 97)
        edge.hunger = 50
        simulation._act(edge, "go_south", simulation._sense(edge)[2])
        self.assertEqual(edge.y, 99)
        self.assertAlmostEqual(edge.hunger, 50 + edge.burn * GO_EFFORT * 1)  # only the two cells it could go
        self.assertNotIn(ACTIONS.index("walk_south"), simulation._sense(edge)[1])

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
        grazer = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM, sight=2), 50, 38)
        simulation.grass.update({(50, 41): 1, (50, 33): 1})
        self.assertEqual(simulation._nearest_grass(grazer, grazer.sight), (50, 33))  # past the nearer, hot one
        hot = simulation._add_agent(encode_dna("grazer", temperature=HOT, sight=2), 50, 42)
        self.assertEqual(simulation._nearest_grass(hot, hot.sight), (50, 41))  # it is at home there

        herder = simulation._add_agent(encode_dna("grazer", temperature=MEDIUM, empathy=7, patience=7, sight=1), 70, 36)  # sees 5 cells
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

    @patch("tools.Simulation.TRUST_HEED", 0.0)  # trust aside
    def test_herd_mates_in_sight_learn_from_a_death_and_feel_it_by_empathy_and_intellect(self):
        bright = encode_dna("grazer", empathy=7, intellect=7, sight=1)  # sees 5 cells
        dying = self.decided(self.simulation._add_agent(bright, 50, 50))
        witness = self.simulation._add_agent(bright, 50, 53)  # three cells off: in sight
        dull = self.simulation._add_agent(encode_dna("grazer", empathy=7, intellect=0, sight=1), 51, 50)
        far = self.simulation._add_agent(bright, 50, 59)  # out of sight
        stranger = self.simulation._add_agent(encode_dna("omnivore", empathy=7, intellect=7, sight=1), 49, 50)
        self.simulation._index_cells()
        before = witness.brain.copy()
        dying.health = 0
        self.simulation._die(dying, "exhausted")  # a sudden death, its last choice's doing
        self.assertEqual(witness.brain, before)  # what the herd saw is learnt at the turn's end, with the rest
        self.assertEqual([(observer, choice, outcome, rate) for observer, _, choice, outcome, rate in self.simulation._witnessed
                          if observer is witness], [(witness, dying.last_choice, -DEATH_PENALTY, LEARNING_RATE)])
        self.simulation._learn_witnessed()
        self.assertNotEqual(witness.brain, before)
        self.assertEqual((witness.reward, witness.seen_failure), (SOCIAL_MOOD * -DEATH_PENALTY, 1.0))
        self.assertEqual(dull.reward, 0.0)  # with no intellect, it sees it but takes nothing in
        for unmoved in (far, stranger):  # out of sight, not of its herd
            self.assertEqual((unmoved.reward, unmoved.seen_failure), (0.0, 0.0))

    def test_an_action_finds_the_world_as_those_before_it_left_it_and_a_stale_plan_fails_softly(self):
        simulation = world(turn=240)
        grazer = simulation._add_agent(encode_dna("grazer"), 50, 50)
        prey = simulation._add_agent(encode_dna("grazer"), 50, 51)
        predator = simulation._add_agent(encode_dna("predator"), 51, 50)
        simulation._index_cells()
        simulation._act(grazer, "eat", {})  # its grass was eaten before it got to it
        self.assertEqual(grazer.last_action, "found it eaten")
        prey.health = 0
        simulation._die(prey, "killed")
        simulation._act(predator, "attack", {"prey": prey})  # it fell to another first
        self.assertEqual(predator.last_action, "missed")
        mate = simulation._add_agent(encode_dna("grazer"), 49, 50)
        simulation._index_cells()
        mate.cooldown = 5  # it bred with another first
        simulation._act(grazer, "breed", {"mate": mate})
        self.assertEqual((grazer.last_action, grazer.womb), ("was spurned", []))

    def test_old_age_teaches_nothing(self):
        bright = encode_dna("grazer", empathy=7, intellect=7, sight=3)
        old = self.decided(self.simulation._add_agent(bright, 50, 50))
        witness = self.simulation._add_agent(bright, 50, 52)
        self.simulation._index_cells()
        old.health = 0
        self.simulation._die(old, "old age")
        self.assertEqual(witness.seen_failure, 0.0)

    @patch("tools.Simulation.TRUST_HEED", 0.0)  # trust aside
    def test_a_pack_learns_from_a_kill_and_enjoys_it(self):
        pack = encode_dna("predator", empathy=7, intellect=7, sight=3, strength=7, weapons=7)
        hunter = self.decided(self.simulation._add_agent(pack, 50, 50))
        mate = self.simulation._add_agent(pack, 52, 50)
        prey = self.simulation._add_agent(encode_dna("grazer"), 50, 51)
        prey.health = 0.1
        self.simulation._index_cells()
        before = mate.brain.copy()
        self.simulation._attack(hunter, prey)
        self.assertFalse(prey.alive)
        gained = (KILL_MEAL + KILL_MEAL_FOR_SIZE * prey.physique("size")) / hunter.stomach / 100
        self.assertEqual([(observer, choice, outcome, rate) for observer, _, choice, outcome, rate in self.simulation._witnessed],
                         [(mate, hunter.last_choice, gained, LEARNING_RATE)])
        self.simulation._learn_witnessed()  # at the turn's end
        self.assertNotEqual(mate.brain, before)
        self.assertEqual(mate.seen_success, 1.0)
        self.assertGreater(mate.reward, SOCIAL_MOOD * gained - 2 * 0.15 * 0.4 - 1e-9)  # less its pity for the prey

    @patch("tools.Simulation.RESILIENCE", 0.0)  # resilience aside
    def test_what_it_saw_is_an_input_that_fades(self):
        grazer = self.simulation._add_agent(encode_dna("grazer"), 50, 50)
        grazer.seen_success, grazer.seen_failure = 1.0, 1.0
        senses = self.simulation._sense(grazer)[0]
        self.assertEqual((senses[SENSES.index("herd_success")], senses[SENSES.index("herd_failure")]), (1.0, 1.0))
        take_turn(self.simulation)
        self.assertAlmostEqual(grazer.seen_failure, EIGHT_TURN_DAYS["SOCIAL_MEMORY"])


class HeatTests(unittest.TestCase):
    def test_an_animal_is_in_heat_only_before_its_winter_the_patient_further_from_it_and_never_through_it(self):
        def in_heat(simulation, y, patience):
            grazer = simulation._add_agent(encode_dna("grazer"), 50, y)
            grazer.mind["patience"] = patience
            return simulation.heat(grazer)
        # The north's winter comes at day 90. The least patient are in heat from 216 turns before it to 36 (days 234
        # to 53), the most patient from 270 turns before it to 90 (days 180 to 359), further from it.
        late, early, both, winter = world(turn=20), world(turn=200), world(turn=300), world(turn=120)
        self.assertTrue(in_heat(late, 30, 0) and not in_heat(late, 30, 7))
        self.assertTrue(not in_heat(early, 30, 0) and in_heat(early, 30, 7))
        self.assertTrue(in_heat(both, 30, 0) and in_heat(both, 30, 7))
        self.assertFalse(in_heat(winter, 30, 0) or in_heat(winter, 30, 7))  # never through its own winter
        self.assertTrue(in_heat(winter, 70, 0))  # the south's winter is half a year off
        self.assertTrue(in_heat(winter, 0, 0))   # the polar cap knows no winter


class SeedTests(unittest.TestCase):
    def test_a_seed_that_falls_where_it_cant_grow_waits_for_its_temperature(self):
        simulation = world(turn=240)
        simulation.body_mutation = 0
        plant = simulation._add_agent(encode_dna("plant", temperature=COLD, fertility=7), 50, 30)
        simulation._index_cells()
        with patch.object(simulation.random, "random", return_value=0.0):
            simulation._seed(plant)
        self.assertEqual(len(simulation.agents), 1)
        [(x, y, dna, generation, sown)] = simulation.seeds
        self.assertEqual((generation, sown, plant.children), (1, 240, 1))

        simulation.turn = 495  # next year's winter in the north, settled, makes its temperate zone cold
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

if __name__ == "__main__":
    unittest.main()
