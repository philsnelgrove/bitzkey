import math
import random
import unittest
from unittest.mock import patch
import tools.Simulation as simulation_module

from tools.Brain import ACTIONS, SENSES, Brain
from tools.Exchange import Imported
from tools.Genome import decode_body, encode_dna
from tools.Simulation import (Death, DEATH_PENALTY, AGENT_BYTES, ATTACK_JOY, FEAST_JOY, FRESH_KILL_MEALS, KILL_MEAL, KILL_MEAL_FOR_SIZE, BIRTH_REACH, PLANT_GESTATION, Bitzkey, gestation, CORPSE_TURNS, GRASS_DIEBACK, HOT, MEDIUM, PLANT_MEAL, PLANT_MEAL_FOR_SIZE, SEED_OFFSETS,
                              SIGHT_OFFSETS, SPAWN_SPREAD, UPROOT_BITE, UPROOT_MEALS, Simulation, litter_size,
                              nearby_offsets)


def setUpModule():  # grown bodies: a Bitzkey made at age 0 isn't feeble here, unless a test says so
    global _grown, _common, _grazers, _plenty
    _grown = patch("tools.Simulation.FEEBLE_YOUTH", 0.0)
    _grown.start()
    _common = patch("tools.Simulation.RARE_BREEDING_COST", 1.0)  # the few Bitzkeys of a test breed at the usual cost
    _common.start()
    _grazers = patch("tools.Simulation.GRAZER_RARE_BREEDING_COST", 1.0)  # grazers too
    _grazers.start()
    _plenty = [patch("tools.Simulation.RARE_PREY", 0),  # a test's lone hunters may breed with no prey alive
               patch("tools.Simulation.FULL_LITTER_FOOD", 2.0),  # and full litters at twice the need, as these figures assume
               patch("tools.Simulation.PREDATOR_FULL_LITTER_FOOD", 2.0)]
    for each in _plenty:
        each.start()


def tearDownModule():
    _grown.stop()
    _common.stop()
    _grazers.stop()
    for each in _plenty:
        each.stop()


def deliver(simulation: Simulation, first: Bitzkey, second: Bitzkey, litter: int | None = 1) -> None:
    """Mate two animals and, rather than wait out the gestation, give birth at once: to a litter of
    `litter` young, or of the size breeding sets with None."""
    if litter is None:
        simulation._breed(first, second)
    else:
        with patch("tools.Simulation.litter_size", return_value=litter):
            simulation._breed(first, second)
    simulation._give_birth(first)


def lay_plenty_of_grass(simulation: Simulation) -> None:
    """Grass on every cell, so animals judge food plentiful enough to breed."""
    simulation.grass.update({(x, y): 3 for x in range(simulation.width) for y in range(simulation.height)})


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.simulation = Simulation(width=10, height=10, population=0, seed=3)
        self.simulation.grass.clear()

    def test_grass_density_sets_starting_grass_and_regrowth(self):
        sparse = Simulation(width=20, height=20, population=0, seed=1, grass_density=5)
        lush = Simulation(width=20, height=20, population=0, seed=1, grass_density=50)

        self.assertEqual(sum(sparse.grass.values()), 20)
        self.assertEqual(sum(lush.grass.values()), 200)
        self.assertAlmostEqual(lush.grass_regrowth, 10 * sparse.grass_regrowth)
        with patch("tools.Simulation.GRASS_DIEBACK", 10 ** 9):  # no grass dies back
            for _ in range(100):
                lush._take_turn()
        self.assertAlmostEqual(sum(lush.grass.values()), 200 + 100 * lush.grass_regrowth, delta=8)

    def test_random_population_spawns_in_clustered_groups_of_three_to_five(self):
        for population in (3, 7, 90, 101):
            simulation = Simulation(width=50, height=50, population=population, seed=population)
            groups = {}
            for agent in simulation.agents:
                groups.setdefault(agent.dna, []).append(agent)

            self.assertEqual(len(simulation.agents), population)
            for members in groups.values():
                self.assertIn(len(members), (3, 4, 5))
                self.assertLessEqual(max(agent.x for agent in members) - min(agent.x for agent in members), 4)
                self.assertLessEqual(max(agent.y for agent in members) - min(agent.y for agent in members), 4)

    def test_a_population_too_small_for_a_group_spawns_together(self):
        simulation = Simulation(width=20, height=20, population=2, seed=1)

        self.assertEqual(len({agent.dna for agent in simulation.agents}), 1)
        self.assertEqual(len(simulation.agents), 2)

    def test_each_imported_genome_spawns_as_a_clustered_group(self):
        genomes = [encode_dna("predator", size=4), encode_dna("grazer", fear=4)]

        simulation = Simulation(width=20, height=20, population=10, seed=1,
                                imported=[Imported(dna) for dna in genomes])

        imported = simulation.agents[10:]
        for dna in genomes:
            members = [agent for agent in imported if agent.dna == dna]
            self.assertIn(len(members), (3, 4, 5))
            self.assertLessEqual(max(agent.x for agent in members) - min(agent.x for agent in members), 4)
            self.assertTrue(all(agent.brain is not None for agent in members))
        self.assertEqual(len(imported), sum(agent.dna in genomes for agent in imported))

    def test_imported_brains_are_copied_to_every_member_of_the_group(self):
        trained = Brain.random(random.Random(5))

        simulation = Simulation(width=20, height=20, population=0, seed=1,
                                imported=[Imported(encode_dna("predator"), trained)])

        brains = [agent.brain for agent in simulation.agents]
        self.assertTrue(all(brain == trained for brain in brains))
        self.assertEqual(len({id(brain) for brain in brains + [trained]}), len(brains) + 1)

    def test_imported_temperament_is_given_to_every_member_of_the_group(self):
        saved = {"fear": 6.5, "curiosity": 1.25}

        simulation = Simulation(width=20, height=20, population=0, seed=1,
                                imported=[Imported(encode_dna("grazer", fear=2, patience=7), mind=saved)])

        for agent in simulation.agents:
            self.assertEqual((agent.mind["fear"], agent.mind["curiosity"]), (6.5, 1.25))
            self.assertEqual(agent.mind["patience"], 7.0)
            self.assertEqual(agent.inborn["fear"], 2)
        self.assertEqual(len({id(agent.mind) for agent in simulation.agents}), len(simulation.agents))

    def test_animals_get_brains_and_plants_do_not(self):
        plant = self.simulation._add_agent(encode_dna("plant"), 1, 1)
        grazer = self.simulation._add_agent(encode_dna("grazer"), 2, 2)

        self.assertIsNone(plant.brain)
        self.assertIsNotNone(grazer.brain)

    def test_thinking_never_changes_genome(self):
        dna = encode_dna("grazer", intellect=7, libido=7)
        agents = [self.simulation._add_agent(dna, 5, 5), self.simulation._add_agent(dna, 5, 6)]

        for _ in range(10):
            self.simulation._take_turn()

        self.assertTrue(all(agent.dna == dna for agent in agents))

    def test_body_sets_health_pool_and_energy_burn(self):
        small = self.simulation._add_agent(encode_dna("grazer"), 1, 1)
        large = self.simulation._add_agent(encode_dna("grazer", size=7, strength=7, speed=7), 2, 2)

        self.assertEqual((small.max_health, small.health), (60, 60))
        self.assertEqual((large.max_health, large.health), (150, 150))
        self.assertGreater(large.burn * large.stomach, small.burn * small.stomach)  # eats more a turn
        self.assertEqual((small.steps, large.steps), (2, 6))  # grazers: 1 and 4, half as far again
        predator = self.simulation._add_agent(encode_dna("predator", speed=7), 3, 3)
        self.assertEqual(predator.steps, 4)

    def test_mind_starts_inborn_and_copies_independently(self):
        agent = self.simulation._add_agent(encode_dna("grazer", fear=3), 1, 1)
        duplicate = agent.copy()

        agent.mind["fear"] = 6.0

        self.assertEqual(duplicate.mind["fear"], 3.0)

    def test_experience_reshapes_temperament_more_in_clever_minds(self):
        dull = self.simulation._add_agent(encode_dna("grazer", fear=2), 1, 1)
        bright = self.simulation._add_agent(encode_dna("grazer", fear=2, intellect=7), 2, 2)

        for agent in (dull, bright):
            self.simulation._experience(agent, "wounded")

        self.assertGreater(dull.mind["fear"], 2)
        self.assertGreater(bright.mind["fear"], dull.mind["fear"])
        self.assertEqual(bright.inborn["fear"], 2)

    def test_temperament_drifts_back_toward_inborn(self):
        agent = self.simulation._add_agent(encode_dna("grazer", fear=2), 1, 1)
        agent.mind["fear"] = 7.0

        for _ in range(100):
            self.simulation._settle(agent)

        self.assertLess(agent.mind["fear"], 7.0)
        self.assertGreater(agent.mind["fear"], 2.0)

    @patch("tools.Simulation.PLANTS_PER_GRAZER", 0.0)  # no plants about: breeding left to the rest
    def test_breeding_needs_a_ready_mate_and_room(self):
        lay_plenty_of_grass(self.simulation)
        dna = encode_dna("grazer")
        agent = self.simulation._add_agent(dna, 5, 5)
        mate = self.simulation._add_agent(dna, 5, 6)
        breed = ACTIONS.index("breed")
        self.simulation._index_cells()

        self.assertIn(breed, self.simulation._sense(agent)[1])
        mate.cooldown = 3
        self.assertNotIn(breed, self.simulation._sense(agent)[1])
        mate.cooldown = 0
        for x in (4, 5, 6):  # the largest plant in every square around, its own and the mate's among them
            for y in (4, 5, 6):
                self.simulation._add_agent(encode_dna("plant", size=7), x, y)
        self.assertIn(breed, self.simulation._sense(agent)[1])  # room still: each takes 8 of a square's 10
        with patch("tools.Simulation.BIRTH_REACH", 1), patch("tools.Simulation.CELL_SPACE", 8):
            self.assertNotIn(breed, self.simulation._sense(agent)[1])  # every square near full
        agent.womb = [(dna, {}, None, 0.0, 1)]
        self.assertNotIn(breed, self.simulation._sense(agent)[1])  # already carrying young

    def test_animals_shun_mates_whose_ring_or_dot_differ_too_far_from_their_own(self):
        dna = encode_dna("grazer", size=2, fear=3)
        animal = self.simulation._add_agent(dna, 1, 1)
        close = self.simulation._add_agent(dna ^ (0x08 << 56) ^ 0x10, 2, 2)  # ring 0.8%, dot 1.6% apart
        ringed = self.simulation._add_agent(dna ^ (0x40 << 56), 3, 3)        # ring 6.3% apart
        dotted = self.simulation._add_agent(dna ^ 0xFF, 4, 4)                # dot 25% apart
        other_species = self.simulation._add_agent(encode_dna("omnivore", size=2, fear=3), 5, 5)

        self.assertTrue(self.simulation._compatible(animal, close))
        self.assertFalse(self.simulation._compatible(animal, ringed))
        self.assertFalse(self.simulation._compatible(animal, dotted))
        self.assertFalse(self.simulation._compatible(animal, other_species))

    def test_tolerance_widens_the_mates_an_animal_accepts_and_both_must_accept(self):
        dna = encode_dna("grazer", size=2)
        animal = self.simulation._add_agent(dna, 1, 1)
        strange = self.simulation._add_agent(dna ^ (0xFF << 56) ^ 0xFF, 2, 2)  # ring and dot 25% apart
        self.assertFalse(self.simulation._willing(animal, strange))
        animal.mind["tolerance"] = 7.0  # accepts any ring or dot
        self.assertTrue(self.simulation._willing(animal, strange))
        self.assertFalse(self.simulation._compatible(animal, strange))  # but the stranger shuns it
        strange.mind["tolerance"] = 2.0  # ring limit 32%, dot limit 43%
        self.assertTrue(self.simulation._compatible(animal, strange))
        animal.mind["tolerance"] = 0.0
        self.assertFalse(self.simulation._compatible(animal, strange))

    def test_predators_cannot_eat_grass(self):
        predator = self.simulation._add_agent(encode_dna("predator", sight=7), 5, 5)
        grazer = self.simulation._add_agent(encode_dna("grazer", sight=7), 2, 2)
        self.simulation.grass.update({(5, 5): 1, (2, 2): 1, (3, 3): 1, (7, 7): 1})
        self.simulation._index_cells()
        grass_actions = {ACTIONS.index("eat"), ACTIONS.index("forage")}

        self.assertFalse(grass_actions & set(self.simulation._sense(predator)[1]))
        self.assertEqual(grass_actions & set(self.simulation._sense(grazer)[1]), grass_actions)

    @patch("tools.Simulation.PREY_PER_PREDATOR", 0.0)  # the prey at home aside
    def test_breeding_is_the_greatest_pleasure_and_courting_into_reach_the_next_and_failing_costs_nothing(self):
        from tools.Simulation import BREED_JOY, COURT_JOY, FEAST_JOY, LIBIDO_IN_JOY, mating_joy
        cold = self.simulation._add_agent(encode_dna("predator", sight=7, libido=0), 2, 2)
        self.assertAlmostEqual(mating_joy(cold, BREED_JOY), 3.0)  # three quarters of it even with no libido
        self.assertAlmostEqual(mating_joy(cold, COURT_JOY), 1.875)
        self.assertGreater(mating_joy(cold, COURT_JOY), FEAST_JOY)  # the strongest pleasure besides
        self.assertEqual((BREED_JOY, COURT_JOY, LIBIDO_IN_JOY), (4.0, 2.5, 0.25))

        mate = self.simulation._add_agent(cold.dna, 2, 8)  # six cells off: a predator closes two a step
        self.simulation._index_cells()
        situation = self.simulation._sense(cold)[2]
        self.simulation._act(cold, "court", situation)
        self.assertEqual(cold.reward, 0.0)  # not yet in reach: nothing, and nothing lost
        situation = self.simulation._sense(cold)[2]
        cold.reward = 0.0
        while abs(cold.y - mate.y) > 2:  # on until in reach
            self.simulation._act(cold, "court", situation)
        self.assertAlmostEqual(cold.reward, 1.875)

        cold.reward = mate.reward = 0.0
        mate.cooldown = 3  # it bred with another first
        self.simulation._act(cold, "breed", {"mate": mate})
        self.assertEqual((cold.last_action, cold.reward), ("was spurned", 0.0))
        mate.cooldown = 0
        with patch("tools.Simulation.litter_size", return_value=1):
            self.simulation._breed(cold, mate)
        self.assertAlmostEqual(cold.reward, 3.0)
        self.assertAlmostEqual(mate.reward, 3.0)  # both parents

    @patch("tools.Simulation.PLANTS_PER_GRAZER", 0)  # the plants at home aside
    def test_ready_animals_court_a_willing_mate_they_can_see(self):
        dna = encode_dna("grazer", sight=7)
        suitor = self.simulation._add_agent(dna, 2, 2)
        mate = self.simulation._add_agent(dna, 2, 7)
        court = ACTIONS.index("court")
        self.simulation._index_cells()
        senses, options, situation = self.simulation._sense(suitor)

        self.assertIn(court, options)
        self.assertNotIn(ACTIONS.index("breed"), options)
        self.simulation._act(suitor, "court", situation)
        self.assertEqual((suitor.x, suitor.y), (2, 4))  # a grazer's two cells a step
        mate.cooldown = 4
        self.assertNotIn(court, self.simulation._sense(suitor)[1])
        mate.cooldown, suitor.cooldown = 0, 4
        self.assertNotIn(court, self.simulation._sense(suitor)[1])

    def test_hunters_chase_prey_they_can_see(self):
        predator = self.simulation._add_agent(encode_dna("predator", sight=7, speed=3), 2, 2)
        grazer = self.simulation._add_agent(encode_dna("grazer"), 2, 7)
        self.simulation._index_cells()
        senses, options, situation = self.simulation._sense(predator)

        self.assertIn(ACTIONS.index("hunt"), options)
        self.assertNotIn(ACTIONS.index("attack"), options)
        self.simulation._act(predator, "hunt", situation)
        self.assertEqual((predator.x, predator.y), (2, 5))  # its two steps, and one more for the chase
        self.assertNotIn(ACTIONS.index("hunt"), self.simulation._sense(grazer)[1])

    def test_hunter_that_reaches_its_prey_pounces(self):
        predator = self.simulation._add_agent(encode_dna("predator", sight=7, speed=5), 2, 2)
        grazer = self.simulation._add_agent(encode_dna("grazer"), 2, 6)
        self.simulation._index_cells()
        situation = self.simulation._sense(predator)[2]

        self.simulation._act(predator, "hunt", situation)

        self.assertEqual(predator.last_action, "pounced")
        self.assertLess(grazer.health, grazer.max_health)

    def test_prey_flees_a_hunter_close_by(self):
        predator = self.simulation._add_agent(encode_dna("predator"), 5, 3)
        grazer = self.simulation._add_agent(encode_dna("grazer", sight=7), 5, 6)
        self.simulation._index_cells()
        senses, options, situation = self.simulation._sense(grazer)

        self.assertIn(ACTIONS.index("flee"), options)
        self.simulation._act(grazer, "flee", situation)
        self.assertGreater(abs(grazer.y - predator.y), 4)
        self.assertNotIn(ACTIONS.index("flee"), self.simulation._sense(predator)[1])

    @patch("tools.Simulation.BREED_RESERVE", 10)  # a reserve, as it may be set to
    @patch("tools.Simulation.FOOD_NEEDED", 3.0)
    @patch("tools.Simulation.PLANTS_PER_GRAZER", 0.0)  # no plants about: breeding left to the rest
    @patch("tools.Simulation.GRAZER_BREEDING_COST", 1.0)  # a grazer's full cost, as these figures assume
    def test_only_animals_that_can_afford_it_breed(self):
        lay_plenty_of_grass(self.simulation)
        dna = encode_dna("grazer", fertility=7)  # breeding costs it 10 hunger
        agent = self.simulation._add_agent(dna, 5, 5)
        mate = self.simulation._add_agent(dna, 5, 6)
        breed = ACTIONS.index("breed")
        self.simulation._index_cells()

        agent.hunger = 80  # 80 + 10 leaves exactly the 10 to spare
        self.assertIn(breed, self.simulation._sense(agent)[1])
        agent.hunger = 81
        self.assertNotIn(breed, self.simulation._sense(agent)[1])
        agent.hunger, mate.hunger = 0, 85
        self.assertNotIn(breed, self.simulation._sense(agent)[1])

    @patch("tools.Simulation.PLANTS_PER_GRAZER", 0)  # the plants at home aside
    @patch("tools.Simulation.GRAZER_BREEDING_COST", 1.0)  # a grazer's full cost, as these figures assume
    def test_nobody_courts_a_partner_who_cannot_afford_to_breed(self):
        dna = encode_dna("grazer", sight=7, fertility=7)
        suitor = self.simulation._add_agent(dna, 2, 2)
        mate = self.simulation._add_agent(dna, 2, 7)
        mate.hunger = 95
        self.simulation._index_cells()

        self.assertNotIn(ACTIONS.index("court"), self.simulation._sense(suitor)[1])
        mate.hunger = 0
        self.assertIn(ACTIONS.index("court"), self.simulation._sense(suitor)[1])

    @patch("tools.Simulation.PLANTS_PER_GRAZER", 0.0)  # no plants about: breeding left to the rest
    def test_population_cap_withdraws_breeding(self):
        lay_plenty_of_grass(self.simulation)
        dna = encode_dna("grazer")
        agent = self.simulation._add_agent(dna, 5, 5)
        self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()
        self.simulation._population = 2

        self.assertIn(ACTIONS.index("breed"), self.simulation._sense(agent)[1])  # no cap by default
        self.simulation.population_cap = 2
        self.assertNotIn(ACTIONS.index("breed"), self.simulation._sense(agent)[1])

    @patch("tools.Simulation.BREED_COOLDOWN", 1.0)  # the usual rest after breeding, as it may be set to
    @patch("tools.Simulation.GRAZER_BREEDING_COST", 1.0)  # a grazer's full cost, as these figures assume
    def test_fertile_parents_breed_cheaply_and_their_child_gets_the_energy(self):
        dna = encode_dna("grazer", fertility=7, libido=7)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()

        deliver(self.simulation, first, second)

        child = self.simulation.agents[-1]
        self.assertEqual((child.x, child.y), (5, 5))  # in its mother's square, with room for it
        self.assertEqual(child.hunger, 100 - 2 * 10)
        self.assertEqual(child.generation, 1)
        self.assertNotIn(child.brain, (first.brain, second.brain))
        self.assertEqual((first.hunger, first.cooldown, first.children, second.children), (10, 5, 1, 1))
        self.assertGreater(first.reward, 0)

    def test_breeding_past_full_hunger_leaves_the_parent_starving_not_dead(self):
        dna = encode_dna("grazer")
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        first.hunger = 90
        self.simulation._index_cells()
        self.simulation._population = 2

        self.simulation._breed(first, second)

        self.assertTrue(first.alive)  # full hunger wears its health away rather than killing it
        self.assertGreater(first.hunger, 100)
        self.assertTrue(first.womb)
        self.assertEqual(self.simulation.deaths, 0)
        with patch.object(Simulation, "_think_all"), patch.object(Simulation, "_dies_of_age", return_value=False):
            self.simulation._take_turn()
        self.assertEqual(first.hunger, 100)  # its hunger stops at full
        self.assertLess(first.health, first.max_health)
        self.assertTrue(second.alive)

    def test_child_brain_is_built_from_its_parents_brains(self):
        dna = encode_dna("grazer", fertility=7)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()

        deliver(self.simulation, first, second)

        child = self.simulation.agents[-1]
        for index, weights in enumerate(child.brain.layers[0][0]):
            nearest = min(max(abs(a - b) for a, b in zip(weights, parent.brain.layers[0][0][index]))
                          for parent in (first, second))
            self.assertLess(nearest, 0.3)  # one parent's neuron, plus small mutations

    def test_a_brain_is_twice_as_deep_as_the_inborn_intellect_and_plants_have_none(self):
        for intellect, depth in ((0, 2), (1, 2), (3, 6), (7, 14)):
            animal = self.simulation._add_agent(encode_dna("grazer", intellect=intellect), 1, 1)
            self.assertEqual(animal.brain.depth, depth)
        self.assertIsNone(self.simulation._add_agent(encode_dna("plant", intellect=7), 2, 2).brain)
        imported = self.simulation._add_agent(encode_dna("omnivore", intellect=4), 3, 3, brain=Brain.random(random.Random(1)))
        self.assertEqual(imported.brain.depth, 8)  # a saved brain is resized to fit

    def test_a_child_brain_is_as_deep_as_its_own_inborn_intellect(self):
        first = self.simulation._add_agent(encode_dna("grazer", fertility=7, intellect=1), 5, 5)
        second = self.simulation._add_agent(encode_dna("grazer", fertility=7, intellect=6), 5, 6)
        self.simulation._index_cells()
        for _ in range(5):
            first.cooldown = second.cooldown = 0
            first.hunger = second.hunger = 0
            deliver(self.simulation, first, second)
            child = self.simulation.agents[-1]
            self.assertEqual(child.brain.depth, max(2, 2 * child.inborn["intellect"]))

    def test_the_brain_senses_its_own_genes_but_not_which_actions_are_open(self):
        grazer = self.simulation._add_agent(encode_dna("grazer", size=7, intellect=3), 5, 5)
        self.simulation.grass[(5, 5)] = 1
        self.simulation._index_cells()
        senses, options, _ = self.simulation._sense(grazer)
        self.assertIn(ACTIONS.index("eat"), options)
        self.assertFalse([name for name in SENSES if name.startswith("open_")])
        self.assertEqual((senses[SENSES.index("size")], senses[SENSES.index("inborn_intellect")]), (1.0, 3 / 7))

    def test_groups_spawn_where_their_home_temperature_is_usual(self):
        from tools.Genome import decode_climate
        from tools.Simulation import zone_temperature
        for seed in range(1, 4):
            simulation = Simulation(width=100, height=100, population=90, seed=seed)
            for agent in simulation.agents:  # each within the spread of a centre at home
                home = decode_climate(agent.dna)[0]
                nearby = [zone_temperature(y, 100) for y in range(agent.y - SPAWN_SPREAD, agent.y + SPAWN_SPREAD + 1)]
                self.assertIn(min(home, 2), nearby, f"#{agent.identifier} at row {agent.y}")  # very cold homes take the arctic
        simulation = Simulation(width=100, height=100, population=0, seed=1)
        self.assertEqual(simulation._home_rows(HOT), list(range(40, 60)))
        self.assertEqual(simulation._home_rows(3), list(range(2, 20)) + list(range(80, 98)))  # very cold: the arctic
        with patch("tools.Simulation.SPAWN_AT_HOME", 0):
            anywhere = Simulation(width=100, height=100, population=90, seed=1)
        self.assertGreater(len({agent.y // 20 for agent in anywhere.agents}), 1)

    def test_every_zone_starts_with_plants_and_an_animal_group_is_always_kept(self):
        for seed in range(1, 6):
            simulation = Simulation(width=100, height=100, population=90, seed=seed)
            plants = [agent for agent in simulation.agents if agent.species == "plant"]
            for rows in simulation._zones():  # arctic, temperate, equator, temperate, arctic
                self.assertTrue(any(rows[0] - SPAWN_SPREAD <= plant.y <= rows[-1] + SPAWN_SPREAD for plant in plants))
            self.assertTrue(any(agent.species != "plant" for agent in simulation.agents))
        few = Simulation(width=100, height=100, population=8, seed=2)  # two groups: not enough for every zone
        self.assertEqual(len({agent.species == "plant" for agent in few.agents}), 2)  # a plant group and an animal group

    def test_a_starving_grazer_with_food_scarce_may_uproot_a_plant_beside_it(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        simulation.grass.clear()
        simulation.grass_regrowth = 0  # no grass to come
        grazer = simulation._add_agent(encode_dna("grazer", size=7), 10, 10)  # a big stomach
        plant = simulation._add_agent(encode_dna("plant", size=0), 11, 10)  # 60 health
        simulation._index_cells()
        grazer.hunger = 50
        self.assertNotIn(ACTIONS.index("uproot"), simulation._sense(grazer)[1])  # not starving
        grazer.hunger = 85
        plant.cooldown = 5  # still recovering from a bite: it can't be browsed, but it can be uprooted
        options, situation = simulation._sense(grazer)[1:]
        self.assertIn(ACTIONS.index("uproot"), options)
        self.assertNotIn(ACTIONS.index("browse"), options)
        meal = UPROOT_MEALS * FRESH_KILL_MEALS * KILL_MEAL / grazer.stomach  # half a feast on a body of its size
        simulation._act(grazer, "uproot", situation)
        self.assertAlmostEqual(grazer.hunger, max(0.0, 85 - meal))  # it may fill it
        self.assertEqual((plant.health, grazer.last_action), (60 - UPROOT_BITE, "uprooted"))
        grazer.hunger = 85
        simulation._act(grazer, "uproot", simulation._sense(grazer)[2])
        self.assertFalse(plant.alive)
        self.assertEqual(plant.last_action, "uprooted")  # it destroyed the plant
        for x in (12, 13, 14):  # with food enough about, it waits for the plants to recover
            simulation._add_agent(encode_dna("plant"), x, 10)
        simulation._add_agent(encode_dna("plant"), 10, 11)
        simulation._index_cells()
        self.assertNotIn(ACTIONS.index("uproot"), simulation._sense(grazer)[1])
        omnivore = simulation._add_agent(encode_dna("omnivore"), 10, 9)
        omnivore.hunger = 85
        simulation._index_cells()
        self.assertNotIn(ACTIONS.index("uproot"), simulation._sense(omnivore)[1])  # grazers only

    def test_a_brain_learns_from_changes_in_its_comfort_not_how_comfortable_it_is(self):
        grazer = self.simulation._add_agent(encode_dna("grazer"), 5, 5)
        self.simulation._index_cells()
        with patch.object(Simulation, "_comfort", return_value=0.4), \
                patch.object(Simulation, "_act"), patch.object(Simulation, "_learn") as learn:
            self.simulation._think(grazer)  # its first decision: nothing to learn from yet
            self.assertEqual(grazer.last_comfort, 0.4)
            grazer.last_wellbeing = self.simulation._wellbeing(grazer)
            self.simulation._think(grazer)  # just as comfortable: no success or failure in that
        target = learn.call_args.args[1]
        values = grazer.brain.values(self.simulation._sense(grazer)[0])
        best = max(values[option] for option in self.simulation._sense(grazer)[1])
        self.assertAlmostEqual(target, (0.7 + 0.27 * grazer.feeling("patience")) * best)

    def test_child_inherits_what_experience_taught_its_parents(self):
        dna = encode_dna("grazer", fertility=7, fear=2, curiosity=5, libido=6)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        first.mind.update(fear=4.0, curiosity=3.0, libido=7.0)
        second.mind.update(fear=3.0, curiosity=5.0, libido=7.0)
        self.simulation._index_cells()

        self.simulation.mind_mutation = 0
        deliver(self.simulation, first, second)

        child = self.simulation.agents[-1]
        # its mind starts at the average of its parents' current minds, not their genes
        self.assertEqual(child.mind["fear"], 3.5)
        self.assertEqual(child.mind["curiosity"], 4.0)
        self.assertEqual(child.mind["libido"], 7.0)
        self.assertEqual(child.mind["patience"], 0.0)
        # and its dot shows it, the half rounding either way
        self.assertIn(child.inborn["fear"], (3, 4))
        self.assertEqual((child.inborn["curiosity"], child.inborn["libido"]), (4, 7))

    @patch("tools.Simulation.GRAZER_BREEDING_COST", 1.0)  # a grazer's full cost, as these figures assume
    def test_a_parent_gives_its_young_no_more_energy_than_it_has_left(self):
        dna = encode_dna("grazer", fertility=7)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        second.hunger = 96  # has 4 left to give, though breeding costs 10
        self.simulation._index_cells()
        self.simulation._population = 2

        deliver(self.simulation, first, second)

        self.assertEqual(self.simulation.agents[-1].hunger, 100 - (10 + 4))
        self.assertTrue(second.alive)  # starving, not dead

    def test_intellect_decides_whether_brain_learns(self):
        slow = self.simulation._add_agent(encode_dna("grazer", intellect=0), 1, 1)
        quick = self.simulation._add_agent(encode_dna("grazer", intellect=7), 8, 8)
        slow_brain, quick_brain = slow.brain.copy(), quick.brain.copy()

        for _ in range(5):
            self.simulation._take_turn()

        self.assertEqual(slow.brain, slow_brain)
        self.assertNotEqual(quick.brain, quick_brain)

    def test_armour_blunts_and_weapons_sharpen_blows(self):
        brute = self.simulation._add_agent(encode_dna("predator", strength=5, weapons=7), 1, 1)
        plain = self.simulation._add_agent(encode_dna("predator", strength=5), 2, 2)
        soft = self.simulation._add_agent(encode_dna("grazer"), 3, 3)
        armoured = self.simulation._add_agent(encode_dna("grazer", armour=7), 4, 4)

        self.assertGreater(self.simulation._damage(brute, soft), self.simulation._damage(plain, soft))
        self.assertLess(self.simulation._damage(plain, armoured), self.simulation._damage(plain, soft))

    def test_kill_is_counted_and_corpse_is_buried(self):
        hunter = self.simulation._add_agent(encode_dna("predator", strength=7), 5, 5)
        victim = self.simulation._add_agent(encode_dna("grazer"), 5, 6)
        victim.health = 1
        self.simulation._index_cells()

        self.simulation._attack(hunter, victim)

        self.assertFalse(victim.alive)
        self.assertIsNone(victim.brain)
        self.assertEqual(self.simulation.deaths, 1)
        self.assertGreater(hunter.mind["aggression"], hunter.inborn["aggression"])
        for _ in range(CORPSE_TURNS):
            self.simulation._take_turn()
        self.assertNotIn(victim, self.simulation.agents)

    def test_armed_prey_stings_its_attacker(self):
        hunter = self.simulation._add_agent(encode_dna("predator", fear=3), 5, 5)
        victim = self.simulation._add_agent(encode_dna("grazer", weapons=7, size=7), 5, 6)
        self.simulation._index_cells()

        self.simulation._attack(hunter, victim)

        self.assertLess(hunter.health, hunter.max_health)
        self.assertGreater(hunter.mind["fear"], 3)
        self.assertGreater(victim.mind["fear"], victim.inborn["fear"])

    def test_the_latest_turns_are_kept_in_memory_and_reached_at_once_exactly(self):
        with patch("tools.Simulation.CLIMATE_DAMAGE", 0):  # a world that lasts
            simulation = Simulation(width=20, height=20, population=30, seed=4)

            def state():  # copies, since the live objects carry on changing
                return ([(agent.identifier, agent.dna, agent.x, agent.y, agent.hunger, agent.health,
                          dict(agent.mind), agent.brain and agent.brain.copy(), agent.meat, list(agent.trail))
                         for agent in simulation.agents],
                        dict(simulation.grass), simulation.deaths, simulation.random.getstate())

            recorded = {0: state()}
            for turn in range(1, 61):
                simulation.advance()
                recorded[turn] = state()
            self.assertEqual([entry["turn"] for entry in simulation.history], list(range(61)))
            with patch.object(Simulation, "_take_turn", side_effect=AssertionError("replayed")):
                for target in (37, 3, 0, 59, 60, 17, 18, 60):  # every one at once, with no replaying
                    simulation.seek(target)
                    self.assertEqual(simulation.turn, target)
                    self.assertEqual(state(), recorded[target], f"turn {target}")
                simulation.seek(57)
                simulation.advance()  # stepping on through the recorded run brings back each turn
                self.assertEqual(state(), recorded[58])
            simulation.seek(60)
            simulation.advance()  # and from the latest, the world carries on
            self.assertEqual(simulation.frontier, 61)
            simulation.close()
            self.assertEqual(len(simulation.history), 0)

    def test_the_history_keeps_only_the_latest_turns_and_fits_its_memory(self):
        with patch("tools.Simulation.CLIMATE_DAMAGE", 0), patch("tools.Simulation.HISTORY_TURNS", 10):
            simulation = Simulation(width=20, height=20, population=30, seed=4)
            for _ in range(40):
                simulation.advance()
            self.assertEqual([entry["turn"] for entry in simulation.history], list(range(31, 41)))
            simulation.seek(0)  # as far back as it reaches
            self.assertEqual(simulation.turn, 31)
            self.assertFalse(simulation.rewind())
            simulation.seek(40)
            with patch("tools.Simulation.HISTORY_TURNS", 5000):  # "all": every turn from here on
                for _ in range(20):
                    simulation.advance()
            self.assertEqual(len(simulation.history), 30)
            simulation.seek(60)
            with patch("tools.Simulation.HISTORY_MEMORY", 1e-9):  # far too little: only the newest is kept
                simulation.advance()
            self.assertEqual([entry["turn"] for entry in simulation.history], [61])
        brains = sum(agent.brain_bytes for agent in simulation.agents)
        self.assertGreater(brains, 0)
        self.assertEqual(simulation._history_bytes, brains + AGENT_BYTES * len(simulation.agents))

    def test_a_history_left_behind_is_cleared_away_once_stale(self):
        import os
        import time
        from tools.Simulation import STALE_HISTORY_HOURS, remove_stale_histories
        import tempfile
        from pathlib import Path
        from tools.Simulation import HISTORY_PREFIX
        old, new = (Path(tempfile.mkdtemp(prefix=HISTORY_PREFIX)) for _ in range(2))  # as earlier versions left them
        long_ago = time.time() - (STALE_HISTORY_HOURS + 1) * 3600
        os.utime(old, (long_ago, long_ago))
        remove_stale_histories()
        self.assertEqual((old.exists(), new.exists()), (False, True))
        new.rmdir()

    def test_plants_seed_less_the_more_plants_shade_them(self):
        plant = self.simulation._add_agent(encode_dna("plant", fertility=7), 5, 5)  # a 10% chance alone
        self.simulation._index_cells()
        seeded = []
        for shade in range(4):
            if shade:
                self.simulation._add_agent(encode_dna("plant"), 5 + shade, 5)  # within three cells
                self.simulation._index_cells()
            count = 0
            for roll in (0.02, 0.05, 0.08):
                with patch.object(self.simulation.random, "random", return_value=roll),                         patch.object(self.simulation.random, "randrange", return_value=0):  # beside it, on the map
                    before = len(self.simulation.agents)
                    self.simulation._seed(plant)
                    count += len(self.simulation.agents) > before or bool(self.simulation.seeds)  # sprouted or waits
                    del self.simulation.agents[before:]
                    self.simulation.seeds.clear()
                    self.simulation._index_cells()
                plant.hunger = 0
            seeded.append(count)
        self.assertEqual(seeded, [3, 2, 1, 0])  # a chance of 10%, then two thirds, a third and none

    def test_rewinding_reaches_turn_zero_while_the_history_fits_its_budget(self):
        with patch("tools.Simulation.CLIMATE_DAMAGE", 0):  # a world that lasts
            simulation = Simulation(width=20, height=20, population=10, seed=4)
            for _ in range(30):
                simulation.advance()

            while simulation.rewind():
                pass

        self.assertEqual((simulation.turn, simulation.frontier), (0, 30))

    def test_the_simulation_ends_when_no_animals_are_left(self):
        simulation = Simulation(width=10, height=10, population=0, seed=1)
        plant = simulation._add_agent(encode_dna("plant"), 1, 1)
        grazer = simulation._add_agent(encode_dna("grazer"), 5, 5)
        self.assertIsNone(simulation.end_reason)

        grazer.health = 0
        simulation._die(grazer, "killed")

        self.assertEqual(simulation.end_reason, "extinct")
        self.assertTrue(plant.alive)

    def test_the_simulation_ends_when_every_cell_is_taken(self):
        simulation = Simulation(width=10, height=10, population=0, seed=1)
        simulation._add_agent(encode_dna("grazer"), 0, 0)
        for x in range(10):
            for y in range(10):
                if (x, y) != (0, 0):
                    simulation._add_agent(encode_dna("plant"), x, y)

        self.assertEqual(simulation.end_reason, "full")

    def test_an_ended_simulation_stops_but_can_still_be_rewound(self):
        with patch("tools.Simulation.CLIMATE_DAMAGE", 0):  # a world that lasts until it is ended
            simulation = Simulation(width=20, height=20, population=30, seed=4)
            for _ in range(16):
                simulation.advance()
            for agent in simulation.living:
                if agent.species != "plant":
                    agent.health = 0
                    simulation._die(agent, "killed")
            simulation._record()  # the end, as a recorded turn: one with a snapshot, as the killing can't be replayed
            simulation.frontier = simulation.turn

            simulation.advance()
            self.assertTrue(simulation.ended)
            self.assertEqual(simulation.turn, 16)
            simulation.seek(2)
            self.assertFalse(simulation.ended)
            for _ in range(20):
                simulation.advance()
            self.assertEqual(simulation.turn, 16)

    def test_rewind_and_replay_are_exact(self):
        simulation = Simulation(width=20, height=20, population=30, seed=4)

        def state():
            return ([(agent.identifier, agent.dna, agent.x, agent.y, agent.hunger, dict(agent.mind),
                      agent.brain and agent.brain.copy())
                     for agent in simulation.agents], simulation.insights.snapshot())

        for _ in range(15):
            simulation.advance()
        expected = state()
        for _ in range(5):
            simulation.rewind()
        for _ in range(5):
            simulation.advance()

        self.assertEqual(state(), expected)



class NewTraitTests(unittest.TestCase):
    def setUp(self):
        self.simulation = Simulation(width=12, height=12, population=0, seed=3)
        self.simulation.grass.clear()

    @patch("tools.Simulation.SMALL_LITTERS", 1.0)
    @patch("tools.Simulation.NEWBORN_FOOD", 0.0)  # no limit from the energy given
    @patch("tools.Simulation.PLANTS_PER_GRAZER", 0.0)  # no plants about: breeding left to the rest
    @patch("tools.Simulation.GRAZER_BREEDING_COST", 1.0)  # a grazer's full cost, as these figures assume
    def test_litter_size_sets_how_many_young_share_the_parents_energy(self):
        lay_plenty_of_grass(self.simulation)
        dna = encode_dna("grazer", fertility=7, litter=7)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()
        self.simulation._population = 2

        deliver(self.simulation, first, second, litter=None)

        young = self.simulation.agents[2:]
        self.assertEqual(len(young), 6)
        for child in young:  # the parents give 10% of their small stomachs each, shared six ways
            self.assertAlmostEqual(child.hunger, 100 - 2 * 10 * 0.5 / 6 / child.stomach)
        self.assertEqual((first.children, second.children), (6, 6))

    @patch("tools.Simulation.SMALL_LITTERS", 1.0)
    @patch("tools.Simulation.NEWBORN_FOOD", 5.0)
    @patch("tools.Simulation.PLANTS_PER_GRAZER", 0.0)  # no plants about: breeding left to the rest
    def test_parents_conceive_only_as_many_young_as_they_can_feed(self):
        dna = encode_dna("grazer", fertility=7, litter=7)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()
        self.simulation._population = 2

        first.hunger = second.hunger = 90  # each with 10% of a stomach left to give

        deliver(self.simulation, first, second, litter=None)

        # six young by their traits, but 10% of a stomach from each parent feeds only four at 5% each
        self.assertEqual(len(self.simulation.agents), 2 + 4)

    @patch("tools.Simulation.SMALL_LITTERS", 1.0)
    @patch("tools.Simulation.PLANTS_PER_GRAZER", 0.0)
    def test_what_parents_could_give_sets_the_litter_not_their_cheap_breeding(self):
        for species in ("grazer", "predator"):  # the fertile pay little, and a predator half that: no smaller litters
            simulation = Simulation(width=12, height=12, population=0, seed=3)
            dna = encode_dna(species, fertility=7, litter=7)
            first = simulation._add_agent(dna, 5, 5)
            second = simulation._add_agent(dna, 5, 6)
            simulation._index_cells()
            simulation._population = 2
            with patch("tools.Simulation.PREY_PER_PREDATOR", 0.0):
                deliver(simulation, first, second, litter=None)
            self.assertEqual(len(simulation.agents), 2 + 6)  # all six of their traits: 25% each could feed ten at 5%

    def test_a_pair_too_poor_to_feed_any_young_at_newborn_food_still_has_one(self):
        dna = encode_dna("grazer", fertility=7, litter=7)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()
        self.simulation._population = 2

        first.hunger = second.hunger = 98  # 2% of a stomach left each: not enough for one young at 5%

        deliver(self.simulation, first, second, litter=None)

        self.assertEqual(len(self.simulation.agents), 2 + 1)  # but always at least one

    @patch("tools.Simulation.SMALL_LITTERS", 1.0)
    def test_large_bodies_with_a_low_litter_trait_have_single_young(self):
        dna = encode_dna("grazer", fertility=7, litter=0, size=7)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()

        deliver(self.simulation, first, second, litter=None)

        self.assertEqual(len(self.simulation.agents), 3)

    def test_smaller_bodies_have_bigger_litters(self):
        for litter in range(8):
            sizes = [litter_size([(litter, size)] * 2) for size in range(8)]
            self.assertEqual(sizes, sorted(sizes, reverse=True), f"litter trait {litter}")
        self.assertEqual((litter_size([(7, 0)] * 2), litter_size([(0, 7)] * 2)), (12, 1))  # twice the usual, smallest
        with patch("tools.Simulation.SMALL_LITTERS", 1.0):
            self.assertEqual(litter_size([(7, 0)] * 2), 6)
        self.assertEqual(litter_size([(4, 3)] * 2), round(4 * (1 + 4 / 7)))  # a middling body, part way
        self.assertEqual(litter_size([(7, 0), (0, 7)]), litter_size([(3, 3)] * 2))  # parents average out

    def test_a_struck_animal_can_defend_against_its_assailant_alongside(self):
        grazer = self.simulation._add_agent(encode_dna("grazer", size=7, strength=7), 5, 5)
        bystander = self.simulation._add_agent(encode_dna("predator"), 4, 5)
        predator = self.simulation._add_agent(encode_dna("predator", size=7, strength=7, armour=7), 5, 6)
        self.simulation._index_cells()
        self.assertNotIn(ACTIONS.index("defend"), self.simulation._sense(grazer)[1])

        self.simulation._attack(predator, grazer)
        senses, options, situation = self.simulation._sense(grazer)
        self.assertIn(ACTIONS.index("defend"), options)
        self.assertIs(situation["assailant"], predator)  # not the bystander, which never struck
        self.assertEqual(senses[SENSES.index("assailant_near")], 1)

        health, hunger = predator.health, grazer.hunger
        damage = self.simulation._damage(grazer, predator)
        self.simulation._act(grazer, "defend", situation)
        self.assertAlmostEqual(predator.health, health - damage)
        self.assertGreater(grazer.hunger, hunger)  # striking back costs effort, and feeds nothing
        self.assertEqual(grazer.last_action, "defended")

    def test_a_counterblow_scales_with_size_and_strength_and_can_kill(self):
        attacker = self.simulation._add_agent(encode_dna("predator"), 5, 5)
        small = self.simulation._add_agent(encode_dna("grazer"), 5, 6)
        large = self.simulation._add_agent(encode_dna("grazer", size=7, strength=7), 6, 5)
        self.assertGreater(self.simulation._damage(large, attacker), 2 * self.simulation._damage(small, attacker))

        attacker.health = 1
        self.simulation._index_cells()
        self.simulation._defend(large, attacker)
        self.assertFalse(attacker.alive)

    def test_the_chance_to_defend_passes_once_the_animal_has_chosen(self):
        grazer = self.simulation._add_agent(encode_dna("grazer"), 5, 5)
        predator = self.simulation._add_agent(encode_dna("predator", weapons=0), 5, 6)
        self.simulation._index_cells()
        self.simulation._attack(predator, grazer)
        self.assertEqual(grazer.assailant, predator.identifier)

        self.simulation._think(grazer)
        self.assertIsNone(grazer.assailant)

    def test_the_fearless_never_break_and_the_fearful_flee_their_assailant(self):
        brave = self.simulation._add_agent(encode_dna("grazer", fear=0), 5, 5)
        timid = self.simulation._add_agent(encode_dna("grazer", fear=7, strength=7, intellect=7), 2, 2)  # strong enough to defend; too clever to freeze
        brute = self.simulation._add_agent(encode_dna("predator", weapons=0), 5, 6)
        bully = self.simulation._add_agent(encode_dna("predator", weapons=0), 2, 3)
        self.simulation._index_cells()
        self.simulation._attack(brute, brave)
        self.assertFalse(brave.panicked)

        timid.health = 1  # all but dead, and fully fearful: certain to break
        self.simulation._struck(timid, bully)
        self.assertTrue(timid.panicked)
        senses, options, situation = self.simulation._sense(timid)
        self.assertIn(ACTIONS.index("defend"), options)  # it could have fought, but fear decides
        self.simulation._think(timid)
        self.assertEqual(timid.last_action, "panicked")
        self.assertGreater(abs(timid.x - bully.x) + abs(timid.y - bully.y), 1)
        self.assertFalse(timid.panicked)

    def test_morale_breaks_more_often_the_more_hurt_and_afraid(self):
        striker = self.simulation._add_agent(encode_dna("predator"), 1, 1)
        def breaks(fear, health_share):
            victim = self.simulation._add_agent(encode_dna("grazer", fear=fear), 3, 3)
            victim.health = victim.max_health * health_share
            broke = 0
            for _ in range(2000):
                victim.panicked = False
                self.simulation._struck(victim, striker)
                broke += victim.panicked
                victim.mind["fear"] = float(fear)  # undo what panicking taught it
            return broke / 2000
        self.assertAlmostEqual(breaks(7, 1.0), 0.3, delta=0.04)
        self.assertGreater(breaks(7, 0.2), 0.8)
        self.assertLess(breaks(2, 0.2), 0.35)

    def test_armed_plants_strike_back_at_browsers_and_unarmed_ones_cannot(self):
        thorny = self.simulation._add_agent(encode_dna("plant", size=7, strength=7, weapons=7), 5, 5)
        bare = self.simulation._add_agent(encode_dna("plant", size=7, strength=7, weapons=0), 9, 9)
        first = self.simulation._add_agent(encode_dna("grazer"), 5, 6)
        second = self.simulation._add_agent(encode_dna("grazer"), 9, 10)
        self.simulation._index_cells()

        self.simulation._browse(first, thorny)
        self.simulation._browse(second, bare)
        self.assertAlmostEqual(first.max_health - first.health, self.simulation._damage(thorny, first))
        self.assertEqual(second.health, second.max_health)
        self.assertEqual(thorny.last_action, "new")  # a plant has no brain to reward

    def test_a_plants_thorns_come_mostly_from_its_weapons_and_never_from_strength(self):
        grazer = self.simulation._add_agent(encode_dna("grazer"), 1, 1)
        def thorns(**traits):
            return self.simulation._damage(self.simulation._add_agent(encode_dna("plant", **traits), 5, 5), grazer)
        self.assertEqual(thorns(weapons=4, strength=0), thorns(weapons=4, strength=7))
        self.assertGreater(thorns(weapons=7, size=0), 3 * thorns(weapons=0, size=7))
        self.assertAlmostEqual(thorns(weapons=7, size=7), 15)

    def test_predators_are_one_kind_within_a_quarter_and_hunt_others_only_when_hungry(self):
        dna = encode_dna("predator", size=3)
        hunter = self.simulation._add_agent(dna, 1, 1)
        near = self.simulation._add_agent(dna ^ (0xFF << 56), 2, 2)  # cyan byte inverted: 25% apart
        far = self.simulation._add_agent(dna ^ (0xFFFF << 48), 3, 3)  # cyan and magenta inverted: 50% apart
        self.assertTrue(self.simulation._same_kind(hunter, near))
        self.assertFalse(self.simulation._same_kind(hunter, far))
        self.assertFalse(self.simulation._is_prey(hunter, far))  # not hungry: no other predator is prey
        hunter.hunger = far.hunger = 95
        self.assertTrue(self.simulation._is_prey(hunter, far))
        self.assertTrue(self.simulation._is_prey(far, hunter))  # and so each hunts, and flees, the other
        self.simulation._index_cells()
        self.assertIn(hunter, self.simulation._look_around(far, 5)[:2])  # as prey, and as a threat

    def test_cannibalism_waits_on_hunger_past_empathy_tolerance_and_patience(self):
        gentle = self.simulation._add_agent(encode_dna("predator", empathy=7, tolerance=7, patience=7), 1, 1)
        ruthless = self.simulation._add_agent(encode_dna("predator", empathy=0, tolerance=0, patience=0), 2, 2)
        middling = self.simulation._add_agent(encode_dna("predator", empathy=7, tolerance=0, patience=0), 3, 3)
        kin = self.simulation._add_agent(gentle.dna ^ (0xFFFF << 48), 4, 4)  # another kind: its own is never prey
        for predator in (gentle, ruthless, middling):
            predator.hunger = 50
        self.assertFalse(self.simulation._is_prey(gentle, kin))
        self.assertTrue(self.simulation._is_prey(ruthless, kin))
        self.assertTrue(self.simulation._is_prey(middling, kin))  # reluctant only to a third
        middling.hunger = 30
        self.assertFalse(self.simulation._is_prey(middling, kin))
        gentle.hunger = 99
        self.assertFalse(self.simulation._is_prey(gentle, kin))  # never, short of starving to death

    def test_killing_its_own_kind_pains_an_empathetic_predator_most(self):
        dna = encode_dna("predator", empathy=7, aggression=0, strength=7, size=7)
        hunter = self.simulation._add_agent(dna, 5, 5)
        kin = self.simulation._add_agent(dna, 5, 6)
        stranger = self.simulation._add_agent(dna ^ (0xFF << 56), 6, 5)
        self.simulation._index_cells()
        for victim in (kin, stranger):
            hunter.reward = 0
            self.simulation._attack(hunter, victim)
            if victim is kin:
                kin_pain = -hunter.reward
        self.assertGreater(kin_pain, -hunter.reward)

    def test_grass_grows_twice_as_fast_on_and_around_corpses(self):
        rows = self.simulation._grass_rows  # grass grows only where it is medium
        self.assertTrue({7, 8, 9} <= set(rows))
        corpse = self.simulation._add_agent(encode_dna("grazer"), 0, 8)
        corpse.health = 0
        self.simulation._die(corpse, "wounded")
        self.simulation.grass.clear()
        self.simulation._sow(20000)
        self.assertTrue(all(y in rows for _, y in self.simulation.grass))
        around = [((dx % 12), 8 + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]  # wrapping round the edge
        fertile = sum(self.simulation.grass.get(cell, 0) for cell in around) / len(around)
        elsewhere = (sum(self.simulation.grass.values()) - fertile * len(around)) / (12 * len(rows) - len(around))
        self.assertAlmostEqual(fertile / elsewhere, 2, delta=0.35)

    @patch("tools.Simulation.PACK_RIVALRY", 1.0)  # rivalry measured among predators counting fully
    def test_a_predators_territory_grows_rivalrous_past_a_few_other_predators(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        predator = simulation._add_agent(encode_dna("predator", empathy=0, sight=1), 15, 15)  # tolerates just one other, within five cells
        for count, x in enumerate((10, 11, 12, 18), start=1):
            simulation._add_agent(encode_dna("predator"), x, 15)
            simulation._add_agent(encode_dna("grazer"), x, 16)  # other animals don't count
            simulation._index_cells()
            self.assertAlmostEqual(simulation._territory(predator)[0], max(0, count - 1) / 3)
        simulation._add_agent(encode_dna("predator"), 15, 25)  # beyond its territory
        simulation._index_cells()
        self.assertEqual(simulation._territory(predator)[0], 1)
        self.assertEqual(simulation._sense(predator)[0][SENSES.index("rivalry")], 1)

    @patch("tools.Simulation.PACK_RIVALRY", 1.0)  # rivalry measured among predators counting fully
    def test_the_more_empathetic_a_predator_the_more_rivals_it_tolerates(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        callous = simulation._add_agent(encode_dna("predator", empathy=0, sight=1), 15, 15)  # territories of five cells
        caring = simulation._add_agent(encode_dna("predator", empathy=7, sight=1), 15, 16)
        for x in (12, 13, 17, 18):
            simulation._add_agent(encode_dna("predator"), x, 15)
        simulation._index_cells()
        # each sees the other and four more: five rivals
        self.assertEqual(simulation._territory(callous)[0], 1)    # tolerates one: four too many
        self.assertEqual(simulation._territory(caring)[0], 0)     # tolerates five
        caring.mind["empathy"] = 3.5                         # learned callousness: tolerates three
        self.assertAlmostEqual(simulation._territory(caring)[0], 2 / 3)

    @patch("tools.Simulation.PACK_RIVALRY", 1.0)  # rivalry measured among predators counting fully
    def test_predators_count_only_predators_as_rivals_for_their_territory(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        predator = simulation._add_agent(encode_dna("predator", empathy=0, sight=1), 15, 15)  # tolerates one rival, within five cells
        for x in (12, 13, 17):
            simulation._add_agent(encode_dna("predator"), x, 15)
        simulation._add_agent(encode_dna("omnivore"), 16, 15)  # omnivores and grazers are prey, not rivals
        simulation._add_agent(encode_dna("grazer"), 18, 15)
        simulation._index_cells()
        self.assertAlmostEqual(simulation._territory(predator)[0], 2 / 3)

    def test_omnivores_are_territorial_at_half_strength_and_more_by_empathy(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        callous = simulation._add_agent(encode_dna("omnivore", empathy=0, sight=1), 15, 15)   # tolerates two, within five cells
        caring = simulation._add_agent(encode_dna("omnivore", empathy=7, sight=1), 15, 16)    # tolerates ten
        for x in (10, 11, 12, 18, 19, 20):
            simulation._add_agent(encode_dna("omnivore"), x, 15)
        simulation._index_cells()
        self.assertEqual(simulation._territory(callous)[0], 0.5)  # seven rivals: five too many, at half strength
        self.assertEqual(simulation._territory(caring)[0], 0)

    def test_omnivores_fight_their_own_only_in_an_overcrowded_territory(self):
        ruthless = self.simulation._add_agent(encode_dna("omnivore", empathy=0, tolerance=0, patience=0), 1, 1)
        other = self.simulation._add_agent(encode_dna("omnivore"), 2, 1)
        ruthless.hunger = 50
        self.assertFalse(self.simulation._is_prey(ruthless, other))
        ruthless.own_rivalry = 0.5
        self.assertTrue(self.simulation._is_prey(ruthless, other))
        gentle = self.simulation._add_agent(encode_dna("omnivore", empathy=7, tolerance=7, patience=7), 3, 3)
        gentle.hunger, gentle.own_rivalry = 80, 0.5  # reluctance worn down by a quarter, to three quarters
        self.assertTrue(self.simulation._is_prey(gentle, other))
        gentle.hunger = 70
        self.assertFalse(self.simulation._is_prey(gentle, other))

    def test_omnivores_around_a_predator_are_prey_not_rivals_and_dont_turn_it_on_its_kin(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        predator = simulation._add_agent(encode_dna("predator", empathy=0), 15, 15)
        for x in (11, 12, 13, 17, 18, 19):
            simulation._add_agent(encode_dna("omnivore"), x, 15)
        simulation._index_cells()
        self.assertEqual(simulation._territory(predator), (0.0, 0.0))  # not crowded by another species
        gentle = simulation._add_agent(encode_dna("predator", empathy=7, tolerance=7, patience=7), 20, 20)
        gentle.rivalry, gentle.hunger = 1.0, 30
        self.assertFalse(simulation._is_prey(gentle, simulation._add_agent(gentle.dna, 20, 21)))

    def test_omnivores_count_only_omnivores_as_rivals(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        omnivore = simulation._add_agent(encode_dna("omnivore", empathy=0), 15, 15)  # tolerates two
        for x in (11, 12, 13, 17, 18):
            simulation._add_agent(encode_dna("predator"), x, 15)
        simulation._index_cells()
        self.assertEqual(simulation._territory(omnivore), (0.0, 0.0))  # predators are prey, not rivals
        self.assertTrue(simulation._is_prey(omnivore, simulation.agents[1]))
        for x in (14, 16, 15):
            simulation._add_agent(encode_dna("omnivore"), x, 14)
        simulation._index_cells()
        rivals, own = simulation._territory(omnivore)  # three omnivores: one more than it tolerates, at half strength
        self.assertAlmostEqual(rivals, 1 / 6)
        self.assertAlmostEqual(own, 1 / 6)

    def test_an_overcrowded_territory_wears_down_reluctance_to_hunt_other_kinds_but_never_its_own(self):
        gentle = self.simulation._add_agent(encode_dna("predator", empathy=7, tolerance=7, patience=7), 1, 1)
        kin = self.simulation._add_agent(gentle.dna, 2, 1)
        stranger = self.simulation._add_agent(gentle.dna ^ (0xFFFF << 48), 3, 1)  # 50% apart: another kind
        gentle.hunger, gentle.own_rivalry = 70, 0.8  # reluctance worn down by RIVALRY_WEAR of 0.8, to 0.6
        self.assertTrue(self.simulation._is_prey(gentle, stranger))
        gentle.hunger = 50
        self.assertFalse(self.simulation._is_prey(gentle, stranger))
        gentle.hunger = 99
        self.assertFalse(self.simulation._is_prey(gentle, kin))  # never its own kind

    def test_crowded_territory_is_uncomfortable_and_killing_a_rival_a_relief(self):
        hunter = self.simulation._add_agent(encode_dna("predator", strength=7, size=7, aggression=0, empathy=0),
                                            5, 5)
        situation = {"kin": 0, "others": 0, "threats": 0}
        calm = self.simulation._comfort(hunter, {**situation, "rivalry": 0.0})
        self.assertLess(self.simulation._comfort(hunter, {**situation, "rivalry": 1.0}), calm)

        rival = self.simulation._add_agent(encode_dna("predator") ^ (0xFF << 56), 5, 6)
        rival.health = 1
        self.simulation._index_cells()
        hunter.rivalry, hunter.reward = 1.0, 0.0
        self.simulation._attack(hunter, rival)
        self.assertGreater(hunter.reward, 0.4)  # all relief: it takes no joy in fighting itself

    def test_a_world_sets_its_own_mutation_chances(self):
        stable = Simulation(width=12, height=12, population=0, seed=3, mind_mutation=0, body_mutation=0)
        dna = encode_dna("grazer", fertility=7, fear=3, size=2)
        first, second = stable._add_agent(dna, 5, 5), stable._add_agent(dna, 5, 6)
        stable._index_cells()
        for _ in range(20):
            first.hunger = second.hunger = first.cooldown = second.cooldown = 0
            first.mind = dict(second.mind)  # alike, so the child's mind is exactly theirs
            before = dict(first.mind)  # breeding itself shifts the parents' minds
            deliver(stable, first, second)
            child = stable.agents.pop()
            self.assertEqual(child.mind, before)
            self.assertEqual(child.body, first.body)
        with self.assertRaises(ValueError):
            Simulation(width=12, height=12, population=0, mind_mutation=101)

    def test_herd_instinct_is_the_lower_of_empathy_and_patience(self):
        animal = self.simulation._add_agent(encode_dna("grazer", empathy=7, patience=2), 1, 1)
        self.assertAlmostEqual(animal.herd_instinct, 2 / 7)
        animal.mind["patience"] = 7.0
        self.assertEqual(animal.herd_instinct, 1)

    def test_herd_mates_are_its_species_and_kind_that_it_would_accept_as_mates(self):
        dna = encode_dna("grazer", empathy=7, patience=7)
        animal = self.simulation._add_agent(dna, 5, 5)
        mate = self.simulation._add_agent(dna, 5, 7)
        stranger = self.simulation._add_agent(dna ^ (0xFF << 56), 7, 5)  # ring 25% apart
        omnivore = self.simulation._add_agent(encode_dna("omnivore", empathy=7, patience=7), 3, 5)
        self.assertTrue(self.simulation._herd_mate(animal, mate))
        self.assertFalse(self.simulation._herd_mate(animal, stranger))
        self.assertFalse(self.simulation._herd_mate(animal, omnivore))
        animal.mind["tolerance"] = 7.0  # the tolerant herd with other kinds of their species
        self.assertTrue(self.simulation._herd_mate(animal, stranger))
        predator = self.simulation._add_agent(encode_dna("predator", tolerance=7), 9, 9)
        other_kind = self.simulation._add_agent(encode_dna("predator", tolerance=7) ^ (0xFFFF << 48), 9, 10)  # 50% apart
        self.assertFalse(self.simulation._herd_mate(predator, other_kind))  # packs keep to their kind

    def test_the_herd_action_moves_toward_the_middle_of_the_herd(self):
        dna = encode_dna("grazer", empathy=7, patience=7)
        animal = self.simulation._add_agent(encode_dna("grazer", empathy=7, patience=7, fear=7, sight=1), 2, 5)  # sees its herd four off
        for x, y in ((5, 4), (5, 6), (6, 5)):
            self.simulation._add_agent(dna, x, y)
        self.simulation._index_cells()
        senses, options, situation = self.simulation._sense(animal)
        self.assertIn(ACTIONS.index("herd"), options)
        self.assertEqual(situation["herd_middle"], (5, 5))
        self.simulation._act(animal, "herd", situation)
        self.assertEqual((animal.x, animal.y, animal.last_action), (4, 5, "herded"))  # two cells a step

    def test_the_fearful_keep_to_the_middle_of_the_herd_and_the_fearless_roam_its_edge(self):
        dna = encode_dna("grazer", empathy=7, patience=7)
        for x, y in ((6, 4), (6, 6), (7, 5)):
            self.simulation._add_agent(dna, x, y)
        fearful = self.simulation._add_agent(encode_dna("grazer", empathy=7, patience=7, fear=7, sight=1), 5, 5)
        fearless = self.simulation._add_agent(encode_dna("grazer", empathy=7, patience=7, fear=0, sight=1), 4, 5)
        self.simulation._index_cells()
        herd = ACTIONS.index("herd")
        self.assertIn(herd, self.simulation._sense(fearful)[1])       # one step from the middle is too far
        self.assertNotIn(herd, self.simulation._sense(fearless)[1])   # two steps out is near enough
        fearless.x = 1
        self.simulation._index_cells()
        self.assertIn(herd, self.simulation._sense(fearless)[1])      # but not five

    def test_the_fearful_feel_their_herd_most(self):
        fearful = self.simulation._add_agent(encode_dna("grazer", empathy=7, patience=7, fear=7), 1, 1)
        fearless = self.simulation._add_agent(encode_dna("grazer", empathy=7, patience=7, fear=0), 3, 3)
        among = {"kin": 0, "others": 0, "threats": 0, "herd": True, "herd_near": 3}
        self.assertAlmostEqual(self.simulation._comfort(fearful, among), 3 * self.simulation._comfort(fearless, among))

    def test_herd_animals_enjoy_their_herd_and_feel_its_absence(self):
        herder = self.simulation._add_agent(encode_dna("grazer", empathy=7, patience=7), 1, 1)
        loner = self.simulation._add_agent(encode_dna("grazer", empathy=0, patience=7), 3, 3)
        among = {"kin": 0, "others": 0, "threats": 0, "herd": True, "herd_near": 3}
        alone = {**among, "herd": False, "herd_near": 0}
        self.assertGreater(self.simulation._comfort(herder, among), self.simulation._comfort(herder, alone))
        self.assertLess(self.simulation._comfort(herder, alone), 0)
        self.assertEqual(self.simulation._comfort(loner, among), self.simulation._comfort(loner, alone))

    def test_a_herd_calms_fear_of_hunters(self):
        herder = self.simulation._add_agent(encode_dna("grazer", empathy=7, patience=7, fear=7), 1, 1)
        threatened = {"kin": 0, "others": 0, "threats": 2, "herd": True}
        self.assertGreater(self.simulation._comfort(herder, {**threatened, "herd_near": 3}),
                           self.simulation._comfort(herder, {**threatened, "herd_near": 0}))

    def test_a_herd_steadies_morale(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        dna = encode_dna("grazer", empathy=7, patience=7, fear=7)
        striker = simulation._add_agent(encode_dna("predator"), 20, 20)
        alone = simulation._add_agent(dna, 5, 5)
        herded = simulation._add_agent(dna, 15, 15)
        for x, y in ((14, 15), (16, 15), (15, 16)):
            simulation._add_agent(dna, x, y)
        simulation._index_cells()
        def breaks(victim):
            broke = 0
            for _ in range(2000):
                victim.panicked = False
                simulation._struck(victim, striker)
                broke += victim.panicked
                victim.mind["fear"] = 7.0
            return broke / 2000
        self.assertAlmostEqual(breaks(alone), 0.3, delta=0.04)
        self.assertAlmostEqual(breaks(herded), 0.3 * (1 - 0.6), delta=0.04)

    def test_striking_a_herd_mate_alarms_the_herd_and_rouses_those_beside_the_attacker_to_defend(self):
        dna = encode_dna("grazer", empathy=7, patience=7, sight=7, strength=7)  # strong enough to defend
        victim = self.simulation._add_agent(dna, 5, 5)
        beside = self.simulation._add_agent(dna, 6, 6)  # alongside the attacker, two cells from the victim
        near = self.simulation._add_agent(dna, 3, 5)
        far = self.simulation._add_agent(dna, 9, 9)
        predator = self.simulation._add_agent(encode_dna("predator", weapons=0), 6, 5)
        self.simulation._index_cells()
        self.simulation._attack(predator, victim)
        self.assertEqual((beside.alarm, near.alarm, far.alarm), (predator.identifier, predator.identifier, None))
        self.assertEqual((beside.herd_assailant, near.herd_assailant), (predator.identifier, None))

        senses, options, situation = self.simulation._sense(beside)
        self.assertEqual(senses[SENSES.index("alarm")], 1)
        self.assertIn(ACTIONS.index("defend"), options)
        beside.reward = 0
        self.simulation._act(beside, "defend", situation)
        self.assertGreater(beside.reward, 0.4)  # defending its herd pleases it
        senses, options, situation = self.simulation._sense(near)
        self.assertIn(ACTIONS.index("flee"), options)  # the alarm: flight from the attacker
        self.assertIs(situation["hunter"], predator)
        self.simulation._think(near)
        self.assertIsNone(near.alarm)

    def test_a_pack_strikes_harder_and_is_no_rival_to_its_own(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        dna = encode_dna("predator", empathy=7, patience=7, sight=1)  # its territory: five cells
        hunter = simulation._add_agent(dna, 10, 10)
        prey = simulation._add_agent(encode_dna("grazer", size=7), 10, 11)
        simulation._index_cells()
        alone = simulation._damage(hunter, prey)
        prey.health = prey.max_health
        simulation._attack(hunter, prey)
        self.assertAlmostEqual(prey.max_health - prey.health, alone)
        simulation._add_agent(dna, 11, 11)  # a pack-mate alongside the prey
        simulation._index_cells()
        prey.health = prey.max_health
        simulation._attack(hunter, prey)
        self.assertAlmostEqual(prey.max_health - prey.health, alone * 1.5)
        for x in range(4, 9):
            simulation._add_agent(dna, x, 10)
        simulation._index_cells()
        self.assertEqual(simulation._territory(hunter), (0.0, 0.0))  # however large, its pack is no rival
        for x, y in [(x, 10) for x in range(11, 17)] + [(x, 9) for x in range(12, 15)]:  # but another kind is
            simulation._add_agent(dna ^ (0xFFFF << 48), x, y)
        simulation._index_cells()
        self.assertGreater(simulation._territory(hunter)[0], 0)

    def test_each_species_weighs_the_food_it_eats_against_the_mouths_sharing_it(self):
        from tools.Simulation import PACK_MOUTHS, PREY_MEALS
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        simulation.grass.clear()
        simulation.grass.update({(15, 13): 4, (16, 15): 2})
        grazer = simulation._add_agent(encode_dna("grazer"), 15, 15)
        omnivore = simulation._add_agent(encode_dna("omnivore"), 14, 15)
        predator = simulation._add_agent(encode_dna("predator"), 17, 15)
        simulation._add_agent(encode_dna("plant", temperature=HOT), 15, 17)  # at home on the equator
        simulation._add_agent(encode_dna("grazer"), 15, 18)
        simulation._index_cells()
        simulation.grass_regrowth = 0  # judge the standing grass alone first
        # grazer: a quarter of 6 tufts and a plant (2 meals), at its specialist's 1.25, shared by
        # itself, another grazer, and an omnivore by half
        self.assertAlmostEqual(simulation._food_supply(grazer), 1.25 * (0.25 * 6 + 2) / (1 + 1 + 0.5))
        # omnivore: grass at half value, the plant at three quarters, two grazers as prey in full;
        # only the predator's half a mouth
        self.assertAlmostEqual(simulation._food_supply(omnivore), (0.5 * 0.25 * 6 + 0.75 * 2 + 2 * PREY_MEALS) / (1 + 0.5))
        # predator: no grass for it; two grazers and an omnivore as prey, and no rivals
        self.assertAlmostEqual(simulation._food_supply(predator), 3 * PREY_MEALS)
        # another kind of predator, though it may be hunted, is a rival for the prey, not food
        simulation._add_agent(encode_dna("predator") ^ (0xFFFF << 48), 18, 16)
        simulation._index_cells()
        self.assertAlmostEqual(simulation._food_supply(predator), 3 * PREY_MEALS / 2)
        # a pack-mate shares its kills, so counts as only part of a mouth
        simulation._add_agent(predator.dna, 18, 14)
        simulation._index_cells()
        self.assertAlmostEqual(simulation._food_supply(predator), 3 * PREY_MEALS / (2 + PACK_MOUTHS))

    def test_grass_is_judged_by_what_will_regrow_as_well_as_what_stands(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        simulation.grass.clear()
        grazer = simulation._add_agent(encode_dna("grazer"), 15, 15)
        simulation._index_cells()
        # every cell within six steps where grass grows now, of all the cells it grows in
        cells = sum(simulation.temperature(15 + dy) == MEDIUM for _, dy in nearby_offsets(6))
        expected = 1.25 * (cells * simulation.grass_regrowth / (30 * simulation._grass_reach[-1]) * 150)
        self.assertAlmostEqual(simulation._food_supply(grazer), expected)
        corner = simulation._add_agent(encode_dna("grazer"), 0, 0)  # a corner sees less of the world
        simulation._index_cells()
        self.assertLess(simulation._food_supply(corner), expected / 2)

    @patch("tools.Simulation.SMALL_LITTERS", 1.0)
    @patch("tools.Simulation.NEWBORN_FOOD", 0.0)  # no limit from the energy given
    @patch("tools.Simulation.PREY_PER_PREDATOR", 1.5)
    def test_a_predators_breeding_follows_the_prey_in_its_home_band_which_it_senses(self):
        from tools.Simulation import PREY_PER_PREDATOR
        simulation = Simulation(width=30, height=30, population=0, seed=3)  # temperate bands: rows 6-11 and 18-23
        dna = encode_dna("predator", fertility=7, litter=7)  # at home in the temperate bands
        first = simulation._add_agent(dna, 15, 8)
        second = simulation._add_agent(dna, 15, 9)
        simulation._add_agent(encode_dna("grazer"), 3, 20)  # the southern temperate band: not its home band
        simulation._add_agent(encode_dna("plant"), 4, 7)  # no prey
        simulation._index_cells()
        simulation._population = 3
        breed, home_food = ACTIONS.index("breed"), SENSES.index("home_food")
        self.assertEqual(simulation._home_prey(first), 0.0)
        senses, options, _ = simulation._sense(first)
        self.assertNotIn(breed, options)  # no prey at home: a hunter, like a grazer, doesn't breed below its need
        self.assertEqual(senses[home_food], 0.0)
        for x in (2, 4, 6):  # three prey at home: a grazer and two omnivores, shared by two predators
            simulation._add_agent(encode_dna("grazer" if x == 2 else "omnivore"), x, 10)
        simulation._index_cells()
        self.assertEqual(simulation._home_prey(first), 1.5)
        senses, options, _ = simulation._sense(first)
        self.assertIn(breed, options)
        self.assertEqual(senses[home_food], 0.5)  # half of the twice it needs for a full litter
        omnivore = simulation.agents[5]
        self.assertEqual(simulation._sense(omnivore)[0][home_food], simulation._home_fed(omnivore))  # its own: both foods

        deliver(simulation, first, second, litter=None)

        self.assertEqual(len(simulation.agents) - 7, 3)  # six young, halved: half the prey of a full litter

    @patch("tools.Simulation.SMALL_LITTERS", 1.0)
    @patch("tools.Simulation.NEWBORN_FOOD", 0.0)  # no limit from the energy given
    @patch("tools.Simulation.PLANTS_PER_GRAZER", 0.5)
    def test_a_grazers_breeding_follows_the_plants_in_its_home_band(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)  # temperate bands: rows 6-11 and 18-23
        dna = encode_dna("grazer", fertility=7, litter=7)  # at home in the temperate bands; a litter of six
        first = simulation._add_agent(dna, 15, 8)
        second = simulation._add_agent(dna, 15, 9)
        simulation._add_agent(encode_dna("plant"), 3, 20)  # the southern temperate band: not its home band
        simulation._index_cells()
        simulation._population = 2
        breed = ACTIONS.index("breed")
        self.assertEqual(simulation._home_plants(first), 0.0)
        self.assertNotIn(breed, simulation._sense(first)[1])
        simulation._add_agent(encode_dna("plant"), 3, 7)
        simulation._index_cells()
        self.assertEqual(simulation._home_plants(first), 0.5)  # one plant shared by two grazers
        self.assertIn(breed, simulation._sense(first)[1])

        deliver(simulation, first, second, litter=None)

        self.assertEqual(len(simulation.agents) - 4, 3)  # six young, halved: half the plants of a full litter

    @patch("tools.Simulation.SMALL_LITTERS", 1.0)
    @patch("tools.Simulation.FOOD_NEEDED", 3.0)  # food limiting breeding, as it may be set to
    @patch("tools.Simulation.NEWBORN_FOOD", 0.0)  # no limit from the energy given
    @patch("tools.Simulation.PLANTS_PER_GRAZER", 0.0)  # no plants about: breeding left to the rest
    def test_scarce_food_stops_breeding_and_plenty_allows_full_litters(self):
        dna = encode_dna("grazer", fertility=7, litter=7)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()
        breed = ACTIONS.index("breed")
        self.simulation.grass_regrowth = 0  # a barren world
        self.assertNotIn(breed, self.simulation._sense(first)[1])  # no grass: no young to feed them on
        lay_plenty_of_grass(self.simulation)
        self.assertIn(breed, self.simulation._sense(first)[1])
        deliver(self.simulation, first, second, litter=None)
        self.assertEqual(len(self.simulation.agents), 2 + 6)

    @patch("tools.Simulation.SMALL_LITTERS", 1.0)
    @patch("tools.Simulation.FOOD_NEEDED", 3.0)  # food limiting breeding, as it may be set to
    @patch("tools.Simulation.NEWBORN_FOOD", 0.0)  # no limit from the energy given
    @patch("tools.Simulation.PLANTS_PER_GRAZER", 0.0)  # no plants about: breeding left to the rest
    def test_a_litter_shrinks_as_food_runs_short(self):
        dna = encode_dna("grazer", fertility=7, litter=7)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()
        with patch.object(Simulation, "_food_supply", return_value=4.5):  # three quarters of plenty
            deliver(self.simulation, first, second, litter=None)
        self.assertEqual(len(self.simulation.agents), 2 + 5)  # six young at three quarters: 4.5, rounded up

    def test_a_distant_hunter_is_watched_but_not_fled(self):
        grazer = self.simulation._add_agent(encode_dna("grazer", sight=7), 5, 6)
        predator = self.simulation._add_agent(encode_dna("predator"), 5, 2)
        self.simulation._index_cells()
        senses, options, situation = self.simulation._sense(grazer)
        self.assertIs(situation["hunter"], predator)  # it is seen
        self.assertGreater(senses[SENSES.index("threat_closeness")], 0)
        self.assertNotIn(ACTIONS.index("flee"), options)  # but four steps off is no reason to run

    def test_a_herd_keeping_watch_flees_only_hunters_closer_still(self):
        dna = encode_dna("grazer", empathy=7, patience=7, sight=7)
        grazer = self.simulation._add_agent(dna, 5, 6)
        self.simulation._add_agent(encode_dna("predator"), 5, 3)  # three steps off
        self.simulation._index_cells()
        flee = ACTIONS.index("flee")
        self.assertIn(flee, self.simulation._sense(grazer)[1])
        for x, y in ((4, 6), (6, 6), (5, 7)):
            self.simulation._add_agent(dna, x, y)
        self.simulation._index_cells()
        self.assertEqual(self.simulation._flee_range(grazer, 3), 1.5)
        self.assertNotIn(flee, self.simulation._sense(grazer)[1])  # its herd watches with it

    @patch("tools.Simulation.KILL_MEAL", 25)  # a meal small enough to measure, short of filling it
    def test_grazers_are_the_plant_and_grass_specialists_and_omnivores_live_on_meat(self):
        grazer = self.simulation._add_agent(encode_dna("grazer", size=7), 1, 1)
        omnivore = self.simulation._add_agent(encode_dna("omnivore", size=7), 3, 3)
        for animal in (grazer, omnivore):
            self.simulation.grass[(animal.x, animal.y)] = 1
            animal.hunger = 90
            self.simulation._act(animal, "eat", {})
        from tools.Simulation import GRASS_MEAL
        self.assertEqual((90 - grazer.hunger, 90 - omnivore.hunger), (1.25 * GRASS_MEAL / 2, GRASS_MEAL / 2 / 2))  # 1.25 and half

        plant = self.simulation._add_agent(encode_dna("plant", size=7), 8, 8)
        omnivore.x, omnivore.y, omnivore.hunger = 8, 9, 90
        self.simulation._index_cells()
        self.simulation._browse(omnivore, plant)
        self.assertEqual(omnivore.hunger, 90 - 0.75 * (PLANT_MEAL + PLANT_MEAL_FOR_SIZE) / 2)  # plants: three quarters

        prey = self.simulation._add_agent(encode_dna("grazer"), 9, 9)
        self.simulation._index_cells()
        omnivore.hunger, prey.health = 90, 1
        self.simulation._attack(omnivore, prey)
        effort = 0.2 * Simulation._damage(omnivore, prey) / omnivore.stomach
        self.assertAlmostEqual(90 - omnivore.hunger + effort, (5 + 25) / 2)  # meat: in full

    @patch("tools.Simulation.SWITCH_SHARE", 0.25)  # the switch measured at a quarter
    def test_hunters_switch_away_from_prey_that_is_scarce_nearby(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        simulation.grass.clear()
        omnivore = simulation._add_agent(encode_dna("omnivore", sight=7), 15, 15)
        grazer = simulation._add_agent(encode_dna("grazer"), 15, 16)
        for x in range(12, 16):  # four plants against one grazer: grazers are a fifth of its food
            simulation._add_agent(encode_dna("plant"), x, 13)
        simulation._index_cells()
        attack = ACTIONS.index("attack")
        self.assertNotIn(attack, simulation._sense(omnivore)[1])
        self.assertNotIn(ACTIONS.index("flee"), simulation._sense(grazer)[1])  # nothing to flee
        simulation._add_agent(encode_dna("grazer"), 16, 16)  # two in six: common enough
        simulation._index_cells()
        self.assertIn(attack, simulation._sense(omnivore)[1])

    def test_primal_fear_grows_with_the_disparity_and_spares_predators(self):
        grazer = self.simulation._add_agent(encode_dna("grazer", size=0), 5, 5)
        brute = self.simulation._add_agent(encode_dna("predator", size=7, strength=7, weapons=7), 5, 8)
        weakling = self.simulation._add_agent(encode_dna("predator", size=0), 9, 9)
        self.assertGreater(Simulation.disparity(grazer, brute), 0.9)
        self.assertEqual(Simulation.disparity(brute, grazer), 0.0)  # the stronger feels none
        self.assertGreater(Simulation.disparity(grazer, brute), Simulation.disparity(grazer, weakling))
        with patch.object(self.simulation.random, "random", return_value=0.8):  # under the grazer's chance, over two thirds
            self.assertTrue(self.simulation._primal_fear(grazer, brute))
            small = self.simulation._add_agent(encode_dna("predator", size=0), 8, 1)
            self.assertFalse(self.simulation._primal_fear(small, brute))  # a predator is two thirds as susceptible
        self.assertTrue(self.simulation._primal_fear(grazer, brute))  # it lasts while a hunter is in sight
        self.assertFalse(self.simulation._primal_fear(grazer, None))  # and passes when none is
        self.assertFalse(grazer.primal_fear)

    def test_in_primal_fear_the_simple_freeze_and_the_clever_flee(self):
        dull = self.simulation._add_agent(encode_dna("grazer", size=0, intellect=0), 1, 1)
        bright = self.simulation._add_agent(encode_dna("grazer", size=0, intellect=7), 8, 6)
        self.simulation._add_agent(encode_dna("predator", size=7, strength=7, weapons=7), 1, 3)
        self.simulation._add_agent(encode_dna("predator", size=7, strength=7, weapons=7), 8, 8)
        self.simulation._index_cells()
        dull.primal_fear = bright.primal_fear = True
        before = (dull.x, dull.y, dull.hunger)
        with patch.object(self.simulation.random, "random", return_value=0.2):  # under the dull's freeze chance
            self.simulation._think_all([dull, bright])
        self.assertEqual(dull.last_action, "froze")
        self.assertEqual((dull.x, dull.y, dull.hunger), before)
        self.assertEqual(bright.last_action, "panicked")

    def test_predators_see_prey_further_than_anything_else(self):
        predator = self.simulation._add_agent(encode_dna("predator", sight=0), 1, 1)  # sees 2, prey 5
        omnivore = self.simulation._add_agent(encode_dna("omnivore", sight=0), 1, 8)
        grazer = self.simulation._add_agent(encode_dna("grazer"), 1, 5)
        self.simulation._index_cells()
        self.assertIs(self.simulation._sense(predator)[2]["quarry"], grazer)
        self.assertIsNone(self.simulation._sense(omnivore)[2]["quarry"])  # three away: beyond an omnivore's two

    def test_a_predators_young_are_predators(self):
        from tools.Simulation import predator_dna
        for species in ("plant", "grazer", "omnivore", "predator"):
            dna = encode_dna(species, size=5)
            made = predator_dna(dna)
            self.assertEqual(decode_body(made)["species"], "predator")
            self.assertEqual(decode_body(made)["size"], 5)
        self.assertEqual(predator_dna(encode_dna("predator")), encode_dna("predator"))

    def test_the_hunting_instinct_draws_a_hungry_predator_to_its_prey(self):
        predator = self.simulation._add_agent(encode_dna("predator", sight=7), 2, 2)
        self.simulation._add_agent(encode_dna("grazer", speed=0), 6, 2)  # along its own row: no further from home
        self.simulation._index_cells()
        predator.brain = Brain.random(random.Random(1))
        hunts = 0
        predator.hunger = 95  # very hungry, but alive
        for seed in range(40):
            self.simulation.random.seed(seed)
            predator.x, predator.y = 2, 2
            self.simulation._index_cells()
            with patch.object(simulation_module, "HUNTING_INSTINCT", 5.0):
                self.simulation._think_all([predator])
            hunts += predator.last_action in ("hunted", "pounced")
        self.assertGreater(hunts, 30)  # an untrained brain, overwhelmingly drawn to the hunt

    def test_deaths_are_counted_by_species_and_cause_and_rewound(self):
        from tools.Insights import death_report_lines
        world = Simulation(width=30, height=30, population=60, seed=4)
        for _ in range(60):
            world.advance()
        counted = sum(sum(causes.values()) for causes in world.death_causes.values())
        self.assertEqual(counted, world.deaths)
        deaths = world.deaths_so_far()
        self.assertEqual(len(deaths), world.deaths)  # every death back to the start
        lines = death_report_lines(deaths)
        total = next(line for line in lines if line.startswith("ALL "))
        self.assertTrue(total.endswith(f"{world.deaths:>10}"))  # the total of totals
        for section in ("WHO KILLED WHOM", "WHAT THE KILLERS WERE DOING", "THE DEADLIEST KILLERS"):
            self.assertTrue(any(line.startswith(section) for line in lines))
        killings = sum(death.killer is not None for death in deaths)
        doing = lines[next(i for i, line in enumerate(lines) if line.startswith("WHAT THE KILLERS")) + 1:]
        self.assertEqual(sum(int(line.split()[-1]) for line in doing[1:doing.index("")]), killings)
        later = {species: dict(causes) for species, causes in world.death_causes.items()}
        world.seek(world.turn - 20)
        self.assertEqual(sum(sum(causes.values()) for causes in world.death_causes.values()), world.deaths)
        while world.turn < world.frontier:
            world.advance()
        self.assertEqual(world.death_causes, later)  # replayed, not counted twice
        self.assertEqual(len(world.deaths_so_far()), world.deaths)
        world.seek(world.turn - 20)
        self.assertEqual(len(world.deaths_so_far()), world.deaths)  # the report as it was then
        world.deploy([], 5, 5)  # a rewound world changed lets its future go, and the deaths in it
        self.assertTrue(all(death.turn <= world.turn for death in world.death_records.values()))

    def test_a_brain_matrix_shows_each_situation_its_species_meets(self):
        from tools.Insights import SCENARIOS, brain_matrix, brain_matrix_lines
        grazer = self.simulation._add_agent(encode_dna("grazer"), 5, 5)
        rows = brain_matrix(grazer)
        self.assertEqual(len(rows), sum("grazer" in species for *_, species in SCENARIOS))
        for _, chances in rows:
            self.assertAlmostEqual(sum(chances.values()), 1.0)
        lines = brain_matrix_lines(grazer)
        self.assertEqual(len(lines), len(rows) + 1)
        self.assertNotIn("hunt", lines[0].split())  # a grazer never meets a chance to hunt

    def test_predators_loom_larger_in_primal_fear_and_feel_half_their_fear(self):
        traits = dict(size=3, strength=3, weapons=0, armour=0)
        grazer = self.simulation._add_agent(encode_dna("grazer", **traits), 5, 5)
        omnivore = self.simulation._add_agent(encode_dna("omnivore", **traits), 6, 5)
        predator = self.simulation._add_agent(encode_dna("predator", **traits), 7, 5)
        self.assertEqual(Simulation.disparity(grazer, omnivore), 0.0)  # an even match
        self.assertGreater(Simulation.disparity(grazer, predator), 0.0)  # the same body, but a predator looms larger
        self.assertEqual(Simulation.disparity(predator, omnivore), 0.0)  # and sizes itself up as larger too
        bigger = self.simulation._add_agent(encode_dna("omnivore", size=5, strength=5, weapons=0, armour=0), 8, 5)
        self.assertAlmostEqual(Simulation.disparity(bigger, predator), 0.0)  # two steps up: the predator's match
        with patch.object(simulation_module, "PREDATOR_LOOMS", 0):
            self.assertEqual(Simulation.disparity(grazer, predator), 0.0)
        fearful = self.simulation._add_agent(encode_dna("predator", fear=7), 9, 9)
        self.assertAlmostEqual(fearful.fearfulness, 0.5)
        self.assertAlmostEqual(self.simulation._add_agent(encode_dna("grazer", fear=7), 9, 8).fearfulness, 1.0)

    def test_hunger_lets_a_predator_ignore_its_fear_until_none_is_left_as_it_starves(self):
        hunter = self.simulation._add_agent(encode_dna("predator", fear=7), 5, 5)
        grazer = self.simulation._add_agent(encode_dna("grazer", fear=7), 6, 5)
        for agent in (hunter, grazer):
            agent.hunger = 0.0
        self.assertAlmostEqual(hunter.fearfulness, 0.5)
        hunter.hunger, grazer.hunger = 60.0, 60.0
        self.assertAlmostEqual(hunter.fearfulness, 0.5 * 0.4)
        self.assertAlmostEqual(grazer.fearfulness, 1.0)  # only predators
        hunter.hunger = 100.0
        self.assertEqual(hunter.fearfulness, 0.0)
        self.assertEqual(hunter.unafraid, 0.0)  # nor will primal fear seize it

    def test_predators_think_twice_as_well_and_good_experiences_sharpen_everyone(self):
        from tools.Simulation import thinking_intellect
        predator = self.simulation._add_agent(encode_dna("predator", intellect=3), 5, 5)
        grazer = self.simulation._add_agent(encode_dna("grazer", intellect=3), 6, 5)
        self.assertAlmostEqual(predator.intelligence, 6 / 7)
        self.assertAlmostEqual(grazer.intelligence, 3 / 7)
        self.assertEqual(self.simulation._add_agent(encode_dna("predator", intellect=5), 7, 7).intelligence, 1.0)  # capped
        self.assertEqual((thinking_intellect(3, "predator"), thinking_intellect(5, "predator"), thinking_intellect(3, "grazer")),
                         (6, 7, 3))
        self.assertEqual(predator.brain.depth, 12)  # twice its doubled intellect
        before = grazer.mind["intellect"]
        self.simulation._experience(grazer, "discovery")
        self.simulation._experience(grazer, "lesson")
        self.assertGreater(grazer.mind["intellect"], before)

    def test_weak_grazers_cannot_strike_back(self):
        weak = self.simulation._add_agent(encode_dna("grazer", strength=5), 5, 5)
        strong = self.simulation._add_agent(encode_dna("grazer", strength=6), 9, 9)
        self.simulation._add_agent(encode_dna("predator"), 5, 6)
        self.simulation._add_agent(encode_dna("predator"), 9, 10)
        self.simulation._index_cells()
        weak.assailant, strong.assailant = self.simulation.agents[2].identifier, self.simulation.agents[3].identifier
        self.assertNotIn(ACTIONS.index("defend"), self.simulation._sense(weak)[1])
        self.assertIn(ACTIONS.index("defend"), self.simulation._sense(strong)[1])

    def test_a_predators_weapons_count_three_times_and_the_best_armed_kill_the_largest_grazer_in_three(self):
        grazer = self.simulation._add_agent(encode_dna("grazer", size=7, armour=7), 5, 5)
        hunter = self.simulation._add_agent(encode_dna("predator", size=7, strength=7, weapons=7), 5, 6)
        omnivore = self.simulation._add_agent(encode_dna("omnivore", size=7, strength=7, weapons=7), 6, 5)
        self.assertGreater(self.simulation._damage(hunter, grazer), self.simulation._damage(omnivore, grazer))
        self.assertGreaterEqual(3 * self.simulation._damage(hunter, grazer), grazer.max_health)
        self.assertLess(2 * self.simulation._damage(hunter, grazer), grazer.max_health)

    def test_in_defence_grazers_bring_half_their_weapons_omnivores_three_quarters_and_predators_all(self):
        from tools.Simulation import STRIKE_FORCE
        target = self.simulation._add_agent(encode_dna("grazer"), 1, 1)
        for i, (species, share) in enumerate((("grazer", 0.5), ("omnivore", 0.75), ("predator", 1.0))):
            animal = self.simulation._add_agent(encode_dna(species, weapons=7), 3 + i, 3)
            force = STRIKE_FORCE * (2 + 9 * animal.might) * (1 - 0.6 * target.toughness)
            armed = self.simulation._armed(animal)
            self.assertAlmostEqual(self.simulation._damage(animal, target), force * (1 + 0.75 * armed))
            self.assertAlmostEqual(self.simulation._damage(animal, target, defending=True), force * (1 + 0.75 * armed * share))

    def test_omnivore_weapons_count_three_quarters_and_grazer_weapons_half_a_predators(self):
        from tools.Simulation import PREDATOR_WEAPONS
        armed = {species: self.simulation._armed(self.simulation._add_agent(encode_dna(species, weapons=7), 5, 5 + i))
                 for i, species in enumerate(("predator", "omnivore", "grazer"))}
        self.assertAlmostEqual(armed["predator"], PREDATOR_WEAPONS)
        self.assertAlmostEqual(armed["omnivore"], 0.75 * PREDATOR_WEAPONS)
        self.assertAlmostEqual(armed["grazer"], 0.5 * PREDATOR_WEAPONS)

    def test_a_grazer_alerts_its_herd_to_a_hunter_and_a_hunter_its_pack_to_a_kill(self):
        lookout = self.simulation._add_agent(encode_dna("grazer", sight=7, trust=7), 2, 2)
        herd_mate = self.simulation._add_agent(lookout.dna, 4, 2)
        hunter = self.simulation._add_agent(encode_dna("predator", sight=7, trust=7), 2, 6)
        pack_mate = self.simulation._add_agent(hunter.dna, 4, 6)
        self.simulation._index_cells()
        options, situation = self.simulation._sense(lookout)[1:]
        self.assertIn(ACTIONS.index("alert"), options)
        self.simulation._act(lookout, "alert", situation)
        self.assertEqual((lookout.last_action, herd_mate.alarm), ("alerted", hunter.identifier))
        self.assertIn(ACTIONS.index("flee"), self.simulation._sense(herd_mate)[1])  # alarmed, it may flee from afar

        kill = self.simulation._add_agent(encode_dna("grazer"), 0, 9)
        kill.health = 0
        self.simulation._die(kill, "killed")
        self.simulation._index_cells()
        options, situation = self.simulation._sense(hunter)[1:]
        self.assertIn(ACTIONS.index("alert"), options)
        self.simulation._act(hunter, "alert", situation)
        self.assertEqual(pack_mate.kill_alert, (0, 9))
        senses, options, situation = self.simulation._sense(pack_mate)
        self.assertIn(ACTIONS.index("feast"), options)
        self.assertIs(situation["kill"], kill)
        self.assertEqual(senses[SENSES.index("kill_alert")], 1.0)  # and it knows it was told
        self.assertEqual(self.simulation._sense(herd_mate)[0][SENSES.index("kill_alert")], 0.0)

    @patch("tools.Simulation.FOOD_NEEDED", 3.0)  # food limiting breeding, as it may be set to
    @patch("tools.Simulation.PREY_PER_PREDATOR", 0.0)  # the prey at home aside
    def test_predators_breed_whatever_the_food_about_them_but_grazers_need_it(self):
        for species in ("predator", "grazer"):
            simulation = Simulation(width=12, height=12, population=0, seed=3)
            simulation.grass.clear()
            simulation.grass_regrowth = 0  # no food of any kind about them
            first = simulation._add_agent(encode_dna(species), 4, 4)
            simulation._add_agent(first.dna, 4, 5)
            simulation._index_cells()
            offered = ACTIONS.index("breed") in simulation._sense(first)[1]
            self.assertEqual(offered, species == "predator", species)

    @patch("tools.Simulation.GRAZER_BREEDING_COST", 1.0)  # a grazer's full cost, as these figures assume
    def test_a_ready_predator_seeks_a_mate_beyond_its_sight_and_hunters_breed_cheaply(self):
        seeker = self.simulation._add_agent(encode_dna("predator", sight=0), 2, 2)  # sees two cells
        mate = self.simulation._add_agent(seeker.dna, 2, 6)  # four away: twice its sight
        grazer = self.simulation._add_agent(encode_dna("grazer", sight=0), 9, 2)
        self.simulation._add_agent(grazer.dna, 9, 6)
        self.simulation._index_cells()
        self.assertIs(self.simulation._sense(seeker)[2]["suitor"], mate)
        self.assertIsNone(self.simulation._sense(grazer)[2]["suitor"])  # only predators
        self.assertAlmostEqual(seeker.breeding_cost, 0.5 * (25 - 15 * seeker.physique("fertility")))
        self.assertAlmostEqual(grazer.breeding_cost, 25 - 15 * grazer.physique("fertility"))

    def test_each_death_records_its_cause_and_any_killer(self):
        from display.ColourGraph import death_text
        hunter = self.simulation._add_agent(encode_dna("predator", size=7, strength=7), 4, 4)
        prey = self.simulation._add_agent(encode_dna("grazer", weapons=0), 4, 5)
        starved = self.simulation._add_agent(encode_dna("grazer"), 9, 9)
        self.simulation._index_cells()
        prey.health = 0.1
        self.simulation._attack(hunter, prey, "hunting")
        self.assertEqual(self.simulation.death_records[prey.identifier],
                         Death("killed", "grazer", self.simulation.turn, hunter.identifier, "predator", "hunting"))
        starved.hunger = 100.5
        self.simulation._die(starved, "starved")
        self.assertEqual(self.simulation.death_records[starved.identifier], Death("starved", "grazer", self.simulation.turn))
        self.assertEqual(death_text(self.simulation, prey.identifier),
                         f"died: killed by #{hunter.identifier:03d} predator, hunting")
        self.assertEqual(death_text(self.simulation, starved.identifier), "died: starved")

        spiny = self.simulation._add_agent(encode_dna("grazer", weapons=7, size=7), 7, 2)
        reckless = self.simulation._add_agent(encode_dna("predator"), 7, 3)
        self.simulation._index_cells()
        reckless.health = 0.1
        self.simulation._attack(reckless, spiny)
        self.assertEqual(self.simulation.death_records[reckless.identifier][3:], (spiny.identifier, "grazer", "spines"))

        defender = self.simulation._add_agent(encode_dna("grazer", weapons=7, size=7, strength=7), 2, 8)
        assailant = self.simulation._add_agent(encode_dna("omnivore"), 2, 9)
        plant = self.simulation._add_agent(encode_dna("plant", size=0), 9, 2)
        browser = self.simulation._add_agent(encode_dna("grazer"), 9, 3)
        self.simulation._index_cells()
        assailant.health = 0.1
        self.simulation._defend(defender, assailant)
        self.assertEqual(self.simulation.death_records[assailant.identifier].activity, "defending")
        plant.health = 0.1
        self.simulation._browse(browser, plant)
        self.assertEqual(self.simulation.death_records[plant.identifier][3:], (browser.identifier, "grazer", "browsing"))

    def test_a_partner_two_cells_away_is_close_enough_to_breed_with(self):
        first = self.simulation._add_agent(encode_dna("predator"), 4, 4)
        second = self.simulation._add_agent(first.dna, 4, 6)
        far = self.simulation._add_agent(first.dna, 8, 8)
        self.simulation._index_cells()
        self.assertIs(self.simulation._sense(first)[2]["mate"], second)
        self.assertIsNone(self.simulation._sense(far)[2]["mate"])

    def test_a_hunter_senses_whether_its_prey_carries_young(self):
        hunter = self.simulation._add_agent(encode_dna("predator", sight=7), 2, 2)
        prey = self.simulation._add_agent(encode_dna("grazer"), 2, 5)
        self.simulation._index_cells()
        self.assertEqual(self.simulation._sense(hunter)[0][SENSES.index("prey_pregnant")], 0.0)
        prey.womb = [(prey.dna, None, None, 0.0, 1)]  # carrying young
        self.assertEqual(self.simulation._sense(hunter)[0][SENSES.index("prey_pregnant")], 1.0)

    def test_a_predator_takes_an_omnivore_for_prey_only_if_it_would_win(self):
        weak = self.simulation._add_agent(encode_dna("predator", size=0, strength=0, weapons=0, armour=0), 5, 5)
        strong = self.simulation._add_agent(encode_dna("predator", size=7, strength=7, weapons=7, armour=7), 6, 5)
        omnivore = self.simulation._add_agent(encode_dna("omnivore", size=5, strength=5, weapons=5, armour=5), 5, 6)
        grazer = self.simulation._add_agent(encode_dna("grazer", size=7, strength=7, armour=7), 6, 6)
        for hunter in (weak, strong):
            hunter.hunts_grazers, hunter.hunts_omnivores = self.simulation._prey_switching(hunter)
        self.assertFalse(Simulation._is_prey(weak, omnivore))  # it would lose
        self.assertTrue(Simulation._is_prey(strong, omnivore))
        self.assertTrue(Simulation._is_prey(weak, grazer))  # grazers are always prey

    def test_predators_enjoy_feasting_and_blows_they_survive(self):
        hunter = self.simulation._add_agent(encode_dna("predator", aggression=0, empathy=0), 5, 6)
        prey = self.simulation._add_agent(encode_dna("grazer", weapons=0, size=7, armour=7), 5, 5)
        self.simulation._index_cells()
        hunter.reward = 0
        self.simulation._attack(hunter, prey)
        self.assertTrue(prey.alive)
        self.assertAlmostEqual(hunter.reward, ATTACK_JOY)  # no joy of fighting at no aggression: just the blow survived
        prey.health = 0
        self.simulation._die(prey, "killed")
        hunter.reward, hunter.hunger = 0, 50
        self.simulation._feast(hunter, prey)
        self.assertEqual(hunter.last_action, "feasted")
        from tools.Simulation import SATED_JOY
        self.assertAlmostEqual(hunter.reward, FEAST_JOY + SATED_JOY * 50 / 100)  # and its hunger, all eased

    def test_any_easing_of_hunger_pleases_by_how_much_is_eased(self):
        from tools.Simulation import SATED_JOY
        grazer = self.simulation._add_agent(encode_dna("grazer"), 5, 5)
        grazer.reward, grazer.hunger = 0, 30
        grazer.relieve(20)
        self.assertAlmostEqual(grazer.reward, SATED_JOY * 0.2)
        grazer.relieve(50)  # only the 10 left to ease
        self.assertEqual(grazer.hunger, 0)
        self.assertAlmostEqual(grazer.reward, SATED_JOY * 0.3)

    def test_predators_breed_with_their_kind(self):
        dna = encode_dna("predator", tolerance=0)
        first = self.simulation._add_agent(dna, 1, 1)
        kin = self.simulation._add_agent(dna ^ (0xFF << 56), 2, 2)  # 25% apart: still its kind
        stranger = self.simulation._add_agent(dna ^ (0xFFFF << 48), 3, 3)  # 50% apart
        self.assertTrue(Simulation._willing(first, kin))
        self.assertFalse(Simulation._willing(first, stranger))

    def test_predators_always_hunt_grazers_and_omnivores(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        simulation.grass.clear()
        predator = simulation._add_agent(encode_dna("predator"), 15, 15)
        for x in range(10, 15):  # all within six steps
            simulation._add_agent(encode_dna("omnivore"), x, 14)
        grazer = simulation._add_agent(encode_dna("grazer"), 16, 15)
        simulation._index_cells()
        self.assertEqual(simulation._prey_switching(predator), (True, True))  # however few grazers are near
        simulation._sense(predator)
        self.assertTrue(simulation._is_prey(predator, grazer))
        rival = simulation._add_agent(encode_dna("predator") ^ (0xFFFF << 48), 14, 15)
        self.assertFalse(simulation._is_prey(predator, rival))  # other kinds only when hungry
        predator.hunger = 95
        self.assertTrue(simulation._is_prey(predator, rival))

    def test_a_blow_that_exhausts_the_striker_leaves_it_starving_and_its_body_rots_once_it_dies(self):
        grazer = self.simulation._add_agent(encode_dna("grazer", size=7, strength=7, weapons=7), 5, 5)
        predator = self.simulation._add_agent(encode_dna("predator", size=7), 5, 6)
        self.simulation._index_cells()
        grazer.hunger = 99.9
        self.simulation._defend(grazer, predator)
        self.assertTrue(grazer.alive)  # past full hunger, but alive: starving wears its health away
        grazer.health = 0.05 * grazer.max_health
        with patch.object(Simulation, "_think_all"), patch.object(Simulation, "_dies_of_age", return_value=False):
            self.simulation._take_turn()
        self.assertFalse(grazer.alive)
        self.assertEqual(grazer.died_turn, self.simulation.turn)
        self.assertIn(grazer.last_action, ("starved", "overheated", "froze"))  # starving, or out of its temperature too
        for _ in range(CORPSE_TURNS):
            self.simulation._take_turn()
        self.assertNotIn(grazer, self.simulation.agents)  # it decayed and was cleared away

    def test_no_body_is_left_without_a_recorded_death(self):
        # Every turn, and with rules that make moving, striking and seeding costly enough to starve.
        harsh = {"GO_EFFORT": 2.0, "STRIKE_EFFORT": 2.0, "SEED_COST": 100, "SEED_HUNGER": 100}
        for rules in ({}, harsh):
            with patch.dict(simulation_module.__dict__, rules):
                for seed in (1, 2, 3):
                    simulation = Simulation(width=40, height=40, population=120, seed=seed)
                    for _ in range(120):
                        simulation.advance()
                        ghosts = [agent for agent in simulation.agents if not agent.alive and agent.died_turn is None]
                        self.assertEqual(ghosts, [], f"seed {seed}, turn {simulation.turn}, rules {rules}")
                        if simulation.ended:
                            break

    def test_a_move_leaves_a_trail_of_the_cells_it_passed_through_that_lasts_the_turn(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        grazer = simulation._add_agent(encode_dna("grazer", speed=7), 10, 10)  # six cells a step
        simulation._index_cells()
        simulation._move(grazer, 16, 10)
        self.assertEqual(grazer.trail, [(10, 10), (11, 10), (12, 10), (13, 10), (14, 10), (15, 10)])
        simulation._move(grazer, 17, 11)
        self.assertEqual(grazer.trail[-1], (16, 10))
        grazer.trail.clear()
        simulation._move(grazer, 1, 11)  # the short way round, east across the edge
        self.assertEqual(grazer.trail, [(x % 30, 11) for x in range(17, 31)])
        copied = grazer.copy()
        simulation.advance()  # a new turn forgets the last one's way
        self.assertNotIn((17, 11), grazer.trail)
        self.assertEqual(copied.trail[0], (17, 11))  # though a copy, as history holds, keeps its own

    def test_starving_is_recorded_and_its_body_handled_however_it_comes(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        grazer = simulation._add_agent(encode_dna("grazer", intellect=7, empathy=7), 10, 10)
        herd_mate = simulation._add_agent(grazer.dna, 11, 10)
        simulation._index_cells()
        simulation._think(grazer)  # a decision to learn from
        herd_mate.x, herd_mate.y = grazer.x + 1, grazer.y  # still alongside, wherever that took it
        simulation._index_cells()
        grazer.hunger = 100.5  # starving, with a turn of health left
        grazer.health = 0.05 * grazer.max_health
        before = grazer.brain.copy()
        deaths = simulation.deaths
        simulation.advance()
        self.assertEqual((grazer.last_action, grazer.died_turn, simulation.deaths), ("starved", 1, deaths + 1))
        self.assertEqual(grazer.meat, 0)  # a starved body leaves nothing to eat
        self.assertIsNone(grazer.brain)
        self.assertNotEqual(before, herd_mate.brain)
        self.assertEqual(herd_mate.seen_failure, 0)  # a slow death: no choice of its own to see fail

        plant = simulation._add_agent(encode_dna("plant", fertility=7), 20, 20)
        plant.hunger = 30
        simulation._index_cells()
        with patch("tools.Simulation.SEED_COST", 100), patch.object(simulation.random, "random", return_value=0.0):
            simulation._seed(plant)
        self.assertEqual((plant.last_action, plant.died_turn), ("seeded", None))  # starving, not dead

    def test_striking_burns_food_by_the_damage_dealt(self):
        target = self.simulation._add_agent(encode_dna("grazer", size=7), 5, 5)
        weak = self.simulation._add_agent(encode_dna("predator"), 5, 6)
        brute = self.simulation._add_agent(encode_dna("predator", strength=7, weapons=7), 6, 5)
        for attacker in (weak, brute):
            attacker.hunger = 50
            damage = self.simulation._damage(attacker, target)
            self.simulation._attack(attacker, target)
            self.assertAlmostEqual(attacker.hunger, max(0, 50 - 5 / 0.5) + 0.2 * damage / 0.5)
        self.assertGreater(brute.hunger, weak.hunger)

    @patch("tools.Simulation.NEWBORN_FOOD", 0.0)  # the full litter, whatever energy is given
    @patch("tools.Simulation.PLANTS_PER_GRAZER", 0.0)  # no plants about: breeding left to the rest
    def test_the_mother_carries_her_young_for_a_gestation_set_by_her_size_then_gives_birth(self):
        self.assertEqual([gestation(size) for size in (0, 1, 4, 7)], [10, 16, 33, 50])
        with patch("tools.Simulation.CLIMATE_DAMAGE", 0):
            simulation = Simulation(width=30, height=30, population=0, seed=3)
            lay_plenty_of_grass(simulation)
            dna = encode_dna("grazer", fertility=7, litter=7)  # the smallest: ten turns, and twelve young
            first, second = simulation._add_agent(dna, 15, 15), simulation._add_agent(dna, 15, 16)
            simulation._index_cells()
            simulation._breed(first, second)
            self.assertEqual((len(first.womb), first.due_turn, len(simulation.agents)), (12, 10, 2))
            self.assertEqual(second.womb, [])  # the parent that chose to breed carries them
            self.assertFalse(first.can_breed)
            with patch.object(Simulation, "_think_all"):
                for _ in range(9):
                    first.hunger = second.hunger = 0
                    simulation._take_turn()
                self.assertEqual(len(simulation.agents), 2)
                simulation._take_turn()
            young = simulation.agents[2:]
            self.assertEqual((len(young), first.womb, first.due_turn), (12, [], None))
            self.assertEqual({child.generation for child in young}, {1})
            reach = [max(abs(child.x - first.x), abs(child.y - first.y)) for child in young]
            self.assertTrue(all(cells <= BIRTH_REACH for cells in reach))
            self.assertEqual(reach.count(0), 9)  # her own square first: nine more of the smallest fit beside her
            self.assertEqual(simulation._population, 14)

    def test_young_die_with_the_mother_carrying_them(self):
        dna = encode_dna("grazer", fertility=7, litter=7)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()
        self.simulation._breed(first, second)
        self.assertTrue(first.womb)
        self.simulation._die(first, "killed")
        self.assertEqual((first.womb, first.due_turn), ([], None))
        self.assertEqual(len(self.simulation.agents), 2)

    @patch("tools.Simulation.BIRTH_REACH", 1)
    @patch("tools.Simulation.CELL_SPACE", 1)  # a square holds one of the smallest
    def test_a_mother_hemmed_in_bears_only_the_young_there_is_room_for(self):
        dna = encode_dna("grazer", fertility=7, litter=7)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        for x, y in ((4, 5), (6, 5), (4, 4), (6, 4), (4, 6), (6, 6)):
            self.simulation._add_agent(encode_dna("plant"), x, y)
        self.simulation._index_cells()

        deliver(self.simulation, first, second, litter=None)

        self.assertEqual([(agent.x, agent.y) for agent in self.simulation.agents[8:]], [(5, 4)])
        self.assertEqual(first.womb, [])  # the rest are lost

    def test_large_bodies_outlive_small_ones_plant_or_animal(self):
        large = self.simulation._add_agent(encode_dna("grazer", size=7), 2, 2)
        small = self.simulation._add_agent(encode_dna("predator", size=0), 3, 3)
        plants = [self.simulation._add_agent(encode_dna("plant", size=size), 1, 1) for size in (0, 7)]

        self.assertEqual([plant.lifespan for plant in plants], [small.lifespan, large.lifespan])
        self.assertGreater(large.lifespan, small.lifespan)
        self.assertEqual((large.lifespan, small.lifespan), (3600, 360))  # ten years and one
        # five years halfway between the smallest and largest bodies, rising steadily either side
        below, above = (self.simulation._add_agent(encode_dna("grazer", size=size), 2, 2) for size in (3, 4))
        self.assertAlmostEqual(below.lifespan, (1 + 4 * 6 / 7) * 360)
        self.assertAlmostEqual(above.lifespan, (5 + 5 * 1 / 7) * 360)

    def test_small_bodies_run_fast_and_short_lives_large_ones_slow_and_long(self):
        small = self.simulation._add_agent(encode_dna("grazer", size=0), 1, 1)
        large = self.simulation._add_agent(encode_dna("grazer", size=7), 2, 2)

        self.assertAlmostEqual(small.metabolism, 1.6)
        self.assertAlmostEqual(large.metabolism, 0.6)
        self.assertGreater(small.burn / small.metabolism, 0)
        self.assertAlmostEqual(small.burn / large.burn, 1.6 / 0.6)  # same build, hungrier sooner
        self.assertLess(small.lifespan, large.lifespan)

    def test_food_needs_grow_with_size_offset_by_metabolism(self):
        small = self.simulation._add_agent(encode_dna("grazer", size=0), 1, 1)
        large = self.simulation._add_agent(encode_dna("grazer", size=7), 2, 2)
        for animal in (small, large):
            animal.hunger = 80
            animal.nourish(20)

        self.assertEqual((small.hunger, large.hunger), (40, 70))  # a meal fills a small stomach more
        food_per_turn = [animal.burn * animal.stomach for animal in (small, large)]
        self.assertGreater(food_per_turn[1], food_per_turn[0])  # but a large body eats more each turn
        self.assertLess(food_per_turn[1], 4 * food_per_turn[0])  # if less than its size alone would ask

    def test_faster_metabolisms_heal_faster(self):
        small = self.simulation._add_agent(encode_dna("grazer", size=0), 1, 1)
        large = self.simulation._add_agent(encode_dna("grazer", size=7), 2, 2)
        for animal in (small, large):
            animal.health = animal.max_health / 2
            animal.heal(0.1)

        self.assertAlmostEqual(small.health - small.max_health / 2, small.max_health * 0.16)
        self.assertAlmostEqual(large.health - large.max_health / 2, large.max_health * 0.06)

    def test_size_strongly_raises_strength_and_armour(self):
        target = self.simulation._add_agent(encode_dna("grazer"), 1, 1)
        small_brute = self.simulation._add_agent(encode_dna("predator", strength=7), 2, 2)
        large_weakling = self.simulation._add_agent(encode_dna("predator", size=7), 3, 3)
        small_shell = self.simulation._add_agent(encode_dna("grazer", armour=7), 4, 4)
        large_bare = self.simulation._add_agent(encode_dna("grazer", size=7), 5, 5)

        self.assertEqual(self.simulation._damage(large_weakling, target),
                         self.simulation._damage(small_brute, target))
        self.assertEqual(self.simulation._damage(target, large_bare), self.simulation._damage(target, small_shell))
        self.assertLess(self.simulation._damage(target, large_bare), self.simulation._damage(target, target))

    def test_young_are_fed_in_absolute_energy_shared_by_stomach_size(self):
        large = [self.simulation._add_agent(encode_dna("grazer", size=7), 5, y) for y in (5, 6)]
        small = [self.simulation._add_agent(encode_dna("grazer", size=0), 1, y) for y in (1, 2)]
        mixed = [self.simulation._add_agent(encode_dna("grazer", size=7, sight=1), 9, 9),
                 self.simulation._add_agent(encode_dna("grazer", size=0, sight=1), 9, 10)]
        for animal in large + small + mixed:
            animal.hunger = 0
        self.simulation._index_cells()

        young = []
        for parents in (large, small, mixed):
            deliver(self.simulation, *parents)
            young.append(self.simulation.agents[-1])

        cost = large[0].breeding_cost
        # each parent gives a share of its own stomach, refilling the young's stomach (which may
        # differ, as genes mutate): a large parent gives four times what a small one does
        for child, parents in zip(young, (large, small, mixed)):
            given = cost * sum(parent.stomach for parent in parents)
            self.assertAlmostEqual(child.hunger, max(0.0, 100 - given / child.stomach))
        self.assertAlmostEqual(large[0].stomach, 4 * small[0].stomach)

    def test_old_age_comes_only_late_in_life(self):
        animal = self.simulation._add_agent(encode_dna("grazer", size=0), 1, 1)

        with patch.object(self.simulation.random, "random", return_value=0.0):
            animal.age = int(0.7 * animal.lifespan)
            self.assertFalse(self.simulation._dies_of_age(animal))
            animal.age += 1
            self.assertTrue(self.simulation._dies_of_age(animal))
        with patch.object(self.simulation.random, "random", return_value=0.02):  # the chance each turn rises with age
            animal.age = int(animal.lifespan)
            self.assertFalse(self.simulation._dies_of_age(animal))
            animal.age = int(1.6 * animal.lifespan)
            self.assertTrue(self.simulation._dies_of_age(animal))

    def test_about_half_live_to_their_lifespan(self):
        simulation = Simulation(width=12, height=12, population=0, seed=8)
        animal = simulation._add_agent(encode_dna("grazer", size=0), 1, 1)
        survivors = 0
        for _ in range(400):
            alive = True
            for age in range(int(0.7 * animal.lifespan), int(animal.lifespan)):
                animal.age = age
                if simulation._dies_of_age(animal):
                    alive = False
                    break
            survivors += alive

        self.assertAlmostEqual(survivors / 400, 0.5, delta=0.08)

    def test_a_very_old_bitzkey_dies_of_old_age(self):
        animal = self.simulation._add_agent(encode_dna("grazer", size=0), 5, 5)
        animal.age = 100 * int(animal.lifespan)  # certain death

        self.simulation._take_turn()
        self.assertEqual((animal.alive, animal.last_action, animal.brain), (False, "old age", None))

        self.simulation._take_turn()  # and it stays dead
        self.assertEqual(animal.last_action, "old age")

    def test_grazers_dont_scavenge(self):
        corpse = self.simulation._add_agent(encode_dna("grazer", size=7), 5, 5)
        corpse.health = 0
        self.simulation._die(corpse, "wounded")
        corpse.died_turn = -2
        grazer = self.simulation._add_agent(encode_dna("grazer", carrion=7, sight=4), 5, 6)
        self.simulation._index_cells()
        senses, options, _ = self.simulation._sense(grazer)
        self.assertNotIn(ACTIONS.index("scavenge"), options)
        self.assertEqual(senses[SENSES.index("carrion_closeness")], 0.0)

    def test_a_body_is_a_fresh_kill_through_its_death_turn_and_three_more_then_carrion(self):
        corpse = self.simulation._add_agent(encode_dna("grazer", size=7), 5, 5)
        corpse.health = 0
        corpse.hunger = 50
        self.simulation._die(corpse, "wounded")
        predator = self.simulation._add_agent(encode_dna("predator", sight=4), 5, 8)
        self.simulation._index_cells()
        self.assertEqual((self.simulation._nearest_kill(predator, 7), self.simulation._nearest_carrion(predator, 7)),
                         (corpse, None))
        self.simulation.turn = 3
        self.simulation._index_cells()
        self.assertIs(self.simulation._nearest_kill(predator, 7), corpse)
        self.simulation.turn = 4
        self.simulation._index_cells()
        self.assertEqual((self.simulation._nearest_kill(predator, 7), self.simulation._nearest_carrion(predator, 7)),
                         (None, corpse))

    @patch("tools.Simulation.KILL_MEAL", 25)  # a meal small enough to measure, short of filling it
    def test_a_hunter_eats_a_fresh_kill_whole_for_three_times_the_kill_meal(self):
        corpse = self.simulation._add_agent(encode_dna("grazer", size=0), 5, 5)
        corpse.health = 0
        self.simulation._die(corpse, "killed")
        predator = self.simulation._add_agent(encode_dna("predator", size=7, sight=4), 5, 8)
        predator.hunger = 90
        self.simulation._index_cells()
        senses, options, situation = self.simulation._sense(predator)
        self.assertIn(ACTIONS.index("feast"), options)
        self.assertGreater(senses[SENSES.index("kill_closeness")], 0)
        while predator.last_action != "feasted":
            self.simulation._act(predator, "feast", situation)
        self.assertAlmostEqual(predator.hunger, 90 - 3 * 25 / predator.stomach)  # the kill meal of the smallest prey
        self.assertEqual((corpse.meat, self.simulation._nearest_kill(predator, 7)), (0.0, None))
        self.simulation._take_turn()
        self.assertNotIn(corpse, self.simulation.agents)  # the body is gone

    def test_only_hunters_feast_and_only_on_what_they_live_on(self):
        for species, eaten in (("grazer", "grazer"), ("omnivore", "predator"), ("predator", "plant")):
            simulation = Simulation(width=12, height=12, population=0, seed=3)
            corpse = simulation._add_agent(encode_dna(eaten), 5, 5)
            corpse.health = 0
            simulation._die(corpse, "killed")
            eater = simulation._add_agent(encode_dna(species, sight=4), 5, 6)
            simulation._index_cells()
            senses, options, _ = simulation._sense(eater)
            self.assertNotIn(ACTIONS.index("feast"), options, f"{species} eating a {eaten}")
            self.assertEqual(senses[SENSES.index("kill_closeness")], 0.0)

    def test_the_top_and_bottom_of_the_map_are_barriers_but_east_and_west_wrap(self):
        simulation = Simulation(width=20, height=20, population=0, seed=2)
        top = simulation._add_agent(encode_dna("grazer", speed=7), 0, 0)
        bottom = simulation._add_agent(encode_dna("predator", speed=7), 19, 19)
        simulation._index_cells()
        for heading in ("go_north", "go_west", "go_south", "go_east") * 50:
            simulation._act(top, heading, {})
            simulation._act(bottom, heading, {})
            self.assertTrue(0 <= top.y < 20 and 0 <= bottom.y < 20 and 0 <= top.x < 20 and 0 <= bottom.x < 20)
        top.x, top.y, bottom.x, bottom.y = 0, 0, 19, 19
        simulation._index_cells()
        self.assertEqual(simulation._within(-1), 0)
        self.assertEqual(simulation._within(20), 19)
        slow = simulation._add_agent(encode_dna("predator", speed=0), 0, 0)  # a cell a step
        hunter = simulation._add_agent(encode_dna("predator"), 1, 1)
        simulation._move_away(slow, hunter)  # it flees up and west: stopped at the top, it wraps round the west
        self.assertEqual((slow.x, slow.y), (19, 0))
        self.assertTrue(all(0 <= y < 20 for _, y in simulation._free_cells_near(bottom)))
        self.assertIn((0, 19), simulation._free_cells_near(bottom))  # east wraps round to the west

    def test_seeds_and_fertile_ground_dont_cross_the_top_or_bottom(self):
        simulation = Simulation(width=20, height=20, population=0, seed=2)
        plant = simulation._add_agent(encode_dna("plant", fertility=7), 10, 0)
        simulation._index_cells()
        offsets = [index for index, (dx, dy) in enumerate(SIGHT_OFFSETS[:SEED_OFFSETS]) if dy < 0]
        with patch.object(simulation.random, "random", return_value=0.0),                 patch.object(simulation.random, "randrange", return_value=offsets[0]):
            simulation._seed(plant)  # northward, off the map
        self.assertEqual((len(simulation.agents), simulation.seeds, plant.children), (1, [], 0))
        corpse = simulation._add_agent(encode_dna("grazer"), 5, 0)
        corpse.health = 0
        simulation._die(corpse, "wounded")
        simulation.grass.clear()
        simulation._grass_rows = list(range(20))  # grass may grow anywhere, to see where it grows faster
        simulation._grass_reach = [float(rows) for rows in range(1, 21)]
        with patch.object(simulation.random, "randrange", return_value=5), \
                patch.object(simulation.random, "random", side_effect=[0.975, 0.0]):  # rows 19 and 0
            simulation._sow(2)
        self.assertEqual((simulation.grass[(5, 19)], simulation.grass[(5, 0)]), (1, 2))

    def test_grass_dies_back_a_tuft_at_a_time_where_it_isnt_medium(self):
        simulation = Simulation(width=20, height=20, population=0, seed=1)
        simulation.grass = {(x, y): 3 for x in range(20) for y in (1, 5)}  # arctic, then temperate
        simulation.turn = 240  # mid-year, mild: the arctic cold, the temperate zone medium
        simulation._update_climate()
        for _ in range(GRASS_DIEBACK):
            simulation.turn += 1
            simulation._wither_grass()
        self.assertEqual(sum(simulation.grass.get((x, 1), 0) for x in range(20)), 20 * 2)  # a tuft each
        self.assertEqual(sum(simulation.grass.get((x, 5), 0) for x in range(20)), 20 * 3)

    def test_corpses_hold_the_energy_their_body_had_left_and_rot(self):
        small = self.simulation._add_agent(encode_dna("grazer", size=0), 2, 2)
        large = self.simulation._add_agent(encode_dna("grazer", size=7), 8, 8)
        starved = self.simulation._add_agent(encode_dna("grazer", size=7), 2, 8)
        small.health = large.health = 0
        large.hunger = 50
        starved.hunger = 100
        self.simulation._die(small, "killed")
        self.simulation._die(large, "wounded")
        self.simulation._die(starved, "starved")

        self.assertEqual((small.meat, large.meat, starved.meat), (15.0, 40.0, 0.0))  # a kill leaves half
        for _ in range(3):  # fresh through these turns, it starts to rot at the third's end
            self.simulation._take_turn()
        self.assertAlmostEqual(large.meat, 40.0 - 40.0 / 27)

    def test_scavengers_eat_carrion_and_their_gut_decides_how_much_it_feeds_them(self):
        corpse = self.simulation._add_agent(encode_dna("grazer", size=7), 5, 5)
        corpse.health = 0
        self.simulation._die(corpse, "wounded")
        corpse.died_turn = -5  # no longer a fresh kill
        picky = self.simulation._add_agent(encode_dna("omnivore", carrion=0, sight=4), 5, 9)
        scavenger = self.simulation._add_agent(encode_dna("predator", carrion=7, sight=4), 5, 1)
        for animal in (picky, scavenger):
            animal.hunger = 60
        self.simulation._index_cells()

        for animal in (picky, scavenger):
            senses, options, situation = self.simulation._sense(animal)
            self.assertIn(ACTIONS.index("scavenge"), options)
            while animal.last_action != "scavenged":
                self.simulation._act(animal, "scavenge", situation)

        # the largest body: eight portions of its 80 meat, each a portion of what is left (ten each, as
        # they come); both are small, with half-size stomachs, so each bite fills them twice as much
        self.assertAlmostEqual(picky.hunger, 60 - 10 * 0.2 / 0.5)
        self.assertAlmostEqual(scavenger.hunger, 60 - 10 * 1.0 / 0.5)
        self.assertAlmostEqual(corpse.meat, 80 - 20)
        self.assertEqual(corpse.portions, 6)

    def test_a_body_is_eaten_a_portion_at_a_time_and_gone_with_the_last(self):
        corpse = self.simulation._add_agent(encode_dna("grazer", size=1), 5, 5)  # two portions
        corpse.health = 0
        self.simulation._die(corpse, "killed")
        self.assertEqual(corpse.portions, 2)
        first = self.simulation._add_agent(encode_dna("predator", size=7), 5, 6)  # a big stomach: not filled
        second = self.simulation._add_agent(encode_dna("predator", size=7), 4, 5)
        self.simulation._index_cells()
        first.hunger = second.hunger = 80
        self.simulation._feast(first, corpse)
        meal = FRESH_KILL_MEALS * (KILL_MEAL + KILL_MEAL_FOR_SIZE * corpse.physique("size"))
        self.assertAlmostEqual(first.hunger, max(0.0, 80 - meal / first.stomach))  # a whole feast: one portion of two
        self.assertEqual(corpse.portions, 1)
        self.assertIsNotNone(corpse.died_turn)
        self.assertEqual(corpse.died_turn, self.simulation.turn)  # still there for the next
        self.simulation._feast(second, corpse)
        self.assertEqual((corpse.portions, corpse.meat), (0, 0.0))
        self.simulation._feast(first, corpse)
        self.assertEqual(first.last_action, "sought a kill")  # nothing left
        self.assertLessEqual(corpse.died_turn, self.simulation.turn - CORPSE_TURNS)  # gone at the turn's end

    def test_species_evolve_along_the_chain_and_plants_stay_plants(self):
        from tools.Simulation import evolved
        expected = {  # (before, the species bit flipped): after
            ("plant", 0): "plant", ("plant", 1): "plant",
            ("grazer", 0): "omnivore", ("grazer", 1): "predator",
            ("omnivore", 0): "predator", ("omnivore", 1): "grazer",
            ("predator", 0): "predator", ("predator", 1): "omnivore",
        }
        for (before, bit), after in expected.items():
            parent = encode_dna(before, size=3)
            child = evolved(parent, parent ^ (1 << (32 + bit)))
            self.assertEqual(decode_body(child)["species"], after, f"{before}, bit {bit}")
            self.assertEqual(decode_body(child)["size"], 3)
        self.assertEqual(evolved(encode_dna("grazer"), encode_dna("grazer")), encode_dna("grazer"))

    def test_a_predator_born_of_grazers_brings_their_eating_to_hunting(self):
        dna = encode_dna("grazer", fertility=7)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()
        with patch.object(simulation_module, "cross", lambda a, b, *rest, **named: encode_dna("predator")):
            self.simulation._breed(first, second)
        young = first.womb[0][2]
        eat, uproot, hunt, feast = (ACTIONS.index(name) for name in ("eat", "uproot", "hunt", "feast"))
        self.assertEqual(list(young.output_weights[hunt]), list(young.output_weights[eat]))
        self.assertEqual(young.output_biases[hunt], young.output_biases[eat])
        self.assertEqual(list(young.output_weights[feast]), list(young.output_weights[uproot]))

    def test_grazers_and_omnivores_browse_plants_for_far_more_than_grass(self):
        plant = self.simulation._add_agent(encode_dna("plant", size=7, temperature=HOT), 5, 5)  # at home on the equator
        grazer = self.simulation._add_agent(encode_dna("grazer", size=7, sight=7), 5, 9)
        predator = self.simulation._add_agent(encode_dna("predator", sight=7), 9, 5)
        grazer.hunger = 90
        self.simulation._index_cells()

        self.assertNotIn(ACTIONS.index("browse"), self.simulation._sense(predator)[1])
        senses, options, situation = self.simulation._sense(grazer)
        self.assertIn(ACTIONS.index("browse"), options)
        with patch("tools.Simulation.GO_EFFORT", 0):  # the meal alone, not the walk to it
            while grazer.last_action != "browsed":
                self.simulation._act(grazer, "browse", situation)

        self.assertEqual(grazer.hunger, 90 - 1.25 * (PLANT_MEAL + PLANT_MEAL_FOR_SIZE) / 2)  # the largest plant, a quarter more
        # to a grazer, into a double stomach
        self.assertEqual(plant.health, plant.max_health - 15)

    def test_a_bitten_plant_is_left_alone_until_it_recovers(self):
        plant = self.simulation._add_agent(encode_dna("plant", size=7), 5, 5)
        grazer = self.simulation._add_agent(encode_dna("grazer", sight=7), 5, 6)
        self.simulation._index_cells()

        self.simulation._browse(grazer, plant)
        self.assertNotIn(ACTIONS.index("browse"), self.simulation._sense(grazer)[1])
        self.simulation._browse(grazer, plant)  # arriving at a recovering plant gets nothing
        self.assertEqual((plant.health, grazer.last_action), (plant.max_health - 15, "sought plants"))

        self.simulation.agents.remove(grazer)  # left alone while it recovers
        for _ in range(8):
            self.simulation._take_turn()
        self.assertEqual(plant.cooldown, 0)

    def test_enough_bites_kill_a_plant_which_leaves_no_meat(self):
        plant = self.simulation._add_agent(encode_dna("plant", size=0), 5, 5)  # 60 health, 15 a bite
        grazer = self.simulation._add_agent(encode_dna("grazer"), 5, 6)
        self.simulation._index_cells()

        for _ in range(4):
            plant.cooldown = 0  # as if it had recovered between bites
            self.simulation._browse(grazer, plant)

        self.assertFalse(plant.alive)
        self.assertEqual((plant.last_action, plant.meat), ("eaten", 0.0))

    def test_well_fed_plants_seed_their_own_kind_nearby(self):
        plant = self.simulation._add_agent(encode_dna("plant", fertility=7, size=3, temperature=HOT), 5, 5)
        self.simulation._index_cells()

        with patch.object(self.simulation.random, "random", return_value=0.0):
            self.simulation._seed(plant)  # onto the equator, where it is at home
        self.assertEqual((len(self.simulation.agents), len(self.simulation.seeds)), (1, 1))  # in the ground
        self.simulation.turn += PLANT_GESTATION - 1
        self.simulation._germinate()
        self.assertEqual(len(self.simulation.agents), 1)  # not yet
        self.simulation.turn += 1
        self.simulation._germinate()

        seedling = self.simulation.agents[-1]
        self.assertIsNot(seedling, plant)
        self.assertEqual(seedling.species, "plant")
        self.assertLessEqual(abs(seedling.x - 5) + abs(seedling.y - 5), 5)
        self.assertEqual((seedling.hunger, seedling.generation, plant.hunger, plant.last_action),
                         (50, 1, 20, "seeded"))

    def test_plants_do_not_seed_when_hungry_or_crowded(self):
        plant = self.simulation._add_agent(encode_dna("plant", fertility=7), 5, 5)
        self.simulation._index_cells()
        with patch.object(self.simulation.random, "random", return_value=0.0):
            plant.hunger = 41
            self.simulation._seed(plant)
            plant.hunger = 0
            for x in range(12):  # a dense checkerboard of plants over the whole world
                for y in range(12):
                    if (x, y) != (5, 5) and (x + y) % 2:
                        self.simulation._add_agent(encode_dna("plant"), x, y)
            self.simulation._index_cells()
            before = len(self.simulation.agents)
            for _ in range(20):
                self.simulation._seed(plant)

        self.assertEqual(len(self.simulation.agents), before)  # no seedlings

    @patch("tools.Simulation.PLANTS_PER_GRAZER", 0.0)  # no plants about: breeding left to the rest
    def test_plants_do_not_take_up_the_animals_population_cap(self):
        lay_plenty_of_grass(self.simulation)
        dna = encode_dna("grazer")
        agent = self.simulation._add_agent(dna, 5, 5)
        mate = self.simulation._add_agent(dna, 5, 6)
        for x in range(0, 12, 3):
            self.simulation._add_agent(encode_dna("plant"), x, 0)
        self.simulation.population_cap = 3
        agent.cooldown = mate.cooldown = 50  # no breeding during the recount turn
        self.simulation._take_turn()  # recounts: 2 animals, 4 plants
        self.assertEqual((self.simulation._population, self.simulation._plants), (2, 4))
        for animal, y in ((agent, 5), (mate, 6)):  # side by side again, ready to breed
            animal.x, animal.y, animal.cooldown, animal.hunger = 5, y, 0, 0
        self.simulation._index_cells()
        self.assertIn(ACTIONS.index("breed"), self.simulation._sense(agent)[1])

    @patch("tools.Simulation.SMALL_LITTERS", 1.0)
    @patch("tools.Simulation.FOOD_NEEDED", 3.0)  # food limiting breeding, as it may be set to
    @patch("tools.Simulation.NEWBORN_FOOD", 0.0)  # no limit from the energy given
    @patch("tools.Simulation.PLANTS_PER_GRAZER", 0.0)  # no plants about: breeding left to the rest
    def test_crowding_shrinks_litters_and_at_its_limit_stops_breeding(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        dna = encode_dna("grazer", fertility=7, litter=7, sight=1)  # a litter of six where there is room; sees five cells
        first = simulation._add_agent(dna, 15, 15)
        second = simulation._add_agent(dna, 15, 16)
        neighbours = [simulation._add_agent(dna, 15 + dx, 12) for dx in (-2, -1, 0, 1, 2)]
        simulation._index_cells()
        simulation._population = len(simulation.agents)

        deliver(simulation, first, second, litter=None)
        self.assertEqual(len(simulation.agents) - 7, 3)  # six young, cut by the five of its kind within its sight

        first.cooldown = second.cooldown = 0
        first.hunger = second.hunger = 0
        for dx in range(-3, 4):  # a crowd of another species neither shrinks litters nor stops breeding
            simulation._add_agent(encode_dna("predator"), 15 + dx, 13)
        simulation._index_cells()
        self.assertGreaterEqual(simulation._crowding(first, second), 10)
        self.assertIn(ACTIONS.index("breed"), simulation._sense(first)[1])
        for dx in range(-3, 4):  # its own kind does
            for y in (17, 18):
                simulation._add_agent(dna, 15 + dx, y)
        simulation._index_cells()
        self.assertGreaterEqual(simulation._crowding(first, second, own_kind=True), 10)
        self.assertNotIn(ACTIONS.index("breed"), simulation._sense(first)[1])

    def test_tolerant_animals_are_at_ease_among_other_species(self):
        tolerant = self.simulation._add_agent(encode_dna("grazer", tolerance=7, sociability=7, fear=7), 1, 1)
        intolerant = self.simulation._add_agent(encode_dna("grazer", tolerance=0, sociability=7, fear=7), 9, 9)
        crowd = {"kin": 0, "others": 3, "threats": 1}

        self.assertGreater(Simulation._comfort(tolerant, crowd), 0)
        self.assertLess(Simulation._comfort(intolerant, crowd), Simulation._comfort(tolerant, crowd))
        alone = {"kin": 0, "others": 0, "threats": 0}
        self.assertEqual(Simulation._comfort(tolerant, alone), Simulation._comfort(intolerant, alone))

    def test_empathy_makes_watching_and_making_a_kill_painful(self):
        hunter = self.simulation._add_agent(encode_dna("predator", strength=7, aggression=7, empathy=7), 5, 5)
        callous = self.simulation._add_agent(encode_dna("predator", strength=7, aggression=7, empathy=0), 1, 1)
        victims = [self.simulation._add_agent(encode_dna("grazer"), x, y) for x, y in ((5, 6), (1, 2))]
        kin_witness = self.simulation._add_agent(encode_dna("grazer", empathy=7), 6, 6)
        cold_witness = self.simulation._add_agent(encode_dna("grazer", empathy=0), 4, 6)
        for victim in victims:
            victim.health = 1
        self.simulation._index_cells()

        self.simulation._attack(hunter, victims[0])
        self.simulation._attack(callous, victims[1])

        self.assertLess(hunter.reward, callous.reward)
        self.assertLess(kin_witness.reward, 0)
        self.assertEqual(cold_witness.reward, 0)



class SocialBackgroundTests(unittest.TestCase):
    """The social background's traits, and the mind's memory, resilience and moodiness."""

    def setUp(self):
        self.simulation = Simulation(width=12, height=12, population=0, seed=3)
        self.simulation.grass.clear()

    def test_loyalty_draws_an_animal_to_its_herd(self):
        loner = self.simulation._add_agent(encode_dna("grazer", sight=1), 2, 2)
        self.simulation._add_agent(loner.dna, 6, 2)  # four cells off: far enough from its herd to herd
        self.simulation._index_cells()
        odds = []
        for loyalty in (0.0, 7.0):
            loner.mind["loyalty"] = loyalty
            with patch.object(Simulation, "_act"):
                self.simulation._think(loner)
            loner.last_senses = None  # nothing learnt between the two
            odds.append(loner.odds["herd"])
        self.assertGreater(odds[1], odds[0])

    def test_the_most_dominant_willing_partner_is_chosen_and_crowding_troubles_the_dominant_less(self):
        lay_plenty_of_grass(self.simulation)
        suitor = self.simulation._add_agent(encode_dna("grazer"), 5, 5)
        first = self.simulation._add_agent(suitor.dna, 5, 6)
        second = self.simulation._add_agent(suitor.dna, 6, 5)
        self.simulation._index_cells()
        for dominant in (first, second):
            first.mind["dominance"], second.mind["dominance"] = 0.0, 0.0
            dominant.mind["dominance"] = 7.0
            self.assertIs(self.simulation._sense(suitor)[2]["mate"], dominant)

        crowded = {"kin": 0, "others": 0, "threats": 0, "rivalry": 1.0}
        comfort = []
        for dominance in (0.0, 7.0):
            suitor.mind["dominance"] = dominance
            comfort.append(Simulation._comfort(suitor, crowded))
        self.assertAlmostEqual(comfort[0], 3 * comfort[1])  # 1.5 times the unease, against half

    def test_trust_sets_how_far_off_an_alert_is_heeded_and_how_much_is_taken_in(self):
        lookout = self.simulation._add_agent(encode_dna("grazer", sight=7), 2, 2)
        near = self.simulation._add_agent(lookout.dna, 3, 2)
        far = self.simulation._add_agent(lookout.dna, 2, 9)  # seven cells off
        hunter = self.simulation._add_agent(encode_dna("predator"), 9, 9)
        near.mind["trust"], far.mind["trust"] = 1.0, 1.0  # heeds herd-mates up to twice its sight times 1/7 off: 6.6 cells
        self.simulation._index_cells()
        self.simulation._alert(lookout, hunter)
        self.assertEqual((near.alarm, far.alarm), (hunter.identifier, None))

        bright = encode_dna("grazer", empathy=7, intellect=7, sight=1, trust=7)
        actor = self.simulation._add_agent(bright, 8, 2)
        actor.last_senses, actor.last_choice = [0.0] * len(SENSES), 0
        watcher = self.simulation._add_agent(bright, 8, 4)
        self.simulation._index_cells()
        self.simulation._witness(actor, -1.0)
        self.assertAlmostEqual(watcher.reward, -simulation_module.SOCIAL_MOOD * 1.5)  # half as much again

    def test_kinship_shuns_herd_mates_and_partners_of_another_social_background(self):
        clannish = self.simulation._add_agent(encode_dna("grazer"), 5, 5)
        stranger = self.simulation._add_agent(clannish.dna ^ (0xFE000000 << 64), 5, 6)  # its social background a quarter apart
        clannish.mind["kinship"] = 7.0
        self.assertFalse(Simulation._willing(clannish, stranger))
        self.assertFalse(self.simulation._herd_mate(clannish, stranger))
        clannish.mind["tolerance"] = 7.0  # tolerance widens it
        self.assertTrue(Simulation._willing(clannish, stranger))
        clannish.mind["tolerance"], clannish.mind["kinship"] = 0.0, 0.0
        self.assertTrue(Simulation._willing(clannish, stranger))

    def test_memory_brings_an_animal_back_to_where_it_fed_and_warns_it_where_it_was_struck(self):
        grazer = self.simulation._add_agent(encode_dna("grazer", memory=7), 5, 5)
        self.simulation.grass[(5, 5)] = 1
        self.simulation._index_cells()
        self.simulation._act(grazer, "eat", {})
        self.assertEqual(grazer.food_memory, (5, 5, self.simulation.turn))
        self.simulation._move(grazer, 5, 9)
        self.simulation.turn += 10
        senses, options, situation = self.simulation._sense(grazer)
        self.assertAlmostEqual(senses[SENSES.index("food_remembered")], 1 - 10 / simulation_module.MEMORY_TURNS)
        self.assertIn(ACTIONS.index("revisit"), options)
        self.simulation._act(grazer, "revisit", situation)
        self.assertEqual((grazer.x, grazer.y, grazer.last_action), (5, 7, "revisited"))
        grazer.mind["memory"] = 0.0  # no memory: nothing to go back to
        senses, options, _ = self.simulation._sense(grazer)
        self.assertEqual(senses[SENSES.index("food_remembered")], 0.0)
        self.assertNotIn(ACTIONS.index("revisit"), options)

        grazer.mind["memory"] = 7.0
        striker = self.simulation._add_agent(encode_dna("predator"), 5, 8)
        self.simulation._index_cells()
        self.simulation._struck(grazer, striker)
        self.assertEqual(self.simulation._sense(grazer)[0][SENSES.index("danger_remembered")], 1.0)

    def test_resilience_eases_fright_and_loss_and_moodiness_sets_how_far_experience_moves_the_mind(self):
        fragile = self.simulation._add_agent(encode_dna("grazer", resilience=0), 3, 3)
        hardy = self.simulation._add_agent(encode_dna("grazer", resilience=7), 8, 8)
        for animal in (fragile, hardy):
            self.simulation._experience(animal, "wounded")
            animal.seen_failure = 1.0
        self.assertAlmostEqual(fragile.mind["fear"], 3 * hardy.mind["fear"])  # 1.5 times the fright, against half
        with patch.object(Simulation, "_think_all"), patch.object(Simulation, "_dies_of_age", return_value=False):
            self.simulation._take_turn()
        memory = simulation_module.SOCIAL_MEMORY
        self.assertAlmostEqual(fragile.seen_failure, memory ** 0.5)
        self.assertAlmostEqual(hardy.seen_failure, memory ** 1.5)

        steady = self.simulation._add_agent(encode_dna("predator", moodiness=0), 3, 8)
        moody = self.simulation._add_agent(encode_dna("predator", moodiness=7), 8, 3)
        for animal in (steady, moody):
            self.simulation._experience(animal, "kill")
        self.assertAlmostEqual(moody.mind["aggression"], 3 * steady.mind["aggression"])



class SquareTests(unittest.TestCase):
    """Squares hold CELL_SPACE spaces, each Bitzkey taking 1 + its size; meat lasts; full hunger wastes."""

    def setUp(self):
        self.simulation = Simulation(width=12, height=12, population=0, seed=3)
        self.simulation.grass.clear()

    def test_a_square_holds_ten_spaces_and_a_full_one_stops_steps_spawning_and_births(self):
        from tools.Simulation import CELL_SPACE
        walker = self.simulation._add_agent(encode_dna("grazer", size=3), 5, 5)  # four spaces
        self.assertEqual((CELL_SPACE, walker.space), (10, 4))
        self.simulation._add_agent(encode_dna("plant", size=7), 6, 5)  # eight of the next square's ten
        self.simulation._index_cells()
        self.assertEqual(self.simulation._room_in((6, 5)), 2)
        self.simulation._move_toward(walker, 8, 5, steps=3)
        self.assertEqual((walker.x, walker.y), (5, 5))  # stopped short of the full square
        self.simulation._act(walker, "walk_east", {})
        self.assertEqual((walker.x, walker.y, walker.last_action), (5, 5, "walked east"))  # it stayed
        small = self.simulation._add_agent(encode_dna("grazer", size=0), 7, 7)
        self.simulation._index_cells()
        self.simulation._move_toward(small, 6, 5, steps=3)
        self.assertEqual((small.x, small.y), (6, 5))  # one space: room enough, sharing the plant's square
        self.assertEqual(self.simulation._spot_near((6, 5), 4), (6, 4))  # the nearest square with room
        self.simulation._spawn_group(encode_dna("grazer", size=7), 3, rows=[1])
        for square in {(agent.x, agent.y) for agent in self.simulation.living}:
            self.assertGreaterEqual(self.simulation._room_in(square), 0)  # none overfilled

    @patch("tools.Simulation.SMALL_LITTERS", 1.0)
    @patch("tools.Simulation.NEWBORN_FOOD", 0.0)
    @patch("tools.Simulation.PLANTS_PER_GRAZER", 0.0)
    def test_several_young_are_born_into_one_square_while_it_has_room(self):
        dna = encode_dna("grazer", fertility=7, litter=7, size=1)  # two spaces each: six young
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()
        deliver(self.simulation, first, second, litter=None)
        young = self.simulation.agents[2:]
        self.assertEqual(len(young), 6)
        self.assertEqual(sum((child.x, child.y) == (5, 5) for child in young), 4)  # 2 + 4 x 2 fills her square
        self.assertEqual(sum((child.x, child.y) == (5, 4) for child in young), 2)  # the rest in the next square, north of her
        self.assertEqual(self.simulation._room_in((5, 5)), 0)

    def test_meat_lasts_twice_as_long_as_plant_food(self):
        from tools.Simulation import MEAT_DIGESTION
        hunter, grazer = (self.simulation._add_agent(encode_dna(kind), x, 5) for kind, x in (("predator", 3), ("grazer", 8)))
        for animal in (hunter, grazer):
            animal.hunger = 50
        hunter.nourish(40 * hunter.stomach, meat=True)
        grazer.nourish(40 * grazer.stomach)
        self.assertEqual((hunter.hunger, hunter.meat_fill, grazer.meat_fill), (10, 40, 0))
        self.assertAlmostEqual(hunter.digest(10), 10 * (1 - 40 / 90 * (1 - MEAT_DIGESTION)))  # its meat share at half the rate
        self.assertAlmostEqual(grazer.digest(10), 10)
        self.assertEqual(MEAT_DIGESTION, 0.5)
        hunter.meat_fill, hunter.hunger = 90, 10  # all its fullness meat: half the hunger a turn
        self.assertAlmostEqual(hunter.digest(10), 5)

    def test_full_hunger_wears_a_tenth_of_its_health_away_each_turn(self):
        from tools.Simulation import STARVING_DAMAGE
        grazer = self.simulation._add_agent(encode_dna("grazer", temperature=1), 5, 3)  # medium: at home here
        self.simulation._add_agent(encode_dna("predator"), 10, 10)
        grazer.hunger = 150
        with patch.object(Simulation, "_think_all"), patch.object(Simulation, "_dies_of_age", return_value=False):
            self.simulation._take_turn()
            self.assertTrue(grazer.alive)
            self.assertEqual(grazer.hunger, 100)
            self.assertAlmostEqual(grazer.health, (1 - STARVING_DAMAGE) * grazer.max_health)
            for _ in range(9):
                self.simulation._take_turn()
        self.assertEqual((grazer.alive, grazer.last_action), (False, "starved"))  # ten turns at full hunger


if __name__ == "__main__":
    unittest.main()


@patch("tools.Simulation.FEEBLE_YOUTH", 0.1)
class FeebleTests(unittest.TestCase):
    """An animal is feeble in its infancy, the first 10% of its lifespan, and its old age, past AGEING of it:
    every body trait, and its health, at half its own."""

    def setUp(self):
        self.simulation = Simulation(width=12, height=12, population=0, seed=3)
        self.simulation.grass.clear()

    def test_an_infant_has_half_its_body_and_health_and_grows_out_of_it(self):
        from tools.Genome import share
        infant = self.simulation._add_agent(encode_dna("predator", size=7, strength=7, speed=6, sight=5), 5, 5)
        lifespan = infant.lifespan
        self.assertTrue(infant.feeble)
        self.assertEqual([infant.trait(name) for name in ("size", "strength", "speed", "sight")], [3, 3, 3, 2])
        self.assertEqual((infant.space, infant.body["size"]), (4, 7))  # its genes are as they were
        self.assertAlmostEqual(infant.max_health, (60 + 90 * share(3)) * 0.5)
        self.assertEqual(infant.health, infant.max_health)  # born with the health of an infant
        infant.age = round(0.1 * lifespan)
        self.assertFalse(infant.feeble)
        self.assertEqual((infant.trait("size"), infant.space, infant.lifespan), (7, 8, lifespan))  # lifespan by its genes
        self.assertAlmostEqual(infant.max_health, 150.0)

    def test_old_age_brings_feebleness_and_its_health_shrinks_to_match(self):
        from tools.Simulation import AGEING
        old = self.simulation._add_agent(encode_dna("grazer", size=7, armour=5), 5, 5)
        old.age = int(AGEING * old.lifespan) - 1
        old.health = old.max_health
        self.assertFalse(old.feeble)
        with patch.object(Simulation, "_think_all"), patch.object(Simulation, "_dies_of_age", return_value=False):
            self.simulation._take_turn()
            self.simulation._take_turn()
        self.assertTrue(old.feeble)
        self.assertEqual(old.trait("armour"), 2)
        self.assertLessEqual(old.health, old.max_health)
        self.assertAlmostEqual(old.max_health, (60 + 90 * 3 / 7) * 0.5)

    def test_plants_are_never_feeble(self):
        seedling = self.simulation._add_agent(encode_dna("plant", size=6, weapons=6), 5, 5)
        self.assertFalse(seedling.feeble)
        self.assertEqual(seedling.trait("weapons"), 6)


class RewardAuditTests(unittest.TestCase):
    """Every reward earned by what came of a choice, not by trying it."""

    def setUp(self):
        self.simulation = Simulation(width=12, height=12, population=0, seed=3)
        self.simulation.grass.clear()

    def test_a_heading_pleases_the_curious_only_if_it_moved(self):
        from tools.Simulation import CURIOSITY_JOY
        walker = self.simulation._add_agent(encode_dna("grazer", curiosity=7), 5, 0)
        self.simulation._index_cells()
        self.simulation._act(walker, "walk_north", {})  # the map's edge
        self.assertEqual((walker.y, walker.reward), (0, 0.0))
        self.simulation._act(walker, "walk_south", {})
        self.assertEqual(walker.y, 1)
        self.assertAlmostEqual(walker.reward, CURIOSITY_JOY)

    def test_an_alert_pleases_only_when_a_herd_mate_takes_it_in(self):
        from tools.Simulation import ALERT_JOY
        dna = encode_dna("grazer", trust=7, empathy=7, patience=7, sight=3)
        lookout = self.simulation._add_agent(dna, 5, 5)
        hunter = self.simulation._add_agent(encode_dna("predator"), 9, 9)
        self.simulation._index_cells()
        self.simulation._alert(lookout, hunter)  # no one to tell
        self.assertEqual(lookout.reward, 0.0)
        mate = self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()
        self.simulation._alert(lookout, hunter)
        self.assertEqual((lookout.reward, mate.alarm), (ALERT_JOY, hunter.identifier))

    def test_learning_weights_make_success_memorable_not_failure(self):
        from tools.Simulation import KILL_FOOD_LEARNING, PLANT_FOOD_LEARNING, lesson_weight
        hunt, eat, rest = (ACTIONS.index(action) for action in ("hunt", "eat", "rest"))
        self.assertEqual((lesson_weight(hunt, 0.3), lesson_weight(hunt, -0.3)), (KILL_FOOD_LEARNING, 1.0))
        self.assertEqual((lesson_weight(eat, 0.3), lesson_weight(eat, -0.3)), (PLANT_FOOD_LEARNING, 1.0))
        self.assertEqual(lesson_weight(rest, 0.3), 1.0)

    def test_a_slow_death_is_not_blamed_on_the_last_choice(self):
        dna = encode_dna("grazer", intellect=7, empathy=7)
        for cause, blamed in (("starved", False), ("froze", False), ("old age", False), ("exhausted", True)):
            dying = self.simulation._add_agent(dna, 5, 5)
            self.simulation._index_cells()
            self.simulation._think(dying)
            dying.reward = 0.0
            dying.last_wellbeing = self.simulation._wellbeing(dying)
            with patch.object(Simulation, "_learn") as learns, patch.object(Simulation, "_witness") as witnessed:
                self.simulation._die(dying, cause)
            outcome = learns.call_args.args[1]
            self.assertAlmostEqual(outcome, -DEATH_PENALTY if blamed else 0.0)
            self.assertEqual(witnessed.called, blamed)

    def test_growing_out_of_infancy_or_into_old_age_is_no_lesson(self):
        from tools.Simulation import AGEING
        with patch("tools.Simulation.FEEBLE_YOUTH", 0.1):
            for start in ("infant", "grown"):
                animal = self.simulation._add_agent(encode_dna("grazer", size=7), 5, 5)
                self.simulation._index_cells()
                animal.age = round(0.1 * animal.lifespan) - 1 if start == "infant" else int(AGEING * animal.lifespan)
                animal.health = 0.8 * animal.max_health
                animal.last_wellbeing = self.simulation._wellbeing(animal)
                feeble = animal.feeble
                with patch.object(Simulation, "_think_all"), patch.object(Simulation, "_dies_of_age", return_value=False):
                    self.simulation._take_turn()
                self.assertNotEqual(animal.feeble, feeble)
                # Only the turn's own hunger and healing move its wellbeing; its health's share jumping with its full
                # health does not.
                self.assertLess(abs(self.simulation._wellbeing(animal) - animal.last_wellbeing), 0.05)
                animal.health = 0  # out of the way of the next
                self.simulation._die(animal, "old age")

    def test_every_way_of_moving_pays_for_the_distance(self):
        from tools.Simulation import GO_EFFORT
        hunter = self.simulation._add_agent(encode_dna("predator", speed=7), 1, 5)
        prey = self.simulation._add_agent(encode_dna("grazer"), 9, 5)
        self.simulation._index_cells()
        hunter.hunger = 10
        self.simulation._act(hunter, "herd", {"herd_middle": (prey.x, prey.y)})  # four cells, at its speed
        self.assertEqual(hunter.x, 5)
        self.assertAlmostEqual(hunter.hunger, 10 + hunter.burn * GO_EFFORT * 3)  # every cell past the first
        prey.hunger = 10
        self.simulation._act(prey, "flee", {"hunter": hunter})
        self.assertAlmostEqual(prey.hunger, 10 + prey.burn * GO_EFFORT * max(0, prey.steps - 1))

    def test_defending_is_damped_by_tolerance_as_attacking_is(self):
        joys = []
        for tolerance in (0, 7):
            defender = self.simulation._add_agent(encode_dna("omnivore", aggression=7, tolerance=tolerance, size=7), 2, 2 + 3 * tolerance // 7)
            assailant = self.simulation._add_agent(encode_dna("grazer", size=7), 3, 2 + 3 * tolerance // 7)
            self.simulation._index_cells()
            self.simulation._defend(defender, assailant)
            joys.append(defender.reward)
        self.assertAlmostEqual(joys[1], joys[0] * 0.25)


class SolitaryOmnivoreTests(unittest.TestCase):
    def test_omnivores_live_alone_in_neither_herd_nor_pack_and_miss_none(self):
        from tools.Simulation import HERD_LONELINESS
        simulation = Simulation(width=12, height=12, population=0, seed=3)
        dna = encode_dna("omnivore", empathy=7, patience=7)
        first = simulation._add_agent(dna, 5, 5)
        second = simulation._add_agent(dna, 6, 5)
        grazers = [simulation._add_agent(encode_dna("grazer", empathy=7, patience=7), x, 9) for x in (2, 3)]
        simulation._index_cells()
        self.assertFalse(simulation._herd_mate(first, second))
        self.assertFalse(simulation._society(first, second))
        self.assertEqual(simulation._herd(first), (False, 0, None))
        self.assertTrue(simulation._herd_mate(*grazers))  # grazers still herd
        _, _, situation = simulation._sense(first)
        lonely = dict(situation, herd=False, herd_near=0)
        with patch("tools.Simulation.HERD_LONELINESS", 1.0):
            alone = simulation._comfort(first, lonely)
            with patch("tools.Simulation.SOLITARY_OMNIVORES", 0):
                self.assertTrue(simulation._herd_mate(first, second))  # the rule off: they herd again
                self.assertLess(simulation._comfort(first, lonely), alone)  # and miss their herd


class CourtingTests(unittest.TestCase):
    def test_courting_is_open_and_pays_only_when_breeding_could_follow(self):
        from tools.Simulation import COURT_JOY
        simulation = Simulation(width=12, height=12, population=0, seed=3)
        dna = encode_dna("grazer", fertility=7, sight=2)
        agent = simulation._add_agent(dna, 2, 5)
        partner = simulation._add_agent(dna, 6, 5)
        simulation._index_cells()
        simulation._population = 2
        court = ACTIONS.index("court")
        with patch("tools.Simulation.PLANTS_PER_GRAZER", 1.0):  # no plants at home: breeding couldn't follow
            self.assertNotIn(court, simulation._sense(agent)[1])
        with patch("tools.Simulation.PLANTS_PER_GRAZER", 0):
            _, options, situation = simulation._sense(agent)
            self.assertIn(court, options)
            simulation._act(agent, "court", situation)
            self.assertGreater(agent.reward, 0.5 * COURT_JOY)  # courted into reach, to breed
            agent.reward = 0.0
            agent.x = 2
            simulation._index_cells()
            _, _, situation = simulation._sense(agent)
            with patch("tools.Simulation.CROWDING_LIMIT", 1), patch.object(Simulation, "_crowding", return_value=1):
                simulation._act(agent, "court", situation)  # into reach, but too crowded to breed there
            self.assertEqual(agent.reward, 0.0)


class PackClimateTests(unittest.TestCase):
    @patch("tools.Simulation.PREY_PER_PREDATOR", 0.0)  # the prey at home aside
    def test_a_pack_courts_and_hunts_only_toward_its_home_temperature_as_a_herd_herds(self):
        from tools.Simulation import HOT
        simulation = Simulation(width=100, height=100, population=0, seed=3)  # cold rows 0-19, medium 20-39, hot 40-59
        dna = encode_dna("predator", temperature=HOT, sight=7, fertility=7)
        predator = simulation._add_agent(dna, 50, 30)  # a step from home
        prey = simulation._add_agent(encode_dna("grazer"), 50, 42)
        partner = simulation._add_agent(dna, 52, 42)
        simulation._index_cells()
        simulation._population = 3
        hunt, court = ACTIONS.index("hunt"), ACTIONS.index("court")
        _, options, situation = simulation._sense(predator)
        self.assertEqual((situation["quarry"], situation["suitor"]), (prey, partner))
        self.assertIn(hunt, options)  # toward home: open
        self.assertIn(court, options)
        prey.y = partner.y = 18  # off toward the cold: closed
        simulation._index_cells()
        _, options, situation = simulation._sense(predator)
        self.assertEqual((situation["quarry"], situation["suitor"]), (prey, partner))
        self.assertNotIn(hunt, options)
        self.assertNotIn(court, options)


class CowardTests(unittest.TestCase):
    def test_an_omnivore_struck_always_breaks_and_a_predator_is_never_overcome(self):
        simulation = Simulation(width=12, height=12, population=0, seed=3)
        bold = simulation._add_agent(encode_dna("omnivore", fear=0), 5, 5)  # fearless, and unhurt
        striker = simulation._add_agent(encode_dna("predator"), 5, 6)
        grazer = simulation._add_agent(encode_dna("grazer", fear=0), 8, 8)
        simulation._index_cells()
        simulation._struck(bold, striker)
        simulation._struck(grazer, striker)
        self.assertTrue(bold.panicked)  # a coward, whatever its fear
        self.assertFalse(grazer.panicked)  # the fearless otherwise never break
        with patch("tools.Simulation.COWARDLY_OMNIVORES", 0):
            bold.panicked = False
            simulation._struck(bold, striker)
            self.assertFalse(bold.panicked)
        small = simulation._add_agent(encode_dna("predator", size=0), 2, 2)
        brute = simulation._add_agent(encode_dna("predator", size=7, strength=7, weapons=7), 2, 4)
        self.assertGreater(Simulation.disparity(small, brute), 0.5)
        with patch.object(simulation.random, "random", return_value=0.0):  # the surest roll
            self.assertFalse(simulation._primal_fear(small, brute))  # a predator is never overcome


class OmnivoreFoodTests(unittest.TestCase):
    @patch("tools.Simulation.SMALL_LITTERS", 1.0)
    @patch("tools.Simulation.NEWBORN_FOOD", 0.0)
    @patch("tools.Simulation.PLANTS_PER_GRAZER", 1.0)
    @patch("tools.Simulation.PREY_PER_PREDATOR", 1.0)
    def test_an_omnivores_litters_follow_plants_and_prey_at_home_half_each(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)  # temperate bands: rows 6-11 and 18-23
        dna = encode_dna("omnivore", fertility=7, litter=7)  # at home in the temperate bands
        first = simulation._add_agent(dna, 15, 8)
        second = simulation._add_agent(dna, 15, 9)
        simulation._index_cells()
        self.assertEqual(simulation._home_fed(first), 0.0)  # nothing to eat at home
        for x in (2, 4, 6, 8):  # two plants each, twice what a grazer needs: half a full litter from plants alone
            simulation._add_agent(encode_dna("plant"), x, 7)
        simulation._index_cells()
        self.assertEqual(simulation._home_fed(first), 0.5)
        for x in (10, 12, 14, 16):  # and as much prey: grazers and a predator
            simulation._add_agent(encode_dna("predator" if x == 16 else "grazer"), x, 11)
        simulation._index_cells()
        self.assertEqual(simulation._home_fed(first), 1.0)
        for plant in [agent for agent in simulation.agents if agent.species == "plant"]:
            plant.health = 0
        simulation._index_cells()
        simulation._population = 2
        self.assertEqual(simulation._home_fed(first), 0.5)  # prey alone: half
        deliver(simulation, first, second, litter=None)
        self.assertEqual(sum(agent.generation == 1 for agent in simulation.agents), 3)  # six by their traits, halved


class SpineTests(unittest.TestCase):
    def test_an_omnivore_has_no_spines_but_an_armed_grazer_does(self):
        simulation = Simulation(width=12, height=12, population=0, seed=3)
        omnivore = simulation._add_agent(encode_dna("omnivore", weapons=7, size=7), 2, 2)
        grazer = simulation._add_agent(encode_dna("grazer", weapons=7, size=7, strength=7), 8, 8)
        first = simulation._add_agent(encode_dna("predator"), 2, 3)
        second = simulation._add_agent(encode_dna("predator"), 8, 9)
        simulation._index_cells()
        simulation._attack(first, omnivore)
        simulation._attack(second, grazer)
        self.assertEqual(first.health, first.max_health)  # struck an omnivore: unstung
        self.assertLess(second.health, second.max_health)  # struck a well-armed grazer: stung


class CorpsePlantTests(unittest.TestCase):
    @patch("tools.Simulation.CORPSE_PLANT", 1.0)  # every body that rots away
    def test_a_body_rotted_away_leaves_the_seed_of_a_plant_at_home_where_it_lay(self):
        from tools.Genome import decode_climate
        from tools.Simulation import CORPSE_TURNS, zone_temperature
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        rotted = simulation._add_agent(encode_dna("grazer"), 5, 8)
        eaten = simulation._add_agent(encode_dna("grazer"), 9, 8)
        simulation._add_agent(encode_dna("predator"), 25, 25)
        simulation._index_cells()
        simulation._die(rotted, "starved")
        simulation._die(eaten, "killed")
        rotted.died_turn = eaten.died_turn = simulation.turn + 1 - CORPSE_TURNS  # both gone by the coming turn's end
        eaten.portions = 0  # eaten to the last portion: nothing left to rot
        with patch.object(Simulation, "_think_all"):
            simulation._take_turn()
        seeds = [seed for seed in simulation.seeds if (seed[0], seed[1]) in ((5, 8), (9, 8))]
        self.assertEqual([(seed[0], seed[1]) for seed in seeds], [(5, 8)])
        self.assertEqual(decode_body(seeds[0][2])["species"], "plant")
        self.assertEqual(decode_climate(seeds[0][2])[0], zone_temperature(8, 30))  # at home where it lay
        self.assertNotIn(rotted, simulation.agents)


class BreedingCostTests(unittest.TestCase):
    def test_grazers_breed_at_half_the_cost_and_predators_take_spines_half_as_hard(self):
        from tools.Simulation import GRAZER_BREEDING_COST, HUNTER_BREEDING_COST, PREDATOR_STING
        simulation = Simulation(width=12, height=12, population=0, seed=3)
        grazer = simulation._add_agent(encode_dna("grazer", fertility=0), 2, 2)
        self.assertEqual((GRAZER_BREEDING_COST, PREDATOR_STING), (0.5, 0.5))
        self.assertEqual(grazer.breeding_cost, 25 * GRAZER_BREEDING_COST)
        self.assertEqual(simulation._add_agent(encode_dna("predator", fertility=0), 4, 4).breeding_cost, 25 * HUNTER_BREEDING_COST)
        stung = []
        for share, x in ((1.0, 2), (PREDATOR_STING, 8)):  # the same blow, its spines taken in full, then half as hard
            spiny = simulation._add_agent(encode_dna("grazer", weapons=7, size=7, strength=7), x, 8)
            striker = simulation._add_agent(encode_dna("predator"), x, 9)
            simulation._index_cells()
            with patch("tools.Simulation.PREDATOR_STING", share):
                simulation._attack(striker, spiny)
            stung.append(striker.max_health - striker.health)
        self.assertGreater(stung[0], 0)
        self.assertAlmostEqual(stung[1], stung[0] * 0.5)


class ScavengerTests(unittest.TestCase):
    def test_a_carrion_eater_scales_into_a_coward_and_away_from_hunting(self):
        simulation = Simulation(width=12, height=12, population=0, seed=3)
        scavenger = simulation._add_agent(encode_dna("grazer", carrion=7, fear=0), 2, 2)  # fearless, but a carrion-eater
        middling = simulation._add_agent(encode_dna("grazer", carrion=3, fear=0), 8, 8)
        striker = simulation._add_agent(encode_dna("predator"), 2, 3)
        simulation._index_cells()
        self.assertEqual((scavenger.scavenger, middling.scavenger), (1.0, 0.0))
        with patch.object(simulation.random, "random", return_value=0.99):  # the least likely roll
            simulation._struck(scavenger, striker)
            simulation._struck(middling, striker)
        self.assertTrue(scavenger.panicked)  # always breaks, like an omnivore
        self.assertFalse(middling.panicked)
        hunter = simulation._add_agent(encode_dna("predator", carrion=7, sight=3), 5, 5)
        simulation._add_agent(encode_dna("grazer"), 5, 8)
        simulation._index_cells()
        hunter.brain = Brain.random(random.Random(1))
        odds = []
        for aversion in (0.0, 1.0):
            hunter.x, hunter.y = 5, 5
            simulation._index_cells()
            simulation.random.seed(1)
            with patch("tools.Simulation.CARRION_AVERSION", aversion), patch.object(Simulation, "_act"):
                simulation._think_all([hunter])
            odds.append(hunter.odds["hunt"])
        self.assertLess(odds[1], odds[0])  # it naturally shies from hunting


class ScarcityTests(unittest.TestCase):
    @patch("tools.Simulation.RARE_BREEDING_COST", 1 / 3)
    @patch("tools.Simulation.GRAZER_RARE_BREEDING_COST", 1 / 9)
    def test_a_scarce_species_breeds_more_cheaply_down_to_a_third_when_rare_and_grazers_to_a_ninth(self):
        from tools.Simulation import scarcity_share
        self.assertEqual((scarcity_share(100), scarcity_share(250)), (1.0, 1.0))
        self.assertEqual((scarcity_share(10), scarcity_share(3)), (1 / 3, 1 / 3))
        self.assertAlmostEqual(scarcity_share(55), 2 / 3)  # halfway
        self.assertEqual((scarcity_share(300, "grazer"), scarcity_share(10, "grazer")), (1.0, 1 / 9))  # grazers' from 300
        self.assertAlmostEqual(scarcity_share(155, "grazer"), 1 / 9 + (8 / 9) * 145 / 290)
        simulation = Simulation(width=12, height=12, population=0, seed=3)
        omnivore = simulation._add_agent(encode_dna("omnivore", fertility=0), 2, 2)
        grazer = simulation._add_agent(encode_dna("grazer", fertility=0), 8, 8)
        usual, usual_grazer = omnivore.breeding_cost, grazer.breeding_cost
        simulation._index_cells()  # one of each: rare
        self.assertAlmostEqual(omnivore.breeding_cost, usual / 3)
        self.assertAlmostEqual(grazer.breeding_cost, usual_grazer / 9)
        for x in range(10):
            for y in range(10):
                simulation._add_agent(encode_dna("omnivore"), x, y + 1) if (x, y + 1) != (2, 2) else None
        simulation._index_cells()  # a hundred omnivores: common again
        self.assertEqual(omnivore.breeding_cost, usual)


class PredatorInfancyTests(unittest.TestCase):
    @patch("tools.Simulation.FEEBLE_YOUTH", 0.1)
    def test_a_predator_is_an_infant_half_as_long(self):
        simulation = Simulation(width=12, height=12, population=0, seed=3)
        predator = simulation._add_agent(encode_dna("predator", size=7), 2, 2)
        grazer = simulation._add_agent(encode_dna("grazer", size=7), 8, 8)
        for animal in (predator, grazer):
            animal.age = int(0.07 * animal.lifespan)  # past a predator's 5%, within the usual 10%
        self.assertFalse(predator.feeble)
        self.assertTrue(grazer.feeble)
        predator.age = int(0.04 * predator.lifespan)
        self.assertTrue(predator.feeble)


class SparseFoodTests(unittest.TestCase):
    @patch("tools.Simulation.FULL_LITTER_FOOD", 4.0)
    @patch("tools.Simulation.PREDATOR_FULL_LITTER_FOOD", 8.0)
    @patch("tools.Simulation.RARE_PREY", 10)
    @patch("tools.Simulation.PREY_PER_PREDATOR", 1.0)
    def test_hunters_need_their_food_full_litters_need_four_times_it_and_the_last_prey_are_spared(self):
        from tools.Simulation import Simulation as World
        simulation = World(width=30, height=30, population=0, seed=3)  # temperate bands: rows 6-11 and 18-23
        dna = encode_dna("predator", fertility=7, litter=7, sight=3)  # at home in the temperate bands
        first = simulation._add_agent(dna, 15, 8)
        second = simulation._add_agent(dna, 15, 9)
        breed = ACTIONS.index("breed")
        far = [simulation._add_agent(encode_dna("grazer"), x, 26) for x in range(12)]  # twelve prey alive, none at home
        simulation._index_cells()
        self.assertNotIn(breed, simulation._sense(first)[1])  # below its need at home: no breeding, as for a grazer
        for x in (2, 4):  # two prey at home for two predators: just its need
            simulation._add_agent(encode_dna("grazer"), x, 10)
        simulation._index_cells()
        self.assertEqual(simulation._home_fed(first), 0.125)  # an eighth of a predator's full litter at the need
        self.assertIn(breed, simulation._sense(first)[1])
        for grazer in far:  # the prey dying out: only the two at home left
            grazer.health = 0
        simulation._index_cells()
        self.assertNotIn(breed, simulation._sense(first)[1])  # under ten of its prey alive: it spares them


class PredatorGlutTests(unittest.TestCase):
    @patch("tools.Simulation.PREDATOR_GLUT", 3)  # a glut of three predators, for a small world
    def test_a_glut_of_predators_turns_them_on_other_kinds_then_their_own(self):
        simulation = Simulation(width=12, height=12, population=0, seed=3)
        hunter = simulation._add_agent(encode_dna("predator", empathy=7, tolerance=7, patience=7), 5, 5)  # loath to
        kin = simulation._add_agent(hunter.dna, 5, 6)
        other = simulation._add_agent(encode_dna("predator", size=7, strength=7, armour=7, weapons=7, speed=7, sight=7), 6, 5)  # another kind: its ring far apart
        self.assertFalse(Simulation._same_kind(hunter, other))
        hunter.hunger = 50
        for count, other_prey, kin_prey in ((3, False, False), (6, True, False), (8, True, True)):  # at it, twice it, past it
            for extra in [agent for agent in simulation.agents if agent.identifier > 2]:
                extra.health = 0
            for x in range(count - 3):
                simulation._add_agent(encode_dna("predator"), x, 10)
            simulation._index_cells()
            self.assertEqual((Simulation._is_prey(hunter, other), Simulation._is_prey(hunter, kin)), (other_prey, kin_prey), count)


class PredatorFamineTests(unittest.TestCase):
    @patch("tools.Simulation.RARE_PREY", 10)
    def test_predators_turn_on_other_kinds_then_their_own_as_their_prey_die_out(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        hunter = simulation._add_agent(encode_dna("predator", empathy=7, tolerance=7, patience=7), 5, 5)  # loath to
        kin = simulation._add_agent(hunter.dna, 5, 6)
        other = simulation._add_agent(encode_dna("predator", size=7, strength=7, armour=7, weapons=7, speed=7, sight=7), 6, 5)
        prey = [simulation._add_agent(encode_dna("grazer"), x % 25, 20 + x // 25) for x in range(20)]
        hunter.hunger = 50
        for alive, other_prey, kin_prey in ((20, False, False), (10, True, False), (0, True, True)):  # twice Rare prey, it, none
            for grazer in prey[alive:]:
                grazer.health = 0
            simulation._index_cells()
            self.assertEqual((Simulation._is_prey(hunter, other), Simulation._is_prey(hunter, kin)), (other_prey, kin_prey), alive)


@patch("tools.Simulation.FEEBLE_YOUTH", 0.1)
class NursingTests(unittest.TestCase):
    def setUp(self):
        self.simulation = Simulation(width=12, height=12, population=0, seed=3)

    @patch("tools.Simulation.SMALL_LITTERS", 1.0)
    @patch("tools.Simulation.PLANTS_PER_GRAZER", 0.0)
    def test_parents_pay_at_least_what_their_young_need(self):
        from tools.Simulation import NEWBORN_FOOD
        dna = encode_dna("grazer", fertility=7, litter=7, size=4)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        for parent in (first, second):
            parent.age = round(parent.lifespan / 2)  # grown
        self.simulation._index_cells()
        self.simulation._population = 2
        cheap = first.breeding_cost
        self.simulation._breed(first, second)
        litter = len(first.womb)
        self.assertGreater(litter * NEWBORN_FOOD / 2, cheap)  # a litter needs more than breeding costs them
        self.assertAlmostEqual(first.hunger, litter * NEWBORN_FOOD / 2)  # so each paid its half of the young's food
        for young in first.womb:  # and each young starts on NEWBORN_FOOD of a stomach of its parents'
            self.assertAlmostEqual(young[3] / first.stomach, NEWBORN_FOOD)

    def test_a_grown_animal_feeds_the_hungriest_infant_of_its_kind_close_by(self):
        from tools.Simulation import NURSING_LIMIT
        adult = self.simulation._add_agent(encode_dna("grazer", size=4), 5, 5)
        adult.age = round(adult.lifespan / 2)
        hungry = self.simulation._add_agent(encode_dna("grazer", size=4), 6, 5)
        fed = self.simulation._add_agent(encode_dna("grazer", size=4), 5, 6)
        stranger = self.simulation._add_agent(encode_dna("omnivore", size=4), 4, 5)  # another species: not its to feed
        far = self.simulation._add_agent(encode_dna("grazer", size=4), 10, 10)
        self.simulation._index_cells()
        self.assertTrue(all(young.infant for young in (hungry, fed, stranger, far)) and not adult.infant)
        hungry.hunger, fed.hunger, stranger.hunger, far.hunger = 60, 20, 90, 99
        adult.hunger = 10
        self.simulation._nurse(adult)
        self.assertAlmostEqual(hungry.hunger, 60 - hungry.burn)  # the hungriest close by, a turn's burn
        self.assertEqual((fed.hunger, stranger.hunger, far.hunger), (20, 90, 99))
        self.assertAlmostEqual(adult.hunger, 10 + hungry.burn * hungry.stomach / adult.stomach)  # from its own stomach
        adult.hunger = NURSING_LIMIT - 0.01  # too hungry itself
        self.simulation._nurse(adult)
        self.assertAlmostEqual(hungry.hunger, 60 - hungry.burn)
        self.simulation._nurse(hungry)  # an infant feeds no one
        self.assertEqual(fed.hunger, 20)

    def test_an_infant_keeps_close_to_its_kind_even_a_loners(self):
        dna = encode_dna("omnivore", empathy=7, patience=7, sight=3)
        infant = self.simulation._add_agent(dna, 2, 5)
        grown = [self.simulation._add_agent(dna, x, 5) for x in (4, 5)]
        for adult in grown:
            adult.age = round(adult.lifespan / 2)
        self.simulation._index_cells()
        self.assertTrue(self.simulation._herd_mate(infant, grown[0]))  # a solitary omnivore's infant herds with its kind
        self.assertFalse(self.simulation._herd_mate(grown[0], grown[1]))  # though grown omnivores still don't
        self.assertIn(ACTIONS.index("herd"), self.simulation._sense(infant)[1])  # two cells off their middle: it herds
