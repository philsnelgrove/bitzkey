import math
import random
import unittest
from unittest.mock import patch

from tools.Brain import ACTIONS, SENSES, Brain
from tools.Exchange import Imported
from tools.Genome import encode_dna
from tools.Simulation import CORPSE_TURNS, Simulation, litter_size


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

    def test_breeding_needs_a_ready_mate_and_room(self):
        dna = encode_dna("grazer")
        agent = self.simulation._add_agent(dna, 5, 5)
        mate = self.simulation._add_agent(dna, 5, 6)
        breed = ACTIONS.index("breed")
        self.simulation._index_cells()

        self.assertIn(breed, self.simulation._sense(agent)[1])
        mate.cooldown = 3
        self.assertNotIn(breed, self.simulation._sense(agent)[1])
        mate.cooldown = 0
        for x, y in ((5, 4), (4, 5), (6, 5), (4, 4), (6, 4), (4, 6), (6, 6)):  # every cell around but the mate's
            self.simulation._add_agent(encode_dna("plant"), x, y)
        self.assertNotIn(breed, self.simulation._sense(agent)[1])

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
        self.simulation.grass.update({(5, 5): 1, (2, 2): 1, (7, 7): 1})
        self.simulation._index_cells()
        grass_actions = {ACTIONS.index("eat"), ACTIONS.index("forage")}

        self.assertFalse(grass_actions & set(self.simulation._sense(predator)[1]))
        self.assertEqual(grass_actions & set(self.simulation._sense(grazer)[1]), grass_actions)

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
        self.assertEqual((predator.x, predator.y), (2, 4))
        self.assertNotIn(ACTIONS.index("hunt"), self.simulation._sense(grazer)[1])

    def test_hunter_that_reaches_its_prey_pounces(self):
        predator = self.simulation._add_agent(encode_dna("predator", sight=7, speed=5), 2, 2)
        grazer = self.simulation._add_agent(encode_dna("grazer"), 2, 6)
        self.simulation._index_cells()
        situation = self.simulation._sense(predator)[2]

        self.simulation._act(predator, "hunt", situation)

        self.assertEqual(predator.last_action, "pounced")
        self.assertLess(grazer.health, grazer.max_health)

    def test_prey_flees_a_hunter_it_can_see(self):
        predator = self.simulation._add_agent(encode_dna("predator"), 5, 2)
        grazer = self.simulation._add_agent(encode_dna("grazer", sight=7), 5, 6)
        self.simulation._index_cells()
        senses, options, situation = self.simulation._sense(grazer)

        self.assertIn(ACTIONS.index("flee"), options)
        self.simulation._act(grazer, "flee", situation)
        self.assertGreater(abs(grazer.y - predator.y), 4)
        self.assertNotIn(ACTIONS.index("flee"), self.simulation._sense(predator)[1])

    def test_only_animals_that_can_afford_it_breed(self):
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

    def test_nobody_courts_a_partner_who_cannot_afford_to_breed(self):
        dna = encode_dna("grazer", sight=7, fertility=7)
        suitor = self.simulation._add_agent(dna, 2, 2)
        mate = self.simulation._add_agent(dna, 2, 7)
        mate.hunger = 95
        self.simulation._index_cells()

        self.assertNotIn(ACTIONS.index("court"), self.simulation._sense(suitor)[1])
        mate.hunger = 0
        self.assertIn(ACTIONS.index("court"), self.simulation._sense(suitor)[1])

    def test_population_cap_withdraws_breeding(self):
        dna = encode_dna("grazer")
        agent = self.simulation._add_agent(dna, 5, 5)
        self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()
        self.simulation._population = 2

        self.assertIn(ACTIONS.index("breed"), self.simulation._sense(agent)[1])  # no cap by default
        self.simulation.population_cap = 2
        self.assertNotIn(ACTIONS.index("breed"), self.simulation._sense(agent)[1])

    def test_fertile_parents_breed_cheaply_and_their_child_gets_the_energy(self):
        dna = encode_dna("grazer", fertility=7, libido=7)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()

        self.simulation._breed(first, second, [(5, 4)])

        child = self.simulation.agents[-1]
        self.assertEqual((child.x, child.y), (5, 4))
        self.assertEqual(child.hunger, 100 - 2 * 10)
        self.assertEqual(child.generation, 1)
        self.assertNotIn(child.brain, (first.brain, second.brain))
        self.assertEqual((first.hunger, first.cooldown, first.children, second.children), (10, 5, 1, 1))
        self.assertGreater(first.reward, 0)

    def test_breeding_past_full_hunger_kills_the_parent(self):
        dna = encode_dna("grazer")
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        first.hunger = 90
        self.simulation._index_cells()
        self.simulation._population = 2

        self.simulation._breed(first, second, [(5, 4)])

        self.assertFalse(first.alive)
        self.assertEqual((first.last_action, first.died_turn, first.brain), ("bred to death", 0, None))
        self.assertEqual(self.simulation.deaths, 1)
        self.assertTrue(second.alive)

    def test_child_brain_is_built_from_its_parents_brains(self):
        dna = encode_dna("grazer", fertility=7)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()

        self.simulation._breed(first, second, [(5, 4)])

        child = self.simulation.agents[-1]
        for index, weights in enumerate(child.brain.hidden_weights):
            nearest = min(max(abs(a - b) for a, b in zip(weights, parent.brain.hidden_weights[index]))
                          for parent in (first, second))
            self.assertLess(nearest, 0.3)  # one parent's neuron, plus small mutations

    def test_child_inherits_what_experience_taught_its_parents(self):
        dna = encode_dna("grazer", fertility=7, fear=2, curiosity=5, libido=6)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        first.mind.update(fear=4.0, curiosity=3.0, libido=7.0)
        second.mind.update(fear=3.0, curiosity=5.0, libido=7.0)
        self.simulation._index_cells()

        self.simulation.mind_mutation = 0
        self.simulation._breed(first, second, [(5, 4)])

        child = self.simulation.agents[-1]
        # its mind starts at the average of its parents' current minds, not their genes
        self.assertEqual(child.mind["fear"], 3.5)
        self.assertEqual(child.mind["curiosity"], 4.0)
        self.assertEqual(child.mind["libido"], 7.0)
        self.assertEqual(child.mind["patience"], 0.0)
        # and its dot shows it, the half rounding either way
        self.assertIn(child.inborn["fear"], (3, 4))
        self.assertEqual((child.inborn["curiosity"], child.inborn["libido"]), (4, 7))

    def test_a_parent_gives_its_young_no_more_energy_than_it_has_left(self):
        dna = encode_dna("grazer", fertility=7)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        first.hunger = 96  # has 4 left to give, though breeding costs 10
        self.simulation._index_cells()
        self.simulation._population = 2

        self.simulation._breed(first, second, [(5, 4)])

        self.assertEqual(self.simulation.agents[-1].hunger, 100 - (4 + 10))
        self.assertEqual(first.last_action, "bred to death")

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

    def test_seeking_rebuilds_any_turn_exactly_even_between_checkpoints(self):
        with patch("tools.Simulation.DENSE_HISTORY", 10), patch("tools.Simulation.CHECKPOINTS", 4):
            simulation = Simulation(width=20, height=20, population=30, seed=4)

            def state():  # copies, since the live objects carry on changing
                return ([(agent.identifier, agent.dna, agent.x, agent.y, agent.hunger, agent.health,
                          dict(agent.mind), agent.brain and agent.brain.copy(), agent.meat)
                         for agent in simulation.agents],
                        dict(simulation.grass), simulation.deaths, simulation.random.getstate())

            recorded = {0: state()}
            for turn in range(1, 81):
                simulation.advance()
                recorded[turn] = state()

            self.assertLessEqual(len(simulation.history), 10 + 4 + 2)
            for target in (37, 3, 0, 61, 79, 80, 17, 18):
                simulation.seek(target)
                self.assertEqual(simulation.turn, target)
                self.assertEqual(state(), recorded[target], f"turn {target}")

    def test_old_turns_are_kept_as_evenly_spaced_checkpoints(self):
        with patch("tools.Simulation.DENSE_HISTORY", 10), patch("tools.Simulation.CHECKPOINTS", 4):
            simulation = Simulation(width=20, height=20, population=10, seed=4)
            for _ in range(60):
                simulation.advance()

        turns = [snapshot["turn"] for snapshot in simulation.history]
        older = [turn for turn in turns if turn < 60 - 10]
        self.assertEqual(turns[0], 0)
        self.assertLessEqual(len(older), 4)
        self.assertTrue(all(turn % simulation._checkpoint_stride == 0 for turn in older))
        self.assertEqual(turns[-11:], list(range(50, 61)))

    def test_rewinding_reaches_turn_zero_however_long_the_run(self):
        with patch("tools.Simulation.DENSE_HISTORY", 5), patch("tools.Simulation.CHECKPOINTS", 2):
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
        simulation = Simulation(width=20, height=20, population=10, seed=1)
        for _ in range(5):
            simulation.advance()
        for agent in simulation.living:
            if agent.species != "plant":
                agent.health = 0
                simulation._die(agent, "killed")
        simulation._record()  # the end, as a recorded turn
        simulation.frontier = simulation.turn

        simulation.advance()
        self.assertTrue(simulation.ended)
        self.assertEqual(simulation.turn, 5)
        simulation.seek(2)
        self.assertFalse(simulation.ended)
        for _ in range(10):
            simulation.advance()
        self.assertEqual(simulation.turn, 5)

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

    def test_litter_size_sets_how_many_young_share_the_parents_energy(self):
        dna = encode_dna("grazer", fertility=7, litter=7)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()
        self.simulation._population = 2
        cells = self.simulation._free_cells_near(first)

        self.simulation._breed(first, second, cells)

        young = self.simulation.agents[2:]
        self.assertEqual(len(cells), 7)  # every cell around the parent but its mate's
        self.assertEqual(len(young), 6)
        for child in young:  # the parents give 10% of their small stomachs each, shared six ways
            self.assertAlmostEqual(child.hunger, 100 - 2 * 10 * 0.5 / 6 / child.stomach)
        self.assertEqual((first.children, second.children), (6, 6))

    def test_large_bodies_with_a_low_litter_trait_have_single_young(self):
        dna = encode_dna("grazer", fertility=7, litter=0, size=7)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()

        self.simulation._breed(first, second, self.simulation._free_cells_near(first))

        self.assertEqual(len(self.simulation.agents), 3)

    def test_smaller_bodies_have_bigger_litters(self):
        for litter in range(8):
            sizes = [litter_size([(litter, size)] * 2) for size in range(8)]
            self.assertEqual(sizes, sorted(sizes, reverse=True), f"litter trait {litter}")
        self.assertEqual((litter_size([(7, 0)] * 2), litter_size([(0, 7)] * 2)), (6, 1))
        self.assertEqual(litter_size([(7, 0), (0, 7)]), litter_size([(3, 3)] * 2))  # parents average out
        # predators have half the litters, rounded up
        self.assertEqual([litter_size([(7, 0)] * 2, species) for species in ("grazer", "omnivore", "predator")],
                         [6, 6, 3])
        self.assertEqual(litter_size([(0, 7)] * 2, "predator"), 1)
        self.assertEqual(litter_size([(4, 3)] * 2, "predator"), math.ceil(litter_size([(4, 3)] * 2) / 2))

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
        timid = self.simulation._add_agent(encode_dna("grazer", fear=7), 2, 2)
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

    def test_predators_of_another_kind_are_prey_judged_by_ring_colour(self):
        dna = encode_dna("predator", size=3)
        hunter = self.simulation._add_agent(dna, 1, 1)
        near = self.simulation._add_agent(dna ^ (0x05 << 56), 2, 2)  # cyan byte off by 5: 0.5% apart
        far = self.simulation._add_agent(dna ^ (0xFF << 56), 3, 3)   # cyan byte inverted: 25% apart
        self.assertTrue(self.simulation._same_kind(hunter, near))
        self.assertFalse(self.simulation._same_kind(hunter, far))
        self.assertTrue(self.simulation._is_prey(hunter, far))
        self.assertTrue(self.simulation._is_prey(far, hunter))  # and so each hunts, and flees, the other
        self.simulation._index_cells()
        self.assertIs(self.simulation._look_around(far, 5)[1], hunter)

    def test_cannibalism_waits_on_hunger_past_empathy_tolerance_and_patience(self):
        gentle = self.simulation._add_agent(encode_dna("predator", empathy=7, tolerance=7, patience=7), 1, 1)
        ruthless = self.simulation._add_agent(encode_dna("predator", empathy=0, tolerance=0, patience=0), 2, 2)
        middling = self.simulation._add_agent(encode_dna("predator", empathy=7, tolerance=0, patience=0), 3, 3)
        kin = self.simulation._add_agent(gentle.dna, 4, 4)
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
        corpse = self.simulation._add_agent(encode_dna("grazer"), 0, 0)
        corpse.health = 0
        self.simulation._die(corpse, "wounded")
        self.simulation.grass.clear()
        self.simulation._sow(20000)
        around = [((dx % 12), (dy % 12)) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]  # wrapping round the edge
        fertile = sum(self.simulation.grass.get(cell, 0) for cell in around) / len(around)
        elsewhere = (sum(self.simulation.grass.values()) - fertile * len(around)) / (144 - len(around))
        self.assertAlmostEqual(fertile / elsewhere, 2, delta=0.35)

    def test_a_predators_territory_grows_rivalrous_past_a_few_other_predators(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        predator = simulation._add_agent(encode_dna("predator", empathy=0), 15, 15)  # tolerates just one other
        for count, x in enumerate((10, 11, 12, 18), start=1):
            simulation._add_agent(encode_dna("predator"), x, 15)
            simulation._add_agent(encode_dna("grazer"), x, 16)  # other animals don't count
            simulation._index_cells()
            self.assertAlmostEqual(simulation._territory(predator)[0], max(0, count - 1) / 3)
        simulation._add_agent(encode_dna("predator"), 15, 25)  # beyond its territory
        simulation._index_cells()
        self.assertEqual(simulation._territory(predator)[0], 1)
        self.assertEqual(simulation._sense(predator)[0][SENSES.index("rivalry")], 1)

    def test_the_more_empathetic_a_predator_the_more_rivals_it_tolerates(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        callous = simulation._add_agent(encode_dna("predator", empathy=0), 15, 15)
        caring = simulation._add_agent(encode_dna("predator", empathy=7), 15, 16)
        for x in (12, 13, 17, 18):
            simulation._add_agent(encode_dna("predator"), x, 15)
        simulation._index_cells()
        # each sees the other and four more: five rivals
        self.assertEqual(simulation._territory(callous)[0], 1)    # tolerates one: four too many
        self.assertEqual(simulation._territory(caring)[0], 0)     # tolerates five
        caring.mind["empathy"] = 3.5                         # learned callousness: tolerates three
        self.assertAlmostEqual(simulation._territory(caring)[0], 2 / 3)

    def test_predators_count_omnivores_as_rivals_for_their_territory(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        predator = simulation._add_agent(encode_dna("predator", empathy=0), 15, 15)  # tolerates one rival
        for x in (12, 13, 17):
            simulation._add_agent(encode_dna("omnivore"), x, 15)
        simulation._add_agent(encode_dna("grazer"), 18, 15)  # grazers are prey, not rivals
        simulation._index_cells()
        self.assertAlmostEqual(simulation._territory(predator)[0], 2 / 3)

    def test_omnivores_are_territorial_at_half_strength_and_more_by_empathy(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        callous = simulation._add_agent(encode_dna("omnivore", empathy=0), 15, 15)   # tolerates two
        caring = simulation._add_agent(encode_dna("omnivore", empathy=7), 15, 16)    # tolerates ten
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
        gentle.hunger, gentle.own_rivalry = 60, 0.5  # reluctance only halved, to a half
        self.assertTrue(self.simulation._is_prey(gentle, other))
        gentle.hunger = 40
        self.assertFalse(self.simulation._is_prey(gentle, other))

    def test_omnivores_crowding_a_predator_unsettle_it_but_dont_turn_it_on_its_kin(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        predator = simulation._add_agent(encode_dna("predator", empathy=0), 15, 15)
        for x in (11, 12, 13, 17, 18, 19):
            simulation._add_agent(encode_dna("omnivore"), x, 15)
        simulation._index_cells()
        self.assertEqual(simulation._territory(predator), (1.0, 0.0))  # fully crowded, but not by predators
        gentle = simulation._add_agent(encode_dna("predator", empathy=7, tolerance=7, patience=7), 20, 20)
        gentle.rivalry, gentle.hunger = 1.0, 30
        self.assertFalse(simulation._is_prey(gentle, simulation._add_agent(gentle.dna, 20, 21)))

    def test_omnivores_count_predators_as_rivals_unless_utterly_empathetic(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        omnivore = simulation._add_agent(encode_dna("omnivore", empathy=0), 15, 15)  # tolerates two
        for x in (11, 12, 13, 17, 18):
            simulation._add_agent(encode_dna("predator"), x, 15)
        simulation._index_cells()
        self.assertEqual(simulation._territory(omnivore), (0.5, 0.0))  # five rivals, none its own species
        omnivore.mind["empathy"] = 7.0  # now tolerates ten, and no predator is a rival
        self.assertEqual(simulation._rival_weight(omnivore, simulation.agents[1]), 0)
        omnivore.mind["empathy"] = 3.5
        self.assertEqual(simulation._rival_weight(omnivore, simulation.agents[1]), 0.5)

    def test_an_overcrowded_territory_turns_even_gentle_predators_on_their_kin(self):
        gentle = self.simulation._add_agent(encode_dna("predator", empathy=7, tolerance=7, patience=7), 1, 1)
        kin = self.simulation._add_agent(gentle.dna, 2, 1)
        gentle.hunger = 30
        self.assertFalse(self.simulation._is_prey(gentle, kin))
        gentle.own_rivalry = 0.8  # reluctance worn down to a fifth
        self.assertTrue(self.simulation._is_prey(gentle, kin))

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

    def test_grazers_get_twice_what_omnivores_do_from_grass_and_plants(self):
        grazer = self.simulation._add_agent(encode_dna("grazer", size=7), 1, 1)
        omnivore = self.simulation._add_agent(encode_dna("omnivore", size=7), 5, 5)
        for animal in (grazer, omnivore):
            self.simulation.grass[(animal.x, animal.y)] = 1
            animal.hunger = 90
            self.simulation._act(animal, "eat", {})
        self.assertEqual((grazer.hunger, omnivore.hunger), (90 - 35 / 2, 90 - 35 / 2 / 2))

        plant = self.simulation._add_agent(encode_dna("plant", size=7), 8, 8)
        omnivore.x, omnivore.y, omnivore.hunger = 8, 9, 90
        self.simulation._index_cells()
        self.simulation._browse(omnivore, plant)
        self.assertEqual(omnivore.hunger, 90 - 70 / 2 / 2)

    def test_a_world_sets_its_own_mutation_chances(self):
        stable = Simulation(width=12, height=12, population=0, seed=3, mind_mutation=0, body_mutation=0)
        dna = encode_dna("grazer", fertility=7, fear=3, size=2)
        first, second = stable._add_agent(dna, 5, 5), stable._add_agent(dna, 5, 6)
        stable._index_cells()
        for _ in range(20):
            first.hunger = second.hunger = first.cooldown = second.cooldown = 0
            first.mind = dict(second.mind)  # alike, so the child's mind is exactly theirs
            before = dict(first.mind)  # breeding itself shifts the parents' minds
            stable._breed(first, second, [(5, 4)])
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
        other_kind = self.simulation._add_agent(encode_dna("predator", tolerance=7) ^ (0xFF << 56), 9, 10)
        self.assertFalse(self.simulation._herd_mate(predator, other_kind))  # packs keep to their kind

    def test_the_herd_action_moves_toward_the_middle_of_the_herd(self):
        dna = encode_dna("grazer", empathy=7, patience=7)
        animal = self.simulation._add_agent(dna, 2, 5)
        for x, y in ((5, 4), (5, 6), (6, 5)):
            self.simulation._add_agent(dna, x, y)
        self.simulation._index_cells()
        senses, options, situation = self.simulation._sense(animal)
        self.assertIn(ACTIONS.index("herd"), options)
        self.assertEqual(situation["herd_middle"], (5, 5))
        self.simulation._act(animal, "herd", situation)
        self.assertEqual((animal.x, animal.y, animal.last_action), (4, 5, "herded"))  # two cells a step

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
        dna = encode_dna("grazer", empathy=7, patience=7, sight=7)
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

    def test_a_pack_strikes_harder_but_still_counts_as_rivals(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        dna = encode_dna("predator", empathy=7, patience=7)
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
        self.assertGreater(simulation._territory(hunter)[1], 0)  # an overgrown pack still crowds it

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

    def test_a_parent_hemmed_in_has_single_young(self):
        dna = encode_dna("grazer", fertility=7, litter=7)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        for x, y in ((4, 5), (6, 5), (4, 4), (6, 4), (4, 6), (6, 6)):
            self.simulation._add_agent(encode_dna("plant"), x, y)
        self.simulation._index_cells()

        self.simulation._breed(first, second, self.simulation._free_cells_near(first))

        self.assertEqual([(agent.x, agent.y) for agent in self.simulation.agents[8:]], [(5, 4)])

    def test_plants_outlive_large_animals_which_outlive_small_ones(self):
        plant = self.simulation._add_agent(encode_dna("plant", size=0), 1, 1)
        large = self.simulation._add_agent(encode_dna("grazer", size=7), 2, 2)
        small = self.simulation._add_agent(encode_dna("predator", size=0), 3, 3)

        self.assertGreater(plant.lifespan, large.lifespan)
        self.assertGreater(large.lifespan, small.lifespan)
        self.assertEqual((large.lifespan, small.lifespan), (900, 250))

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
        for parents, cell in ((large, (6, 5)), (small, (2, 1)), (mixed, (10, 9))):
            self.simulation._breed(*parents, [cell])
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
        with patch.object(self.simulation.random, "random", return_value=0.04):  # the chance each turn rises with age
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
        self.simulation._take_turn()
        self.assertAlmostEqual(large.meat, 40.0 - 40.0 / 30)

    def test_scavengers_eat_carrion_and_their_gut_decides_how_much_it_feeds_them(self):
        corpse = self.simulation._add_agent(encode_dna("grazer", size=7), 5, 5)
        corpse.health = 0
        self.simulation._die(corpse, "wounded")
        picky = self.simulation._add_agent(encode_dna("grazer", carrion=0, sight=4), 5, 9)
        scavenger = self.simulation._add_agent(encode_dna("predator", carrion=7, sight=4), 5, 1)
        for animal in (picky, scavenger):
            animal.hunger = 60
        self.simulation._index_cells()

        for animal in (picky, scavenger):
            senses, options, situation = self.simulation._sense(animal)
            self.assertIn(ACTIONS.index("scavenge"), options)
            while animal.last_action != "scavenged":
                self.simulation._act(animal, "scavenge", situation)

        # both are small, with half-size stomachs, so each bite fills them twice as much
        self.assertAlmostEqual(picky.hunger, 60 - 25 * 0.2 / 0.5)
        self.assertAlmostEqual(scavenger.hunger, 60 - 25 * 1.0 / 0.5)
        self.assertAlmostEqual(corpse.meat, 80 - 50)

    def test_grazers_and_omnivores_browse_plants_for_far_more_than_grass(self):
        plant = self.simulation._add_agent(encode_dna("plant", size=7), 5, 5)
        grazer = self.simulation._add_agent(encode_dna("grazer", size=7, sight=7), 5, 9)
        predator = self.simulation._add_agent(encode_dna("predator", sight=7), 9, 5)
        grazer.hunger = 90
        self.simulation._index_cells()

        self.assertNotIn(ACTIONS.index("browse"), self.simulation._sense(predator)[1])
        senses, options, situation = self.simulation._sense(grazer)
        self.assertIn(ACTIONS.index("browse"), options)
        while grazer.last_action != "browsed":
            self.simulation._act(grazer, "browse", situation)

        self.assertEqual(grazer.hunger, 90 - 70 / 2)  # the largest plant gives 70, into a double stomach
        self.assertEqual(plant.health, plant.max_health - 15)

    def test_a_bitten_plant_is_left_alone_until_it_recovers(self):
        plant = self.simulation._add_agent(encode_dna("plant", size=7), 5, 5)
        grazer = self.simulation._add_agent(encode_dna("grazer", sight=7), 5, 6)
        self.simulation._index_cells()

        self.simulation._browse(grazer, plant)
        self.assertNotIn(ACTIONS.index("browse"), self.simulation._sense(grazer)[1])
        self.simulation._browse(grazer, plant)  # arriving at a recovering plant gets nothing
        self.assertEqual((plant.health, grazer.last_action), (plant.max_health - 15, "sought plants"))

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
        plant = self.simulation._add_agent(encode_dna("plant", fertility=7, size=3), 5, 5)
        self.simulation._index_cells()

        with patch.object(self.simulation.random, "random", return_value=0.0):
            self.simulation._seed(plant)

        seedling = self.simulation.agents[-1]
        self.assertIsNot(seedling, plant)
        self.assertEqual(seedling.species, "plant")
        self.assertLessEqual(abs(seedling.x - 5) + abs(seedling.y - 5), 5)
        self.assertEqual((seedling.hunger, seedling.generation, plant.hunger, plant.last_action),
                         (50, 1, 20, "seeded"))

    def test_plants_do_not_seed_when_hungry_crowded_or_too_many(self):
        plant = self.simulation._add_agent(encode_dna("plant", fertility=7), 5, 5)
        self.simulation._index_cells()
        with patch.object(self.simulation.random, "random", return_value=0.0):
            plant.hunger = 41
            self.simulation._seed(plant)
            plant.hunger = 0
            with patch("tools.Simulation.MAX_PLANTS", 1):
                self.simulation._plants = 1
                self.simulation._seed(plant)
            self.simulation._plants = 0
            for x in range(12):  # a dense checkerboard of plants over the whole world
                for y in range(12):
                    if (x, y) != (5, 5) and (x + y) % 2:
                        self.simulation._add_agent(encode_dna("plant"), x, y)
            self.simulation._index_cells()
            before = len(self.simulation.agents)
            for _ in range(20):
                self.simulation._seed(plant)

        self.assertEqual(len(self.simulation.agents), before)  # no seedlings

    def test_plants_do_not_take_up_the_animals_population_cap(self):
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

    def test_crowding_shrinks_litters_and_at_its_limit_stops_breeding(self):
        simulation = Simulation(width=30, height=30, population=0, seed=3)
        dna = encode_dna("grazer", fertility=7, litter=7)  # a litter of six where there is room
        first = simulation._add_agent(dna, 15, 15)
        second = simulation._add_agent(dna, 15, 16)
        neighbours = [simulation._add_agent(encode_dna("predator"), 15 + dx, 12) for dx in (-2, -1, 0, 1, 2)]
        simulation._index_cells()
        simulation._population = len(simulation.agents)

        simulation._breed(first, second, simulation._free_cells_near(first))
        self.assertEqual(len(simulation.agents) - 7, 4)  # six young, cut by the three predators within four cells

        for dx in range(-4, 5):
            simulation._add_agent(encode_dna("grazer"), 15 + dx, 17)
        simulation._index_cells()
        first.cooldown = second.cooldown = 0
        first.hunger = second.hunger = 0
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


if __name__ == "__main__":
    unittest.main()
