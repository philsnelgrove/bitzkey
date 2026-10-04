import math
import random
import unittest
from unittest.mock import patch

from tools.Brain import ACTIONS, Brain
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
        self.assertEqual((small.steps, large.steps), (1, 4))

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

    def test_compatibility_looks_only_at_outer_rings(self):
        calm = self.simulation._add_agent(encode_dna("grazer", size=2), 1, 1)
        wild = self.simulation._add_agent(encode_dna("grazer", size=2) | 0xFFFFFFFF, 2, 2)
        distant = self.simulation._add_agent(
            encode_dna("grazer", size=7, strength=7, armour=7, weapons=7, speed=7, sight=7), 3, 3)

        self.assertTrue(self.simulation._compatible(calm, wild))
        self.assertFalse(self.simulation._compatible(calm, distant))

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
        self.assertEqual((suitor.x, suitor.y), (2, 3))
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

        with patch("tools.Genome.MUTATION_CHANCE", 0):
            self.simulation._breed(first, second, [(5, 4)])

        child = self.simulation.agents[-1]
        self.assertEqual(child.inborn["fear"], 2)
        self.assertEqual(child.mind["fear"], 3.5)        # genes 2, parents learned +2 and +1
        self.assertEqual(child.mind["curiosity"], 4.0)   # genes 5, parents learned -2 and 0
        self.assertEqual(child.mind["libido"], 7.0)      # genes 6 + 1, within the 7 limit
        self.assertEqual(child.mind["patience"], 0.0)    # nothing learned, so the gene value

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
        self.assertTrue(all(child.hunger == 100 - 20 / 6 for child in young))
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
        animal.age = 10 * int(animal.lifespan)

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
