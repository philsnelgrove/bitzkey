import unittest
from unittest.mock import patch

from tools.Brain import ACTIONS
from tools.Genome import encode_dna
from tools.Simulation import CORPSE_TURNS, Simulation


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.simulation = Simulation(width=10, height=10, population=0, seed=3)
        self.simulation.grass.clear()

    def test_animals_get_brains_and_plants_do_not(self):
        plant = self.simulation._add_agent(encode_dna("plant"), 1, 1)
        grazer = self.simulation._add_agent(encode_dna("grazer"), 2, 2)

        self.assertIsNone(plant.brain)
        self.assertIsNotNone(grazer.brain)

    def test_thinking_never_changes_genome(self):
        dna = encode_dna("grazer", intellect=15, libido=15)
        agents = [self.simulation._add_agent(dna, 5, 5), self.simulation._add_agent(dna, 5, 6)]

        for _ in range(10):
            self.simulation._take_turn()

        self.assertTrue(all(agent.dna == dna for agent in agents))

    def test_body_sets_health_pool_and_energy_burn(self):
        small = self.simulation._add_agent(encode_dna("grazer"), 1, 1)
        large = self.simulation._add_agent(encode_dna("grazer", size=15, strength=15, speed=15), 2, 2)

        self.assertEqual((small.max_health, small.health), (60, 60))
        self.assertEqual((large.max_health, large.health), (150, 150))
        self.assertGreater(large.burn, small.burn)
        self.assertEqual((small.steps, large.steps), (1, 4))

    def test_mind_starts_inborn_and_copies_independently(self):
        agent = self.simulation._add_agent(encode_dna("grazer", fear=6), 1, 1)
        duplicate = agent.copy()

        agent.mind["fear"] = 12.0

        self.assertEqual(duplicate.mind["fear"], 6.0)

    def test_experience_reshapes_temperament_more_in_clever_minds(self):
        dull = self.simulation._add_agent(encode_dna("grazer", fear=5), 1, 1)
        bright = self.simulation._add_agent(encode_dna("grazer", fear=5, intellect=15), 2, 2)

        for agent in (dull, bright):
            self.simulation._experience(agent, "wounded")

        self.assertGreater(dull.mind["fear"], 5)
        self.assertGreater(bright.mind["fear"], dull.mind["fear"])
        self.assertEqual(bright.inborn["fear"], 5)

    def test_temperament_drifts_back_toward_inborn(self):
        agent = self.simulation._add_agent(encode_dna("grazer", fear=5), 1, 1)
        agent.mind["fear"] = 15.0

        for _ in range(100):
            self.simulation._settle(agent)

        self.assertLess(agent.mind["fear"], 15.0)
        self.assertGreater(agent.mind["fear"], 5.0)

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
        for x, y in ((5, 4), (4, 5), (6, 5)):
            self.simulation._add_agent(encode_dna("plant"), x, y)
        self.assertNotIn(breed, self.simulation._sense(agent)[1])

    def test_compatibility_looks_only_at_outer_rings(self):
        calm = self.simulation._add_agent(encode_dna("grazer", size=5), 1, 1)
        wild = self.simulation._add_agent(encode_dna("grazer", size=5) | 0xFFFFFFFF, 2, 2)
        distant = self.simulation._add_agent(
            encode_dna("grazer", size=15, strength=15, armour=15, weapons=15, speed=15), 3, 3)

        self.assertTrue(self.simulation._compatible(calm, wild))
        self.assertFalse(self.simulation._compatible(calm, distant))

    def test_predators_cannot_eat_grass(self):
        predator = self.simulation._add_agent(encode_dna("predator", sight=15), 5, 5)
        grazer = self.simulation._add_agent(encode_dna("grazer", sight=15), 2, 2)
        self.simulation.grass.update({(5, 5): 1, (2, 2): 1, (7, 7): 1})
        self.simulation._index_cells()
        grass_actions = {ACTIONS.index("eat"), ACTIONS.index("forage")}

        self.assertFalse(grass_actions & set(self.simulation._sense(predator)[1]))
        self.assertEqual(grass_actions & set(self.simulation._sense(grazer)[1]), grass_actions)

    def test_ready_animals_court_a_willing_mate_they_can_see(self):
        dna = encode_dna("grazer", sight=15)
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
        predator = self.simulation._add_agent(encode_dna("predator", sight=15, speed=5), 2, 2)
        grazer = self.simulation._add_agent(encode_dna("grazer"), 2, 7)
        self.simulation._index_cells()
        senses, options, situation = self.simulation._sense(predator)

        self.assertIn(ACTIONS.index("hunt"), options)
        self.assertNotIn(ACTIONS.index("attack"), options)
        self.simulation._act(predator, "hunt", situation)
        self.assertEqual((predator.x, predator.y), (2, 4))
        self.assertNotIn(ACTIONS.index("hunt"), self.simulation._sense(grazer)[1])

    def test_hunter_that_reaches_its_prey_pounces(self):
        predator = self.simulation._add_agent(encode_dna("predator", sight=15, speed=10), 2, 2)
        grazer = self.simulation._add_agent(encode_dna("grazer"), 2, 6)
        self.simulation._index_cells()
        situation = self.simulation._sense(predator)[2]

        self.simulation._act(predator, "hunt", situation)

        self.assertEqual(predator.last_action, "pounced")
        self.assertLess(grazer.health, grazer.max_health)

    def test_prey_flees_a_hunter_it_can_see(self):
        predator = self.simulation._add_agent(encode_dna("predator"), 5, 2)
        grazer = self.simulation._add_agent(encode_dna("grazer", sight=15), 5, 6)
        self.simulation._index_cells()
        senses, options, situation = self.simulation._sense(grazer)

        self.assertIn(ACTIONS.index("flee"), options)
        self.simulation._act(grazer, "flee", situation)
        self.assertGreater(abs(grazer.y - predator.y), 4)
        self.assertNotIn(ACTIONS.index("flee"), self.simulation._sense(predator)[1])

    def test_population_cap_withdraws_breeding(self):
        dna = encode_dna("grazer")
        agent = self.simulation._add_agent(dna, 5, 5)
        self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()
        self.simulation._population = 2

        with patch("tools.Simulation.MAX_POPULATION", 2):
            self.assertNotIn(ACTIONS.index("breed"), self.simulation._sense(agent)[1])

    def test_fertile_parents_breed_cheaply_and_their_child_gets_the_energy(self):
        dna = encode_dna("grazer", fertility=15, libido=15)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()

        self.simulation._breed(first, second, (5, 4))

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

        self.simulation._breed(first, second, (5, 4))

        self.assertFalse(first.alive)
        self.assertEqual((first.last_action, first.died_turn, first.brain), ("bred to death", 0, None))
        self.assertEqual(self.simulation.deaths, 1)
        self.assertTrue(second.alive)

    def test_intellect_decides_whether_brain_learns(self):
        slow = self.simulation._add_agent(encode_dna("grazer", intellect=0), 1, 1)
        quick = self.simulation._add_agent(encode_dna("grazer", intellect=15), 8, 8)
        slow_brain, quick_brain = slow.brain.copy(), quick.brain.copy()

        for _ in range(5):
            self.simulation._take_turn()

        self.assertEqual(slow.brain, slow_brain)
        self.assertNotEqual(quick.brain, quick_brain)

    def test_armour_blunts_and_weapons_sharpen_blows(self):
        brute = self.simulation._add_agent(encode_dna("predator", strength=10, weapons=15), 1, 1)
        plain = self.simulation._add_agent(encode_dna("predator", strength=10), 2, 2)
        soft = self.simulation._add_agent(encode_dna("grazer"), 3, 3)
        armoured = self.simulation._add_agent(encode_dna("grazer", armour=15), 4, 4)

        self.assertGreater(self.simulation._damage(brute, soft), self.simulation._damage(plain, soft))
        self.assertLess(self.simulation._damage(plain, armoured), self.simulation._damage(plain, soft))

    def test_kill_is_counted_and_corpse_is_buried(self):
        hunter = self.simulation._add_agent(encode_dna("predator", strength=15), 5, 5)
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
        victim = self.simulation._add_agent(encode_dna("grazer", weapons=15, size=15), 5, 6)
        self.simulation._index_cells()

        self.simulation._attack(hunter, victim)

        self.assertLess(hunter.health, hunter.max_health)
        self.assertGreater(hunter.mind["fear"], 3)
        self.assertGreater(victim.mind["fear"], victim.inborn["fear"])

    def test_rewind_and_replay_are_exact(self):
        simulation = Simulation(width=20, height=20, population=30, seed=4)

        def state():
            return ([(agent.identifier, agent.dna, agent.x, agent.y, agent.hunger, agent.mind, agent.brain)
                     for agent in simulation.agents], simulation.insights.snapshot())

        for _ in range(15):
            simulation.advance()
        expected = state()
        for _ in range(5):
            simulation.rewind()
        for _ in range(5):
            simulation.advance()

        self.assertEqual(state(), expected)


if __name__ == "__main__":
    unittest.main()
