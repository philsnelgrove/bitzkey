import unittest
from unittest.mock import patch

from tools.Brain import ACTIONS
from tools.Simulation import BIRTH_HUNGER, BREED_COST, CORPSE_TURNS, Simulation


def genome(species: int, **traits: int) -> int:
    shifts = {"metabolism": 0, "aggression": 4, "constitution": 8, "fertility": 16,
              "sight": 20, "social": 24, "learning": 28}
    dna = species << 12
    for name, value in traits.items():
        dna |= value << shifts[name]
    return dna


GRAZER = 1
PREDATOR = 3


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.simulation = Simulation(width=10, height=10, population=0, seed=3)
        self.simulation.food.clear()

    def test_animals_get_brains_and_plants_do_not(self):
        plant = self.simulation._add_agent(genome(0), 1, 1)
        grazer = self.simulation._add_agent(genome(GRAZER), 2, 2)

        self.assertIsNone(plant.brain)
        self.assertIsNotNone(grazer.brain)

    def test_thinking_never_changes_genome(self):
        dna = genome(GRAZER, learning=15, fertility=15)
        agents = [self.simulation._add_agent(dna, 5, 5), self.simulation._add_agent(dna, 5, 6)]

        for _ in range(10):
            self.simulation._take_turn()

        self.assertTrue(all(agent.dna == dna for agent in agents))

    def test_breeding_needs_a_ready_mate_and_room(self):
        dna = genome(GRAZER)
        agent = self.simulation._add_agent(dna, 5, 5)
        mate = self.simulation._add_agent(dna, 5, 6)
        breed = ACTIONS.index("breed")
        self.simulation._index_cells()

        self.assertIn(breed, self.simulation._sense(agent)[1])
        mate.cooldown = 3
        self.assertNotIn(breed, self.simulation._sense(agent)[1])
        mate.cooldown = 0
        for x, y in ((5, 4), (4, 5), (6, 5)):
            self.simulation._add_agent(genome(0), x, y)
        self.assertNotIn(breed, self.simulation._sense(agent)[1])

    def test_population_cap_withdraws_breeding(self):
        dna = genome(GRAZER)
        agent = self.simulation._add_agent(dna, 5, 5)
        self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()
        self.simulation._population = 2

        with patch("tools.Simulation.MAX_POPULATION", 2):
            self.assertNotIn(ACTIONS.index("breed"), self.simulation._sense(agent)[1])

    def test_child_is_born_beside_parent_with_energy_from_both(self):
        dna = genome(GRAZER, fertility=15)
        first = self.simulation._add_agent(dna, 5, 5)
        second = self.simulation._add_agent(dna, 5, 6)
        self.simulation._index_cells()

        self.simulation._breed(first, second, (5, 4))

        child = self.simulation.agents[-1]
        self.assertEqual((child.x, child.y), (5, 4))
        self.assertEqual(child.hunger, BIRTH_HUNGER)
        self.assertEqual(child.generation, 1)
        self.assertNotIn(child.brain, (first.brain, second.brain))
        self.assertEqual((first.hunger, first.children, second.children), (BREED_COST, 1, 1))
        self.assertGreater(first.reward, 0)

    def test_learning_trait_decides_whether_brain_changes(self):
        slow = self.simulation._add_agent(genome(GRAZER, learning=0), 1, 1)
        quick = self.simulation._add_agent(genome(GRAZER, learning=15), 8, 8)
        slow_brain, quick_brain = slow.brain.copy(), quick.brain.copy()

        for _ in range(5):
            self.simulation._take_turn()

        self.assertEqual(slow.brain, slow_brain)
        self.assertNotEqual(quick.brain, quick_brain)

    def test_kill_is_counted_and_corpse_is_buried(self):
        hunter = self.simulation._add_agent(genome(PREDATOR, constitution=15), 5, 5)
        victim = self.simulation._add_agent(genome(GRAZER), 5, 6)
        victim.health = 1
        self.simulation._index_cells()

        self.simulation._attack(hunter, victim)

        self.assertFalse(victim.alive)
        self.assertIsNone(victim.brain)
        self.assertEqual(self.simulation.deaths, 1)
        for _ in range(CORPSE_TURNS):
            self.simulation._take_turn()
        self.assertNotIn(victim, self.simulation.agents)

    def test_rewind_and_replay_are_exact(self):
        simulation = Simulation(width=20, height=20, population=30, seed=4)

        def state():
            return ([(agent.identifier, agent.dna, agent.x, agent.y, agent.hunger, agent.brain)
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
