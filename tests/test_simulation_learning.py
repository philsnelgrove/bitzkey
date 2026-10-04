import unittest
from unittest.mock import patch

from tools.AggressionNetwork import INPUT_SIZE
from tools.Simulation import AGGRESSION_BATCH_SIZE, Simulation


class SimulationLearningTests(unittest.TestCase):
    def test_aggression_policy_keeps_genome_unchanged(self):
        simulation = Simulation(width=10, height=10, population=0, seed=3)
        dna = (3 << 12) | (8 << 4) | (15 << 28)
        agent = simulation._add_agent(dna, 5, 5)

        simulation._take_turn()

        self.assertEqual(agent.dna, dna)
        self.assertGreaterEqual(simulation.aggression_score(agent), 0)
        self.assertLessEqual(simulation.aggression_score(agent), 15)

    def test_low_learning_trait_keeps_genome_aggression(self):
        simulation = Simulation(width=10, height=10, population=0, seed=3)
        dna = (3 << 12) | (8 << 4) | (3 << 28)
        agent = simulation._add_agent(dna, 5, 5)

        simulation._take_turn()

        self.assertEqual(simulation.aggression_score(agent), 8)

    def test_kill_generates_signed_examples_without_mutating_genomes(self):
        simulation = Simulation(width=10, height=10, population=0, seed=3)
        hunter_dna = (3 << 12) | (15 << 4) | (15 << 8) | (15 << 28)
        victim_dna = (1 << 12) | (15 << 28)
        hunter = simulation._add_agent(hunter_dna, 5, 5)
        victim = simulation._add_agent(victim_dna, 5, 5)
        victim.health = 1

        with patch.object(simulation.random, "shuffle", side_effect=lambda _: None):
            simulation._take_turn()

        self.assertFalse(victim.alive)
        self.assertEqual(hunter.dna, hunter_dna)
        self.assertEqual(victim.dna, victim_dna)
        self.assertEqual(
            [target for _, target in simulation.aggression_training_samples],
            [1.0, -1.0],
        )

    def test_turn_trains_full_conflict_minibatch(self):
        simulation = Simulation(width=10, height=10, population=0, seed=5)
        hunter_dna = (3 << 12) | (15 << 4) | (15 << 8) | (15 << 28)
        grazer_dna = (1 << 12) | (15 << 28)
        simulation._add_agent(hunter_dna, 5, 5)
        grazers = [
            simulation._add_agent(grazer_dna, 5, 5)
            for _ in range(AGGRESSION_BATCH_SIZE)
        ]
        grazers[0].health = 1
        initial_network = simulation.aggression_network.snapshot()

        with patch.object(simulation.random, "shuffle", side_effect=lambda _: None):
            simulation._take_turn()

        self.assertNotEqual(simulation.aggression_network.snapshot(), initial_network)
        self.assertEqual(len(simulation.aggression_training_samples), 1)

    def test_learning_waits_for_full_minibatch(self):
        simulation = Simulation(width=10, height=10, population=0, seed=3)
        sample = ([0.5] * INPUT_SIZE, 1.0)
        simulation.aggression_training_samples = [sample] * (AGGRESSION_BATCH_SIZE - 1)
        initial_state = simulation.aggression_network.snapshot()

        simulation._train_aggression_network()
        self.assertEqual(simulation.aggression_network.snapshot(), initial_state)
        self.assertEqual(len(simulation.aggression_training_samples), AGGRESSION_BATCH_SIZE - 1)

        simulation.aggression_training_samples.append(sample)
        simulation._train_aggression_network()
        self.assertNotEqual(simulation.aggression_network.snapshot(), initial_state)
        self.assertEqual(simulation.aggression_training_samples, [])

    def test_rewind_restores_network_and_pending_examples(self):
        simulation = Simulation(width=10, height=10, population=0, seed=3)
        initial_network = simulation.aggression_network.snapshot()
        simulation.aggression_training_samples.append(([0.5] * INPUT_SIZE, -1.0))
        simulation._record()

        self.assertTrue(simulation.rewind())
        self.assertEqual(simulation.aggression_network.snapshot(), initial_network)
        self.assertEqual(simulation.aggression_training_samples, [])


if __name__ == "__main__":
    unittest.main()
