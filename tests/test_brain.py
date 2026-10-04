import random
import unittest

from tools.Brain import ACTIONS, INPUT_SIZE, Brain, choice_odds, make_senses


class BrainTests(unittest.TestCase):
    def test_seeded_brains_value_every_action_identically(self):
        senses = [0.5] * INPUT_SIZE
        first = Brain.random(random.Random(7))
        second = Brain.random(random.Random(7))

        self.assertEqual(first.values(senses), second.values(senses))
        self.assertEqual(len(first.values(senses)), len(ACTIONS))

    def test_learning_moves_value_toward_target(self):
        brain = Brain.random(random.Random(7))
        senses = [0.5] * INPUT_SIZE
        initial_error = abs(brain.values(senses)[3] - 2.0)

        for _ in range(50):
            brain.learn(senses, 3, 2.0, learning_rate=0.05)

        self.assertLess(abs(brain.values(senses)[3] - 2.0), initial_error)

    def test_zero_learning_rate_leaves_brain_unchanged(self):
        brain = Brain.random(random.Random(7))
        before = brain.copy()

        brain.learn([0.5] * INPUT_SIZE, 0, 5.0, learning_rate=0)

        self.assertEqual(brain, before)

    def test_child_takes_each_neuron_whole_from_one_parent(self):
        rng = random.Random(3)
        mother, father = Brain.random(rng), Brain.random(rng)

        child = Brain.inherit(mother, father, rng, mutation=0)

        for index, weights in enumerate(child.hidden_weights):
            donor = mother if weights == mother.hidden_weights[index] else father
            self.assertEqual(weights, donor.hidden_weights[index])
            self.assertEqual(child.hidden_biases[index], donor.hidden_biases[index])
            for action in range(len(ACTIONS)):
                self.assertEqual(child.output_weights[action][index], donor.output_weights[action][index])

    def test_copy_is_independent(self):
        brain = Brain.random(random.Random(7))
        duplicate = brain.copy()

        brain.learn([1.0] * INPUT_SIZE, 0, 5.0, learning_rate=0.1)

        self.assertNotEqual(brain, duplicate)

    def test_wrong_input_size_is_rejected(self):
        with self.assertRaises(ValueError):
            Brain.random(random.Random(7)).values([0.0])

    def test_unknown_sense_is_rejected(self):
        with self.assertRaises(ValueError):
            make_senses(smell=1)

    def test_odds_cover_only_available_actions(self):
        odds = choice_odds([0.0, 1.0, 5.0, 2.0], [0, 1, 3])

        self.assertEqual(set(odds), {0, 1, 3})
        self.assertAlmostEqual(sum(odds.values()), 1.0)
        self.assertGreater(odds[3], odds[1])


if __name__ == "__main__":
    unittest.main()
