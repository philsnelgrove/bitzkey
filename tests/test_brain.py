import random
import unittest

import numpy as np

from tools.Brain import (ACTIONS, HIDDEN_SIZE, INPUT_SIZE, MIN_DEPTH, SENSES, Brain, choice_odds, depth_for, gene_senses,
                         make_senses)
from tools.Genome import BODY_TRAITS, MIND_TRAITS, encode_dna


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

    def test_depth_is_twice_the_inborn_intellect_and_at_least_two(self):
        self.assertEqual([depth_for(intellect) for intellect in range(8)], [2, 2, 4, 6, 8, 10, 12, 14])
        brain = Brain.random(random.Random(7), 6)
        self.assertEqual((brain.depth, brain.layers[0][0].shape, brain.layers[5][0].shape),
                         (6, (HIDDEN_SIZE, INPUT_SIZE), (HIDDEN_SIZE, HIDDEN_SIZE)))
        self.assertEqual(Brain.random(random.Random(7)).depth, MIN_DEPTH)

    def test_there_is_an_input_for_each_action_and_each_ring_and_dot_trait(self):
        for name in ([f"open_{action}" for action in ACTIONS] + [f"inborn_{trait}" for trait in MIND_TRAITS]
                     + list(BODY_TRAITS) + ["home_temperature", "heat_tolerance", "cold_tolerance"]):
            self.assertIn(name, SENSES)
        genes = gene_senses(encode_dna("predator", temperature=2, cold_tolerance=True, size=7, intellect=7, fear=0))
        self.assertEqual((genes["species"], genes["size"], genes["inborn_intellect"], genes["inborn_fear"]),
                         (1.0, 1.0, 1.0, 0.0))
        self.assertEqual((genes["home_temperature"], genes["heat_tolerance"], genes["cold_tolerance"]), (2 / 3, 0.0, 1.0))

    def test_child_takes_each_neuron_whole_from_one_parent(self):
        rng = random.Random(3)
        mother, father = Brain.random(rng, 4), Brain.random(rng, 4)

        child = Brain.inherit(mother, father, rng, 4, mutation=0)

        for layer in range(4):
            for index, weights in enumerate(child.layers[layer][0].tolist()):
                donor = mother if weights == mother.layers[layer][0][index].tolist() else father
                self.assertEqual(weights, donor.layers[layer][0][index].tolist())
                self.assertEqual(child.layers[layer][1][index], donor.layers[layer][1][index])
        for action in range(len(ACTIONS)):
            donor = mother if child.output_weights[action].tolist() == mother.output_weights[action].tolist() else father
            self.assertEqual(child.output_weights[action].tolist(), donor.output_weights[action].tolist())
            self.assertEqual(child.output_biases[action], donor.output_biases[action])

    def test_a_child_deeper_than_a_parent_keeps_the_deeper_parents_layers_then_passes_input_on(self):
        rng = random.Random(3)
        shallow, deep = Brain.random(rng, 2), Brain.random(rng, 4)

        child = Brain.inherit(shallow, deep, rng, 6, mutation=0)

        self.assertEqual(child.depth, 6)
        for layer in (2, 3):  # only the deeper parent has these
            self.assertTrue(np.array_equal(child.layers[layer][0], deep.layers[layer][0]))
        for layer in (4, 5):  # neither has these
            self.assertTrue(np.array_equal(child.layers[layer][0], np.eye(HIDDEN_SIZE)))
            self.assertFalse(child.layers[layer][1].any())
        self.assertEqual(Brain.inherit(deep, deep, rng, 2, mutation=0).depth, 2)

    def test_resizing_deepens_with_pass_through_layers_or_drops_the_last(self):
        brain = Brain.random(random.Random(7), 4)
        deeper, shallower = brain.resized(6), brain.resized(2)
        self.assertEqual((deeper.depth, shallower.depth), (6, 2))
        self.assertTrue(np.array_equal(deeper.layers[3][0], brain.layers[3][0]))
        self.assertTrue(np.array_equal(deeper.layers[5][0], np.eye(HIDDEN_SIZE)))
        self.assertTrue(np.array_equal(shallower.layers[1][0], brain.layers[1][0]))

    def test_copy_is_independent(self):
        brain = Brain.random(random.Random(7))
        duplicate = brain.copy()

        brain.learn([1.0] * INPUT_SIZE, 0, 5.0, learning_rate=0.1)

        self.assertNotEqual(brain, duplicate)

    def test_exported_brain_loads_back_with_the_same_values(self):
        brain = Brain.random(random.Random(7))
        brain.learn([0.3] * INPUT_SIZE, 2, 1.5, learning_rate=0.05)

        loaded = Brain.from_record(brain.to_record())

        senses = [0.4] * INPUT_SIZE
        for original, restored in zip(brain.values(senses), loaded.values(senses), strict=True):
            self.assertAlmostEqual(original, restored, places=4)

    def test_brain_weights_are_matched_by_sense_and_action_name(self):
        record = Brain.random(random.Random(7)).to_record()
        record["senses"] = ["smell"] + record["senses"][1:]  # an unknown sense replaces "hunger"
        record["layers"][0]["weights"] = [[9.0] + row[1:] for row in record["layers"][0]["weights"]]
        record["actions"] = list(reversed(record["actions"]))
        record["output_biases"] = list(reversed(record["output_biases"]))
        record["output_weights"] = list(reversed(record["output_weights"]))
        original = Brain.random(random.Random(7))

        loaded = Brain.from_record(record)

        self.assertTrue(all(row[SENSES.index("hunger")] == 0.0 for row in loaded.layers[0][0]))
        self.assertEqual(loaded.output_biases.tolist(), [round(bias, 6) for bias in original.output_biases.tolist()])

    def test_a_brain_saved_with_one_hidden_layer_loads_one_layer_deep(self):
        brain = Brain.random(random.Random(7))
        record = brain.to_record()
        first = record.pop("layers")[0]
        record.update(hidden_weights=first["weights"], hidden_biases=first["biases"])

        loaded = Brain.from_record(record)

        self.assertEqual(loaded.depth, 1)
        self.assertTrue(np.allclose(loaded.layers[0][0], brain.layers[0][0], atol=1e-6))

    def test_malformed_brains_are_rejected(self):
        record = Brain.random(random.Random(7)).to_record()
        layers = record["layers"]
        for broken in ({**record, "layers": [{**layers[0], "weights": layers[0]["weights"][:3]}, layers[1]]},
                       {**record, "layers": [{**layers[0], "biases": "lots"}, layers[1]]},
                       {**record, "layers": [layers[0], {**layers[1], "weights": [[0.0] * 5] * HIDDEN_SIZE}]},
                       {**record, "layers": []},
                       {key: value for key, value in record.items() if key != "actions"}):
            with self.assertRaises(ValueError):
                Brain.from_record(broken)

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
