import random
import unittest

from unittest.mock import patch

import numpy as np

from tools.Brain import (ACTIONS, BASE_WIDTH, INPUT_SIZE, MAX_WIDTH, SENSES, Brain, choice_odds, depth_for, gene_senses,
                         make_senses, width_for)
from tools.Genome import BODY_TRAITS, TEMPERAMENT, encode_dna


from tools.Simulation import EIGHT_TURN_DAYS

def setUpModule():  # grown bodies: a Bitzkey made at age 0 isn't feeble here, unless a test says so
    global _grown
    _grown = patch.multiple("tools.Simulation", FEEBLE_YOUTH=0.0, SMALL_HIDING=0.0, DEFAULT_YEAR_LENGTH=360, SLEEP=0, **EIGHT_TURN_DAYS)  # and 360-turn years, no sleeping, spans in eight-turn days
    _grown.start()


def tearDownModule():
    _grown.stop()


class BrainTests(unittest.TestCase):
    def test_food_that_takes_a_kill_teaches_more_than_grass_and_mating_as_its_rules_say(self):
        from tools.Brain import ACTIONS
        from tools.Simulation import (BREED_LEARNING, COURT_LEARNING, KILL_FOOD_LEARNING, PLANT_FOOD_LEARNING,
                                      choice_learning)
        self.assertEqual(choice_learning(ACTIONS.index("breed")), BREED_LEARNING)
        self.assertEqual(choice_learning(ACTIONS.index("court")), COURT_LEARNING)
        for action in ("hunt", "attack", "feast"):
            self.assertEqual(choice_learning(ACTIONS.index(action)), KILL_FOOD_LEARNING)
        for action in ("eat", "forage", "browse", "uproot"):
            self.assertEqual(choice_learning(ACTIONS.index(action)), PLANT_FOOD_LEARNING)
        for action in ("scavenge", "rest", "flee", "walk_north"):
            self.assertEqual(choice_learning(ACTIONS.index(action)), 1.0)
        self.assertEqual((BREED_LEARNING, COURT_LEARNING, KILL_FOOD_LEARNING, PLANT_FOOD_LEARNING), (1.0, 1.0, 2.0, 0.5))

    def test_the_clever_learn_faster_while_the_least_intelligent_learn_as_ever(self):
        from tools.Brain import SMART_TRACE_DECAY, TRACE_DECAY
        from tools.Genome import encode_dna
        from tools.Simulation import LEARNING_BOOST, LEARNING_RATE, REPLAY_PER_INTELLECT, Simulation
        simulation = Simulation(width=20, height=20, population=0, seed=1)
        dull = simulation._add_agent(encode_dna("grazer", intellect=0), 5, 5)
        sharp = simulation._add_agent(encode_dna("grazer", intellect=7), 9, 9)
        self.assertEqual(simulation._learning_rate(dull), 0.0)  # as ever
        self.assertAlmostEqual(simulation._learning_rate(sharp), LEARNING_RATE * (1 + LEARNING_BOOST))
        from tools.Simulation import PREDATOR_LEARNING
        hunter = simulation._add_agent(encode_dna("predator", intellect=7), 13, 13)
        self.assertAlmostEqual(simulation._learning_rate(hunter), LEARNING_RATE * (1 + LEARNING_BOOST) * PREDATOR_LEARNING)
        eater = simulation._add_agent(encode_dna("omnivore", intellect=7), 15, 15)
        self.assertAlmostEqual(simulation._learning_rate(eater),
                               LEARNING_RATE * (1 + LEARNING_BOOST) * (1 + (PREDATOR_LEARNING - 1) / 2))
        simulation._index_cells()
        for _ in range(20):  # each decides, learns from the last decision, and remembers it
            previous = (sharp.last_senses, sharp.last_choice)
            for agent in (dull, sharp):
                simulation._think(agent)
        self.assertEqual(dull.memories, [])  # replays nothing
        self.assertEqual(len(sharp.memories), REPLAY_PER_INTELLECT * 7)  # its latest experiences, to replay
        self.assertEqual(sharp.memories[-1][:2], previous)  # the newest: the decision it just learnt from
        # lambda runs from TRACE_DECAY for the least to SMART_TRACE_DECAY for the most intelligent
        brain = Brain.random(random.Random(2))
        senses = [0.3] * len(SENSES)
        decays = []
        for sharpness in (0.0, 1.0):
            traced = brain.copy()
            traced.learn_traced(senses, 0, 1.0, 0.0001, 1.0, sharpness)
            traced.learn_traced(senses, 0, 1.0, 0.0001, 1.0, sharpness)
            fresh = brain.copy()
            fresh.learn_traced(senses, 0, 1.0, 0.0001, 1.0, sharpness)
            decays.append(traced.traces[-1][1][0] - fresh.traces[-1][1][0])  # the bias trace: decay + 1, less 1
        self.assertAlmostEqual(decays[0], TRACE_DECAY)
        self.assertAlmostEqual(decays[1], SMART_TRACE_DECAY)

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

    def test_intellect_deepens_the_brain_to_four_layers_then_widens_it(self):
        self.assertEqual([depth_for(intellect) for intellect in range(8)], [1, 1, 2, 3, 4, 4, 4, 4])
        self.assertEqual([width_for(intellect) for intellect in range(8)], [38, 38, 38, 38, 38, 43, 48, 53])
        self.assertEqual((BASE_WIDTH, MAX_WIDTH), (38, 53))
        brain = Brain.random(random.Random(7), 4, 53)
        self.assertEqual((brain.depth, brain.width, brain.layers[0][0].shape, brain.layers[3][0].shape, brain.output_weights.shape),
                         (4, 53, (53, INPUT_SIZE), (53, 53), (len(ACTIONS), 53)))
        self.assertEqual((Brain.random(random.Random(7)).depth, Brain.random(random.Random(7)).width), (1, BASE_WIDTH))

    def test_there_is_an_input_for_each_action_and_each_ring_dot_and_social_trait(self):
        for name in ([f"inborn_{trait}" for trait in TEMPERAMENT] + list(TEMPERAMENT)
                     + list(BODY_TRAITS) + ["home_temperature", "heat_tolerance", "cold_tolerance"]):
            self.assertIn(name, SENSES)
        genes = gene_senses(encode_dna("predator", temperature=2, cold_tolerance=True, size=7, intellect=7, fear=0, trust=7))
        self.assertEqual((genes["species"], genes["size"], genes["inborn_intellect"], genes["inborn_fear"],
                          genes["inborn_trust"]), (1.0, 1.0, 1.0, 0.0, 1.0))
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
            self.assertTrue(np.array_equal(child.layers[layer][0], np.eye(BASE_WIDTH)))
            self.assertFalse(child.layers[layer][1].any())
        self.assertEqual(Brain.inherit(deep, deep, rng, 2, mutation=0).depth, 2)

    def test_a_child_wider_than_a_parent_takes_the_neurons_only_the_wider_parent_has_from_it(self):
        rng = random.Random(3)
        narrow, wide = Brain.random(rng, 2, 38), Brain.random(rng, 2, 48)

        child = Brain.inherit(narrow, wide, rng, 2, 53, mutation=0)

        self.assertEqual((child.depth, child.width, child.layers[1][0].shape), (2, 53, (53, 53)))
        for neuron in range(38, 48):  # only the wider parent has these
            self.assertTrue(np.array_equal(child.layers[0][0][neuron], wide.layers[0][0][neuron]))
            self.assertTrue(np.array_equal(child.layers[1][0][neuron][:48], wide.layers[1][0][neuron]))
        for neuron in range(48, 53):  # neither has these: idle
            self.assertFalse(child.layers[0][0][neuron].any() or child.layers[1][0][neuron].any())
        self.assertFalse(child.output_weights[:, 48:].any())
        narrower = Brain.inherit(wide, wide, rng, 1, 38, mutation=0)
        self.assertEqual(narrower.output_weights.shape, (len(ACTIONS), 38))
        self.assertTrue(np.array_equal(narrower.layers[0][0], wide.layers[0][0][:38]))

    def test_resizing_deepens_with_pass_through_layers_or_drops_the_last(self):
        brain = Brain.random(random.Random(7), 4)
        deeper, shallower = brain.resized(6, BASE_WIDTH), brain.resized(2, BASE_WIDTH)
        self.assertEqual((deeper.depth, shallower.depth), (6, 2))
        self.assertTrue(np.array_equal(deeper.layers[3][0], brain.layers[3][0]))
        self.assertTrue(np.array_equal(deeper.layers[5][0], np.eye(BASE_WIDTH)))
        self.assertTrue(np.array_equal(shallower.layers[1][0], brain.layers[1][0]))

    def test_resizing_wider_adds_idle_neurons_that_change_nothing_and_narrower_drops_the_last(self):
        brain = Brain.random(random.Random(7), 3)
        wider = brain.resized(3, 53)
        self.assertEqual((wider.width, wider.layers[2][0].shape), (53, (53, 53)))
        senses = [0.4] * INPUT_SIZE
        for mine, theirs in zip(wider.values(senses), brain.values(senses), strict=True):
            self.assertAlmostEqual(mine, theirs, places=12)
        narrower = brain.resized(3, 30)
        self.assertTrue(np.array_equal(narrower.layers[1][0], brain.layers[1][0][:30, :30]))

    def test_a_new_brain_is_reluctant_to_set_off_in_any_direction(self):
        reluctant = Brain.random(random.Random(7))
        with patch("tools.Brain.HEADING_RELUCTANCE", 0.0):
            eager = Brain.random(random.Random(7))
        for action, (mine, theirs) in zip(ACTIONS, zip(reluctant.output_biases, eager.output_biases), strict=True):
            self.assertAlmostEqual(mine, theirs - (0.5 if action.startswith(("go_", "walk_")) else 0.0))

    def test_a_first_traced_lesson_is_an_ordinary_lesson(self):
        traced, plain = Brain.random(random.Random(7), 3), Brain.random(random.Random(7), 3)
        senses = [0.3] * INPUT_SIZE
        traced.learn_traced(senses, 2, 1.5, learning_rate=0.05, discount=0.9)
        plain.learn(senses, 2, 1.5, learning_rate=0.05)
        for mine, theirs in zip(traced.values([0.6] * INPUT_SIZE), plain.values([0.6] * INPUT_SIZE), strict=True):
            self.assertAlmostEqual(mine, theirs, places=12)

    def test_a_traced_lesson_reaches_back_to_the_choice_before(self):
        earlier, later = [0.2] * INPUT_SIZE, [0.7] * INPUT_SIZE
        changes = []
        for decay in (0.0, 0.9):
            with patch("tools.Brain.TRACE_DECAY", decay):
                brain = Brain.random(random.Random(7))
                before = brain.values(earlier)[1]
                brain.learn_traced(earlier, 1, before, learning_rate=0.05, discount=1.0)  # no surprise yet
                brain.learn_traced(later, 4, brain.values(later)[4] - 1, learning_rate=0.05, discount=1.0)
                changes.append(brain.values(earlier)[1] - before)
        # Without a trace, the earlier choice moves only a little, either way, through the layers both
        # share; with one, it shares the blame.
        self.assertLess(changes[1], changes[0] - 0.1)

    def test_there_are_105_senses(self):
        self.assertEqual(INPUT_SIZE, 105)

    def test_a_childs_brain_is_made_exactly_as_the_cpp_version_makes_it_from_the_same_seed(self):
        rng = random.Random(7)
        first, second = Brain.random(rng, 2, 40), Brain.random(rng, 3, 45)
        child = Brain.inherit(first, second, rng, 3, 45, 0.05)
        # As bitzkey_cpp's Brain::inherit makes them, to the last bit.
        self.assertEqual([float(child.layers[0][0][3][11]), float(child.layers[0][0][44][89]), float(child.layers[1][1][20]),
                          float(child.layers[2][0][5][7]), float(child.output_weights[23][44]), float(child.output_biases[3])],
                         [-0.15983579101927251, 0.092250751175896234, -0.097734976568545115, -0.097151955572367998,
                          -0.011420547423109582, -0.019905365281007829])
        self.assertEqual(rng.random(), 0.20169522729508749)  # one draw of the world's for the child

    def test_a_childs_brain_is_its_parents_nudged_by_normal_mutations_of_the_mutations_size(self):
        rng = random.Random(5)
        parent = Brain.random(rng, 4, MAX_WIDTH)
        nudges = np.concatenate([(Brain.inherit(parent, parent, rng, 4, MAX_WIDTH, 0.05).layers[layer][0]
                                  - parent.layers[layer][0]).ravel() for _ in range(5) for layer in range(4)])
        self.assertLess(abs(nudges.mean()), 0.001)
        self.assertAlmostEqual(nudges.std(), 0.05, delta=0.001)
        self.assertAlmostEqual(np.mean(np.abs(nudges) > 0.15), 0.0027, delta=0.0006)  # a normal's tails beyond 3 sigma

    def test_a_brain_saved_with_narrower_layers_loads_as_saved_and_values_alike(self):
        rng = np.random.default_rng(5)
        narrow = 12
        layers = [(rng.uniform(-0.5, 0.5, (narrow, INPUT_SIZE)), rng.uniform(-0.2, 0.2, narrow)),
                  (rng.uniform(-0.5, 0.5, (narrow, narrow)), rng.uniform(-0.2, 0.2, narrow))]
        output_weights, output_biases = rng.uniform(-0.3, 0.3, (len(ACTIONS), narrow)), rng.uniform(-0.2, 0.2, len(ACTIONS))
        record = {"senses": list(SENSES), "actions": list(ACTIONS),
                  "layers": [{"weights": weights.tolist(), "biases": biases.tolist()} for weights, biases in layers],
                  "output_weights": output_weights.tolist(), "output_biases": output_biases.tolist()}
        loaded = Brain.from_record(record)
        self.assertEqual([weights.shape for weights, _ in loaded.layers], [(narrow, INPUT_SIZE), (narrow, narrow)])
        senses = rng.uniform(0, 1, INPUT_SIZE)
        activation = senses
        for weights, biases in layers:
            activation = np.tanh(weights @ activation + biases)
        expected = output_weights @ activation + output_biases
        for mine, theirs in zip(loaded.values(senses.tolist()), expected, strict=True):
            self.assertAlmostEqual(mine, theirs, places=12)
        record["layers"][1]["biases"] = record["layers"][1]["biases"][:5]
        with self.assertRaisesRegex(ValueError, "neurons in a layer"):
            Brain.from_record(record)

    def test_copy_is_independent(self):
        brain = Brain.random(random.Random(7))
        duplicate = brain.copy()

        brain.learn([1.0] * INPUT_SIZE, 0, 5.0, learning_rate=0.1)

        self.assertNotEqual(brain, duplicate)

    def test_copy_keeps_its_own_traces(self):
        brain = Brain.random(random.Random(7))
        brain.learn_traced([0.5] * INPUT_SIZE, 0, 5.0, learning_rate=0.1, discount=0.9)
        duplicate = brain.copy()
        brain.learn_traced([0.5] * INPUT_SIZE, 0, 5.0, learning_rate=0.1, discount=0.9)
        duplicate.learn_traced([0.5] * INPUT_SIZE, 0, 5.0, learning_rate=0.1, discount=0.9)
        self.assertEqual(brain, duplicate)  # the same traces, kept apart
        self.assertIsNone(Brain.from_record(brain.to_record()).traces)  # traces belong to one life

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
        record = Brain.random(random.Random(7), 2).to_record()
        layers = record["layers"]
        too_wide = Brain.random(random.Random(7), 1, MAX_WIDTH + 1).to_record()
        for broken in ({**record, "layers": [{**layers[0], "weights": layers[0]["weights"][:3]}, layers[1]]},
                       {**record, "layers": [{**layers[0], "biases": "lots"}, layers[1]]},
                       {**record, "layers": [layers[0], {**layers[1], "weights": [[0.0] * 5] * BASE_WIDTH}]},
                       too_wide,
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
