import random
import unittest

from tools.AggressionNetwork import AggressionNetwork, INPUT_SIZE


class AggressionNetworkTests(unittest.TestCase):
    def test_batch_predictions_are_bounded_and_seeded(self):
        observations = [[0.5] * INPUT_SIZE, [0.2] * INPUT_SIZE]
        first = AggressionNetwork(random.Random(7))
        second = AggressionNetwork(random.Random(7))

        predictions = first.predict_batch(observations)

        self.assertEqual(predictions, second.predict_batch(observations))
        self.assertTrue(all(-1 <= value <= 1 for value in predictions))

    def test_training_batch_reduces_prediction_error(self):
        network = AggressionNetwork(random.Random(7))
        batch = [
            ([0.0] * INPUT_SIZE, -1.0),
            ([1.0] * INPUT_SIZE, 1.0),
        ]
        initial_error = sum(
            (prediction - target) ** 2
            for prediction, (_, target) in zip(
                network.predict_batch([inputs for inputs, _ in batch]), batch, strict=True
            )
        )

        for _ in range(100):
            network.train_batch(batch)

        final_error = sum(
            (prediction - target) ** 2
            for prediction, (_, target) in zip(
                network.predict_batch([inputs for inputs, _ in batch]), batch, strict=True
            )
        )
        self.assertLess(final_error, initial_error)

    def test_wrong_input_size_is_rejected(self):
        network = AggressionNetwork(random.Random(7))
        with self.assertRaises(ValueError):
            network.predict_batch([[0.0]])


if __name__ == "__main__":
    unittest.main()
