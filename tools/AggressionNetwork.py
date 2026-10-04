"""Small deterministic neural network for learned aggression policy."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import TypedDict


INPUT_SIZE = 11
HIDDEN_SIZE = 8


class AggressionNetworkState(TypedDict):
    hidden_weights: list[list[float]]
    hidden_biases: list[float]
    output_weights: list[float]
    output_bias: float


class AggressionNetwork:
    """One-hidden-layer tanh network with minibatch gradient descent."""

    def __init__(self, rng: random.Random):
        hidden_bound = math.sqrt(6 / (INPUT_SIZE + HIDDEN_SIZE))
        self.hidden_weights = [
            [rng.uniform(-hidden_bound, hidden_bound) for _ in range(INPUT_SIZE)]
            for _ in range(HIDDEN_SIZE)
        ]
        self.hidden_biases = [0.0] * HIDDEN_SIZE
        self.output_weights = [0.0] * HIDDEN_SIZE
        self.output_bias = 0.0

    def _forward(self, inputs: Sequence[float]) -> tuple[list[float], float]:
        if len(inputs) != INPUT_SIZE:
            raise ValueError(f"Expected {INPUT_SIZE} inputs, got {len(inputs)}.")
        hidden = [
            math.tanh(sum(weight * value for weight, value
                          in zip(weights, inputs, strict=True)) + bias)
            for weights, bias in zip(self.hidden_weights, self.hidden_biases, strict=True)
        ]
        output = math.tanh(
            sum(weight * value for weight, value
                in zip(self.output_weights, hidden, strict=True))
            + self.output_bias
        )
        return hidden, output

    def predict_batch(self, batch: Sequence[Sequence[float]]) -> list[float]:
        """Return aggression adjustments in [-1, 1] for one observation batch."""
        return [self._forward(inputs)[1] for inputs in batch]

    def train_batch(
        self,
        batch: Sequence[tuple[Sequence[float], float]],
        learning_rate: float = 0.03,
    ) -> None:
        """Apply one mean-gradient update over a batch of labeled outcomes."""
        if not batch:
            return
        hidden_weight_gradients = [[0.0] * INPUT_SIZE for _ in range(HIDDEN_SIZE)]
        hidden_bias_gradients = [0.0] * HIDDEN_SIZE
        output_weight_gradients = [0.0] * HIDDEN_SIZE
        output_bias_gradient = 0.0

        for inputs, target in batch:
            hidden, output = self._forward(inputs)
            output_delta = (output - target) * (1 - output * output)
            output_bias_gradient += output_delta
            for index, activation in enumerate(hidden):
                output_weight_gradients[index] += output_delta * activation
            for index in range(HIDDEN_SIZE):
                hidden_delta = (
                    output_delta * self.output_weights[index] * (1 - hidden[index] ** 2)
                )
                hidden_bias_gradients[index] += hidden_delta
                for input_index, value in enumerate(inputs):
                    hidden_weight_gradients[index][input_index] += hidden_delta * value

        scale = learning_rate / len(batch)
        for index in range(HIDDEN_SIZE):
            self.hidden_biases[index] -= scale * hidden_bias_gradients[index]
            for input_index in range(INPUT_SIZE):
                self.hidden_weights[index][input_index] -= (
                    scale * hidden_weight_gradients[index][input_index]
                )
            self.output_weights[index] -= scale * output_weight_gradients[index]
        self.output_bias -= scale * output_bias_gradient

    def snapshot(self) -> AggressionNetworkState:
        return {
            "hidden_weights": [weights.copy() for weights in self.hidden_weights],
            "hidden_biases": self.hidden_biases.copy(),
            "output_weights": self.output_weights.copy(),
            "output_bias": self.output_bias,
        }

    def restore(self, state: AggressionNetworkState) -> None:
        self.hidden_weights = [weights.copy() for weights in state["hidden_weights"]]
        self.hidden_biases = state["hidden_biases"].copy()
        self.output_weights = state["output_weights"].copy()
        self.output_bias = state["output_bias"]
