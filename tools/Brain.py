"""Per-animal neural network: what a Bitzkey senses, what it can do, and how it learns."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from tools.Genome import MIND_TRAITS


SENSES = (
    "hunger",          # 0..1
    "health",          # 0..1
    "grass_here",       # 1 when grass lies on the animal's own cell
    "grass_seen",       # 1 when grass is visible elsewhere
    "grass_closeness",  # 1 next to the nearest visible grass, falling to 0 at the edge of sight
    "mate_near",       # 1 when a compatible, ready mate is adjacent
    "mate_closeness",  # 1 when a willing mate is adjacent, falling to 0 beyond sight; 0 when none is seen
    "ready_to_breed",  # 1 when the animal's breeding cooldown has passed
    "prey_near",       # adjacent prey, 0..4 scaled to 0..1
    "prey_frailty",    # share of the frailest adjacent prey's health one blow would take
    "prey_closeness",  # 1 when prey is adjacent, falling to 0 beyond sight; 0 when none is seen
    "threat_near",     # adjacent hunters, 0..4 scaled to 0..1
    "threat_danger",   # share of the animal's own health the worst adjacent hunter's blow would take
    "threat_closeness",  # 1 when a hunter is adjacent, falling to 0 beyond sight; 0 when none is seen
    "kin_near",        # same-species animals within two cells, 0..3 scaled to 0..1
    "age",             # 0 at birth, 1 at 400 turns or older
) + MIND_TRAITS        # then its current temperament, each trait 0..15 scaled to 0..1
ACTIONS = ("eat", "forage", "rest", "court", "breed", "hunt", "attack", "flee", "explore")
INPUT_SIZE = len(SENSES)
HIDDEN_SIZE = 12
TEMPERATURE = 0.2
MUTATION = 0.05


def make_senses(**values: float) -> list[float]:
    """Build an input vector by sense name; unnamed senses are zero."""
    unknown = set(values) - set(SENSES)
    if unknown:
        raise ValueError(f"Unknown senses: {', '.join(sorted(unknown))}.")
    return [float(values.get(name, 0.0)) for name in SENSES]


def choice_odds(values: Sequence[float], options: Sequence[int]) -> dict[int, float]:
    """Softmax over the available actions' values."""
    best = max(values[option] for option in options)
    weights = {option: math.exp((values[option] - best) / TEMPERATURE) for option in options}
    total = sum(weights.values())
    return {option: weight / total for option, weight in weights.items()}


class Brain:
    """One-hidden-layer tanh network estimating the long-run value of each action.

    It learns online from temporal-difference targets and is inherited neuron by
    neuron, so a child keeps whole learned features from one parent or the other.
    """

    def __init__(self, hidden_weights: list[list[float]], hidden_biases: list[float],
                 output_weights: list[list[float]], output_biases: list[float]):
        self.hidden_weights = hidden_weights
        self.hidden_biases = hidden_biases
        self.output_weights = output_weights
        self.output_biases = output_biases

    @classmethod
    def random(cls, rng: random.Random) -> Brain:
        hidden_bound = math.sqrt(6 / (INPUT_SIZE + HIDDEN_SIZE))
        return cls(
            [[rng.uniform(-hidden_bound, hidden_bound) for _ in range(INPUT_SIZE)]
             for _ in range(HIDDEN_SIZE)],
            [0.0] * HIDDEN_SIZE,
            [[rng.uniform(-0.3, 0.3) for _ in range(HIDDEN_SIZE)] for _ in ACTIONS],
            [rng.uniform(-0.2, 0.2) for _ in ACTIONS],
        )

    @classmethod
    def inherit(cls, first: Brain, second: Brain, rng: random.Random,
                mutation: float = MUTATION) -> Brain:
        """Take each hidden neuron, with its outgoing weights, from one parent, then mutate."""
        donors = [first if rng.random() < 0.5 else second for _ in range(HIDDEN_SIZE)]
        child = cls(
            [donor.hidden_weights[index].copy() for index, donor in enumerate(donors)],
            [donor.hidden_biases[index] for index, donor in enumerate(donors)],
            [[donor.output_weights[action][index] for index, donor in enumerate(donors)]
             for action in range(len(ACTIONS))],
            [(first if rng.random() < 0.5 else second).output_biases[action]
             for action in range(len(ACTIONS))],
        )
        if mutation:
            for row in child.hidden_weights + child.output_weights:
                for index in range(len(row)):
                    row[index] += rng.gauss(0, mutation)
            for biases in (child.hidden_biases, child.output_biases):
                for index in range(len(biases)):
                    biases[index] += rng.gauss(0, mutation)
        return child

    def _forward(self, inputs: Sequence[float]) -> tuple[list[float], list[float]]:
        if len(inputs) != INPUT_SIZE:
            raise ValueError(f"Expected {INPUT_SIZE} inputs, got {len(inputs)}.")
        hidden = [
            math.tanh(sum(weight * value for weight, value in zip(weights, inputs, strict=True)) + bias)
            for weights, bias in zip(self.hidden_weights, self.hidden_biases, strict=True)
        ]
        values = [
            sum(weight * activation for weight, activation in zip(weights, hidden, strict=True)) + bias
            for weights, bias in zip(self.output_weights, self.output_biases, strict=True)
        ]
        return hidden, values

    def values(self, inputs: Sequence[float]) -> list[float]:
        """Return the estimated value of every action in this situation."""
        return self._forward(inputs)[1]

    def learn(self, inputs: Sequence[float], action: int, target: float, learning_rate: float) -> None:
        """Move one action's value toward a target, with the error clipped for stability."""
        if learning_rate <= 0:
            return
        hidden, values = self._forward(inputs)
        error = max(-1.0, min(1.0, values[action] - target))
        output_weights = self.output_weights[action]
        for index, activation in enumerate(hidden):
            hidden_delta = error * output_weights[index] * (1 - activation * activation)
            output_weights[index] -= learning_rate * error * activation
            self.hidden_biases[index] -= learning_rate * hidden_delta
            weights = self.hidden_weights[index]
            for input_index, value in enumerate(inputs):
                weights[input_index] -= learning_rate * hidden_delta * value
        self.output_biases[action] -= learning_rate * error

    def copy(self) -> Brain:
        return Brain([weights.copy() for weights in self.hidden_weights], self.hidden_biases.copy(),
                     [weights.copy() for weights in self.output_weights], self.output_biases.copy())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Brain) and (
            (self.hidden_weights, self.hidden_biases, self.output_weights, self.output_biases)
            == (other.hidden_weights, other.hidden_biases, other.output_weights, other.output_biases))
