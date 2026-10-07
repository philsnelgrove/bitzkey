"""Per-animal neural network: what a Bitzkey senses, what it can do, and how it learns."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

import numpy as np

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
    "others_near",     # animals of other species within two cells, 0..3 scaled to 0..1
    "carrion_closeness",  # 1 when meat lies alongside, falling to 0 beyond sight; 0 when none is seen
    "plant_closeness",    # 1 when a plant stands alongside, falling to 0 beyond sight (grazers and omnivores)
    "assailant_near",     # 1 when the animal that just struck it is still alongside
    "counter_frailty",    # share of that assailant's health one counterblow would take
    "rivalry",            # predators and omnivores: how overcrowded its territory is with rivals, 0..1
    "herd_near",          # herd-mates within two cells, 0..3 scaled to 0..1
    "herd_closeness",     # 1 at the middle of its herd in range, falling to 0 at the range's edge; 0 alone
    "alarm",              # 1 when a herd-mate close by was struck since it last acted
    "age",             # 0 at birth, 1 at 400 turns or older
) + MIND_TRAITS        # then its current temperament, each trait 0..7 scaled to 0..1
ACTIONS = ("eat", "forage", "browse", "scavenge", "rest", "court", "breed", "hunt", "attack", "defend", "flee", "herd", "explore")
INPUT_SIZE = len(SENSES)
HIDDEN_SIZE = 12
TEMPERATURE = 0.2
MUTATION = 0.05
RECORD_DIGITS = 6  # decimal places kept when a brain is exported


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
    Its weights are NumPy arrays: hidden weights are one row per hidden neuron, output
    weights one row per action.
    """

    def __init__(self, hidden_weights: Sequence, hidden_biases: Sequence,
                 output_weights: Sequence, output_biases: Sequence):
        self.hidden_weights = np.array(hidden_weights, dtype=float)
        self.hidden_biases = np.array(hidden_biases, dtype=float)
        self.output_weights = np.array(output_weights, dtype=float)
        self.output_biases = np.array(output_biases, dtype=float)

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
                mutation: float | None = None) -> Brain:
        """Take each hidden neuron, with its outgoing weights, from one parent, then mutate (by
        MUTATION unless given)."""
        mutation = MUTATION if mutation is None else mutation
        from_first = np.array([rng.random() < 0.5 for _ in range(HIDDEN_SIZE)])
        bias_from_first = np.array([rng.random() < 0.5 for _ in ACTIONS])
        child = cls(
            np.where(from_first[:, None], first.hidden_weights, second.hidden_weights),
            np.where(from_first, first.hidden_biases, second.hidden_biases),
            np.where(from_first[None, :], first.output_weights, second.output_weights),
            np.where(bias_from_first, first.output_biases, second.output_biases),
        )
        if mutation:
            for weights in (child.hidden_weights, child.output_weights, child.hidden_biases, child.output_biases):
                weights += np.array([rng.gauss(0, mutation) for _ in range(weights.size)]).reshape(weights.shape)
        return child

    def _forward(self, inputs: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inputs = np.asarray(inputs, dtype=float)
        if inputs.shape != (INPUT_SIZE,):
            raise ValueError(f"Expected {INPUT_SIZE} inputs, got {inputs.size}.")
        hidden = np.tanh(self.hidden_weights @ inputs + self.hidden_biases)
        return inputs, hidden, self.output_weights @ hidden + self.output_biases

    def values(self, inputs: Sequence[float]) -> list[float]:
        """Return the estimated value of every action in this situation."""
        return self._forward(inputs)[2].tolist()

    def learn(self, inputs: Sequence[float], action: int, target: float, learning_rate: float) -> None:
        """Move one action's value toward a target, with the error clipped for stability."""
        if learning_rate <= 0:
            return
        inputs, hidden, values = self._forward(inputs)
        error = max(-1.0, min(1.0, float(values[action]) - target))
        hidden_delta = error * self.output_weights[action] * (1 - hidden * hidden)
        self.output_weights[action] -= learning_rate * error * hidden
        self.hidden_biases -= learning_rate * hidden_delta
        self.hidden_weights -= learning_rate * np.outer(hidden_delta, inputs)
        self.output_biases[action] -= learning_rate * error

    def copy(self) -> Brain:
        return Brain(self.hidden_weights, self.hidden_biases, self.output_weights, self.output_biases)

    def to_record(self) -> dict:
        """The brain's weights for export, labelled with the senses and actions they belong to."""
        def rounded(values: np.ndarray) -> list[float]:
            return [round(value, RECORD_DIGITS) for value in values.tolist()]
        return {
            "senses": list(SENSES),
            "actions": list(ACTIONS),
            "hidden_weights": [rounded(weights) for weights in self.hidden_weights],
            "hidden_biases": rounded(self.hidden_biases),
            "output_weights": [rounded(weights) for weights in self.output_weights],
            "output_biases": rounded(self.output_biases),
        }

    @classmethod
    def from_record(cls, record: Mapping) -> Brain:
        """Rebuild an exported brain. Weights are matched to senses and actions by name, so a
        brain saved before a sense or action was added still loads; what it never had starts
        neutral (zero) and what this version no longer has is dropped. Raises ValueError."""
        try:
            senses = [str(name) for name in record["senses"]]
            actions = [str(name) for name in record["actions"]]
            hidden_weights = [[float(weight) for weight in row] for row in record["hidden_weights"]]
            hidden_biases = [float(bias) for bias in record["hidden_biases"]]
            output_weights = [[float(weight) for weight in row] for row in record["output_weights"]]
            output_biases = [float(bias) for bias in record["output_biases"]]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"The brain is incomplete or not numeric ({error}).") from None
        if len(hidden_weights) != HIDDEN_SIZE or len(hidden_biases) != HIDDEN_SIZE:
            raise ValueError(f"The brain has {len(hidden_weights)} hidden neurons; this version uses {HIDDEN_SIZE}.")
        if (any(len(row) != len(senses) for row in hidden_weights) or len(output_weights) != len(actions)
                or len(output_biases) != len(actions) or any(len(row) != HIDDEN_SIZE for row in output_weights)):
            raise ValueError("The brain's weights don't match its listed senses and actions.")

        sense_at = {name: index for index, name in enumerate(senses)}
        action_at = {name: index for index, name in enumerate(actions)}
        return cls(
            [[row[sense_at[name]] if name in sense_at else 0.0 for name in SENSES] for row in hidden_weights],
            hidden_biases,
            [output_weights[action_at[name]] if name in action_at else [0.0] * HIDDEN_SIZE for name in ACTIONS],
            [output_biases[action_at[name]] if name in action_at else 0.0 for name in ACTIONS],
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Brain) and all(
            np.array_equal(mine, theirs) for mine, theirs in
            ((self.hidden_weights, other.hidden_weights), (self.hidden_biases, other.hidden_biases),
             (self.output_weights, other.output_weights), (self.output_biases, other.output_biases)))
