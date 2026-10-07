"""Per-animal neural network: what a Bitzkey senses, what it can do, and how it learns."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

import numpy as np

from tools.Genome import BODY_TRAITS, MIND_TRAITS, SPECIES, TEMPERATURES, decode_body, decode_climate, decode_mind, share

ACTIONS = ("eat", "forage", "browse", "scavenge", "feast", "rest", "court", "breed", "hunt", "attack", "defend", "flee", "herd", "explore")
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
    "carrion_closeness",  # 1 when meat lies alongside, falling to 0 beyond sight; 0 when none is seen (not grazers)
    "kill_closeness",     # 1 when a fresh kill it eats lies alongside, falling to 0 beyond sight (hunters only)
    "plant_closeness",    # 1 when a plant stands alongside, falling to 0 beyond sight (grazers and omnivores)
    "assailant_near",     # 1 when the animal that just struck it is still alongside
    "counter_frailty",    # share of that assailant's health one counterblow would take
    "rivalry",            # predators and omnivores: how overcrowded its territory is with rivals, 0..1
    "herd_near",          # herd-mates within two cells, 0..3 scaled to 0..1
    "herd_closeness",     # 1 at the middle of its herd in range, falling to 0 at the range's edge; 0 alone
    "alarm",              # 1 when a herd-mate close by was struck since it last acted
    "climate_discomfort",  # how far the temperature here is from home, as it feels it: 0, 0.5 a step, 1 two or more
    "comfort_closeness",  # 1 where it feels at home or next to such a row, falling to 0 beyond sight; 0 when none is seen
    "age",             # 0 at birth, 1 at 400 turns or older
) + MIND_TRAITS + (    # then its current temperament, each trait 0..7 scaled to 0..1
    *(f"open_{action}" for action in ACTIONS),            # 1 for each action open to it now
    *(f"inborn_{trait}" for trait in MIND_TRAITS),        # its dot's traits, as born, 0..1
    *BODY_TRAITS,                                         # its ring's traits, 0..1 (species 0 plant to 1 predator)
    "home_temperature", "heat_tolerance", "cold_tolerance",  # its climate genes: home 0 hot to 1 very cold, flags 0 or 1
)
INPUT_SIZE = len(SENSES)
HIDDEN_SIZE = 12  # neurons in each hidden layer
MIN_DEPTH = 2     # hidden layers: twice the inborn intellect, and never fewer than this
TEMPERATURE = 0.2
MUTATION = 0.05
RECORD_DIGITS = 6  # decimal places kept when a brain is exported


def depth_for(intellect: int) -> int:
    """How many hidden layers a brain has, by its owner's inborn intellect."""
    return max(MIN_DEPTH, 2 * intellect)


def make_senses(**values: float) -> list[float]:
    """Build an input vector by sense name; unnamed senses are zero."""
    unknown = set(values) - set(SENSES)
    if unknown:
        raise ValueError(f"Unknown senses: {', '.join(sorted(unknown))}.")
    return [float(values.get(name, 0.0)) for name in SENSES]


def gene_senses(dna: int) -> dict[str, float]:
    """The inputs for a genome's traits, ring and dot, and its climate genes."""
    inborn, body = decode_mind(dna), decode_body(dna)
    temperature, heat, cold = decode_climate(dna)
    values = {f"inborn_{trait}": share(inborn[trait]) for trait in MIND_TRAITS}
    values["species"] = SPECIES.index(str(body["species"])) / (len(SPECIES) - 1)
    values.update({trait: share(int(body[trait])) for trait in BODY_TRAITS[1:]})
    values.update(home_temperature=temperature / (len(TEMPERATURES) - 1), heat_tolerance=float(heat),
                  cold_tolerance=float(cold))
    return values


def open_senses(actions: Sequence[str]) -> dict[str, float]:
    """The inputs saying which actions are open to it."""
    return {f"open_{action}": 1.0 for action in actions}


def choice_odds(values: Sequence[float], options: Sequence[int]) -> dict[int, float]:
    """Softmax over the available actions' values."""
    best = max(values[option] for option in options)
    weights = {option: math.exp((values[option] - best) / TEMPERATURE) for option in options}
    total = sum(weights.values())
    return {option: weight / total for option, weight in weights.items()}


def _identity_layer() -> tuple[np.ndarray, np.ndarray]:
    """A hidden layer that passes its input on, near enough: added to deepen a brain without
    changing what it has learned."""
    return np.eye(HIDDEN_SIZE), np.zeros(HIDDEN_SIZE)


class Brain:
    """A tanh network of hidden layers estimating the long-run value of each action.

    It learns online from temporal-difference targets and is inherited neuron by neuron, so a
    child keeps whole learned features from one parent or the other. Its weights are NumPy
    arrays: each hidden layer's weights are one row per neuron, the output weights one row per
    action. The first hidden layer reads the senses; each later one reads the layer before.
    """

    def __init__(self, layers: Sequence[tuple[Sequence, Sequence]], output_weights: Sequence, output_biases: Sequence):
        self.layers = [(np.array(weights, dtype=float), np.array(biases, dtype=float)) for weights, biases in layers]
        self.output_weights = np.array(output_weights, dtype=float)
        self.output_biases = np.array(output_biases, dtype=float)

    @property
    def depth(self) -> int:
        return len(self.layers)

    @classmethod
    def random(cls, rng: random.Random, depth: int = MIN_DEPTH) -> Brain:
        layers = []
        for index in range(depth):
            reads = INPUT_SIZE if index == 0 else HIDDEN_SIZE
            bound = math.sqrt(6 / (reads + HIDDEN_SIZE))
            layers.append(([[rng.uniform(-bound, bound) for _ in range(reads)] for _ in range(HIDDEN_SIZE)],
                           [0.0] * HIDDEN_SIZE))
        return cls(layers, [[rng.uniform(-0.3, 0.3) for _ in range(HIDDEN_SIZE)] for _ in ACTIONS],
                   [rng.uniform(-0.2, 0.2) for _ in ACTIONS])

    @classmethod
    def inherit(cls, first: Brain, second: Brain, rng: random.Random, depth: int = MIN_DEPTH,
                mutation: float | None = None) -> Brain:
        """A child's brain, `depth` layers deep. Where both parents have a layer, each of its
        neurons, with its incoming weights, comes from one parent or the other; where only one
        has it, from that one; deeper still, it starts passing its input on. Each action's output
        comes from one parent or the other. Then it mutates (by MUTATION unless given)."""
        mutation = MUTATION if mutation is None else mutation
        layers = []
        for index in range(depth):
            donors = [parent for parent in (first, second) if index < parent.depth]
            if len(donors) == 2:
                from_first = np.array([rng.random() < 0.5 for _ in range(HIDDEN_SIZE)])
                layers.append((np.where(from_first[:, None], first.layers[index][0], second.layers[index][0]),
                               np.where(from_first, first.layers[index][1], second.layers[index][1])))
            elif donors:
                layers.append(tuple(part.copy() for part in donors[0].layers[index]))
            else:
                layers.append(_identity_layer())
        from_first = np.array([rng.random() < 0.5 for _ in ACTIONS])
        child = cls(layers, np.where(from_first[:, None], first.output_weights, second.output_weights),
                    np.where(from_first, first.output_biases, second.output_biases))
        if mutation:
            for weights in ([weights for weights, _ in child.layers] + [child.output_weights]
                            + [biases for _, biases in child.layers] + [child.output_biases]):
                weights += np.array([rng.gauss(0, mutation) for _ in range(weights.size)]).reshape(weights.shape)
        return child

    def resized(self, depth: int) -> Brain:
        """A copy `depth` layers deep: deeper with layers that pass their input on, or shallower
        without its last layers."""
        layers = [(weights.copy(), biases.copy()) for weights, biases in self.layers[:depth]]
        layers += [_identity_layer() for _ in range(depth - len(layers))]
        return Brain(layers, self.output_weights, self.output_biases)

    def _activations(self, inputs: Sequence[float]) -> list[np.ndarray]:
        """The senses, then each hidden layer's output."""
        activation = np.asarray(inputs, dtype=float)
        if activation.shape != (INPUT_SIZE,):
            raise ValueError(f"Expected {INPUT_SIZE} inputs, got {activation.size}.")
        activations = [activation]
        for weights, biases in self.layers:
            activation = np.tanh(weights @ activation + biases)
            activations.append(activation)
        return activations

    def values(self, inputs: Sequence[float]) -> list[float]:
        """Return the estimated value of every action in this situation."""
        return (self.output_weights @ self._activations(inputs)[-1] + self.output_biases).tolist()

    def learn(self, inputs: Sequence[float], action: int, target: float, learning_rate: float) -> None:
        """Move one action's value toward a target, with the error clipped for stability, back
        through every layer."""
        if learning_rate <= 0:
            return
        activations = self._activations(inputs)
        top = activations[-1]
        value = float(self.output_weights[action] @ top + self.output_biases[action])
        error = max(-1.0, min(1.0, value - target))
        gradient = error * self.output_weights[action]
        self.output_weights[action] -= learning_rate * error * top
        for index in range(self.depth - 1, -1, -1):
            weights, biases = self.layers[index]
            delta = gradient * (1 - activations[index + 1] * activations[index + 1])
            if index:
                gradient = weights.T @ delta
            biases -= learning_rate * delta
            weights -= learning_rate * np.outer(delta, activations[index])
        self.output_biases[action] -= learning_rate * error

    def copy(self) -> Brain:
        return Brain(self.layers, self.output_weights, self.output_biases)

    def to_record(self) -> dict:
        """The brain's weights for export, labelled with the senses and actions they belong to."""
        def rounded(values: np.ndarray) -> list[float]:
            return [round(value, RECORD_DIGITS) for value in values.tolist()]
        return {
            "senses": list(SENSES),
            "actions": list(ACTIONS),
            "layers": [{"weights": [rounded(row) for row in weights], "biases": rounded(biases)}
                       for weights, biases in self.layers],
            "output_weights": [rounded(weights) for weights in self.output_weights],
            "output_biases": rounded(self.output_biases),
        }

    @classmethod
    def from_record(cls, record: Mapping) -> Brain:
        """Rebuild an exported brain. Weights are matched to senses and actions by name, so a
        brain saved before a sense or action was added still loads; what it never had starts
        neutral (zero) and what this version no longer has is dropped. A brain saved with a single
        hidden layer, before brains grew deeper, loads as one layer deep. Raises ValueError."""
        try:
            senses = [str(name) for name in record["senses"]]
            actions = [str(name) for name in record["actions"]]
            if "layers" in record:
                layers = [([[float(weight) for weight in row] for row in layer["weights"]],
                           [float(bias) for bias in layer["biases"]]) for layer in record["layers"]]
            else:
                layers = [([[float(weight) for weight in row] for row in record["hidden_weights"]],
                           [float(bias) for bias in record["hidden_biases"]])]
            output_weights = [[float(weight) for weight in row] for row in record["output_weights"]]
            output_biases = [float(bias) for bias in record["output_biases"]]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"The brain is incomplete or not numeric ({error}).") from None
        if not layers:
            raise ValueError("The brain has no hidden layers.")
        for index, (weights, biases) in enumerate(layers):
            if len(weights) != HIDDEN_SIZE or len(biases) != HIDDEN_SIZE:
                raise ValueError(f"The brain has {len(weights)} neurons in a layer; this version uses {HIDDEN_SIZE}.")
            reads = len(senses) if index == 0 else HIDDEN_SIZE
            if any(len(row) != reads for row in weights):
                raise ValueError("The brain's weights don't match its listed senses and actions.")
        if (len(output_weights) != len(actions) or len(output_biases) != len(actions)
                or any(len(row) != HIDDEN_SIZE for row in output_weights)):
            raise ValueError("The brain's weights don't match its listed senses and actions.")

        sense_at = {name: index for index, name in enumerate(senses)}
        action_at = {name: index for index, name in enumerate(actions)}
        first_weights, first_biases = layers[0]
        layers[0] = ([[row[sense_at[name]] if name in sense_at else 0.0 for name in SENSES] for row in first_weights],
                     first_biases)
        return cls(
            layers,
            [output_weights[action_at[name]] if name in action_at else [0.0] * HIDDEN_SIZE for name in ACTIONS],
            [output_biases[action_at[name]] if name in action_at else 0.0 for name in ACTIONS],
        )

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Brain) and self.depth == other.depth
                and all(np.array_equal(mine[0], theirs[0]) and np.array_equal(mine[1], theirs[1])
                        for mine, theirs in zip(self.layers, other.layers))
                and np.array_equal(self.output_weights, other.output_weights)
                and np.array_equal(self.output_biases, other.output_biases))
