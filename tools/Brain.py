"""Per-animal neural network: what a Bitzkey senses, what it can do, and how it learns."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

import numpy as np

from tools.Genome import BODY_TRAITS, SPECIES, TEMPERAMENT, TEMPERATURES, TRAIT_MAX, decode_body, decode_climate, decode_temperament, share

ACTIONS = ("eat", "forage", "browse", "uproot", "scavenge", "feast", "rest", "court", "breed", "hunt", "attack", "defend",
           "flee", "herd", "go_north", "go_south", "go_east", "go_west", "walk_north", "walk_south", "walk_east",
           "walk_west", "alert", "revisit")
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
    "prey_pregnant",   # 1 when the nearest prey in sight carries young
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
    "alarm",              # 1 when a herd-mate close by was struck, or alerted it to a hunter, since it last acted
    "kill_alert",         # 1 when a pack-mate alerted it to a fresh kill, still there, since it last acted
    "herd_success",       # 1 when it has just seen a herd-mate succeed (a kill), fading turn by turn
    "herd_failure",       # 1 when it has just seen a herd-mate fail (die), fading turn by turn
    "climate_discomfort",  # how far the temperature here is from home, as it feels it: 0.5 a step, to 1 two or more
    "comfort_closeness",  # 1 where it feels at home or next to such a row, falling to 0 beyond sight; 0 when none is seen
    "age",             # 0 at birth, 1 at 400 turns or older
    # The time of year, as two hands of a clock a quarter of a year apart, so the year's turn is smooth:
    "time_of_year_a",  # 1 at the start of the year, falling to 0 halfway through, and rising back to 1
    "time_of_year_b",  # the same a quarter of a year later: 0 a quarter in, 1 three quarters in
    "latitude",        # -1 at the top (north) edge of the map, 0 on the equator, 1 at the bottom (south) edge
    # How much warmer than home it is, in steps, 1 two or more too hot, -1 two or more too cold, part
    # way while the weather turns, as the zones' colours show:
    "warmth",          # here
    "warmth_north",    # at the edge of its sight to the north, or the map's edge if nearer
    "warmth_south",    # and to the south
    "food_remembered",    # 1 just after it fed somewhere, fading as its memory lets the place go; 0 when forgotten
    "danger_remembered",  # 1 at the place it was last struck, while it remembers it, falling to 0 beyond sight
    "home_food",          # how well fed its home band is: plants per grazer, prey per predator, 1 at twice what breeding needs; 0 for omnivores
    # Where the nearest row it would feel at home in lies, however far beyond sight:
    "home_direction",     # -1 to the north, 1 to the south, 0 at home or with no such row
    "home_distance",      # 0 at home, rising to 1 half the map's height away or with no such row
    "young_near",         # its own young, still infants, within its sight, 0..3 scaled to 0..1
) + TEMPERAMENT + (    # then its current temperament, mind and social traits, each 0..7 scaled to 0..1
    *(f"inborn_{trait}" for trait in TEMPERAMENT),        # its dot's and social background's traits, as born, 0..1
    *BODY_TRAITS,                                         # its ring's traits, 0..1 (species 0 plant to 1 predator)
    "home_temperature", "heat_tolerance", "cold_tolerance",  # its climate genes: home 0 hot to 1 very cold, flags 0 or 1
)
INPUT_SIZE = len(SENSES)
MAX_DEPTH = 4      # hidden layers: one for each point of inborn intellect, at least one and up to this many
BASE_WIDTH = 38    # neurons in each hidden layer, up to an intellect of MAX_DEPTH
WIDTH_STEP = 5     # neurons more in each layer for each point of intellect beyond MAX_DEPTH
MAX_WIDTH = BASE_WIDTH + WIDTH_STEP * (TRAIT_MAX - MAX_DEPTH)
TEMPERATURE = 0.2
MUTATION = 0.05
TRACE_DECAY = 0.7  # lambda: how much of each earlier choice's credit carries on to the next, beside the discount
SMART_TRACE_DECAY = 0.95  # lambda at the highest intellect: the clever trace credit further back; the least, TRACE_DECAY
# A new brain values setting off in a direction this much below its other actions, so an untaught
# brain doesn't march straight away from its range, its herd and its food before it has learned where to go.
HEADING_RELUCTANCE = 0.5
HEADINGS = ("go_north", "go_south", "go_east", "go_west", "walk_north", "walk_south", "walk_east", "walk_west")
RECORD_DIGITS = 6  # decimal places kept when a brain is exported


def output_spread(width: int) -> float:
    """A new brain's output weights are drawn within this of zero: smaller the wider its layers, so its
    untaught values spread as they did when layers were twelve neurons wide."""
    return 0.3 * math.sqrt(12 / width)


def depth_for(intellect: int) -> int:
    """How many hidden layers a brain has, by its owner's inborn intellect: one a point, from one to MAX_DEPTH."""
    return max(1, min(MAX_DEPTH, intellect))


def width_for(intellect: int) -> int:
    """How many neurons each of its hidden layers has: BASE_WIDTH, and WIDTH_STEP more for each point of
    intellect beyond MAX_DEPTH."""
    return BASE_WIDTH + WIDTH_STEP * max(0, intellect - MAX_DEPTH)


def make_senses(**values: float) -> list[float]:
    """Build an input vector by sense name; unnamed senses are zero."""
    unknown = set(values) - set(SENSES)
    if unknown:
        raise ValueError(f"Unknown senses: {', '.join(sorted(unknown))}.")
    return [float(values.get(name, 0.0)) for name in SENSES]


def gene_senses(dna: int) -> dict[str, float]:
    """The inputs for a genome's traits, ring, dot and social background, and its climate genes."""
    inborn, body = decode_temperament(dna), decode_body(dna)
    temperature, heat, cold = decode_climate(dna)
    values = {f"inborn_{trait}": share(inborn[trait]) for trait in TEMPERAMENT}
    values["species"] = SPECIES.index(str(body["species"])) / (len(SPECIES) - 1)
    values.update({trait: share(int(body[trait])) for trait in BODY_TRAITS[1:]})
    values.update(home_temperature=temperature / (len(TEMPERATURES) - 1), heat_tolerance=float(heat),
                  cold_tolerance=float(cold))
    return values


def choice_odds(values: Sequence[float], options: Sequence[int]) -> dict[int, float]:
    """Softmax over the available actions' values."""
    best = max(values[option] for option in options)
    weights = {option: math.exp((values[option] - best) / TEMPERATURE) for option in options}
    total = sum(weights.values())
    return {option: weight / total for option, weight in weights.items()}


def _tanh(values: np.ndarray) -> np.ndarray:
    """tanh as the C library works it out, which the C++ version uses too; numpy's own differs in the
    last bit now and then."""
    return np.fromiter(map(math.tanh, values), float, len(values))


def _identity_layer(width: int) -> tuple[np.ndarray, np.ndarray]:
    """A hidden layer that passes its input on, near enough: added to deepen a brain without
    changing what it has learned."""
    return np.eye(width), np.zeros(width)


def _fitted(array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """A copy of `array` cut down or padded with zeros to `shape`: neurons, and weights from neurons, a
    layer no longer has are dropped, and those it gains are idle until mutation moves them."""
    fitted = np.zeros(shape)
    common = tuple(slice(0, min(have, want)) for have, want in zip(array.shape, shape))
    fitted[common] = array[common]
    return fitted


_MASK = (1 << 64) - 1


def _ziggurat() -> tuple[list[int], list[float], list[float]]:
    """Marsaglia and Tsang's ziggurat for the normal distribution, 128 layers, as the C++ version builds it."""
    k, w, f = [0] * 128, [0.0] * 128, [0.0] * 128
    m1, vn = 2147483648.0, 9.91256303526217e-3
    dn = tn = 3.442619855899
    q = vn / math.exp(-0.5 * dn * dn)
    k[0] = int((dn / q) * m1)
    k[1] = 0
    w[0], w[127] = q / m1, dn / m1
    f[0], f[127] = 1.0, math.exp(-0.5 * dn * dn)
    for i in range(126, 0, -1):
        dn = math.sqrt(-2.0 * math.log(vn / dn + math.exp(-0.5 * dn * dn)))
        k[i + 1] = int((dn / tn) * m1)
        tn = dn
        f[i] = math.exp(-0.5 * dn * dn)
        w[i] = dn / m1
    return k, w, f


_ZIGGURAT = _ziggurat()


class _Stream:
    """A fast stream of random numbers of a child's own: xoshiro256+, seeded by splitmix64, with normal draws by the
    ziggurat. It draws exactly what the C++ version's does from the same seed."""

    def __init__(self, seed: int) -> None:
        self.state = []
        for _ in range(4):
            seed = (seed + 0x9E3779B97F4A7C15) & _MASK
            z = seed
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
            self.state.append(z ^ (z >> 31))

    def next(self) -> int:
        s = self.state
        result = (s[0] + s[3]) & _MASK
        shifted = (s[1] << 17) & _MASK
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= shifted
        s[3] = ((s[3] << 45) & _MASK) | (s[3] >> 19)
        return result

    def random(self) -> float:
        """In [0, 1), from the top 53 bits."""
        return (self.next() >> 11) * (1.0 / 9007199254740992.0)

    def normal(self) -> float:
        """Of mean 0 and spread 1."""
        k, w, f = _ZIGGURAT
        tail = 3.442620
        while True:
            hz = self.next() >> 32
            if hz >= 1 << 31:
                hz -= 1 << 32
            iz = hz & 127
            x = hz * w[iz]
            if abs(hz) < k[iz]:
                return x
            if iz == 0:  # beyond the base: the tail, by Marsaglia's method
                while True:
                    tx = -math.log(1.0 - self.random()) / tail
                    ty = -math.log(1.0 - self.random())
                    if not ty + ty < tx * tx:
                        break
                return tail + tx if hz > 0 else -tail - tx
            if f[iz] + self.random() * (f[iz - 1] - f[iz]) < math.exp(-0.5 * x * x):
                return x


class Brain:
    """A tanh network of hidden layers estimating the long-run value of each action.

    Its sums are worked out as products added up by numpy's sum, in its fixed pairwise order, and
    its tanh by the C library, so that the C++ version, which follows the same order, matches it to
    the last bit; numpy's matrix product adds up in an order that varies with the machine.

    It learns online from temporal-difference targets and is inherited neuron by neuron, so a
    child keeps whole learned features from one parent or the other. Its weights are NumPy
    arrays: each hidden layer's weights are one row per neuron, the output weights one row per
    action. The first hidden layer reads the senses; each later one reads the layer before.
    Learning from its own choices, it keeps an eligibility trace for every weight (TD(lambda)), so
    what comes of a choice reaches back to the choices that led to it; the traces belong to one
    life and are neither inherited nor exported.
    """

    def __init__(self, layers: Sequence[tuple[Sequence, Sequence]], output_weights: Sequence, output_biases: Sequence):
        self.layers = [(np.array(weights, dtype=float), np.array(biases, dtype=float)) for weights, biases in layers]
        self.output_weights = np.array(output_weights, dtype=float)
        self.output_biases = np.array(output_biases, dtype=float)
        # Each hidden layer's (weights, biases) traces, then the output layer's; None until it first learns.
        self.traces: list[tuple[np.ndarray, np.ndarray]] | None = None

    @property
    def depth(self) -> int:
        return len(self.layers)

    @property
    def width(self) -> int:
        """Neurons in each hidden layer."""
        return len(self.layers[0][1])

    @classmethod
    def random(cls, rng: random.Random, depth: int = 1, width: int = BASE_WIDTH) -> Brain:
        layers = []
        for index in range(depth):
            reads = INPUT_SIZE if index == 0 else width
            bound = math.sqrt(6 / (reads + width))
            layers.append(([[rng.uniform(-bound, bound) for _ in range(reads)] for _ in range(width)], [0.0] * width))
        spread = output_spread(width)
        return cls(layers, [[rng.uniform(-spread, spread) for _ in range(width)] for _ in ACTIONS],
                   [rng.uniform(-0.2, 0.2) - (HEADING_RELUCTANCE if action in HEADINGS else 0.0) for action in ACTIONS])

    @classmethod
    def inherit(cls, first: Brain, second: Brain, rng: random.Random, depth: int = 1, width: int = BASE_WIDTH,
                mutation: float | None = None) -> Brain:
        """A child's brain, `depth` layers of `width` neurons. Where both parents have a layer, each of
        its neurons, with its incoming weights, comes from one parent or the other, or from the one that
        has it where only one does; where only one has the layer, from that one; deeper still, it starts
        passing its input on. A parent's layers are cut down or padded to the child's width (see
        _fitted). Each action's output comes from one parent or the other. Then it mutates (by MUTATION
        unless given)."""
        mutation = MUTATION if mutation is None else mutation
        # Thousands of draws a brain, so they come from a stream of the child's own, seeded from the world's: one
        # draw of the world's a child, as in the C++ version, which makes the same child from the same seed.
        stream = _Stream(rng.getrandbits(64))
        layers = []
        for index in range(depth):
            shape = (width, INPUT_SIZE if index == 0 else width)
            donors = [parent for parent in (first, second) if index < parent.depth]
            if len(donors) == 2:
                from_first = np.array([stream.random() < 0.5 for _ in range(width)])
                neuron = np.arange(width)
                from_first = (from_first & (neuron < first.width)) | (neuron >= second.width)
                layers.append((np.where(from_first[:, None], _fitted(first.layers[index][0], shape),
                                        _fitted(second.layers[index][0], shape)),
                               np.where(from_first, _fitted(first.layers[index][1], (width,)),
                                        _fitted(second.layers[index][1], (width,)))))
            elif donors:
                layers.append((_fitted(donors[0].layers[index][0], shape), _fitted(donors[0].layers[index][1], (width,))))
            else:
                layers.append(_identity_layer(width))
        from_first = np.array([stream.random() < 0.5 for _ in ACTIONS])
        outputs = (len(ACTIONS), width)
        child = cls(layers, np.where(from_first[:, None], _fitted(first.output_weights, outputs),
                                     _fitted(second.output_weights, outputs)),
                    np.where(from_first, first.output_biases, second.output_biases))
        if mutation:
            for weights in ([weights for weights, _ in child.layers] + [child.output_weights]
                            + [biases for _, biases in child.layers] + [child.output_biases]):
                weights += mutation * np.array([stream.normal() for _ in range(weights.size)]).reshape(weights.shape)
        return child

    def resized(self, depth: int, width: int) -> Brain:
        """A copy `depth` layers of `width` neurons: deeper with layers that pass their input on, or
        shallower without its last layers, and each layer cut down or padded (see _fitted)."""
        layers = [(_fitted(weights, (width, INPUT_SIZE if index == 0 else width)), _fitted(biases, (width,)))
                  for index, (weights, biases) in enumerate(self.layers[:depth])]
        layers += [_identity_layer(width) for _ in range(depth - len(layers))]
        return Brain(layers, _fitted(self.output_weights, (len(ACTIONS), width)), self.output_biases)

    def _activations(self, inputs: Sequence[float]) -> list[np.ndarray]:
        """The senses, then each hidden layer's output."""
        activation = np.asarray(inputs, dtype=float)
        if activation.shape != (INPUT_SIZE,):
            raise ValueError(f"Expected {INPUT_SIZE} inputs, got {activation.size}.")
        activations = [activation]
        for weights, biases in self.layers:
            activation = _tanh((weights * activation).sum(axis=1) + biases)
            activations.append(activation)
        return activations

    def values(self, inputs: Sequence[float]) -> list[float]:
        """Return the estimated value of every action in this situation."""
        return ((self.output_weights * self._activations(inputs)[-1]).sum(axis=1) + self.output_biases).tolist()

    def learn(self, inputs: Sequence[float], action: int, target: float, learning_rate: float) -> None:
        """Move one action's value toward a target, with the error clipped for stability, back
        through every layer."""
        if learning_rate <= 0:
            return
        activations = self._activations(inputs)
        top = activations[-1]
        value = float((self.output_weights[action] * top).sum() + self.output_biases[action])
        error = max(-1.0, min(1.0, value - target))
        gradient = error * self.output_weights[action]
        self.output_weights[action] -= learning_rate * error * top
        for index in range(self.depth - 1, -1, -1):
            weights, biases = self.layers[index]
            delta = gradient * (1 - activations[index + 1] * activations[index + 1])
            if index:
                gradient = (weights * delta[:, None]).sum(axis=0)
            biases -= learning_rate * delta
            weights -= learning_rate * np.outer(delta, activations[index])
        self.output_biases[action] -= learning_rate * error

    def learn_traced(self, inputs: Sequence[float], action: int, target: float, learning_rate: float,
                     discount: float, sharpness: float = 0.0) -> None:
        """Learn from its own choice by TD(lambda): every weight's eligibility fades by the discount
        times lambda and gains how this action's value moves with it; then every weight moves by the
        clipped error times its eligibility, so the choices before this one share in it. Lambda runs
        from TRACE_DECAY up to SMART_TRACE_DECAY with `sharpness`, its owner's intellect, 0 to 1."""
        if learning_rate <= 0:
            return
        decay = discount * (TRACE_DECAY + (SMART_TRACE_DECAY - TRACE_DECAY) * sharpness)
        if self.traces is None:
            self.traces = [(np.zeros_like(weights), np.zeros_like(biases)) for weights, biases in self.layers]
            self.traces.append((np.zeros_like(self.output_weights), np.zeros_like(self.output_biases)))
        *hidden, (output_weights, output_biases) = self.traces
        activations = self._activations(inputs)
        top = activations[-1]
        value = float((self.output_weights[action] * top).sum() + self.output_biases[action])
        error = max(-1.0, min(1.0, value - target))
        output_weights *= decay
        output_weights[action] += top
        output_biases *= decay
        output_biases[action] += 1.0
        gradient = self.output_weights[action]
        for index in range(self.depth - 1, -1, -1):
            weights, _ = self.layers[index]
            delta = gradient * (1 - activations[index + 1] * activations[index + 1])
            if index:
                gradient = (weights * delta[:, None]).sum(axis=0)
            weight_trace, bias_trace = hidden[index]
            weight_trace *= decay
            weight_trace += np.outer(delta, activations[index])
            bias_trace *= decay
            bias_trace += delta
        step = learning_rate * error
        for (weights, biases), (weight_trace, bias_trace) in zip(self.layers, hidden):
            weights -= step * weight_trace
            biases -= step * bias_trace
        self.output_weights -= step * output_weights
        self.output_biases -= step * output_biases

    @property
    def nbytes(self) -> int:
        """The memory its weights and traces take."""
        arrays = [array for layer in self.layers for array in layer] + [self.output_weights, self.output_biases]
        if self.traces is not None:
            arrays += [array for layer in self.traces for array in layer]
        return sum(array.nbytes for array in arrays)

    def copy(self) -> Brain:
        brain = Brain(self.layers, self.output_weights, self.output_biases)
        if self.traces is not None:
            brain.traces = [(weights.copy(), biases.copy()) for weights, biases in self.traces]
        return brain

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
        hidden layer, before brains grew deeper, loads as one layer deep. It keeps the width it was
        saved with, and the Bitzkey given it resizes it to its own. Raises ValueError."""
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
        width = len(layers[0][0])
        for index, (weights, biases) in enumerate(layers):
            if len(weights) != width or len(biases) != width or not 0 < width <= MAX_WIDTH:
                raise ValueError(f"The brain has {len(weights)} neurons in a layer; this version uses up to {MAX_WIDTH}, "
                                 "the same in every layer.")
            reads = len(senses) if index == 0 else width
            if any(len(row) != reads for row in weights):
                raise ValueError("The brain's weights don't match its listed senses and actions.")
        if (len(output_weights) != len(actions) or len(output_biases) != len(actions)
                or any(len(row) != width for row in output_weights)):
            raise ValueError("The brain's weights don't match its listed senses and actions.")

        sense_at = {name: index for index, name in enumerate(senses)}
        action_at = {name: index for index, name in enumerate(actions)}
        matched = [([[row[sense_at[name]] if name in sense_at else 0.0 for name in SENSES] for row in weights]
                    if index == 0 else weights, biases) for index, (weights, biases) in enumerate(layers)]
        return cls(
            matched,
            [output_weights[action_at[name]] if name in action_at else [0.0] * width for name in ACTIONS],
            [output_biases[action_at[name]] if name in action_at else 0.0 for name in ACTIONS],
        )

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Brain) and self.depth == other.depth
                and all(np.array_equal(mine[0], theirs[0]) and np.array_equal(mine[1], theirs[1])
                        for mine, theirs in zip(self.layers, other.layers))
                and np.array_equal(self.output_weights, other.output_weights)
                and np.array_equal(self.output_biases, other.output_biases))
