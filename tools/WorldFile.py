"""Saving a whole world to a file and loading it back, to carry on exactly as it would have.

A world file is the same for the Python and C++ versions, so a world saved by either loads in both:

    WORLD_MAGIC (16 bytes), the JSON part's length (8 bytes, little-endian), the JSON part (UTF-8),
    then the numbers part: 64-bit little-endian floats.

The JSON part holds the world's settings, its rules, and everything in it: the random generator's state,
the grass, the seeds waiting to sprout, every Bitzkey and every field of it, the insights, the census, the
genomes of all who ever lived and the record of every death. Brains and senses would make the JSON part
enormous, and must come back exactly, so they go in the numbers part, and the JSON part names where:

- a brain is {"at": its first number, "depth": hidden layers, "traced": whether learning traces follow}.
  Its numbers are each hidden layer's weights (HIDDEN_SIZE rows, each reading the senses or the layer
  before) and biases, then the output weights (a row of HIDDEN_SIZE for each action) and biases; then,
  if traced, its traces, laid out the same.
- a set of senses (its last, or a memory's) is the position of INPUT_SIZE numbers.

A genome is 24 hex digits: the social background word, then the ring, then the dot.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from tools import Rules
from tools import Simulation as world
from tools.Brain import ACTIONS, HIDDEN_SIZE, INPUT_SIZE, SENSES, Brain
from tools.Genome import TEMPERAMENT
from tools.Simulation import Bitzkey, Death, Simulation

WORLD_MAGIC = b"BITZKEY WORLD 1\n"
WORLD_FORMAT = 1
WORLD_SUFFIX = ".bzworld"
# Rules that say how this computer runs a world rather than how the world unfolds: loading keeps the user's own.
MACHINE_RULES = ("USE_GPU", "HISTORY_TURNS", "HISTORY_MEMORY")


def _optional_list(value) -> list | None:
    return None if value is None else list(value)


class _Numbers:
    """The numbers part, as it is written: each array's numbers appended, and where they start."""

    def __init__(self) -> None:
        self.parts: list[np.ndarray] = []
        self.count = 0

    def add(self, values) -> int:
        array = np.ascontiguousarray(values, dtype="<f8").ravel()
        self.parts.append(array)
        at = self.count
        self.count += array.size
        return at

    def brain(self, brain: Brain | None) -> dict | None:
        if brain is None:
            return None
        sets = [(brain.layers, brain.output_weights, brain.output_biases)]
        if brain.traces is not None:
            *hidden, (weights, biases) = brain.traces
            sets.append((hidden, weights, biases))
        at = self.count
        for layers, output_weights, output_biases in sets:
            for weights, biases in layers:
                self.add(weights)
                self.add(biases)
            self.add(output_weights)
            self.add(output_biases)
        return {"at": at, "depth": brain.depth, "traced": brain.traces is not None}

    def senses(self, senses) -> int | None:
        return None if senses is None else self.add(senses)

    def data(self) -> bytes:
        return b"".join(part.tobytes() for part in self.parts)


class _Reader:
    """The numbers part, as it is read back."""

    def __init__(self, data: bytes) -> None:
        if len(data) % 8:
            raise ValueError("The world file's numbers are cut short.")
        self.numbers = np.frombuffer(data, dtype="<f8")

    def take(self, at: int, count: int) -> np.ndarray:
        if at < 0 or at + count > self.numbers.size:
            raise ValueError("The world file's numbers are cut short.")
        return self.numbers[at:at + count].astype(float)

    def brain(self, record: Mapping | None) -> Brain | None:
        if record is None:
            return None
        at, depth = int(record["at"]), int(record["depth"])
        sets = []
        for _ in range(2 if record["traced"] else 1):
            layers = []
            for index in range(depth):
                reads = INPUT_SIZE if index == 0 else HIDDEN_SIZE
                weights = self.take(at, HIDDEN_SIZE * reads).reshape(HIDDEN_SIZE, reads)
                at += HIDDEN_SIZE * reads
                biases = self.take(at, HIDDEN_SIZE)
                at += HIDDEN_SIZE
                layers.append((weights, biases))
            output_weights = self.take(at, len(ACTIONS) * HIDDEN_SIZE).reshape(len(ACTIONS), HIDDEN_SIZE)
            at += len(ACTIONS) * HIDDEN_SIZE
            output_biases = self.take(at, len(ACTIONS))
            at += len(ACTIONS)
            sets.append((layers, output_weights, output_biases))
        brain = Brain(*sets[0])
        if len(sets) > 1:
            layers, output_weights, output_biases = sets[1]
            brain.traces = [(weights.copy(), biases.copy()) for weights, biases in layers]
            brain.traces.append((output_weights.copy(), output_biases.copy()))
        return brain

    def senses(self, at: int | None) -> list[float] | None:
        return None if at is None else self.take(int(at), INPUT_SIZE).tolist()


def _genome(dna: int) -> str:
    return f"{dna:024X}"


def _mind(mind: Mapping[str, float]) -> list[float]:
    return [float(mind[trait]) for trait in TEMPERAMENT]


def _mind_from(values) -> dict[str, float]:
    return {trait: float(value) for trait, value in zip(TEMPERAMENT, values)}


def _agent_record(agent: Bitzkey, numbers: _Numbers) -> dict:
    return {
        "identifier": agent.identifier, "dna": _genome(agent.dna), "x": agent.x, "y": agent.y,
        "hunger": float(agent.hunger), "health": float(agent.health), "age": agent.age, "cooldown": agent.cooldown,
        "last_action": agent.last_action, "brain": numbers.brain(agent.brain), "generation": agent.generation,
        "children": agent.children, "died_turn": agent.died_turn, "assailant": agent.assailant,
        "panicked": agent.panicked, "primal_fear": agent.primal_fear, "rivalry": float(agent.rivalry),
        "own_rivalry": float(agent.own_rivalry), "hunts_grazers": agent.hunts_grazers,
        "hunts_omnivores": agent.hunts_omnivores, "hunts_predators": agent.hunts_predators, "guarded": agent.guarded,
        "alarm": agent.alarm, "kill_alert": _optional_list(agent.kill_alert),
        "seen_success": float(agent.seen_success), "seen_failure": float(agent.seen_failure),
        "herd_assailant": agent.herd_assailant, "food_memory": _optional_list(agent.food_memory),
        "danger_memory": _optional_list(agent.danger_memory), "meat": float(agent.meat),
        "meat_fill": float(agent.meat_fill), "population": agent.population, "cost_share": float(agent.cost_share),
        "glut": float(agent.glut), "rot": float(agent.rot), "portions": agent.portions,
        "trail": [list(cell) for cell in agent.trail],
        "memories": [[numbers.senses(senses), int(choice), float(target), float(weight)]
                     for senses, choice, target, weight in agent.memories],
        "womb": [[_genome(dna), _mind(mind), numbers.brain(brain), float(energy), int(generation)]
                 for dna, mind, brain, energy, generation in agent.womb],
        "due_turn": agent.due_turn, "mind": _mind(agent.mind), "last_senses": numbers.senses(agent.last_senses),
        "last_choice": int(agent.last_choice), "last_wellbeing": float(agent.last_wellbeing),
        "reward": float(agent.reward), "last_comfort": float(agent.last_comfort),
        "odds": {action: float(chance) for action, chance in agent.odds.items()},
    }


def _agent_from(record: Mapping, numbers: _Reader) -> Bitzkey:
    agent = Bitzkey(int(record["identifier"]), int(record["dna"], 16), int(record["x"]), int(record["y"]),
                    mind=_mind_from(record["mind"]))
    for name in ("hunger", "health", "rivalry", "own_rivalry", "seen_success", "seen_failure", "meat", "meat_fill",
                 "cost_share", "glut", "rot", "last_wellbeing", "reward", "last_comfort"):
        setattr(agent, name, float(record[name]))
    for name in ("age", "cooldown", "generation", "children", "population", "portions", "last_choice"):
        setattr(agent, name, int(record[name]))
    for name in ("panicked", "primal_fear", "hunts_grazers", "hunts_omnivores", "hunts_predators", "guarded"):
        setattr(agent, name, bool(record[name]))
    for name in ("died_turn", "assailant", "alarm", "herd_assailant", "due_turn"):
        setattr(agent, name, None if record[name] is None else int(record[name]))
    for name in ("kill_alert", "food_memory", "danger_memory"):
        setattr(agent, name, None if record[name] is None else tuple(int(value) for value in record[name]))
    agent.last_action = str(record["last_action"])
    agent.brain = numbers.brain(record["brain"])
    agent.trail = [(int(x), int(y)) for x, y in record["trail"]]
    agent.memories = [(numbers.senses(senses), int(choice), float(target), float(weight))
                      for senses, choice, target, weight in record["memories"]]
    agent.womb = [(int(dna, 16), _mind_from(mind), numbers.brain(brain), float(energy), int(generation))
                  for dna, mind, brain, energy, generation in record["womb"]]
    agent.last_senses = numbers.senses(record["last_senses"])
    agent.odds = {str(action): float(chance) for action, chance in record["odds"].items()}
    return agent


def world_bytes(simulation: Simulation) -> bytes:
    """The whole world, as a world file holds it."""
    numbers = _Numbers()
    version, internal, gauss_next = simulation.random.getstate()
    record = {
        "format": WORLD_FORMAT, "senses": list(SENSES), "actions": list(ACTIONS), "temperament": list(TEMPERAMENT),
        "settings": {"width": simulation.width, "height": simulation.height,
                     "population_cap": simulation.population_cap, "mind_mutation": simulation.mind_mutation,
                     "body_mutation": simulation.body_mutation, "year_length": simulation.year_length,
                     "initial_grass": simulation._initial_grass},
        "rules": Rules.current(),
        "turn": simulation.turn, "deaths": simulation.deaths, "next_identifier": simulation.next_identifier,
        "death_causes": {species: dict(causes) for species, causes in simulation.death_causes.items()},
        "random": {"mt": list(internal[:624]), "index": internal[624], "gauss_next": gauss_next},
        "grass": [simulation.grass.get((x, y), 0) for y in range(simulation.height) for x in range(simulation.width)],
        "seeds": [[x, y, _genome(dna), generation, sown] for x, y, dna, generation, sown in simulation.seeds],
        "insights": [[*key, offered, simulation.insights.chosen.get(key, 0.0)]
                     for key, offered in simulation.insights.offered.items()]
                    + [[*key, 0.0, chosen] for key, chosen in simulation.insights.chosen.items()
                       if key not in simulation.insights.offered],
        "census": {"stride": simulation.census.stride,
                   "snapshots": [[turn, list(alive)] for turn, alive in simulation.census.snapshots]},
        "lineage": [_genome(simulation.lineage.get(identifier, 0))
                    for identifier in range(max(simulation.lineage, default=-1) + 1)],
        "death_records": [[identifier, *death] for identifier, death in sorted(simulation.death_records.items())],
        "agents": [_agent_record(agent, numbers) for agent in simulation.agents],
    }
    text = json.dumps(record, separators=(",", ":")).encode("utf-8")
    return WORLD_MAGIC + struct.pack("<Q", len(text)) + text + numbers.data()


def world_from_bytes(data: bytes) -> Simulation:
    """The world a world file holds, ready to carry on as it would have; its rules become the rules, all but
    those saying how this computer runs it. Raises ValueError, naming the problem, if it can't be read."""
    if not data.startswith(WORLD_MAGIC):
        raise ValueError("This isn't a Bitzkey world file.")
    start = len(WORLD_MAGIC) + 8
    if len(data) < start:
        raise ValueError("The world file is cut short.")
    (length,) = struct.unpack("<Q", data[len(WORLD_MAGIC):start])
    try:
        record = json.loads(data[start:start + length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"The world file is damaged ({error}).") from None
    try:
        return _world_from(record, _Reader(data[start + length:]))
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"The world file is incomplete ({error}).") from None


def _world_from(record: Mapping, numbers: _Reader) -> Simulation:
    if record.get("format") != WORLD_FORMAT:
        raise ValueError(f"This world file is format {record.get('format')}; this version reads format {WORLD_FORMAT}.")
    if (record["senses"], record["actions"], record["temperament"]) != (list(SENSES), list(ACTIONS), list(TEMPERAMENT)):
        raise ValueError("This world was saved by a version of Bitzkey whose brains sense or act differently.")
    rules = {key: value for key, value in record["rules"].items() if key in Rules.BY_KEY}
    rules.update({key: Rules.get(key) for key in MACHINE_RULES})
    Rules.apply(rules)
    settings = record["settings"]
    simulation = Simulation(settings["width"], settings["height"], population=0, seed=0, grass_density=0,
                            population_cap=settings["population_cap"], mind_mutation=100 * settings["mind_mutation"],
                            body_mutation=100 * settings["body_mutation"], year_length=settings["year_length"])
    simulation.mind_mutation, simulation.body_mutation = settings["mind_mutation"], settings["body_mutation"]
    simulation._initial_grass = settings["initial_grass"]
    simulation.grass_regrowth = simulation._initial_grass * world.GRASS_REGROWTH
    simulation.turn = record["turn"]
    simulation.deaths = record["deaths"]
    simulation.next_identifier = record["next_identifier"]
    simulation.death_causes = {species: dict(causes) for species, causes in record["death_causes"].items()}
    state = record["random"]
    simulation.random.setstate((3, (*state["mt"], state["index"]), state["gauss_next"]))
    width = simulation.width
    simulation.grass = {(index % width, index // width): tufts for index, tufts in enumerate(record["grass"]) if tufts}
    simulation.seeds = [(x, y, int(dna, 16), generation, sown) for x, y, dna, generation, sown in record["seeds"]]
    simulation.insights.offered = {}
    simulation.insights.chosen = {}
    for species, trait, level, action, offered, chosen in record["insights"]:
        key = (species, trait, level, action)
        if offered:
            simulation.insights.offered[key] = offered
        if chosen:
            simulation.insights.chosen[key] = chosen
    simulation.census.stride = record["census"]["stride"]
    simulation.census.snapshots = [(turn, tuple(alive)) for turn, alive in record["census"]["snapshots"]]
    simulation.lineage = {identifier: int(dna, 16) for identifier, dna in enumerate(record["lineage"])}
    simulation.death_records = {identifier: Death(*death) for identifier, *death in record["death_records"]}
    simulation.agents = [_agent_from(agent, numbers) for agent in record["agents"]]
    simulation._update_climate()
    simulation._index_cells()
    # What each knew as of its turn's start, which indexing the cells afresh would count again as things now stand.
    for agent, saved in zip(simulation.agents, record["agents"]):
        agent.population, agent.cost_share = int(saved["population"]), float(saved["cost_share"])
        agent.glut, agent.guarded = float(saved["glut"]), bool(saved["guarded"])
    simulation.history.clear()
    simulation._history_bytes = 0
    simulation._record()
    return simulation


def save_world(simulation: Simulation, path: str | Path) -> None:
    Path(path).write_bytes(world_bytes(simulation))


def load_world(path: str | Path) -> Simulation:
    return world_from_bytes(Path(path).read_bytes())
