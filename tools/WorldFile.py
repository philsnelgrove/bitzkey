"""Saving a whole world to a file and loading it back, to carry on exactly as it would have.

A world file is the same for the Python and C++ versions, so a world saved by either loads in both.
Format 2, written now:

    WORLD_MAGIC (16 bytes), the JSON part's length (8 bytes, little-endian), the JSON part (UTF-8),
    the count of numbers (8 bytes), the numbers part: 64-bit little-endian floats,
    the count of whole numbers (8 bytes), the whole-numbers part: 32-bit little-endian integers.

The JSON part holds the world's settings, its rules, and everything in it: the random generator's state,
the grass, the seeds waiting to sprout, every Bitzkey and every field of it, and the insights. What would
make it enormous goes in the other parts, and the JSON part names where:

- a brain is {"at": its first number, "depth": hidden layers, "width": neurons in each, "traced": whether
  learning traces follow}. Its numbers are each hidden layer's weights (width rows, each reading the senses
  or the layer before) and biases, then the output weights (a row of width for each action) and biases;
  then, if traced, its traces, laid out the same. A brain saved without a width, before brains grew wider
  with intellect, is OLD_WIDTH wide, and loads resized to its owner's shape, without its traces.
- a set of senses (its last, or a memory's) is the position of INPUT_SIZE numbers.
- the census's snapshots are [turn, the first whole number, how many]: the identifiers alive then.
- "lineage" {"at", "count", "first"}: the genome of every Bitzkey from identifier "first" on (0 if missing;
  those before it are off the graph and forgotten), three whole numbers each (the social background word,
  the ring, the dot, as unsigned words).
- "death_records" {"at", "count"}: every death since the graph's start, seven whole numbers each: the dead's identifier, its cause
  (an index into "death_strings"), its species (an index into SPECIES), the turn, the killer's identifier,
  its species and what it was doing (an index into "death_strings"); the last three -1 for none.

Elsewhere a genome is 24 hex digits: the social background word, then the ring, then the dot. Format 1 had no
counts and no whole-numbers part, the numbers running to the file's end, and kept the census, lineage and
deaths in the JSON part; it still loads.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from tools import Rules
from tools import Simulation as world
from tools.Brain import ACTIONS, INPUT_SIZE, MAX_WIDTH, SENSES, Brain
from tools.Genome import TEMPERAMENT
from tools.Simulation import SPECIES, Bitzkey, Death, Simulation, brain_shape

WORLD_MAGIC = b"BITZKEY WORLD 2\n"
WORLD_MAGIC_1 = b"BITZKEY WORLD 1\n"  # format 1's, still read
WORLD_FORMAT = 2
WORD = 0xFFFFFFFF
WORLD_SUFFIX = ".bzworld"
OLD_WIDTH = 34  # neurons in each hidden layer of a brain saved without a width
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
        return {"at": at, "depth": brain.depth, "width": brain.width, "traced": brain.traces is not None}

    def senses(self, senses) -> int | None:
        return None if senses is None else self.add(senses)

    def data(self) -> bytes:
        return b"".join(part.tobytes() for part in self.parts)


class _Reader:
    """The numbers part, as it is read back, and the whole numbers."""

    def __init__(self, data: bytes, ints: bytes = b"") -> None:
        if len(data) % 8:
            raise ValueError("The world file's numbers are cut short.")
        self.numbers = np.frombuffer(data, dtype="<f8")
        self.ints = np.frombuffer(ints, dtype="<i4")
        # For a world saved with other senses: how many it saved, and where each sensed now lies among them, or -1.
        self.saved_inputs = INPUT_SIZE
        self.sense_from: list[int] | None = None

    def whole(self, at: int, count: int) -> np.ndarray:
        if at < 0 or at + count > self.ints.size:
            raise ValueError("The world file's whole numbers are cut short.")
        return self.ints[at:at + count]

    def take(self, at: int, count: int) -> np.ndarray:
        if at < 0 or at + count > self.numbers.size:
            raise ValueError("The world file's numbers are cut short.")
        return self.numbers[at:at + count].astype(float)

    def brain(self, record: Mapping | None, dna: int) -> Brain | None:
        """The brain saved at `record`, resized to fit the genome `dna` if it was saved another shape."""
        if record is None:
            return None
        at, depth, width = int(record["at"]), int(record["depth"]), int(record.get("width", OLD_WIDTH))
        if depth < 1 or not 0 < width <= MAX_WIDTH:
            raise ValueError("The world file holds a brain of no sensible shape.")
        sets = []
        for _ in range(2 if record["traced"] else 1):
            layers = []
            for index in range(depth):
                if index == 0:  # a row of senses for each neuron
                    weights = np.array([self._as_senses(self.take(at + neuron * self.saved_inputs, self.saved_inputs))
                                        for neuron in range(width)])
                    at += width * self.saved_inputs
                else:
                    weights = self.take(at, width * width).reshape(width, width)
                    at += width * width
                biases = self.take(at, width)
                at += width
                layers.append((weights, biases))
            output_weights = self.take(at, len(ACTIONS) * width).reshape(len(ACTIONS), width)
            at += len(ACTIONS) * width
            output_biases = self.take(at, len(ACTIONS))
            at += len(ACTIONS)
            sets.append((layers, output_weights, output_biases))
        brain = Brain(*sets[0])
        if len(sets) > 1:
            layers, output_weights, output_biases = sets[1]
            brain.traces = [(weights.copy(), biases.copy()) for weights, biases in layers]
            brain.traces.append((output_weights.copy(), output_biases.copy()))
        shape = brain_shape(dna)
        return brain if (brain.depth, brain.width) == shape else brain.resized(*shape)

    def senses(self, at: int | None) -> list[float] | None:
        return None if at is None else self._as_senses(self.take(int(at), self.saved_inputs)).tolist()

    def _as_senses(self, saved: np.ndarray) -> np.ndarray:
        """One set of saved senses, or one neuron's weights from them, as this version's senses."""
        if self.sense_from is None:
            return saved
        return np.array([0.0 if index < 0 else saved[index] for index in self.sense_from])


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
    agent.brain = numbers.brain(record["brain"], agent.dna)
    agent.trail = [(int(x), int(y)) for x, y in record["trail"]]
    agent.memories = [(numbers.senses(senses), int(choice), float(target), float(weight))
                      for senses, choice, target, weight in record["memories"]]
    agent.womb = [(int(dna, 16), _mind_from(mind), numbers.brain(brain, int(dna, 16)), float(energy), int(generation))
                  for dna, mind, brain, energy, generation in record["womb"]]
    agent.last_senses = numbers.senses(record["last_senses"])
    agent.odds = {str(action): float(chance) for action, chance in record["odds"].items()}
    return agent


def world_bytes(simulation: Simulation) -> bytes:
    """The whole world, as a world file holds it."""
    numbers = _Numbers()
    ints: list[np.ndarray] = []  # the census, the lineage and the deaths
    held = 0

    def whole(values) -> int:
        nonlocal held
        array = np.ascontiguousarray(values, dtype="<i4").ravel()
        ints.append(array)
        at, held = held, held + array.size
        return at

    snapshots = [[turn, whole(alive), len(alive)] for turn, alive in simulation.census.snapshots]
    first = simulation.lineage_first
    lineage = [simulation.lineage.get(identifier, 0)
               for identifier in range(first, max(simulation.lineage, default=first - 1) + 1)]
    lineage_at = whole(np.array([[dna >> 64 & WORD, dna >> 32 & WORD, dna & WORD] for dna in lineage],
                                dtype="<u4").reshape(-1, 3).view("<i4"))
    strings: dict[str, int] = {}  # the causes and doings, each once
    code = lambda text: strings.setdefault(text, len(strings))  # noqa: E731
    deaths = sorted(simulation.death_records.items())
    rows = [[identifier, code(death.cause), SPECIES.index(death.species), death.turn,
             -1 if death.killer is None else death.killer,
             -1 if death.killer_species is None else SPECIES.index(death.killer_species),
             -1 if death.activity is None else code(death.activity)] for identifier, death in deaths]
    deaths_at = whole(np.array(rows, dtype="<i4").reshape(-1, 7))
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
        "census": {"stride": simulation.census.stride, "snapshots": snapshots},
        "lineage": {"at": lineage_at, "count": len(lineage), "first": first},
        "death_strings": list(strings), "death_records": {"at": deaths_at, "count": len(rows)},
        "agents": [_agent_record(agent, numbers) for agent in simulation.agents],
    }
    text = json.dumps(record, separators=(",", ":")).encode("utf-8")
    whole_part = b"".join(array.tobytes() for array in ints)
    return (WORLD_MAGIC + struct.pack("<Q", len(text)) + text + struct.pack("<Q", numbers.count) + numbers.data()
            + struct.pack("<Q", held) + whole_part)


def world_from_bytes(data: bytes) -> Simulation:
    """The world a world file holds, ready to carry on as it would have; its rules become the rules, all but
    those saying how this computer runs it. Raises ValueError, naming the problem, if it can't be read."""
    first_format = data.startswith(WORLD_MAGIC_1)
    if not first_format and not data.startswith(WORLD_MAGIC):
        raise ValueError("This isn't a Bitzkey world file.")
    at = len(WORLD_MAGIC)

    def counted(width: int) -> int:  # a count, then the room for that many values `width` bytes each
        nonlocal at
        if len(data) < at + 8:
            raise ValueError("The world file is cut short.")
        (count,) = struct.unpack("<Q", data[at:at + 8])
        at += 8
        if count > (len(data) - at) // width:
            raise ValueError("The world file is cut short.")
        return count

    length = counted(1)
    text, at = data[at:at + length], at + length
    if first_format:  # the numbers run to the end
        reader = _Reader(data[at:])
    else:
        count = counted(8)
        numbers, at = data[at:at + 8 * count], at + 8 * count
        count = counted(4)
        reader = _Reader(numbers, data[at:at + 4 * count])
    try:
        record = json.loads(text.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"The world file is damaged ({error}).") from None
    try:
        return _world_from(record, reader, 1 if first_format else WORLD_FORMAT)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"The world file is incomplete ({error}).") from None


def _world_from(record: Mapping, numbers: _Reader, format_: int) -> Simulation:
    if record.get("format") != format_:
        raise ValueError(f"This world file is format {record.get('format')}; this version reads format {WORLD_FORMAT}.")
    if (record["actions"], record["temperament"]) != (list(ACTIONS), list(TEMPERAMENT)):
        raise ValueError("This world was saved by a version of Bitzkey whose brains sense or act differently.")
    # Brains saved with other senses are read by name: a sense they lacked starts with no weight, and one no longer
    # sensed is let go.
    if record["senses"] != list(SENSES):
        saved_at = {name: index for index, name in enumerate(record["senses"])}
        numbers.saved_inputs = len(record["senses"])
        numbers.sense_from = [saved_at.get(name, -1) for name in SENSES]
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
    if format_ == 1:
        simulation.census.snapshots = [(turn, tuple(alive)) for turn, alive in record["census"]["snapshots"]]
        simulation.lineage = {identifier: int(dna, 16) for identifier, dna in enumerate(record["lineage"])}
        simulation.death_records = {identifier: Death(*death) for identifier, *death in record["death_records"]}
    else:
        simulation.census.snapshots = [(turn, tuple(numbers.whole(first, count).tolist()))
                                       for turn, first, count in record["census"]["snapshots"]]
        lineage = record["lineage"]
        words = numbers.whole(lineage["at"], 3 * lineage["count"]).view("<u4").reshape(-1, 3).tolist()
        simulation.lineage_first = lineage.get("first", 0)
        simulation.lineage = {identifier: social << 64 | ring << 32 | dot
                              for identifier, (social, ring, dot) in enumerate(words, start=simulation.lineage_first)}
        strings, deaths = record["death_strings"], record["death_records"]
        species = lambda index: None if index < 0 else SPECIES[index]  # noqa: E731
        simulation.death_records = {
            identifier: Death(strings[cause], SPECIES[kind], turn, None if killer < 0 else killer, species(killer_kind),
                              None if doing < 0 else strings[doing])
            for identifier, cause, kind, turn, killer, killer_kind, doing
            in numbers.whole(deaths["at"], 7 * deaths["count"]).reshape(-1, 7).tolist()}
    simulation._order_deaths()
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
