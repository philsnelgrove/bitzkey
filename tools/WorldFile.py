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
- "death_tally": every death since the world began, counted: "causes" and "activities" {label: four counts, by species},
  "whom" (four rows of four counts: the dead's species, by the killers'), "killers" (how many ever killed), and
  "by_killer" [[identifier, species, kills, {doing: kills}]...], the killers kept. Missing, as from versions before it,
  it is tallied from the deaths the file keeps whole.
- "death_records" {"at", "count"}: every death of the last RECENT_DEATH_TURNS turns, seven whole numbers each: the dead's identifier, its cause
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
from tools.Brain import ACTIONS, INPUT_SIZE, MAX_WIDTH, OUTPUTS, SENSES, VALUE_SCALE, Brain
from tools.Genome import GENOME_VERSION, TEMPERAMENT, convert_v3
from tools.Insights import DeathTally
from tools.Simulation import SPECIES, Bitzkey, Death, Simulation, brain_shape

WORLD_MAGIC = b"BITZKEY WORLD 2\n"
WORLD_MAGIC_1 = b"BITZKEY WORLD 1\n"  # format 1's, still read
WORLD_FORMAT = 2
WORD = 0xFFFFFFFF
WORLD_SUFFIX = ".bzworld"
OLD_WIDTH = 34  # neurons in each hidden layer of a brain saved without a width
# Rules that say how this computer runs a world rather than how the world unfolds: loading keeps the user's own.
MACHINE_RULES = ("USE_GPU",)


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
        # For a world saved with other outputs (actions, calls): how many it saved, where each output now lies among
        # them, or -1, and what each action it saved is now, or -1 for one no longer an action.
        self.saved_outputs = len(OUTPUTS)
        self.output_from: list[int] | None = None
        self.action_now: list[int] | None = None
        self.uncentred = False  # saved before brains read their senses centred: each is made to, choosing as before
        self.unscaled = False  # saved before brains held their values in hundredths: each is made to

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
            # An output added since starts unweighted, and one no longer had is let go.
            saved = self.saved_outputs
            saved_weights = self.take(at, saved * width).reshape(saved, width)
            at += saved * width
            saved_biases = self.take(at, saved)
            at += saved
            output_from = self.output_from if self.output_from is not None else range(len(OUTPUTS))
            output_weights, output_biases = np.zeros((len(OUTPUTS), width)), np.zeros(len(OUTPUTS))
            for output, source in enumerate(output_from):
                if source >= 0:
                    output_weights[output], output_biases[output] = saved_weights[source], saved_biases[source]
            sets.append((layers, output_weights, output_biases))
        brain = Brain(*sets[0])
        if len(sets) > 1:
            layers, output_weights, output_biases = sets[1]
            brain.traces = [(weights.copy(), biases.copy()) for weights, biases in layers]
            brain.traces.append((output_weights.copy(), output_biases.copy()))
        if self.uncentred:
            brain.centre_inputs()
        if self.unscaled:
            brain.scale_values()
        shape = brain_shape(dna)
        return brain if (brain.depth, brain.width) == shape else brain.resized(*shape)

    def action(self, saved: int) -> int:
        """What an action it saved is now, or -1 for one no longer an action."""
        if self.action_now is None:
            return saved
        return self.action_now[saved] if 0 <= saved < len(self.action_now) else -1

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
        "hunger": float(agent.hunger), "health": float(agent.health), "stamina": float(agent.stamina), "age": agent.age,
        "cooldown": agent.cooldown,
        "last_action": agent.last_action, "brain": numbers.brain(agent.brain), "generation": agent.generation,
        "children": agent.children, "mother": agent.mother, "father": agent.father, "died_turn": agent.died_turn, "assailant": agent.assailant,
        "panicked": agent.panicked, "primal_fear": agent.primal_fear, "rivalry": float(agent.rivalry),
        "own_rivalry": float(agent.own_rivalry), "hunts_grazers": agent.hunts_grazers,
        "hunts_omnivores": agent.hunts_omnivores, "hunts_predators": agent.hunts_predators, "guarded": agent.guarded,
        "alarm": agent.alarm, "kill_alert": _optional_list(agent.kill_alert),
        "seen_success": float(agent.seen_success), "seen_failure": float(agent.seen_failure),
        "weathered": float(agent.weathered),
        "herd_assailant": agent.herd_assailant, "food_memory": _optional_list(agent.food_memory),
        "danger_memory": _optional_list(agent.danger_memory), "meat": float(agent.meat),
        "meat_fill": float(agent.meat_fill), "population": agent.population, "cost_share": float(agent.cost_share),
        "glut": float(agent.glut), "rot": float(agent.rot), "portions": agent.portions,
        "trail": [list(cell) for cell in agent.trail],
        "memories": [[numbers.senses(senses), int(choice), float(target), float(weight)]
                     for senses, choice, target, weight in agent.memories],
        "diary": [[numbers.senses(senses), int(choice), float(target), float(weight), float(vividness), int(turn)]
                  for (senses, choice, target, weight), vividness, turn in agent.diary],
        "womb": [[_genome(dna), _mind(mind), numbers.brain(brain), float(energy), int(generation), int(father)]
                 for dna, mind, brain, energy, generation, father in agent.womb],
        "due_turn": agent.due_turn, "mind": _mind(agent.mind), "last_senses": numbers.senses(agent.last_senses),
        "last_choice": int(agent.last_choice), "last_forced": bool(agent.last_forced), "last_panicked": bool(agent.last_panicked), "last_wellbeing": float(agent.last_wellbeing),
        "reward": float(agent.reward), "last_comfort": float(agent.last_comfort),
        "odds": {action: float(chance) for action, chance in agent.odds.items()},
        "call": agent.call, "call_turn": agent.call_turn, "last_call": agent.last_call,
        "held": [float(held) for held in agent.held], "last_wrote": bool(agent.last_wrote),
        "past": [numbers.senses(senses) for senses in agent.past],
        # each [column, row, last, then for each season: seen, food, danger, comfort]
        "places": [[place.column, place.row, place.last,
                    *(value for quarter in range(4) for value in (place.seen[quarter], float(place.food[quarter]),
                                                                  float(place.danger[quarter]), float(place.comfort[quarter])))]
                   for place in agent.places],
        "known": [[one.identifier, float(one.regard), one.last] for one in agent.known],  # each [identifier, regard, last]
    }


def _agent_from(record: Mapping, numbers: _Reader, before_terrain: bool = False) -> Bitzkey:
    """A Bitzkey as saved; one saved before the terrain gene is terrestrial, as are the young it carries."""
    genome = (lambda text: convert_v3(int(text, 16))) if before_terrain else (lambda text: int(text, 16))  # noqa: E731
    agent = Bitzkey(int(record["identifier"]), genome(record["dna"]), int(record["x"]), int(record["y"]),
                    mind=_mind_from(record["mind"]))
    agent.stamina = float(record.get("stamina", world.FULL_STAMINA))  # fresh, in a world saved before stamina
    for name in ("hunger", "health", "rivalry", "own_rivalry", "seen_success", "seen_failure", "meat", "meat_fill",
                 "cost_share", "glut", "rot", "last_wellbeing", "reward", "last_comfort"):
        setattr(agent, name, float(record[name]))
    for name in ("age", "cooldown", "generation", "children", "population", "portions", "last_choice"):
        setattr(agent, name, int(record[name]))
    agent.weathered = float(record.get("weathered", 0.0))  # worlds saved before the weather's grace
    agent.last_forced = bool(record.get("last_forced", False))  # chosen, in a world saved before it was kept
    agent.last_panicked = bool(record.get("last_panicked", False))
    for name in ("panicked", "primal_fear", "hunts_grazers", "hunts_omnivores", "hunts_predators", "guarded"):
        setattr(agent, name, bool(record[name]))
    for name in ("died_turn", "assailant", "alarm", "herd_assailant", "due_turn"):
        setattr(agent, name, None if record[name] is None else int(record[name]))
    for name in ("kill_alert", "food_memory", "danger_memory"):
        setattr(agent, name, None if record[name] is None else tuple(int(value) for value in record[name]))
    agent.last_action = str(record["last_action"])
    agent.brain = numbers.brain(record["brain"], agent.dna)
    agent.trail = [(int(x), int(y)) for x, y in record["trail"]]
    agent.memories = [(numbers.senses(senses), numbers.action(int(choice)), float(target), float(weight))  # one of an action
                      for senses, choice, target, weight in record["memories"] if numbers.action(int(choice)) >= 0]  # no longer had is forgotten
    agent.diary = [[(numbers.senses(entry[0]), numbers.action(int(entry[1])), float(entry[2]), float(entry[3])),
                    float(entry[4]), int(entry[5])]
                   for entry in record.get("diary", []) if numbers.action(int(entry[1])) >= 0]
    agent.womb = [(genome(young[0]), _mind_from(young[1]), numbers.brain(young[2], genome(young[0])), float(young[3]),
                   int(young[4]), int(young[5]) if len(young) > 5 else -1)
                  for young in record["womb"]]
    agent.mother = int(record.get("mother", -1))  # unknown in a world saved before parents were kept
    agent.father = int(record.get("father", -1))
    agent.last_senses = numbers.senses(record["last_senses"])
    agent.last_choice = numbers.action(agent.last_choice)
    if agent.last_choice < 0:  # an action no longer had: nothing to learn from it
        agent.last_choice, agent.last_senses = ACTIONS.index("rest"), None
    agent.call = int(record.get("call", 0))  # silent, in a world saved before calls
    agent.call_turn = int(record.get("call_turn", -1))
    agent.last_call = int(record.get("last_call", -1))
    if "held" in record:  # blank, in a world saved before memory
        agent.held = [float(held) for held in record["held"]]
        agent.last_wrote = bool(record["last_wrote"])
        agent.past = [numbers.senses(senses) for senses in record["past"]]
    agent.places = [world.MapPlace(int(entry[0]), int(entry[1]), int(entry[2]),  # none, in a world saved before the map
                                   [bool(entry[3 + 4 * quarter]) for quarter in range(4)],
                                   [float(entry[4 + 4 * quarter]) for quarter in range(4)],
                                   [float(entry[5 + 4 * quarter]) for quarter in range(4)],
                                   [float(entry[6 + 4 * quarter]) for quarter in range(4)])
                    for entry in record.get("places", [])]
    agent.known = [world.Acquaintance(int(entry[0]), float(entry[1]), int(entry[2]))  # none, in a world saved before
                   for entry in record.get("known", [])]
    agent.odds = {str(action): float(chance) for action, chance in record["odds"].items() if action in ACTIONS}
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
    sea: list[int] = []  # the sea's runs, row by row: land, then sea, then land, and so on
    if simulation.has_sea:
        under, run = 0, 0
        for square in simulation.sea_map:
            if square != under:
                sea.append(run)
                under, run = square, 0
            run += 1
        sea.append(run)
    record = {
        "format": WORLD_FORMAT, "genome_version": GENOME_VERSION, "senses": list(SENSES), "centred_senses": True, "value_scale": VALUE_SCALE, "actions": list(OUTPUTS),
        "temperament": list(TEMPERAMENT),
        "settings": {"width": simulation.width, "height": simulation.height,
                     "population_cap": simulation.population_cap, "mind_mutation": simulation.mind_mutation,
                     "body_mutation": simulation.body_mutation, "year_length": simulation.year_length,
                     "initial_grass": simulation._initial_grass, "ocean": simulation.ocean},
        "sea": sea,
        "rules": Rules.current(),
        "turn": simulation.turn, "deaths": simulation.deaths, "next_identifier": simulation.next_identifier,
        "death_causes": {species: dict(causes) for species, causes in simulation.death_causes.items()},
        "death_tally": _tally_record(simulation.death_tally),
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
    if not isinstance(record["actions"], list) or record["temperament"] != list(TEMPERAMENT):
        raise ValueError("This world was saved by a version of Bitzkey whose brains sense or act differently.")
    # Brains' outputs saved otherwise are read by name, as their senses are; so are the actions chosen.
    if record["actions"] != list(OUTPUTS):
        saved_at = {name: index for index, name in enumerate(record["actions"])}
        numbers.saved_outputs = len(record["actions"])
        numbers.output_from = [saved_at.get(name, -1) for name in OUTPUTS]
        numbers.action_now = [ACTIONS.index(name) if name in ACTIONS else -1 for name in record["actions"]]
    numbers.uncentred = not record.get("centred_senses", False)
    numbers.unscaled = "value_scale" not in record
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
    simulation.ocean = settings.get("ocean", 0.0)  # none in a world saved before the sea
    if record.get("sea"):
        squares = bytearray()
        for index, run in enumerate(record["sea"]):
            squares.extend((index % 2,) * run)
        if len(squares) != simulation.width * simulation.height:
            raise ValueError("The world file's sea doesn't fit its map.")
        simulation.sea_map = squares
    # Genomes saved before the terrain gene had nothing but spare bits there: they are terrestrial.
    before_terrain = record.get("genome_version", 3) < 4
    simulation.turn = record["turn"]
    simulation.deaths = record["deaths"]
    simulation.next_identifier = record["next_identifier"]
    simulation.death_causes = {species: dict(causes) for species, causes in record["death_causes"].items()}
    state = record["random"]
    simulation.random.setstate((3, (*state["mt"], state["index"]), state["gauss_next"]))
    width = simulation.width
    simulation.grass = {(index % width, index // width): tufts for index, tufts in enumerate(record["grass"]) if tufts}
    simulation.seeds = [(x, y, convert_v3(int(dna, 16)) if before_terrain else int(dna, 16), generation, sown)
                        for x, y, dna, generation, sown in record["seeds"]]
    simulation.insights.offered = {}
    simulation.insights.chosen = {}
    for species, trait, level, action, offered, chosen in record["insights"]:
        if action not in ACTIONS:  # no longer an action
            continue
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
    if before_terrain:
        simulation.lineage = {identifier: convert_v3(dna) for identifier, dna in simulation.lineage.items()}
    simulation._order_deaths()
    untallied = "death_tally" not in record  # saved before deaths were tallied
    if untallied:  # tallied from those it kept whole
        simulation.death_tally = DeathTally()
        for _, identifier in simulation._death_order:
            simulation.death_tally.add(simulation.death_records[identifier])
    else:
        simulation.death_tally = _tally_from(record["death_tally"])
    simulation.agents = [_agent_from(agent, numbers, before_terrain) for agent in record["agents"]]
    simulation._update_climate()
    simulation._index_cells()
    # What each knew as of its turn's start, which indexing the cells afresh would count again as things now stand.
    for agent, saved in zip(simulation.agents, record["agents"]):
        agent.population, agent.cost_share = int(saved["population"]), float(saved["cost_share"])
        agent.glut, agent.guarded = float(saved["glut"]), bool(saved["guarded"])
    if untallied:  # and then, as now, only the latest deaths kept whole, and only the killers still alive and the deadliest
        simulation._forget_old_deaths()
        simulation.death_tally.let_go(simulation._alive)
    return simulation


def _tally_record(tally: DeathTally) -> dict:
    counts = lambda rows: {label: [row.get(species, 0) for species in SPECIES] for label, row in sorted(rows.items())}  # noqa: E731
    return {"causes": counts(tally.causes), "whom": [[tally.whom.get(dead, {}).get(killer, 0) for killer in SPECIES] for dead in SPECIES],
            "activities": counts(tally.activities), "killers": tally.killers,
            "by_killer": [[identifier, SPECIES.index(kind), kills, dict(sorted(doing.items()))]
                          for identifier, (kind, kills, doing) in sorted(tally.by_killer.items())],
            "ages": [[band.get(species, 0) for species in SPECIES] for band in tally.ages]}


def _tally_from(record: Mapping) -> DeathTally:
    tally = DeathTally()
    rows = lambda table: {label: {species: count for species, count in zip(SPECIES, counts) if count}  # noqa: E731
                          for label, counts in table.items()}
    tally.causes, tally.activities = rows(record["causes"]), rows(record["activities"])
    tally.whom = {dead: {killer: count for killer, count in zip(SPECIES, counts) if count}
                  for dead, counts in zip(SPECIES, record["whom"]) if any(counts)}
    tally.killers = int(record["killers"])
    tally.by_killer = {identifier: [SPECIES[kind], kills, dict(doing)] for identifier, kind, kills, doing in record["by_killer"]}
    if "ages" in record:  # counted since
        tally.ages = [{species: count for species, count in zip(SPECIES, counts) if count} for counts in record["ages"]]
    return tally


def save_world(simulation: Simulation, path: str | Path) -> None:
    Path(path).write_bytes(world_bytes(simulation))


def load_world(path: str | Path) -> Simulation:
    return world_from_bytes(Path(path).read_bytes())
