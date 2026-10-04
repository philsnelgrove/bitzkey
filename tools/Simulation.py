"""Turn-based Bitzkey world and its deterministic history."""

from __future__ import annotations

import bisect
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property

from tools.Brain import ACTIONS, SENSES, Brain, choice_odds, make_senses
from tools.Genome import (GENOME, MIND_TRAITS, SPECIES, TRAIT_MAX, body_word, color_from_word, cross, dot_difference, inherit_mind, ring_difference,
                          BODY_MUTATION_CHANCE, MIND_MUTATION_CHANCE,
                          decode_body, decode_mind, make_dna, mind_word, share)
from tools.Census import Census
from tools.Exchange import Imported
from tools.Insights import Insights


GRID_SIZE = 100
INITIAL_POPULATION = 90
GRASS_DENSITY = 50      # percent of cells that start with grass
GRASS_REGROWTH = 0.004  # share of the starting grass that grows back each turn
MAX_POPULATION = 600   # the most Bitzkeys spawned at the start; a world's population cap is a setting
CROWDING_RADIUS = 4    # animals within this many cells count as crowding a would-be parent
CROWDING_LIMIT = 10    # litters shrink as crowding nears this; at it, an animal doesn't breed
SPAWN_GROUP = (3, 5)  # Bitzkeys spawn in groups of this many sharing one genome
MAX_IMPORTED = MAX_POPULATION // SPAWN_GROUP[1]  # imported Bitzkeys, so their groups fit that number
SPAWN_SPREAD = 2      # a group's members appear within this many cells of each other's centre
DENSE_HISTORY = 500   # the latest turns are each kept as a snapshot of the world
CHECKPOINTS = 200     # older turns are kept as at most this many evenly spaced snapshots
CORPSE_FERTILITY = 2   # grass grows this many times as fast on and around a corpse
CORPSE_TURNS = 30
BREED_RESERVE = 10  # hunger a parent must still have spare after paying to breed
# The most a partner's ring and dot colours may differ from an animal's own before it shuns
# them, at no tolerance; full tolerance widens both to 1, accepting any partner of its species.
MATE_RING_DIFFERENCE = 0.05
MATE_DOT_DIFFERENCE = 0.20
LITTER_RANGE = (1, 6)  # young per birth, while there is room around the parent
LITTER_SIZE_WEIGHT = 0.5  # how much of litter size comes from small bodies; the rest from the litter trait
PLANT_LIFESPAN = 4000
ANIMAL_LIFESPAN = (250, 900)  # expected turns of life at the fastest and slowest metabolisms
METABOLISM = (1.6, 0.6)  # metabolic rate of the smallest and the largest bodies
STOMACH = (0.5, 2.0)     # food a body holds, relative to the middle, from the smallest to the largest;
                         # hunger is how empty it is, so a meal fills a small stomach more
SIZE_IN_MIGHT = 0.5      # share of fighting strength and toughness that comes from body size
AGEING = 0.7  # from this share of its lifespan, an animal may die of old age

DEATH_PENALTY = 2.0
BREED_JOY = 1.0
FIGHT_JOY = 0.5
COMPANY_JOY = 0.1
CURIOSITY_JOY = 0.15
FEAR_DREAD = 0.3
UNEASE = 0.1           # how much an intolerant animal minds other species close by
EMPATHY_PAIN = 0.15    # how much an empathetic animal feels another's wound; twice that for a death
OTHER_KIND = 0.4       # how much of that pain reaches it when the victim is of another species
LEARNING_RATE = 0.045  # at the highest intellect
PANIC = (0.3, 1.0)     # chance a fully fearful animal breaks when struck, unhurt and near death;
                       # its current fear scales it, so the fearless never break
THORN_FORCE = 15       # a plant's counterblow at full weapons and size, before the defender's toughness
THORN_SIZE_SHARE = 0.2  # share of that from the plant's size; the rest is its weapons, and strength counts for nothing
TERRITORY_RADIUS = 6   # a hunter's territory: the cells within this many steps
# For each territorial species: how many rivals it tolerates from the least to the most
# empathetic, and how strongly overcrowding moves it (its rivalry at most). Omnivores are less
# territorial, and their empathy counts for more. Who counts as a rival is _rival_weight's to say.
TERRITORY = {
    "predator": ((1, 5), 1.0),
    "omnivore": ((2, 10), 0.5),
}
TERRITORY_RAMP = 3     # each rival past that limit adds a third of the species' strength to rivalry
RIVAL_UNEASE = 0.3     # discomfort of a fully overcrowded territory, less for the tolerant
RIVAL_KILL_JOY = 1.0   # relief at killing a predator in an overcrowded territory
KIND_DIFFERENCE = 0.05  # predators whose ring colours differ by more than this are different kinds, and prey
STRIKE_EFFORT = 0.2    # food burned for each point of damage a blow deals
PREDATOR_LITTERS = 0.5  # predators' litters are this share of the usual size, rounded up
STRIKE_FORCE = 1.5     # scales every blow
KILL_MEAL = 25         # hunger a kill sates, plus up to KILL_MEAL_FOR_SIZE for the largest prey
KILL_MEAL_FOR_SIZE = 45
CARRION_MEAT = 30      # meat a well-fed corpse leaves, plus up to CARRION_MEAT_FOR_SIZE for the largest;
CARRION_MEAT_FOR_SIZE = 50  # a hungrier body leaves less, and a starved one nothing
SCAVENGE_BITE = 25     # meat eaten in one go
GRAZER_SPEED = 1.5     # grazers move this many times as far a step, rounded half up
GRAZING = {"grazer": 1.0, "omnivore": 0.5}  # share of grass and plant food each species gets
GRASS_MEAL = 35        # hunger a tuft of grass sates
PLANT_MEAL = 45        # hunger a bite of a plant sates, plus up to PLANT_MEAL_FOR_SIZE for the largest
PLANT_MEAL_FOR_SIZE = 25
PLANT_BITE = 15        # health a plant loses to each bite
PLANT_REGROWTH = 0.02  # share of its health pool a plant regrows each turn
PLANT_RECOVERY = 8     # turns after a bite before a plant can be bitten again
SEED_CHANCE = (0.03, 0.10)  # a well-fed plant's chance each turn of seeding, from least to most fertile
SEED_COST = 20         # hunger seeding costs a plant
SEED_HUNGER = 40       # a plant hungrier than this doesn't seed
SEEDLING_HUNGER = 50
SEED_REACH = 5         # how far, in cells, a seed may land from its plant
PLANT_CROWDING = 3     # a seed doesn't take where more than this many plants grow within two cells
MAX_PLANTS = 200
TEMPERAMENT_RELAPSE = 0.005  # share of the gap back to the inborn value closed each turn

# How experiences reshape temperament, in shares of a trait's full range before scaling by
# plasticity.
EXPERIENCES = {
    "kill": {"aggression": 0.04, "fear": -0.02},
    "hit": {"aggression": 0.0067},
    "stung": {"aggression": -0.0133, "fear": 0.0133},
    "wounded": {"fear": 0.0333, "aggression": -0.0067},
    "wounded_alone": {"sociability": 0.02},
    "bred": {"libido": 0.02, "sociability": 0.0133},
    "bred_into_hunger": {"libido": -0.0533},
    "starving": {"appetite": 0.0033, "patience": -0.0033},
    "foraged_meal": {"patience": 0.0067},
    "discovery": {"curiosity": 0.01},
    "ambushed": {"curiosity": -0.02, "fear": 0.0133},
    "defended": {"fear": -0.0133, "aggression": 0.0067},
    "panicked": {"fear": 0.0067},
}

GRASS_SEEN = SENSES.index("grass_seen")
MAX_SIGHT = 2 + TRAIT_MAX
SIGHT_OFFSETS = sorted(
    ((dx, dy) for dx in range(-MAX_SIGHT, MAX_SIGHT + 1) for dy in range(-MAX_SIGHT, MAX_SIGHT + 1)
     if 0 < abs(dx) + abs(dy) <= MAX_SIGHT),
    key=lambda offset: (abs(offset[0]) + abs(offset[1]), offset),
)
SEEN_OFFSETS = [(0, 0), *SIGHT_OFFSETS]
SEED_OFFSETS = sum(abs(dx) + abs(dy) <= SEED_REACH for dx, dy in SIGHT_OFFSETS)  # the first offsets: those a seed reaches


@dataclass
class Bitzkey:
    identifier: int
    dna: int
    x: int
    y: int
    hunger: float = 0.0
    health: float = 100.0
    age: int = 0
    cooldown: int = 0
    last_action: str = "new"
    brain: Brain | None = None
    generation: int = 0
    children: int = 0
    died_turn: int | None = None
    assailant: int | None = None  # the animal that struck it since it last acted, which it may defend against
    panicked: bool = False  # failed a morale check when struck, so it will flee its assailant
    rivalry: float = 0.0    # hunters: how overcrowded its territory was with rivals when it last looked
    own_rivalry: float = 0.0  # the same, counting only its own species: what wears down its reluctance
    meat: float = 0.0  # a corpse's meat left for scavengers
    rot: float = 0.0   # meat lost to rot each turn
    # Current temperament: starts at the inborn values and is reshaped by experience.
    mind: dict[str, float] = field(default_factory=dict)
    # The last decision, kept until the next one so the brain can learn from how it turned out.
    last_senses: list[float] | None = None
    last_choice: int = 0
    last_wellbeing: float = 0.0
    reward: float = 0.0
    odds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.mind:
            self.mind = {trait: float(value) for trait, value in self.inborn.items()}

    @cached_property
    def inborn(self) -> dict[str, int]:
        return decode_mind(self.dna)

    @cached_property
    def body(self) -> dict[str, int | str]:
        return decode_body(self.dna)

    @property
    def species(self) -> str:
        return str(self.body["species"])

    @property
    def profile(self) -> dict[str, float]:
        """Current temperament and physical traits, keyed by trait name."""
        return {**self.mind, **{trait: float(value) for trait, value in self.body.items()
                                if trait != "species"}}

    def physique(self, trait: str) -> float:
        """A body trait as a share of its full range."""
        return share(int(self.body[trait]))

    def feeling(self, trait: str) -> float:
        """A temperament trait, as it is now, as a share of its full range."""
        return share(self.mind[trait])

    @property
    def max_health(self) -> float:
        return 60.0 + 90 * self.physique("size")

    @property
    def metabolism(self) -> float:
        """Metabolic rate: fast in small bodies, slow in large ones."""
        fastest, slowest = METABOLISM
        return fastest + (slowest - fastest) * self.physique("size")

    @property
    def stomach(self) -> float:
        """How much food the body holds: hunger is a share of it, so food needs grow with size."""
        smallest, largest = STOMACH
        return smallest + (largest - smallest) * self.physique("size")

    @property
    def burn(self) -> float:
        """Hunger gained each turn, as a share of its stomach: a faster metabolism, and a stronger,
        faster, better-armed body, empty it sooner."""
        return self.metabolism * (0.45 + 0.18 * self.physique("strength") + 0.24 * self.physique("speed")
                                  + 0.12 * (self.physique("weapons") + self.physique("armour")))

    @property
    def might(self) -> float:
        """Fighting strength: its strength trait and, strongly, its size."""
        return (1 - SIZE_IN_MIGHT) * self.physique("strength") + SIZE_IN_MIGHT * self.physique("size")

    @property
    def toughness(self) -> float:
        """How well its body turns blows: its armour trait and, strongly, its size."""
        return (1 - SIZE_IN_MIGHT) * self.physique("armour") + SIZE_IN_MIGHT * self.physique("size")

    def nourish(self, food: float) -> None:
        """Eat: the same food fills a small stomach more than a large one."""
        self.hunger = max(0.0, self.hunger - food / self.stomach)

    def heal(self, share: float) -> None:
        """Heal a share of the health pool, faster with a faster metabolism."""
        self.health = min(self.max_health, self.health + self.max_health * share * self.metabolism)

    @property
    def steps(self) -> int:
        """Cells moved a step: 1 to 4 by speed, and half as many again for grazers."""
        steps = 1 + int(self.body["speed"]) * 3 // TRAIT_MAX
        return int(steps * GRAZER_SPEED + 0.5) if self.species == "grazer" else steps

    @property
    def grazing(self) -> float:
        """How much of grass and plant food nourishes it: grazers are far better at it than omnivores."""
        return GRAZING.get(self.species, 0.0)

    @property
    def sight(self) -> int:
        return 2 + int(self.body["sight"])

    @property
    def lifespan(self) -> float:
        """Expected turns of life: plants outlive animals, and among animals a slower metabolism,
        that is a larger body, lives longer."""
        if self.species == "plant":
            return PLANT_LIFESPAN
        shortest, longest = ANIMAL_LIFESPAN
        fastest, slowest = METABOLISM
        return shortest + (longest - shortest) * (fastest - self.metabolism) / (fastest - slowest)

    @property
    def breeding_cost(self) -> float:
        """Hunger it pays to breed: less the more fertile it is."""
        return 25 - 15 * self.physique("fertility")

    @property
    def can_breed(self) -> bool:
        """Ready, and able to pay for breeding with energy to spare."""
        return self.cooldown == 0 and self.hunger + self.breeding_cost <= 100 - BREED_RESERVE

    @property
    def digestion(self) -> float:
        """How much of the meat in carrion nourishes it."""
        return 0.2 + 0.8 * self.physique("carrion")

    @property
    def alive(self) -> bool:
        return self.health > 0 and self.hunger < 100

    @property
    def dot_color(self) -> tuple[int, int, int]:
        return color_from_word(mind_word(self.dna))

    @property
    def ring_color(self) -> tuple[int, int, int]:
        return color_from_word(body_word(self.dna))

    def copy(self) -> Bitzkey:
        return replace(self, brain=self.brain.copy() if self.brain else None, mind=self.mind.copy())


def litter_size(parents: Sequence[tuple[int, int]], species: str = "") -> int:
    """Young per birth, before space limits it, from each parent's (litter trait, size): a high
    litter trait and a small body both make for bigger litters. Predators have half as many,
    rounded up."""
    smallest, largest = LITTER_RANGE
    tendency = sum((1 - LITTER_SIZE_WEIGHT) * share(litter) + LITTER_SIZE_WEIGHT * (1 - share(size))
                   for litter, size in parents) / len(parents)
    litter = smallest + int(tendency * (largest - smallest) + 0.5)
    return math.ceil(litter * PREDATOR_LITTERS) if species == "predator" else litter


def _turn_of(snapshot: dict) -> int:
    return snapshot["turn"]


class Simulation:
    def __init__(self, width: int = GRID_SIZE, height: int = GRID_SIZE,
                 population: int = INITIAL_POPULATION, seed: int | None = None,
                 grass_density: float = GRASS_DENSITY, imported: Sequence[Imported] = (),
                 population_cap: int | None = None, mind_mutation: float = 100 * MIND_MUTATION_CHANCE,
                 body_mutation: float = 100 * BODY_MUTATION_CHANCE):
        """Spawn `population` random Bitzkeys in groups sharing a genome, plus a group for each
        `imported` Bitzkey, whose members all start with copies of its saved brain if it has one.
        `population_cap` limits living animals; None leaves crowding alone to limit them.
        `mind_mutation` and `body_mutation` are the percent chances that a birth mutates the
        child's mind or body."""
        if width < 10 or height < 10:
            raise ValueError("World dimensions must both be at least 10 cells.")
        if population < 0:
            raise ValueError("Population cannot be negative.")
        if not 0 <= grass_density <= 100:
            raise ValueError("Grass density must be between 0 and 100 percent.")
        if not (0 <= mind_mutation <= 100 and 0 <= body_mutation <= 100):
            raise ValueError("Mutation chances must be between 0 and 100 percent.")
        self.width = width
        self.height = height
        self.random = random.Random(seed)
        self.insights = Insights()
        # Who was alive, over the whole run, for the colour graph. It outlives the rewind
        # history; replaying reads it, and a new turn discards any abandoned future.
        self.census = Census()
        # The genome of every Bitzkey ever spawned, by identifier, so the graph can colour and
        # export Bitzkeys long gone. A branch taken after rewinding reuses identifiers and
        # overwrites them.
        self.lineage: dict[int, int] = {}
        self.turn = 0
        self.deaths = 0
        self.next_identifier = 0
        self.agents: list[Bitzkey] = []
        self._cells: dict[tuple[int, int], list[Bitzkey]] = {}
        self._carrion: dict[tuple[int, int], list[Bitzkey]] = {}
        self._population = 0
        self.population_cap = population_cap
        self.mind_mutation = mind_mutation / 100
        self.body_mutation = body_mutation / 100
        self._plants = 0
        self.grass: dict[tuple[int, int], int] = {}
        initial_grass = round(width * height * grass_density / 100)
        self.grass_regrowth = initial_grass * GRASS_REGROWTH
        self._sow(initial_grass)
        for size in self._group_sizes(population):
            self._spawn_group(make_dna(self.random.randrange(len(SPECIES)), self.random), size)
        for bitzkey in imported:
            self._spawn_group(bitzkey.dna, self.random.randint(*SPAWN_GROUP), bitzkey.brain, bitzkey.mind)
        # Snapshots of the world by turn: every recent turn, then evenly spaced checkpoints back
        # to turn 0. The simulation is deterministic, so any turn between snapshots is rebuilt
        # exactly by replaying from the snapshot before it.
        self.history: list[dict] = []
        self.frontier = 0  # the latest turn simulated; seeking never goes past it
        self._checkpoint_stride = 1
        self._take_census()
        self._record()

    @property
    def living(self) -> list[Bitzkey]:
        return [agent for agent in self.agents if agent.alive]

    def _group_sizes(self, population: int) -> list[int]:
        """Random group sizes adding up to the population, never leaving a group too small.
        Only a population below the smallest group size spawns as one smaller group."""
        smallest, largest = SPAWN_GROUP
        sizes = []
        remaining = population
        while remaining > largest:
            size = self.random.randint(smallest, min(largest, remaining - smallest))
            sizes.append(size)
            remaining -= size
        if remaining:
            sizes.append(remaining)
        return sizes

    def _spawn_group(self, dna: int, size: int, brain: Brain | None = None,
                     mind: Mapping[str, float] | None = None) -> None:
        """Place `size` Bitzkeys of one genome around a random centre, kept clear of the edges
        so the group is never split across them. Each gets its own copy of `brain` and starts
        with the temperament in `mind`, if given."""
        x = self.random.randrange(SPAWN_SPREAD, self.width - SPAWN_SPREAD)
        y = self.random.randrange(SPAWN_SPREAD, self.height - SPAWN_SPREAD)
        for _ in range(size):
            self._add_agent(dna, x + self.random.randint(-SPAWN_SPREAD, SPAWN_SPREAD),
                            y + self.random.randint(-SPAWN_SPREAD, SPAWN_SPREAD),
                            brain=brain.copy() if brain else None, mind=mind)

    def _add_agent(self, dna: int, x: int, y: int, brain: Brain | None = None,
                   mind: Mapping[str, float] | None = None) -> Bitzkey:
        agent = Bitzkey(self.next_identifier, dna & GENOME, x, y)
        agent.health = agent.max_health
        if mind:
            agent.mind.update({trait: mind[trait] for trait in MIND_TRAITS if trait in mind})
        if agent.species != "plant":
            agent.brain = brain or Brain.random(self.random)
        self.next_identifier += 1
        self.agents.append(agent)
        self.lineage[agent.identifier] = agent.dna
        self._cells.setdefault((x, y), []).append(agent)
        return agent

    def _record(self) -> None:
        """Snapshot the newest turn, then thin the old ones: a turn leaving the recent window
        stays only if it falls on the checkpoint spacing, and when there are too many
        checkpoints the spacing doubles."""
        self.history.append({
            "turn": self.turn,
            "deaths": self.deaths,
            "next_identifier": self.next_identifier,
            "agents": [agent.copy() for agent in self.agents],
            "grass": self.grass.copy(),
            "random_state": self.random.getstate(),
            "insights": self.insights.snapshot(),
        })
        self.frontier = self.turn
        recent = self.frontier - DENSE_HISTORY
        leaving = bisect.bisect_left(self.history, recent - 1, key=_turn_of)
        if (recent - 1 > 0 and leaving < len(self.history) and self.history[leaving]["turn"] == recent - 1
                and (recent - 1) % self._checkpoint_stride):
            del self.history[leaving]
        if bisect.bisect_left(self.history, recent, key=_turn_of) > CHECKPOINTS:
            self._checkpoint_stride *= 2
            self.history = [snapshot for snapshot in self.history
                            if snapshot["turn"] >= recent or snapshot["turn"] % self._checkpoint_stride == 0]

    def _restore(self, snapshot: dict) -> None:
        self.turn = snapshot["turn"]
        self.deaths = snapshot["deaths"]
        self.next_identifier = snapshot["next_identifier"]
        self.agents = [agent.copy() for agent in snapshot["agents"]]
        self.grass = snapshot["grass"].copy()
        self.random.setstate(snapshot["random_state"])
        self.insights.restore(snapshot["insights"])
        self._index_cells()

    def _start_for(self, target: int) -> tuple[dict | None, int]:
        """Where seeking `target` would start (a snapshot to restore, or None to carry on from the
        present) and how many turns must then be replayed."""
        snapshot = self.history[bisect.bisect_right(self.history, target, key=_turn_of) - 1]
        if snapshot["turn"] <= self.turn <= target:
            return None, target - self.turn
        return snapshot, target - snapshot["turn"]

    def replay_cost(self, target: int) -> int:
        """How many turns seeking `target` would have to re-simulate."""
        return self._start_for(max(0, min(self.frontier, target)))[1]

    def seek(self, target: int) -> None:
        """Move the world to any turn already simulated: restore the nearest snapshot at or
        before it and replay forward. Replaying repeats the recorded run exactly."""
        target = max(0, min(self.frontier, target))
        snapshot, _ = self._start_for(target)
        if snapshot is not None:
            self._restore(snapshot)
        while self.turn < target:
            self._take_turn(replay=True)

    def rewind(self) -> bool:
        if self.turn == 0:
            return False
        self.seek(self.turn - 1)
        return True

    def advance(self) -> None:
        """Step one turn: replaying the recorded run, or simulating a new turn, unless the
        simulation has ended."""
        if self.turn < self.frontier:
            self.seek(self.turn + 1)
            return
        if self.end_reason:
            return
        self._take_turn()
        self._record()

    @property
    def end_reason(self) -> str | None:
        """Why the simulation is over, if it is: no animals left alive, or no cell left free."""
        living = self.living
        if not any(agent.species != "plant" for agent in living):
            return "extinct"
        if len({(agent.x, agent.y) for agent in living}) >= self.width * self.height:
            return "full"
        return None

    @property
    def ended(self) -> bool:
        """At the end of the run: the latest turn, with an end reason."""
        return self.turn == self.frontier and self.end_reason is not None

    def _take_turn(self, replay: bool = False) -> None:
        """One turn. A replay re-simulates a turn already recorded, so the census, which already
        holds it, is left alone."""
        self.turn += 1
        self._index_cells()
        order = self.living
        self._plants = sum(agent.species == "plant" for agent in order)
        self._population = len(order) - self._plants  # animals; plants have their own limit
        self.random.shuffle(order)
        for agent in order:
            if not agent.alive:
                continue
            agent.age += 1
            agent.hunger += agent.burn
            if agent.cooldown:
                agent.cooldown -= 1
            if agent.health < agent.max_health and agent.hunger < 65:
                agent.heal(0.005)
            if agent.hunger >= 100:
                self._die(agent, "starved")
                continue
            if self._dies_of_age(agent):
                self._die(agent, "old age")
                continue
            if agent.species == "plant":
                agent.hunger = max(0, agent.hunger - 1.5)
                agent.health = min(agent.max_health, agent.health + agent.max_health * PLANT_REGROWTH)
                agent.last_action = "photosynthesized"
                self._seed(agent)
                continue
            self._settle(agent)
            if agent.hunger >= 80:
                self._experience(agent, "starving")
            self._think(agent)

        whole, fraction = divmod(self.grass_regrowth, 1)
        self._sow(int(whole) + (1 if fraction and self.random.random() < fraction else 0))
        for agent in self.agents:
            if agent.died_turn is not None:
                agent.meat = max(0.0, agent.meat - agent.rot)
        self.agents = [agent for agent in self.agents
                       if agent.died_turn is None or self.turn - agent.died_turn < CORPSE_TURNS]
        self.insights.decay()
        if not replay:
            self._take_census()

    def _seed(self, plant: Bitzkey) -> None:
        """A well-fed plant may sprout a seedling of its own kind on a free cell up to SEED_REACH away,
        unless plants already crowd that spot or there are already MAX_PLANTS plants."""
        if plant.hunger > SEED_HUNGER or self._plants >= MAX_PLANTS:
            return
        least, most = SEED_CHANCE
        if self.random.random() >= least + (most - least) * plant.physique("fertility"):
            return
        dx, dy = SIGHT_OFFSETS[self.random.randrange(SEED_OFFSETS)]
        cell = ((plant.x + dx) % self.width, (plant.y + dy) % self.height)
        if any(other.alive for other in self._cells.get(cell, ())):
            return
        spot = Bitzkey(-1, plant.dna, *cell)
        if sum(other.species == "plant" for other in self._nearby(spot, 2)) > PLANT_CROWDING:
            return
        seedling = self._add_agent(cross(plant.dna, plant.dna, self.random, None, self.mind_mutation,
                                         self.body_mutation), *cell)
        seedling.hunger = SEEDLING_HUNGER
        seedling.generation = plant.generation + 1
        plant.hunger += SEED_COST
        plant.children += 1
        plant.last_action = "seeded"
        self._plants += 1

    def _sow(self, tufts: int) -> None:
        """Scatter new grass at random; on and around a corpse, it grows CORPSE_FERTILITY times as fast."""
        fertile = {((corpse.x + dx) % self.width, (corpse.y + dy) % self.height)
                   for corpse in self.agents if corpse.died_turn is not None
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        for _ in range(tufts):
            cell = (self.random.randrange(self.width), self.random.randrange(self.height))
            self.grass[cell] = self.grass.get(cell, 0) + (CORPSE_FERTILITY if cell in fertile else 1)

    def _take_census(self) -> None:
        self.census.record(self.turn, (agent.identifier for agent in self.living))

    # Temperament

    def _experience(self, agent: Bitzkey, event: str) -> None:
        """Shift temperament after an experience; intellect sets how far."""
        plasticity = 0.2 + 0.8 * agent.feeling("intellect")
        for trait, change in EXPERIENCES[event].items():
            agent.mind[trait] = max(0.0, min(float(TRAIT_MAX), agent.mind[trait] + change * TRAIT_MAX * plasticity))

    @staticmethod
    def _settle(agent: Bitzkey) -> None:
        """Let learned temperament drift slowly back toward the inborn values."""
        for trait in MIND_TRAITS:
            agent.mind[trait] += (agent.inborn[trait] - agent.mind[trait]) * TEMPERAMENT_RELAPSE

    # Decisions and learning

    def _think(self, agent: Bitzkey) -> None:
        """Sense, learn from the previous decision's outcome, then choose and act."""
        senses, options, situation = self._sense(agent)
        if agent.last_action == "explored":
            if situation["threat"]:
                self._experience(agent, "ambushed")
            elif (situation["grass"] or situation["grass_here"]) and not agent.last_senses[GRASS_SEEN]:
                self._experience(agent, "discovery")
        values = agent.brain.values(senses)
        if agent.last_senses is not None:
            reward = (agent.reward + self._wellbeing(agent) - agent.last_wellbeing
                      + self._comfort(agent, situation))
            discount = 0.7 + 0.27 * agent.feeling("patience")
            self._learn(agent, reward + discount * max(values[option] for option in options))

        if agent.panicked and situation["assailant"]:
            # Fight or flight, decided by fear: it breaks off and flees its assailant.
            situation["hunter"] = situation["assailant"]
            situation["panic"] = True
            options = [ACTIONS.index("flee")]

        odds = choice_odds(values, options)
        roll = self.random.random()
        choice = options[-1]
        for option in options:
            roll -= odds[option]
            if roll < 0:
                choice = option
                break
        action = ACTIONS[choice]
        if not situation.get("panic"):  # a panicked flight was no choice
            self.insights.record(agent.species, agent.profile, (ACTIONS[option] for option in options), action)
        agent.odds = {ACTIONS[option]: chance for option, chance in odds.items()}
        agent.last_senses = senses
        agent.last_choice = choice
        agent.last_wellbeing = self._wellbeing(agent)
        agent.reward = 0.0
        agent.assailant = None  # the chance to strike back, or need to flee, passes with this turn's choice
        agent.panicked = False
        self._act(agent, action, situation)

    def _sense(self, agent: Bitzkey) -> tuple[list[float], list[int], dict]:
        if agent.species in TERRITORY:
            agent.rivalry, agent.own_rivalry = self._territory(agent)
        neighbors = self._nearby(agent, 1)
        mate = next((other for other in neighbors if self._compatible(agent, other) and other.can_breed), None)
        nurseries = self._free_cells_near(agent)
        prey = [other for other in neighbors if self._is_prey(agent, other)]
        threats = [other for other in neighbors if self._is_prey(other, agent)]
        target = max(prey, key=lambda other: self._damage(agent, other) / other.health, default=None)
        threat = max(threats, key=lambda other: self._damage(other, agent), default=None)
        around = self._nearby(agent, 2)
        kin = sum(other.species == agent.species for other in around)
        others = sum(other.species not in (agent.species, "plant") for other in around)
        sight = agent.sight
        grazes = agent.species != "predator"
        grass_here = grazes and self.grass.get((agent.x, agent.y), 0) > 0
        grass = self._nearest_grass(agent, sight) if grazes else None
        carrion = self._nearest_carrion(agent, sight)
        quarry, hunter, suitor, flora = self._look_around(agent, sight)
        assailant = next((other for other in neighbors if other.identifier == agent.assailant), None)

        available = {"rest", "explore"}
        if grass_here:
            available.add("eat")
        if grass:
            available.add("forage")
        if flora:
            available.add("browse")
        if carrion:
            available.add("scavenge")
        if suitor:
            available.add("court")
        if (mate and agent.can_breed and nurseries and self._room() > 0
                and self._crowding(agent, mate) < CROWDING_LIMIT):
            available.add("breed")
        if quarry:
            available.add("hunt")
        if target:
            available.add("attack")
        if hunter:
            available.add("flee")
        if assailant:
            available.add("defend")

        senses = make_senses(
            hunger=min(1.0, agent.hunger / 100),
            health=max(0.0, agent.health / agent.max_health),
            grass_here=float(grass_here),
            grass_seen=float(grass is not None),
            grass_closeness=self._closeness(agent, grass, sight),
            mate_near=float(mate is not None),
            mate_closeness=self._closeness(agent, suitor and (suitor.x, suitor.y), sight),
            ready_to_breed=float(agent.cooldown == 0),
            prey_near=min(4, len(prey)) / 4,
            prey_frailty=0.0 if target is None else min(1.0, self._damage(agent, target) / target.health),
            prey_closeness=self._closeness(agent, quarry and (quarry.x, quarry.y), sight),
            threat_near=min(4, len(threats)) / 4,
            threat_danger=0.0 if threat is None else min(1.0, self._damage(threat, agent) / agent.health),
            threat_closeness=self._closeness(agent, hunter and (hunter.x, hunter.y), sight),
            kin_near=min(3, kin) / 3,
            others_near=min(3, others) / 3,
            carrion_closeness=self._closeness(agent, carrion and (carrion.x, carrion.y), sight),
            plant_closeness=self._closeness(agent, flora and (flora.x, flora.y), sight),
            assailant_near=float(assailant is not None),
            rivalry=agent.rivalry,
            counter_frailty=0.0 if assailant is None else min(1.0, self._damage(agent, assailant) / assailant.health),
            age=min(1.0, agent.age / 400),
            **{trait: agent.feeling(trait) for trait in MIND_TRAITS},
        )
        situation = {"mate": mate, "nurseries": nurseries, "prey": target, "threat": threat,
                     "threats": len(threats), "quarry": quarry, "hunter": hunter, "suitor": suitor,
                     "grass": grass, "grass_here": grass_here, "carrion": carrion, "flora": flora, "assailant": assailant, "rivalry": agent.rivalry, "kin": kin,
                     "others": others}
        return senses, [ACTIONS.index(action) for action in ACTIONS if action in available], situation

    def _wellbeing(self, agent: Bitzkey) -> float:
        """How comfortable an animal feels; appetite sets how much hunger hurts."""
        appetite = 0.5 + agent.feeling("appetite")
        return (-appetite * min(100.0, agent.hunger) / 100
                - (1 - max(0.0, agent.health) / agent.max_health))

    @staticmethod
    def _comfort(agent: Bitzkey, situation: dict) -> float:
        """Company pleases the sociable, and the tolerant count other species as company.
        Hunters close by frighten the fearful, less so the tolerant; the intolerant are
        uneasy around anything unlike them."""
        tolerance = agent.feeling("tolerance")
        company_around = min(3.0, situation["kin"] + tolerance * situation["others"])
        company = COMPANY_JOY * agent.feeling("sociability") * company_around / 3
        dread = FEAR_DREAD * agent.feeling("fear") * (1 - 0.5 * tolerance) * min(2, situation["threats"]) / 2
        unease = UNEASE * (1 - tolerance) * min(3, situation["others"]) / 3
        crowded = RIVAL_UNEASE * (1 - 0.5 * tolerance) * situation.get("rivalry", 0.0)
        return company - dread - unease - crowded

    def _learn(self, agent: Bitzkey, target: float) -> None:
        agent.brain.learn(agent.last_senses, agent.last_choice, target, LEARNING_RATE * agent.feeling("intellect"))

    def _dies_of_age(self, agent: Bitzkey) -> bool:
        """Past AGEING of its lifespan, the chance of dying each turn grows with age, so that about
        half live to their lifespan and almost none beyond 1.3 times it."""
        onset = AGEING * agent.lifespan
        if agent.age <= onset:
            return False
        steepness = 2 * math.log(2) / (agent.lifespan - onset) ** 2
        return self.random.random() < steepness * (agent.age - onset)

    def _die(self, agent: Bitzkey, cause: str) -> None:
        """Record a death. The body becomes carrion holding the energy it had left: a starved body
        leaves nothing, and a killed one half, as the killer has already eaten. Dying of old age
        is not held against the brain's last choice."""
        if agent.brain is not None and agent.last_senses is not None:
            penalty = 0.0 if cause == "old age" else DEATH_PENALTY
            self._learn(agent, agent.reward + self._wellbeing(agent) - agent.last_wellbeing - penalty)
        agent.brain = None
        agent.health = min(agent.health, 0.0)  # whatever the cause, a recorded death leaves it dead
        agent.died_turn = self.turn
        agent.last_action = cause
        nourished = 0.0 if agent.species == "plant" else max(0.0, 1 - agent.hunger / 100)
        agent.meat = ((CARRION_MEAT + CARRION_MEAT_FOR_SIZE * agent.physique("size")) * nourished
                      * (0.5 if cause == "killed" else 1))
        agent.rot = agent.meat / CORPSE_TURNS
        self._carrion.setdefault((agent.x, agent.y), []).append(agent)
        self.deaths += 1
        if agent.species == "plant":
            self._plants -= 1
        else:
            self._population -= 1

    # Actions

    def _act(self, agent: Bitzkey, action: str, situation: dict) -> None:
        if action == "eat":
            cell = (agent.x, agent.y)
            self.grass[cell] -= 1
            if self.grass[cell] == 0:
                del self.grass[cell]
            agent.nourish(GRASS_MEAL * agent.grazing)
            if agent.last_action == "foraged":
                self._experience(agent, "foraged_meal")
            agent.last_action = "ate"
        elif action == "forage":
            self._move_toward(agent, *situation["grass"])
            agent.last_action = "foraged"
        elif action == "browse":
            self._browse(agent, situation["flora"])
        elif action == "scavenge":
            self._scavenge(agent, situation["carrion"])
        elif action == "rest":
            agent.hunger = max(0, agent.hunger - agent.burn / 2)
            agent.heal(0.015)
            agent.last_action = "rested"
        elif action == "court":
            suitor = situation["suitor"]
            self._move_toward(agent, suitor.x, suitor.y)
            agent.last_action = "courted"
        elif action == "breed":
            agent.last_action = "mated"  # before breeding, so a parent that dies of it says so
            self._breed(agent, situation["mate"], situation["nurseries"])
        elif action == "hunt":
            quarry = situation["quarry"]
            self._move_toward(agent, quarry.x, quarry.y)
            agent.last_action = "hunted"
            if abs(quarry.x - agent.x) + abs(quarry.y - agent.y) <= 1:
                self._attack(agent, quarry)
                if agent.alive:
                    agent.last_action = "pounced"
        elif action == "attack":
            self._attack(agent, situation["prey"])
        elif action == "defend":
            self._defend(agent, situation["assailant"])
        elif action == "flee":
            self._move_away(agent, situation["hunter"])
            agent.last_action = "panicked" if situation.get("panic") else "fled"
        else:
            for _ in range(agent.steps):
                self._move(agent, (agent.x + self.random.choice((-1, 0, 1))) % self.width,
                           (agent.y + self.random.choice((-1, 0, 1))) % self.height)
            agent.reward += CURIOSITY_JOY * agent.feeling("curiosity")
            agent.last_action = "explored"

    def _browse(self, agent: Bitzkey, plant: Bitzkey) -> None:
        """Go to the plant and, once alongside, take a bite: far more filling than grass, and more
        so from a bigger plant. Each bite wounds the plant, and enough of them kill it. A bitten
        plant can't be bitten again until it has recovered."""
        self._move_toward(agent, plant.x, plant.y)
        if abs(plant.x - agent.x) + abs(plant.y - agent.y) > 1 or plant.cooldown:
            agent.last_action = "sought plants"
            return
        agent.nourish((PLANT_MEAL + PLANT_MEAL_FOR_SIZE * plant.physique("size")) * agent.grazing)
        plant.health -= PLANT_BITE
        plant.cooldown = PLANT_RECOVERY
        agent.last_action = "browsed"
        if not plant.alive:
            self._die(plant, "eaten")
        elif plant.physique("weapons") > 0:
            self._defend(plant, agent)  # a plant can't flee, but an armed one strikes back

    def _scavenge(self, agent: Bitzkey, corpse: Bitzkey) -> None:
        """Go to the carrion and, once alongside, eat; a carrion-eater's gut gets the most from it."""
        self._move_toward(agent, corpse.x, corpse.y)
        if abs(corpse.x - agent.x) + abs(corpse.y - agent.y) > 1:
            agent.last_action = "sought carrion"
            return
        bite = min(corpse.meat, SCAVENGE_BITE)
        corpse.meat -= bite
        agent.nourish(bite * agent.digestion)
        agent.last_action = "scavenged"

    @staticmethod
    def _damage(attacker: Bitzkey, defender: Bitzkey) -> float:
        """One blow: the attacker's might, sharpened by weapons, blunted by the defender's toughness;
        size counts strongly in both. A plant's thorns are mostly its weapons, a little its size,
        and owe nothing to strength."""
        if attacker.species == "plant":
            thorns = THORN_SIZE_SHARE * attacker.physique("size") + (1 - THORN_SIZE_SHARE) * attacker.physique("weapons")
            return THORN_FORCE * thorns * (1 - 0.6 * defender.toughness)
        force = STRIKE_FORCE * (2 + 9 * attacker.might)
        return force * (1 + 0.75 * attacker.physique("weapons")) * (1 - 0.6 * defender.toughness)

    def _attack(self, agent: Bitzkey, victim: Bitzkey) -> None:
        """A blow. The intolerant enjoy fighting other species most; the empathetic feel their
        victim's pain, and so does any empathetic animal watching, most for its own kind."""
        empathy = agent.feeling("empathy")
        joy = FIGHT_JOY * agent.feeling("aggression") * (1 - 0.6 * empathy) * (1 - 0.75 * agent.feeling("tolerance"))
        damage = self._damage(agent, victim)
        victim.health -= damage
        agent.nourish(5)
        agent.hunger += STRIKE_EFFORT * damage / agent.stomach  # the harder the blow, the more it costs
        agent.reward += joy
        agent.last_action = "attacked"
        self._experience(agent, "hit")
        killed = not victim.alive
        if not killed:
            victim.last_action = "wounded"
            self._struck(victim, agent)
            self._experience(victim, "wounded")
            if not any(other.species == victim.species for other in self._nearby(victim, 2)):
                self._experience(victim, "wounded_alone")
        else:
            agent.nourish(KILL_MEAL + KILL_MEAL_FOR_SIZE * victim.physique("size"))
            agent.reward += 2 * joy
            if self._rival_weight(agent, victim) > 0:  # one fewer rival for the feeding ground
                agent.reward += RIVAL_KILL_JOY * agent.rivalry * (1 - 0.5 * agent.feeling("tolerance"))
            self._experience(agent, "kill")
        pain = EMPATHY_PAIN * (2 if killed else 1)
        agent.reward -= pain * empathy * (1.0 if self._same_kind(agent, victim) else OTHER_KIND)
        for witness in self._nearby(victim, 2):
            if witness is not agent and witness.brain is not None:
                kinship = 1.0 if witness.species == victim.species else OTHER_KIND
                witness.reward -= pain * witness.feeling("empathy") * kinship
        if killed:
            self._die(victim, "killed")

        sting = 4.5 * victim.physique("weapons") * (1 - 0.6 * agent.toughness)
        if sting > 0:
            agent.health -= sting
            self._experience(agent, "stung")
            if not agent.alive:
                self._die(agent, "killed")

    def _defend(self, agent: Bitzkey, assailant: Bitzkey) -> None:
        """Strike back at the animal that just struck it, or, for a plant, browsed it, with the
        usual force of might and weapons against the assailant's toughness. The blow costs effort
        like any other, and feeds the defender nothing, but it can kill, and a struck assailant
        must check its own morale."""
        damage = self._damage(agent, assailant)
        assailant.health -= damage
        agent.hunger += STRIKE_EFFORT * damage / agent.stomach
        self._experience(assailant, "stung")
        if agent.species != "plant":
            agent.reward += FIGHT_JOY * agent.feeling("aggression") * (1 - 0.6 * agent.feeling("empathy"))
            agent.last_action = "defended"
            self._experience(agent, "defended")
        if not assailant.alive:
            self._die(assailant, "killed")
        elif assailant.species != "plant":
            self._struck(assailant, agent)

    def _struck(self, victim: Bitzkey, striker: Bitzkey) -> None:
        """A struck animal may strike back on its next turn, unless it fails a morale check
        against its current fear, more likely the more hurt it is, and so must flee instead."""
        victim.assailant = striker.identifier
        unhurt, dying = PANIC
        hurt = 1 - max(0.0, victim.health) / victim.max_health
        if self.random.random() < victim.feeling("fear") * (unhurt + (dying - unhurt) * hurt):
            victim.panicked = True
            self._experience(victim, "panicked")

    def _breed(self, first: Bitzkey, second: Bitzkey, cells: list[tuple[int, int]]) -> None:
        """Fertile parents breed cheaply and often. Their litter, 1 to 6 young (bigger for a high
        litter trait and small bodies, smaller where animals crowd around, and no more than the
        free cells around), shares the energy
        both parents give up. Animals only choose to breed when both can afford it, but a parent
        can only give what it has left. Each child inherits a mix of its parents' brains, and
        starts its temperament at the average of its parents' current temperaments, which its
        dot then shows."""
        litter = litter_size([(int(parent.body["litter"]), int(parent.body["size"])) for parent in (first, second)],
                             first.species)
        litter = max(1, int(litter * (1 - self._crowding(first, second) / CROWDING_LIMIT) + 0.5))
        cells = cells[:max(1, min(litter, self._room()))]
        costs = [parent.breeding_cost for parent in (first, second)]
        # Energy in absolute terms: a share of each parent's stomach, refilled in the young's.
        given = sum(min(cost, max(0.0, 100 - parent.hunger)) * parent.stomach
                    for cost, parent in zip(costs, (first, second)))
        for cell in cells:
            brain = Brain.inherit(first.brain, second.brain, self.random)
            # The child's mind starts at the average of its parents' current minds, mutated, and
            # its dot is written from that mind, so experience is inherited and shows.
            mind = inherit_mind(first.mind, second.mind, self.random, self.mind_mutation)
            child = self._add_agent(cross(first.dna, second.dna, self.random, mind, body_chance=self.body_mutation),
                                    *cell, brain=brain)
            child.mind = mind
            child.hunger = max(0.0, 100 - given / len(cells) / child.stomach)
            child.generation = max(first.generation, second.generation) + 1
            self._population += 1
        for parent, cost in zip((first, second), costs, strict=True):
            parent.hunger += cost
            parent.cooldown = round(20 - 15 * parent.physique("fertility"))
            parent.children += len(cells)
            parent.reward += BREED_JOY * parent.feeling("libido")
            if not parent.alive:
                self._die(parent, "bred to death")
            else:
                self._experience(parent, "bred_into_hunger" if parent.hunger >= 75 else "bred")

    # Space

    def _index_cells(self) -> None:
        self._cells = {}
        self._carrion = {}
        for agent in self.agents:
            if agent.alive:
                self._cells.setdefault((agent.x, agent.y), []).append(agent)
            elif agent.meat > 0:
                self._carrion.setdefault((agent.x, agent.y), []).append(agent)

    def _nearby(self, agent: Bitzkey, radius: int) -> list[Bitzkey]:
        found = []
        for dx in range(-radius, radius + 1):
            for dy in range(-(radius - abs(dx)), radius - abs(dx) + 1):
                for other in self._cells.get((agent.x + dx, agent.y + dy), ()):
                    if other is not agent and other.alive:
                        found.append(other)
        return found

    def _room(self) -> float:
        """How many more animals the population cap allows; unlimited without a cap."""
        return float("inf") if self.population_cap is None else self.population_cap - self._population

    def _crowding(self, agent: Bitzkey, partner: Bitzkey) -> int:
        """How many other animals, besides its partner, live within CROWDING_RADIUS cells."""
        return sum(other.species != "plant" and other is not partner
                   for other in self._nearby(agent, CROWDING_RADIUS))

    def _free_cells_near(self, agent: Bitzkey) -> list[tuple[int, int]]:
        """The empty cells around an animal, sides before corners, where young can be born."""
        cells = []
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0), (1, -1), (1, 1), (-1, 1), (-1, -1)):
            cell = ((agent.x + dx) % self.width, (agent.y + dy) % self.height)
            if not any(other.alive for other in self._cells.get(cell, ())):
                cells.append(cell)
        return cells

    def _nearest_grass(self, agent: Bitzkey, sight: int) -> tuple[int, int] | None:
        for dx, dy in SIGHT_OFFSETS:
            if abs(dx) + abs(dy) > sight:
                return None
            cell = (agent.x + dx, agent.y + dy)
            if self.grass.get(cell, 0) > 0:
                return cell
        return None

    def _nearest_carrion(self, agent: Bitzkey, sight: int) -> Bitzkey | None:
        if not self._carrion:
            return None
        for dx, dy in SEEN_OFFSETS:
            if abs(dx) + abs(dy) > sight:
                return None
            for corpse in self._carrion.get((agent.x + dx, agent.y + dy), ()):
                if corpse.meat > 0:
                    return corpse
        return None

    def _look_around(self, agent: Bitzkey,
                     sight: int) -> tuple[Bitzkey | None, Bitzkey | None, Bitzkey | None, Bitzkey | None]:
        """The nearest prey, hunter, willing mate and edible plant in sight, for the roles this
        animal can have."""
        quarry = hunter = suitor = flora = None
        hunts = agent.species in ("omnivore", "predator")
        hunted = agent.species != "plant"
        browses = agent.species in ("grazer", "omnivore")
        courts = agent.can_breed
        for dx, dy in SEEN_OFFSETS:
            if abs(dx) + abs(dy) > sight or ((quarry or not hunts) and (hunter or not hunted)
                                             and (suitor or not courts) and (flora or not browses)):
                break
            for other in self._cells.get((agent.x + dx, agent.y + dy), ()):
                if other is agent or not other.alive:
                    continue
                if quarry is None and self._is_prey(agent, other):
                    quarry = other
                elif hunter is None and self._is_prey(other, agent):
                    hunter = other
                elif browses and flora is None and other.species == "plant" and not other.cooldown:
                    flora = other
                elif courts and suitor is None and other.can_breed and self._compatible(agent, other):
                    suitor = other
        return quarry, hunter, suitor, flora

    @staticmethod
    def _closeness(agent: Bitzkey, cell: tuple[int, int] | None, sight: int) -> float:
        """1 at or next to the cell, falling toward 0 at the edge of sight; 0 when nothing is seen."""
        if cell is None:
            return 0.0
        return min(1.0, 1 - (abs(cell[0] - agent.x) + abs(cell[1] - agent.y) - 1) / sight)

    def _move(self, agent: Bitzkey, x: int, y: int) -> None:
        self._cells[(agent.x, agent.y)].remove(agent)
        agent.x, agent.y = x, y
        self._cells.setdefault((x, y), []).append(agent)

    def _move_toward(self, agent: Bitzkey, x: int, y: int) -> None:
        for _ in range(agent.steps):
            if agent.x != x:
                self._move(agent, agent.x + (1 if x > agent.x else -1), agent.y)
            elif agent.y != y:
                self._move(agent, agent.x, agent.y + (1 if y > agent.y else -1))

    def _move_away(self, agent: Bitzkey, threat: Bitzkey) -> None:
        for _ in range(agent.steps):
            dx = 1 if agent.x >= threat.x else -1
            dy = 1 if agent.y >= threat.y else -1
            self._move(agent, (agent.x + dx) % self.width, (agent.y + dy) % self.height)

    @staticmethod
    def _compatible(first: Bitzkey, second: Bitzkey) -> bool:
        """Two animals of one species that are each willing to breed with the other."""
        return (first.species != "plant" and first.species == second.species
                and Simulation._willing(first, second) and Simulation._willing(second, first))

    @staticmethod
    def _willing(agent: Bitzkey, other: Bitzkey) -> bool:
        """Whether an animal would breed with another: it shuns a partner whose ring or dot
        colour differs from its own by more than it accepts, which its tolerance widens."""
        tolerance = agent.feeling("tolerance")
        return (ring_difference(agent.dna, other.dna) <= MATE_RING_DIFFERENCE + (1 - MATE_RING_DIFFERENCE) * tolerance
                and dot_difference(agent.dna, other.dna) <= MATE_DOT_DIFFERENCE + (1 - MATE_DOT_DIFFERENCE) * tolerance)

    @staticmethod
    def _same_kind(first: Bitzkey, second: Bitzkey) -> bool:
        """Of one kind: the same species, and, for predators, ring colours within KIND_DIFFERENCE."""
        return first.species == second.species and (
            first.species != "predator" or ring_difference(first.dna, second.dna) <= KIND_DIFFERENCE)

    @staticmethod
    def cannibal_reluctance(predator: Bitzkey) -> float:
        """How hungry, as a share, a predator must be before it will hunt its own kind: the
        empathetic feel for them, the tolerant bear them, and the patient can wait for other prey."""
        return (predator.feeling("empathy") + predator.feeling("tolerance") + predator.feeling("patience")) / 3

    @staticmethod
    def _rival_weight(hunter: Bitzkey, other: Bitzkey) -> float:
        """How much another animal counts as a rival for a hunter's feeding ground. Predators
        count other predators and omnivores; omnivores count other omnivores, and predators
        too unless they have the greatest empathy."""
        if hunter.species == "predator":
            return 1.0 if other.species in ("predator", "omnivore") else 0.0
        if hunter.species == "omnivore":
            if other.species == "omnivore":
                return 1.0
            return 1 - hunter.feeling("empathy") if other.species == "predator" else 0.0
        return 0.0

    def _territory(self, hunter: Bitzkey) -> tuple[float, float]:
        """How overcrowded a hunter's territory is, with all its rivals and with its own species
        alone: nothing up to the rivals it will tolerate, fewer the less empathetic it is, then
        rising over TERRITORY_RAMP more to its species' strength of territorial feeling."""
        (fewest, most), strength = TERRITORY[hunter.species]
        tolerated = fewest + (most - fewest) * hunter.feeling("empathy")
        rivals = own = 0.0
        for other in self._nearby(hunter, TERRITORY_RADIUS):
            rivals += self._rival_weight(hunter, other)
            own += other.species == hunter.species
        return tuple(strength * min(1.0, max(0.0, count - tolerated) / TERRITORY_RAMP) for count in (rivals, own))

    @classmethod
    def _is_prey(cls, hunter: Bitzkey, other: Bitzkey) -> bool:
        if hunter.species == "omnivore":
            # Grazers, and other omnivores only in an overcrowded territory, once hungry past its
            # reluctance as that rivalry wears it down.
            return other.species == "grazer" or (
                other.species == "omnivore" and hunter.own_rivalry > 0
                and hunter.hunger / 100 > cls.cannibal_reluctance(hunter) * (1 - hunter.own_rivalry))
        if hunter.species != "predator":
            return False
        if other.species in ("grazer", "omnivore"):
            return True
        if other.species != "predator":
            return False
        # Its own kind only when hungry past its reluctance, which a territory overcrowded with
        # predators wears away.
        return (not cls._same_kind(hunter, other)
                or hunter.hunger / 100 > cls.cannibal_reluctance(hunter) * (1 - hunter.own_rivalry))
