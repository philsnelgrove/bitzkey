"""Turn-based Bitzkey world and its deterministic history."""

from __future__ import annotations

import bisect
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property

from tools.Brain import ACTIONS, SENSES, Brain, choice_odds, make_senses
from tools.Genome import (GENOME, MIND_TRAITS, SPECIES, TRAIT_MAX, body_word, color_from_word, cross,
                          decode_body, decode_mind, make_dna, mind_word, share)
from tools.Census import Census
from tools.Exchange import Imported
from tools.Insights import Insights


GRID_SIZE = 100
INITIAL_POPULATION = 90
GRASS_DENSITY = 5       # percent of cells that start with grass
GRASS_REGROWTH = 0.004  # share of the starting grass that grows back each turn
MAX_POPULATION = 600
SPAWN_GROUP = (3, 5)  # Bitzkeys spawn in groups of this many sharing one genome
MAX_IMPORTED = MAX_POPULATION // SPAWN_GROUP[1]  # imported Bitzkeys, so their groups fit the cap
SPAWN_SPREAD = 2      # a group's members appear within this many cells of each other's centre
DENSE_HISTORY = 500   # the latest turns are each kept as a snapshot of the world
CHECKPOINTS = 200     # older turns are kept as at most this many evenly spaced snapshots
CORPSE_TURNS = 30
COMPATIBLE_BITS = 14  # partners' body words (outer rings) may differ in at most this many bits
MAX_LITTER = 4

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
STRIKE_FORCE = 1.5     # scales every blow
KILL_MEAL = 25         # hunger a kill sates, plus up to KILL_MEAL_FOR_SIZE for the largest prey
KILL_MEAL_FOR_SIZE = 45
CARRION_MEAT = 30      # meat a well-fed corpse leaves, plus up to CARRION_MEAT_FOR_SIZE for the largest;
CARRION_MEAT_FOR_SIZE = 50  # a hungrier body leaves less, and a starved one nothing
SCAVENGE_BITE = 25     # meat eaten in one go
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
}

GRASS_SEEN = SENSES.index("grass_seen")
MAX_SIGHT = 2 + TRAIT_MAX
SIGHT_OFFSETS = sorted(
    ((dx, dy) for dx in range(-MAX_SIGHT, MAX_SIGHT + 1) for dy in range(-MAX_SIGHT, MAX_SIGHT + 1)
     if 0 < abs(dx) + abs(dy) <= MAX_SIGHT),
    key=lambda offset: (abs(offset[0]) + abs(offset[1]), offset),
)
SEEN_OFFSETS = [(0, 0), *SIGHT_OFFSETS]


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
    def burn(self) -> float:
        """Hunger gained each turn: a bigger, stronger, faster, better-armed body costs more."""
        return (0.3 + 0.3 * self.physique("size") + 0.18 * self.physique("strength")
                + 0.24 * self.physique("speed") + 0.12 * (self.physique("weapons") + self.physique("armour")))

    @property
    def steps(self) -> int:
        return 1 + int(self.body["speed"]) * 3 // TRAIT_MAX

    @property
    def sight(self) -> int:
        return 2 + int(self.body["sight"])

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


def _turn_of(snapshot: dict) -> int:
    return snapshot["turn"]


class Simulation:
    def __init__(self, width: int = GRID_SIZE, height: int = GRID_SIZE,
                 population: int = INITIAL_POPULATION, seed: int | None = None,
                 grass_density: float = GRASS_DENSITY, imported: Sequence[Imported] = ()):
        """Spawn `population` random Bitzkeys in groups sharing a genome, plus a group for each
        `imported` Bitzkey, whose members all start with copies of its saved brain if it has one."""
        if width < 10 or height < 10:
            raise ValueError("World dimensions must both be at least 10 cells.")
        if population < 0:
            raise ValueError("Population cannot be negative.")
        if not 0 <= grass_density <= 100:
            raise ValueError("Grass density must be between 0 and 100 percent.")
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
        if self.turn < self.frontier:
            self.seek(self.turn + 1)
            return
        self._take_turn()
        self._record()

    def _take_turn(self, replay: bool = False) -> None:
        """One turn. A replay re-simulates a turn already recorded, so the census, which already
        holds it, is left alone."""
        self.turn += 1
        self._index_cells()
        order = self.living
        self._population = len(order)
        self.random.shuffle(order)
        for agent in order:
            if not agent.alive:
                continue
            agent.age += 1
            agent.hunger += agent.burn
            if agent.cooldown:
                agent.cooldown -= 1
            if agent.health < agent.max_health and agent.hunger < 65:
                agent.health = min(agent.max_health, agent.health + agent.max_health * 0.005)
            if agent.hunger >= 100:
                self._die(agent, "starved")
                continue
            if agent.species == "plant":
                agent.hunger = max(0, agent.hunger - 1.5)
                agent.last_action = "photosynthesized"
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

    def _sow(self, tufts: int) -> None:
        for _ in range(tufts):
            cell = (self.random.randrange(self.width), self.random.randrange(self.height))
            self.grass[cell] = self.grass.get(cell, 0) + 1

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

        odds = choice_odds(values, options)
        roll = self.random.random()
        choice = options[-1]
        for option in options:
            roll -= odds[option]
            if roll < 0:
                choice = option
                break
        action = ACTIONS[choice]
        self.insights.record(agent.species, agent.profile, (ACTIONS[option] for option in options), action)
        agent.odds = {ACTIONS[option]: chance for option, chance in odds.items()}
        agent.last_senses = senses
        agent.last_choice = choice
        agent.last_wellbeing = self._wellbeing(agent)
        agent.reward = 0.0
        self._act(agent, action, situation)

    def _sense(self, agent: Bitzkey) -> tuple[list[float], list[int], dict]:
        neighbors = self._nearby(agent, 1)
        mate = next((other for other in neighbors
                     if self._compatible(agent, other) and other.cooldown == 0), None)
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
        quarry, hunter, suitor = self._look_around(agent, sight)

        available = {"rest", "explore"}
        if grass_here:
            available.add("eat")
        if grass:
            available.add("forage")
        if carrion:
            available.add("scavenge")
        if suitor:
            available.add("court")
        if mate and agent.cooldown == 0 and nurseries and self._population < MAX_POPULATION:
            available.add("breed")
        if quarry:
            available.add("hunt")
        if target:
            available.add("attack")
        if hunter:
            available.add("flee")

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
            age=min(1.0, agent.age / 400),
            **{trait: agent.feeling(trait) for trait in MIND_TRAITS},
        )
        situation = {"mate": mate, "nurseries": nurseries, "prey": target, "threat": threat,
                     "threats": len(threats), "quarry": quarry, "hunter": hunter, "suitor": suitor,
                     "grass": grass, "grass_here": grass_here, "carrion": carrion, "kin": kin,
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
        return company - dread - unease

    def _learn(self, agent: Bitzkey, target: float) -> None:
        agent.brain.learn(agent.last_senses, agent.last_choice, target, LEARNING_RATE * agent.feeling("intellect"))

    def _die(self, agent: Bitzkey, cause: str) -> None:
        """Record a death. The body becomes carrion holding the energy it had left: a starved body
        leaves nothing, and a killed one half, as the killer has already eaten."""
        if agent.brain is not None and agent.last_senses is not None:
            self._learn(agent, agent.reward + self._wellbeing(agent) - agent.last_wellbeing - DEATH_PENALTY)
        agent.brain = None
        agent.died_turn = self.turn
        agent.last_action = cause
        nourished = max(0.0, 1 - agent.hunger / 100)
        agent.meat = ((CARRION_MEAT + CARRION_MEAT_FOR_SIZE * agent.physique("size")) * nourished
                      * (0.5 if cause == "killed" else 1))
        agent.rot = agent.meat / CORPSE_TURNS
        self._carrion.setdefault((agent.x, agent.y), []).append(agent)
        self.deaths += 1
        self._population -= 1

    # Actions

    def _act(self, agent: Bitzkey, action: str, situation: dict) -> None:
        if action == "eat":
            cell = (agent.x, agent.y)
            self.grass[cell] -= 1
            if self.grass[cell] == 0:
                del self.grass[cell]
            agent.hunger = max(0, agent.hunger - 35)
            if agent.last_action == "foraged":
                self._experience(agent, "foraged_meal")
            agent.last_action = "ate"
        elif action == "forage":
            self._move_toward(agent, *situation["grass"])
            agent.last_action = "foraged"
        elif action == "scavenge":
            self._scavenge(agent, situation["carrion"])
        elif action == "rest":
            agent.hunger = max(0, agent.hunger - agent.burn / 2)
            agent.health = min(agent.max_health, agent.health + agent.max_health * 0.015)
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
        elif action == "flee":
            self._move_away(agent, situation["hunter"])
            agent.last_action = "fled"
        else:
            for _ in range(agent.steps):
                self._move(agent, (agent.x + self.random.choice((-1, 0, 1))) % self.width,
                           (agent.y + self.random.choice((-1, 0, 1))) % self.height)
            agent.reward += CURIOSITY_JOY * agent.feeling("curiosity")
            agent.last_action = "explored"

    def _scavenge(self, agent: Bitzkey, corpse: Bitzkey) -> None:
        """Go to the carrion and, once alongside, eat; a carrion-eater's gut gets the most from it."""
        self._move_toward(agent, corpse.x, corpse.y)
        if abs(corpse.x - agent.x) + abs(corpse.y - agent.y) > 1:
            agent.last_action = "sought carrion"
            return
        bite = min(corpse.meat, SCAVENGE_BITE)
        corpse.meat -= bite
        agent.hunger = max(0, agent.hunger - bite * agent.digestion)
        agent.last_action = "scavenged"

    @staticmethod
    def _damage(attacker: Bitzkey, defender: Bitzkey) -> float:
        """One blow: strength and size, sharpened by weapons, blunted by armour."""
        force = STRIKE_FORCE * (2 + 6 * attacker.physique("strength") + 3 * attacker.physique("size"))
        return force * (1 + 0.75 * attacker.physique("weapons")) * (1 - 0.6 * defender.physique("armour"))

    def _attack(self, agent: Bitzkey, victim: Bitzkey) -> None:
        """A blow. The intolerant enjoy fighting other species most; the empathetic feel their
        victim's pain, and so does any empathetic animal watching, most for its own kind."""
        empathy = agent.feeling("empathy")
        joy = FIGHT_JOY * agent.feeling("aggression") * (1 - 0.6 * empathy) * (1 - 0.75 * agent.feeling("tolerance"))
        victim.health -= self._damage(agent, victim)
        agent.hunger = max(0, agent.hunger - 5)
        agent.reward += joy
        agent.last_action = "attacked"
        self._experience(agent, "hit")
        killed = not victim.alive
        if not killed:
            victim.last_action = "wounded"
            self._experience(victim, "wounded")
            if not any(other.species == victim.species for other in self._nearby(victim, 2)):
                self._experience(victim, "wounded_alone")
        else:
            agent.hunger = max(0, agent.hunger - KILL_MEAL - KILL_MEAL_FOR_SIZE * victim.physique("size"))
            agent.reward += 2 * joy
            self._experience(agent, "kill")
        pain = EMPATHY_PAIN * (2 if killed else 1)
        agent.reward -= pain * empathy * OTHER_KIND
        for witness in self._nearby(victim, 2):
            if witness is not agent and witness.brain is not None:
                kinship = 1.0 if witness.species == victim.species else OTHER_KIND
                witness.reward -= pain * witness.feeling("empathy") * kinship
        if killed:
            self._die(victim, "killed")

        sting = 4.5 * victim.physique("weapons") * (1 - 0.6 * agent.physique("armour"))
        if sting > 0:
            agent.health -= sting
            self._experience(agent, "stung")
            if not agent.alive:
                self._die(agent, "killed")

    def _breed(self, first: Bitzkey, second: Bitzkey, cells: list[tuple[int, int]]) -> None:
        """Fertile parents breed cheaply and often. Their litter, 1 to 4 young by their average
        litter trait and the free cells around, shares the energy both parents give up; a parent
        can only give what it has left. Each child inherits a mix of its parents' brains, and
        starts its temperament at its own genes shifted by the average of what experience
        taught its parents."""
        litter = 1 + (int(first.body["litter"]) + int(second.body["litter"])) * (MAX_LITTER - 1) // (2 * TRAIT_MAX)
        cells = cells[:max(1, min(litter, MAX_POPULATION - self._population))]
        costs = [25 - 15 * parent.physique("fertility") for parent in (first, second)]
        given = sum(min(cost, max(0.0, 100 - parent.hunger)) for cost, parent in zip(costs, (first, second)))
        for cell in cells:
            brain = Brain.inherit(first.brain, second.brain, self.random)
            child = self._add_agent(cross(first.dna, second.dna, self.random), *cell, brain=brain)
            for trait in MIND_TRAITS:
                learned = sum(parent.mind[trait] - parent.inborn[trait] for parent in (first, second)) / 2
                child.mind[trait] = max(0.0, min(float(TRAIT_MAX), child.inborn[trait] + learned))
            child.hunger = 100 - given / len(cells)
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

    def _free_cells_near(self, agent: Bitzkey) -> list[tuple[int, int]]:
        """The empty cells beside an animal, where young can be born."""
        cells = []
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
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
                     sight: int) -> tuple[Bitzkey | None, Bitzkey | None, Bitzkey | None]:
        """The nearest prey, hunter and willing mate in sight, for the roles this animal can have."""
        quarry = hunter = suitor = None
        hunts = agent.species in ("omnivore", "predator")
        hunted = agent.species in ("grazer", "omnivore")
        courts = agent.cooldown == 0
        for dx, dy in SEEN_OFFSETS:
            if abs(dx) + abs(dy) > sight or ((quarry or not hunts) and (hunter or not hunted)
                                             and (suitor or not courts)):
                break
            for other in self._cells.get((agent.x + dx, agent.y + dy), ()):
                if other is agent or not other.alive:
                    continue
                if quarry is None and self._is_prey(agent, other):
                    quarry = other
                elif hunter is None and self._is_prey(other, agent):
                    hunter = other
                elif (courts and suitor is None and other.cooldown == 0
                      and self._compatible(agent, other)):
                    suitor = other
        return quarry, hunter, suitor

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
        return (first.species != "plant" and first.species == second.species
                and (body_word(first.dna) ^ body_word(second.dna)).bit_count() <= COMPATIBLE_BITS)

    @staticmethod
    def _is_prey(hunter: Bitzkey, other: Bitzkey) -> bool:
        return ((hunter.species == "predator" and other.species in ("grazer", "omnivore"))
                or (hunter.species == "omnivore" and other.species == "grazer"))
