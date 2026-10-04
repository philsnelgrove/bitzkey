"""Turn-based Bitzkey world and its deterministic history."""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from functools import cached_property

from tools.Brain import ACTIONS, SENSES, Brain, choice_odds, make_senses
from tools.Genome import (GENOME, MIND_TRAITS, SPECIES, body_word, color_from_word, cross, decode_body,
                          decode_mind, make_dna, mind_word)
from tools.Census import PARTS, family, tally
from tools.Insights import Insights


GRID_SIZE = 100
INITIAL_POPULATION = 90
INITIAL_GRASS = 500
GRASS_PER_TURN = 2
MAX_POPULATION = 600
MAX_HISTORY = 500
CORPSE_TURNS = 30
COMPATIBLE_BITS = 14  # partners' body words (outer rings) may differ in at most this many bits

DEATH_PENALTY = 2.0
BREED_JOY = 1.0
FIGHT_JOY = 0.5
COMPANY_JOY = 0.1
CURIOSITY_JOY = 0.15
FEAR_DREAD = 0.3
LEARNING_RATE_PER_POINT = 0.003
STRIKE_FORCE = 1.5       # scales every blow
KILL_MEAL = 25           # hunger a kill sates, plus KILL_MEAL_PER_SIZE for each point of the prey's size
KILL_MEAL_PER_SIZE = 3
TEMPERAMENT_RELAPSE = 0.005  # share of the gap back to the inborn value closed each turn

# How experiences reshape temperament, in trait points before scaling by plasticity.
EXPERIENCES = {
    "kill": {"aggression": 0.6, "fear": -0.3},
    "hit": {"aggression": 0.1},
    "stung": {"aggression": -0.2, "fear": 0.2},
    "wounded": {"fear": 0.5, "aggression": -0.1},
    "wounded_alone": {"sociability": 0.3},
    "bred": {"libido": 0.3, "sociability": 0.2},
    "bred_into_hunger": {"libido": -0.8},
    "starving": {"appetite": 0.05, "patience": -0.05},
    "foraged_meal": {"patience": 0.1},
    "discovery": {"curiosity": 0.15},
    "ambushed": {"curiosity": -0.3, "fear": 0.2},
}

GRASS_SEEN = SENSES.index("grass_seen")
MAX_SIGHT = 2 + 15 // 2
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

    @property
    def max_health(self) -> float:
        return 60.0 + 6 * int(self.body["size"])

    @property
    def burn(self) -> float:
        """Hunger gained each turn: a bigger, stronger, faster, better-armed body costs more."""
        body = self.body
        return (0.3 + 0.02 * int(body["size"]) + 0.012 * int(body["strength"])
                + 0.016 * int(body["speed"]) + 0.008 * (int(body["weapons"]) + int(body["armour"])))

    @property
    def steps(self) -> int:
        return 1 + int(self.body["speed"]) // 5

    @property
    def alive(self) -> bool:
        return self.health > 0 and self.hunger < 100

    @property
    def dot_color(self) -> tuple[int, int, int]:
        return color_from_word(mind_word(self.dna))

    @property
    def ring_color(self) -> tuple[int, int, int]:
        return color_from_word(body_word(self.dna))

    @cached_property
    def colour_families(self) -> dict[str, str]:
        """The colour family of the ring and of the dot."""
        return {"ring": family(self.ring_color), "dot": family(self.dot_color)}

    def copy(self) -> Bitzkey:
        return replace(self, brain=self.brain.copy() if self.brain else None, mind=self.mind.copy())


class Simulation:
    def __init__(self, width: int = GRID_SIZE, height: int = GRID_SIZE,
                 population: int = INITIAL_POPULATION, seed: int | None = None):
        if width < 10 or height < 10:
            raise ValueError("World dimensions must both be at least 10 cells.")
        if population < 0:
            raise ValueError("Population cannot be negative.")
        self.width = width
        self.height = height
        self.random = random.Random(seed)
        self.insights = Insights()
        # Living Bitzkeys per colour family, for ring and dot, indexed by turn. It outlives the
        # rewind history; replaying reads it, and a new turn discards any abandoned future.
        self.census: list[dict[str, dict[str, int]]] = []
        self.turn = 0
        self.deaths = 0
        self.next_identifier = 0
        self.agents: list[Bitzkey] = []
        self._cells: dict[tuple[int, int], list[Bitzkey]] = {}
        self._population = 0
        self.grass: dict[tuple[int, int], int] = {}
        for _ in range(INITIAL_GRASS):
            cell = (self.random.randrange(width), self.random.randrange(height))
            self.grass[cell] = self.grass.get(cell, 0) + 1
        for _ in range(population):
            self._add_agent(make_dna(self.random.randrange(len(SPECIES)), self.random),
                            self.random.randrange(width), self.random.randrange(height))
        self.history: list[dict] = []
        self.history_index = 0
        self._take_census()
        self._record()

    @property
    def living(self) -> list[Bitzkey]:
        return [agent for agent in self.agents if agent.alive]

    def _add_agent(self, dna: int, x: int, y: int, brain: Brain | None = None) -> Bitzkey:
        agent = Bitzkey(self.next_identifier, dna & GENOME, x, y)
        agent.health = agent.max_health
        if agent.species != "plant":
            agent.brain = brain or Brain.random(self.random)
        self.next_identifier += 1
        self.agents.append(agent)
        self._cells.setdefault((x, y), []).append(agent)
        return agent

    def _record(self) -> None:
        snapshot = {
            "turn": self.turn,
            "deaths": self.deaths,
            "next_identifier": self.next_identifier,
            "agents": [agent.copy() for agent in self.agents],
            "grass": self.grass.copy(),
            "random_state": self.random.getstate(),
            "insights": self.insights.snapshot(),
        }
        self.history = self.history[:self.history_index + 1]
        self.history.append(snapshot)
        if len(self.history) > MAX_HISTORY:
            self.history.pop(0)
        self.history_index = len(self.history) - 1

    def _restore(self, index: int) -> None:
        state = self.history[index]
        self.turn = state["turn"]
        self.deaths = state["deaths"]
        self.next_identifier = state["next_identifier"]
        self.agents = [agent.copy() for agent in state["agents"]]
        self.grass = state["grass"].copy()
        self.random.setstate(state["random_state"])
        self.insights.restore(state["insights"])
        self.history_index = index
        self._index_cells()

    def rewind(self) -> bool:
        if self.history_index == 0:
            return False
        self._restore(self.history_index - 1)
        return True

    def advance(self) -> None:
        if self.history_index + 1 < len(self.history):
            self._restore(self.history_index + 1)
            return
        self._take_turn()
        self._record()

    def _take_turn(self) -> None:
        self.turn += 1
        del self.census[self.turn:]
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

        for _ in range(GRASS_PER_TURN):
            cell = (self.random.randrange(self.width), self.random.randrange(self.height))
            self.grass[cell] = self.grass.get(cell, 0) + 1
        self.agents = [agent for agent in self.agents
                       if agent.died_turn is None or self.turn - agent.died_turn < CORPSE_TURNS]
        self.insights.decay()
        self._take_census()

    def _take_census(self) -> None:
        living = self.living
        self.census.append({part: tally(agent.colour_families[part] for agent in living) for part in PARTS})

    # Temperament

    def _experience(self, agent: Bitzkey, event: str) -> None:
        """Shift temperament after an experience; intellect sets how far."""
        plasticity = 0.2 + 0.8 * agent.mind["intellect"] / 15
        for trait, change in EXPERIENCES[event].items():
            agent.mind[trait] = max(0.0, min(15.0, agent.mind[trait] + change * plasticity))

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
            discount = 0.7 + 0.27 * agent.mind["patience"] / 15
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
        nursery = self._free_cell_near(agent)
        prey = [other for other in neighbors if self._is_prey(agent, other)]
        threats = [other for other in neighbors if self._is_prey(other, agent)]
        target = max(prey, key=lambda other: self._damage(agent, other) / other.health, default=None)
        threat = max(threats, key=lambda other: self._damage(other, agent), default=None)
        kin = sum(other.species == agent.species for other in self._nearby(agent, 2))
        sight = 2 + int(agent.body["sight"]) // 2
        grazes = agent.species != "predator"
        grass_here = grazes and self.grass.get((agent.x, agent.y), 0) > 0
        grass = self._nearest_grass(agent, sight) if grazes else None
        quarry, hunter, suitor = self._look_around(agent, sight)

        available = {"rest", "explore"}
        if grass_here:
            available.add("eat")
        if grass:
            available.add("forage")
        if suitor:
            available.add("court")
        if mate and agent.cooldown == 0 and nursery and self._population < MAX_POPULATION:
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
            age=min(1.0, agent.age / 400),
            **{trait: agent.mind[trait] / 15 for trait in MIND_TRAITS},
        )
        situation = {"mate": mate, "nursery": nursery, "prey": target, "threat": threat,
                     "threats": len(threats), "quarry": quarry, "hunter": hunter, "suitor": suitor,
                     "grass": grass, "grass_here": grass_here, "kin": kin}
        return senses, [ACTIONS.index(action) for action in ACTIONS if action in available], situation

    def _wellbeing(self, agent: Bitzkey) -> float:
        """How comfortable an animal feels; appetite sets how much hunger hurts."""
        appetite = 0.5 + agent.mind["appetite"] / 15
        return (-appetite * min(100.0, agent.hunger) / 100
                - (1 - max(0.0, agent.health) / agent.max_health))

    @staticmethod
    def _comfort(agent: Bitzkey, situation: dict) -> float:
        """Company pleases the sociable; hunters close by frighten the fearful."""
        company = COMPANY_JOY * agent.mind["sociability"] / 15 * min(3, situation["kin"]) / 3
        dread = FEAR_DREAD * agent.mind["fear"] / 15 * min(2, situation["threats"]) / 2
        return company - dread

    def _learn(self, agent: Bitzkey, target: float) -> None:
        learning_rate = LEARNING_RATE_PER_POINT * agent.mind["intellect"]
        agent.brain.learn(agent.last_senses, agent.last_choice, target, learning_rate)

    def _die(self, agent: Bitzkey, cause: str) -> None:
        if agent.brain is not None and agent.last_senses is not None:
            self._learn(agent, agent.reward + self._wellbeing(agent) - agent.last_wellbeing - DEATH_PENALTY)
        agent.brain = None
        agent.died_turn = self.turn
        agent.last_action = cause
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
        elif action == "rest":
            agent.hunger = max(0, agent.hunger - agent.burn / 2)
            agent.health = min(agent.max_health, agent.health + agent.max_health * 0.015)
            agent.last_action = "rested"
        elif action == "court":
            suitor = situation["suitor"]
            self._move_toward(agent, suitor.x, suitor.y)
            agent.last_action = "courted"
        elif action == "breed":
            self._breed(agent, situation["mate"], situation["nursery"])
            agent.last_action = "mated"
        elif action == "hunt":
            quarry = situation["quarry"]
            self._move_toward(agent, quarry.x, quarry.y)
            if abs(quarry.x - agent.x) + abs(quarry.y - agent.y) <= 1:
                self._attack(agent, quarry)
                agent.last_action = "pounced"
            else:
                agent.last_action = "hunted"
        elif action == "attack":
            self._attack(agent, situation["prey"])
        elif action == "flee":
            self._move_away(agent, situation["hunter"])
            agent.last_action = "fled"
        else:
            for _ in range(agent.steps):
                self._move(agent, (agent.x + self.random.choice((-1, 0, 1))) % self.width,
                           (agent.y + self.random.choice((-1, 0, 1))) % self.height)
            agent.reward += CURIOSITY_JOY * agent.mind["curiosity"] / 15
            agent.last_action = "explored"

    @staticmethod
    def _damage(attacker: Bitzkey, defender: Bitzkey) -> float:
        """One blow: strength and size, sharpened by weapons, blunted by armour."""
        body = attacker.body
        force = STRIKE_FORCE * (2 + 0.4 * int(body["strength"]) + 0.2 * int(body["size"]))
        return force * (1 + int(body["weapons"]) / 20) * (1 - int(defender.body["armour"]) / 25)

    def _attack(self, agent: Bitzkey, victim: Bitzkey) -> None:
        joy = FIGHT_JOY * agent.mind["aggression"] / 15
        victim.health -= self._damage(agent, victim)
        agent.hunger = max(0, agent.hunger - 5)
        agent.reward += joy
        agent.last_action = "attacked"
        self._experience(agent, "hit")
        if victim.alive:
            victim.last_action = "wounded"
            self._experience(victim, "wounded")
            if not any(other.species == victim.species for other in self._nearby(victim, 2)):
                self._experience(victim, "wounded_alone")
        else:
            agent.hunger = max(0, agent.hunger - KILL_MEAL - KILL_MEAL_PER_SIZE * int(victim.body["size"]))
            agent.reward += 2 * joy
            self._experience(agent, "kill")
            self._die(victim, "killed")

        sting = 0.3 * int(victim.body["weapons"]) * (1 - int(agent.body["armour"]) / 25)
        if sting > 0:
            agent.health -= sting
            self._experience(agent, "stung")
            if not agent.alive:
                self._die(agent, "killed")

    def _breed(self, first: Bitzkey, second: Bitzkey, cell: tuple[int, int]) -> None:
        """Fertile parents breed cheaply and often; each gives its newborn the energy it spends."""
        costs = [25 - int(parent.body["fertility"]) for parent in (first, second)]
        brain = Brain.inherit(first.brain, second.brain, self.random)
        child = self._add_agent(cross(first.dna, second.dna, self.random), *cell, brain=brain)
        child.hunger = 100 - sum(costs)
        child.generation = max(first.generation, second.generation) + 1
        self._population += 1
        for parent, cost in zip((first, second), costs, strict=True):
            parent.hunger += cost
            parent.cooldown = 20 - int(parent.body["fertility"])
            parent.children += 1
            parent.reward += BREED_JOY * parent.mind["libido"] / 15
            if not parent.alive:
                self._die(parent, "bred to death")
            else:
                self._experience(parent, "bred_into_hunger" if parent.hunger >= 75 else "bred")

    # Space

    def _index_cells(self) -> None:
        self._cells = {}
        for agent in self.living:
            self._cells.setdefault((agent.x, agent.y), []).append(agent)

    def _nearby(self, agent: Bitzkey, radius: int) -> list[Bitzkey]:
        found = []
        for dx in range(-radius, radius + 1):
            for dy in range(-(radius - abs(dx)), radius - abs(dx) + 1):
                for other in self._cells.get((agent.x + dx, agent.y + dy), ()):
                    if other is not agent and other.alive:
                        found.append(other)
        return found

    def _free_cell_near(self, agent: Bitzkey) -> tuple[int, int] | None:
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            cell = ((agent.x + dx) % self.width, (agent.y + dy) % self.height)
            if not any(other.alive for other in self._cells.get(cell, ())):
                return cell
        return None

    def _nearest_grass(self, agent: Bitzkey, sight: int) -> tuple[int, int] | None:
        for dx, dy in SIGHT_OFFSETS:
            if abs(dx) + abs(dy) > sight:
                return None
            cell = (agent.x + dx, agent.y + dy)
            if self.grass.get(cell, 0) > 0:
                return cell
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
