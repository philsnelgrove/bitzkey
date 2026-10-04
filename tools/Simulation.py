"""Turn-based Bitzkey world and its deterministic history."""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

from tools.Brain import ACTIONS, Brain, choice_odds, make_senses
from tools.Insights import Insights


GRID_SIZE = 100
INITIAL_POPULATION = 90
INITIAL_FOOD = 500
FOOD_PER_TURN = 2
MAX_POPULATION = 600
MAX_HISTORY = 500
CORPSE_TURNS = 30
SPECIES = ("plant", "grazer", "omnivore", "predator")

DISCOUNT = 0.9
DEATH_PENALTY = 2.0
BREED_JOY = 1.0
FIGHT_JOY = 0.5
COMPANY_JOY = 0.1
LEARNING_RATE_PER_POINT = 0.003
BREED_COST = 20
BIRTH_HUNGER = 100 - 2 * BREED_COST  # a newborn holds exactly the energy its parents gave up
BREED_COOLDOWN = 12

MAX_SIGHT = 2 + 15 // 2
SIGHT_OFFSETS = sorted(
    ((dx, dy) for dx in range(-MAX_SIGHT, MAX_SIGHT + 1) for dy in range(-MAX_SIGHT, MAX_SIGHT + 1)
     if 0 < abs(dx) + abs(dy) <= MAX_SIGHT),
    key=lambda offset: (abs(offset[0]) + abs(offset[1]), offset),
)


def decode_dna(dna: int) -> dict[str, int | str]:
    """Decode the 32-bit genome as eight four-bit traits."""
    dna &= 0xFFFFFFFF
    values = [(dna >> shift) & 0xF for shift in range(0, 32, 4)]
    return {
        "metabolism": values[0],
        "aggression": values[1],
        "constitution": values[2],
        "species": SPECIES[values[3] % len(SPECIES)],
        "fertility": values[4],
        "sight": values[5],
        "social": values[6],
        "learning": values[7],
    }


def color_from_dna(dna: int) -> tuple[int, int, int]:
    """Convert the four DNA bytes (CMYK) to the displayed RGB color."""
    cyan, magenta, yellow, black = ((dna >> shift) & 0xFF for shift in (24, 16, 8, 0))
    return (round((255 - cyan) * (255 - black) / 255),
            round((255 - magenta) * (255 - black) / 255),
            round((255 - yellow) * (255 - black) / 255))


def make_dna(species: int | None = None, rng: random.Random | None = None) -> int:
    dna = (rng or random).getrandbits(32)
    if species is not None:
        dna = (dna & ~(0xF << 12)) | ((species % 4) << 12)
    return dna


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
    # The last decision, kept until the next one so the brain can learn from how it turned out.
    last_senses: list[float] | None = None
    last_choice: int = 0
    last_wellbeing: float = 0.0
    reward: float = 0.0
    odds: dict[str, float] = field(default_factory=dict)

    @property
    def traits(self) -> dict[str, int | str]:
        return decode_dna(self.dna)

    @property
    def species(self) -> str:
        return str(self.traits["species"])

    @property
    def alive(self) -> bool:
        return self.health > 0 and self.hunger < 100

    @property
    def color(self) -> tuple[int, int, int]:
        return color_from_dna(self.dna)

    def copy(self) -> Bitzkey:
        return replace(self, brain=self.brain.copy() if self.brain else None)


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
        self.turn = 0
        self.deaths = 0
        self.next_identifier = 0
        self.agents: list[Bitzkey] = []
        self._cells: dict[tuple[int, int], list[Bitzkey]] = {}
        self._population = 0
        self.food: dict[tuple[int, int], int] = {}
        for _ in range(INITIAL_FOOD):
            cell = (self.random.randrange(width), self.random.randrange(height))
            self.food[cell] = self.food.get(cell, 0) + 1
        for _ in range(population):
            self._add_agent(make_dna(self.random.randrange(4), self.random),
                            self.random.randrange(width), self.random.randrange(height))
        self.history: list[dict] = []
        self.history_index = 0
        self._record()

    @property
    def living(self) -> list[Bitzkey]:
        return [agent for agent in self.agents if agent.alive]

    def _add_agent(self, dna: int, x: int, y: int, brain: Brain | None = None) -> Bitzkey:
        agent = Bitzkey(self.next_identifier, dna & 0xFFFFFFFF, x, y)
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
            "food": self.food.copy(),
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
        self.food = state["food"].copy()
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
        self._index_cells()
        order = self.living
        self._population = len(order)
        self.random.shuffle(order)
        for agent in order:
            if not agent.alive:
                continue
            agent.age += 1
            agent.hunger += self._hunger_rate(agent)
            if agent.cooldown:
                agent.cooldown -= 1
            if agent.health < 100 and agent.hunger < 65:
                agent.health = min(100, agent.health + 0.5)
            if agent.hunger >= 100:
                self._die(agent, "starved")
                continue
            if agent.species == "plant":
                agent.hunger = max(0, agent.hunger - 1.5)
                agent.last_action = "photosynthesized"
                continue
            self._think(agent)

        for _ in range(FOOD_PER_TURN):
            cell = (self.random.randrange(self.width), self.random.randrange(self.height))
            self.food[cell] = self.food.get(cell, 0) + 1
        self.agents = [agent for agent in self.agents
                       if agent.died_turn is None or self.turn - agent.died_turn < CORPSE_TURNS]
        self.insights.decay()

    # Decisions and learning

    def _think(self, agent: Bitzkey) -> None:
        """Sense, learn from the previous decision's outcome, then choose and act."""
        senses, options, situation = self._sense(agent)
        values = agent.brain.values(senses)
        if agent.last_senses is not None:
            reward = (agent.reward + self._wellbeing(agent) - agent.last_wellbeing
                      + self._company(agent, situation["kin"]))
            self._learn(agent, reward + DISCOUNT * max(values[option] for option in options))

        odds = choice_odds(values, options)
        roll = self.random.random()
        choice = options[-1]
        for option in options:
            roll -= odds[option]
            if roll < 0:
                choice = option
                break
        action = ACTIONS[choice]
        self.insights.record(agent.species, agent.traits, (ACTIONS[option] for option in options), action)
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
        kin = sum(other.species == agent.species for other in self._nearby(agent, 2))
        sight = 2 + int(agent.traits["sight"]) // 2
        food_here = self.food.get((agent.x, agent.y), 0) > 0
        target = self._nearest_food(agent, sight)
        closeness = 0.0 if target is None else 1 - (abs(target[0] - agent.x) + abs(target[1] - agent.y) - 1) / sight

        available = {"rest", "explore"}
        if food_here:
            available.add("eat")
        if target:
            available.add("forage")
        if mate and agent.cooldown == 0 and nursery and self._population < MAX_POPULATION:
            available.add("breed")
        if prey:
            available.add("attack")
        if threats:
            available.add("flee")

        senses = make_senses(
            hunger=min(1.0, agent.hunger / 100),
            health=max(0.0, agent.health / 100),
            food_here=float(food_here),
            food_seen=float(target is not None),
            food_closeness=closeness,
            mate_near=float(mate is not None),
            ready_to_breed=float(agent.cooldown == 0),
            prey_near=min(4, len(prey)) / 4,
            threat_near=min(4, len(threats)) / 4,
            kin_near=min(3, kin) / 3,
            age=min(1.0, agent.age / 400),
        )
        situation = {"mate": mate, "nursery": nursery, "prey": prey, "threats": threats,
                     "food": target, "kin": kin}
        return senses, [ACTIONS.index(action) for action in ACTIONS if action in available], situation

    def _wellbeing(self, agent: Bitzkey) -> float:
        """How comfortable an animal feels; metabolism sets how much hunger hurts."""
        appetite = 0.5 + int(agent.traits["metabolism"]) / 15
        return -appetite * min(100.0, agent.hunger) / 100 - (100 - max(0.0, agent.health)) / 100

    @staticmethod
    def _company(agent: Bitzkey, kin: int) -> float:
        return COMPANY_JOY * int(agent.traits["social"]) / 15 * min(3, kin) / 3

    def _learn(self, agent: Bitzkey, target: float) -> None:
        learning_rate = LEARNING_RATE_PER_POINT * int(agent.traits["learning"])
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
        traits = agent.traits
        if action == "eat":
            cell = (agent.x, agent.y)
            self.food[cell] -= 1
            if self.food[cell] == 0:
                del self.food[cell]
            agent.hunger = max(0, agent.hunger - 35)
            agent.last_action = "ate"
        elif action == "forage":
            self._move_toward(agent, *situation["food"])
            agent.last_action = "foraged"
        elif action == "rest":
            agent.hunger = max(0, agent.hunger - self._hunger_rate(agent) / 2)
            agent.health = min(100, agent.health + 1.5)
            agent.last_action = "rested"
        elif action == "breed":
            self._breed(agent, situation["mate"], situation["nursery"])
            agent.last_action = "mated"
        elif action == "attack":
            self._attack(agent, situation["prey"][0])
        elif action == "flee":
            self._move_away(agent, situation["threats"][0])
            agent.last_action = "fled"
        else:
            distance = 1 + int(traits["constitution"]) // 8
            for _ in range(distance):
                self._move(agent, (agent.x + self.random.choice((-1, 0, 1))) % self.width,
                           (agent.y + self.random.choice((-1, 0, 1))) % self.height)
            agent.last_action = "explored"

    def _attack(self, agent: Bitzkey, victim: Bitzkey) -> None:
        traits = agent.traits
        joy = FIGHT_JOY * int(traits["aggression"]) / 15
        victim.health -= 3 + int(traits["constitution"]) * 0.8
        agent.hunger = max(0, agent.hunger - 15)
        agent.reward += joy
        agent.last_action = "attacked"
        if victim.alive:
            victim.last_action = "wounded"
        else:
            agent.hunger = max(0, agent.hunger - 30)
            agent.reward += 2 * joy
            self._die(victim, "killed")

    def _breed(self, first: Bitzkey, second: Bitzkey, cell: tuple[int, int]) -> None:
        split = self.random.randrange(1, 32)
        mask = (1 << split) - 1
        dna = (first.dna & mask) | (second.dna & ~mask & 0xFFFFFFFF)
        if self.random.random() < 0.2:
            dna ^= 1 << self.random.randrange(32)
        brain = Brain.inherit(first.brain, second.brain, self.random)
        child = self._add_agent(dna, *cell, brain=brain)
        child.hunger = BIRTH_HUNGER
        child.generation = max(first.generation, second.generation) + 1
        self._population += 1
        for parent in (first, second):
            parent.hunger += BREED_COST
            parent.cooldown = BREED_COOLDOWN
            parent.children += 1
            parent.reward += BREED_JOY * int(parent.traits["fertility"]) / 15

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

    def _nearest_food(self, agent: Bitzkey, sight: int) -> tuple[int, int] | None:
        for dx, dy in SIGHT_OFFSETS:
            if abs(dx) + abs(dy) > sight:
                return None
            cell = (agent.x + dx, agent.y + dy)
            if self.food.get(cell, 0) > 0:
                return cell
        return None

    def _move(self, agent: Bitzkey, x: int, y: int) -> None:
        self._cells[(agent.x, agent.y)].remove(agent)
        agent.x, agent.y = x, y
        self._cells.setdefault((x, y), []).append(agent)

    def _move_toward(self, agent: Bitzkey, x: int, y: int) -> None:
        distance = 1 + int(agent.traits["constitution"]) // 6
        for _ in range(distance):
            if agent.x != x:
                self._move(agent, agent.x + (1 if x > agent.x else -1), agent.y)
            elif agent.y != y:
                self._move(agent, agent.x, agent.y + (1 if y > agent.y else -1))

    def _move_away(self, agent: Bitzkey, threat: Bitzkey) -> None:
        dx = 1 if agent.x >= threat.x else -1
        dy = 1 if agent.y >= threat.y else -1
        self._move(agent, (agent.x + dx) % self.width, (agent.y + dy) % self.height)

    def _hunger_rate(self, agent: Bitzkey) -> float:
        return 0.35 + int(agent.traits["metabolism"]) * 0.055

    @staticmethod
    def _compatible(first: Bitzkey, second: Bitzkey) -> bool:
        return (first.species != "plant" and first.species == second.species
                and (first.dna ^ second.dna).bit_count() <= 12)

    @staticmethod
    def _is_prey(hunter: Bitzkey, other: Bitzkey) -> bool:
        return ((hunter.species == "predator" and other.species in ("grazer", "omnivore"))
                or (hunter.species == "omnivore" and other.species == "grazer"))
