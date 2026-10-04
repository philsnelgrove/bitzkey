"""Turn-based Bitzkey world and its deterministic history."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from tools.AggressionNetwork import AggressionNetwork


GRID_SIZE = 100
INITIAL_POPULATION = 90
INITIAL_FOOD = 500
MAX_HISTORY = 500
AGGRESSION_BATCH_SIZE = 32
SPECIES = ("plant", "grazer", "omnivore", "predator")


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

@dataclass
class Food:
    x: int
    y: int
    quantity: int = 1


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
        network_seed = None if seed is None else seed ^ 0xA66E5510
        self.aggression_network = AggressionNetwork(random.Random(network_seed))
        self.aggression_training_samples: list[tuple[list[float], float]] = []
        self._policy_scores: dict[int, float] = {}
        self.turn = 0
        self.next_identifier = 0
        self.agents: list[Bitzkey] = []
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

    def aggression_score(self, agent: Bitzkey) -> float:
        """Return the latest policy score, or the genome baseline before inference."""
        return self._policy_scores.get(agent.identifier, float(agent.traits["aggression"]))

    def _add_agent(self, dna: int, x: int, y: int) -> Bitzkey:
        agent = Bitzkey(self.next_identifier, dna & 0xFFFFFFFF, x, y)
        self.next_identifier += 1
        self.agents.append(agent)
        return agent

    def _record(self) -> None:
        snapshot = {
            "turn": self.turn,
            "next_identifier": self.next_identifier,
            "agents": copy.deepcopy(self.agents),
            "food": self.food.copy(),
            "random_state": self.random.getstate(),
            "aggression_network": self.aggression_network.snapshot(),
            "aggression_training_samples": copy.deepcopy(self.aggression_training_samples),
            "policy_scores": self._policy_scores.copy(),
        }
        self.history = self.history[:self.history_index + 1]
        self.history.append(snapshot)
        if len(self.history) > MAX_HISTORY:
            self.history.pop(0)
        self.history_index = len(self.history) - 1

    def _restore(self, index: int) -> None:
        state = self.history[index]
        self.turn = state["turn"]
        self.next_identifier = state["next_identifier"]
        self.agents = copy.deepcopy(state["agents"])
        self.food = state["food"].copy()
        self.random.setstate(state["random_state"])
        self.aggression_network.restore(state["aggression_network"])
        self.aggression_training_samples = copy.deepcopy(state["aggression_training_samples"])
        self._policy_scores = state["policy_scores"].copy()
        self.history_index = index

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
        observations = self._collect_aggression_observations()
        self._policy_scores = {
            agent.identifier: max(0.0, min(15.0, int(agent.traits["aggression"]) + 5 * adjustment))
            for agent, adjustment in zip(
                (agent for agent, _ in observations),
                self.aggression_network.predict_batch(
                    [observation for _, observation in observations]
                ),
                strict=True,
            )
        }
        order = self.living[:]
        self.random.shuffle(order)
        for agent in order:
            if not agent.alive:
                continue
            traits = agent.traits
            agent.age += 1
            agent.hunger += 0.35 + int(traits["metabolism"]) * 0.055
            if agent.hunger >= 100:
                agent.last_action = "starved"
                continue
            if agent.cooldown:
                agent.cooldown -= 1
            if agent.health < 100 and agent.hunger < 65:
                agent.health = min(100, agent.health + 0.5)
            if agent.species == "plant":
                agent.hunger = max(0, agent.hunger - 1.5)
                agent.last_action = "photosynthesized"
                continue

            neighbors = [other for other in self.living if other is not agent
                         and self._distance(agent, other) <= 1]
            mate = next((other for other in neighbors if self._compatible(agent, other)), None)
            if mate and agent.cooldown == 0 and agent.hunger < 45:
                if self.random.random() < 0.1 + int(traits["fertility"]) / 25:
                    self._breed(agent, mate)
                    agent.last_action = "mated"
                    agent.cooldown = 12
                    continue

            threat = next((other for other in neighbors if self._is_prey(agent, other)), None)
            if threat and self.aggression_score(agent) >= 7:
                damage = 3 + int(traits["constitution"]) * 0.8
                threat.health -= damage
                agent.hunger = max(0, agent.hunger - 15)
                agent.last_action = "attacked"
                threat.last_action = "attacked"
                if not threat.alive:
                    agent.hunger = max(0, agent.hunger - 30)
                    self._learn_from_conflict(agent, 1.0)
                    for witness in neighbors:
                        self._learn_from_conflict(witness, -1.0)
                continue
            if any(self._is_prey(other, agent) for other in neighbors):
                agent.last_action = "fled" if self._move_away(agent, neighbors) else "defended"
                if agent.last_action == "defended":
                    agent.health = max(0, agent.health - 1)
                continue

            if agent.hunger > 20 and self._forage(agent):
                continue
            if self.random.random() < int(traits["social"]) / 30:
                agent.last_action = "rested"
                continue
            self._wander(agent)
            agent.last_action = "explored"

        if self.turn % 8 == 0:
            cell = (self.random.randrange(self.width), self.random.randrange(self.height))
            self.food[cell] = self.food.get(cell, 0) + 1
        self._train_aggression_network()

    def _collect_aggression_observations(self) -> list[tuple[Bitzkey, list[float]]]:
        agents = [
            agent for agent in self.living
            if agent.species != "plant" and int(agent.traits["learning"]) >= 4
        ]
        observations = [(agent, self._observation_for(agent)) for agent in agents]
        self._turn_observations = {
            agent.identifier: observation
            for agent, observation in observations
        }
        return observations

    def _observation_for(self, agent: Bitzkey) -> list[float]:
        traits = agent.traits
        neighbors = [
            other for other in self.living
            if other is not agent and self._distance(agent, other) <= 1
        ]
        prey_count = sum(self._is_prey(agent, other) for other in neighbors)
        threat_count = sum(self._is_prey(other, agent) for other in neighbors)
        species = agent.species
        return [
            int(traits["aggression"]) / 15,
            int(traits["learning"]) / 15,
            int(traits["metabolism"]) / 15,
            int(traits["constitution"]) / 15,
            max(0.0, min(1.0, agent.hunger / 100)),
            max(0.0, min(1.0, agent.health / 100)),
            min(4, prey_count) / 4,
            min(4, threat_count) / 4,
            float(species == "grazer"),
            float(species == "omnivore"),
            float(species == "predator"),
        ]

    def _learn_from_conflict(self, agent: Bitzkey, target: float) -> None:
        if int(agent.traits["learning"]) < 4:
            return
        observation = self._turn_observations.get(agent.identifier)
        if observation is not None:
            self.aggression_training_samples.append((observation.copy(), target))

    def _train_aggression_network(self) -> None:
        while len(self.aggression_training_samples) >= AGGRESSION_BATCH_SIZE:
            batch = self.aggression_training_samples[:AGGRESSION_BATCH_SIZE]
            self.aggression_network.train_batch(batch)
            del self.aggression_training_samples[:AGGRESSION_BATCH_SIZE]

    @staticmethod
    def _distance(first: Bitzkey, second: Bitzkey) -> int:
        return abs(first.x - second.x) + abs(first.y - second.y)

    @staticmethod
    def _compatible(first: Bitzkey, second: Bitzkey) -> bool:
        return (first.species != "plant" and first.species == second.species
                and (first.dna ^ second.dna).bit_count() <= 12)

    @staticmethod
    def _is_prey(hunter: Bitzkey, other: Bitzkey) -> bool:
        return ((hunter.species == "predator" and other.species in ("grazer", "omnivore"))
                or (hunter.species == "omnivore" and other.species == "grazer"))

    def _breed(self, first: Bitzkey, second: Bitzkey) -> None:
        split = self.random.randrange(1, 32)
        mask = (1 << split) - 1
        dna = (first.dna & mask) | (second.dna & ~mask & 0xFFFFFFFF)
        if self.random.random() < 0.2:
            dna ^= 1 << self.random.randrange(32)
        child = self._add_agent(dna, first.x, first.y)
        child.hunger = 5
        first.hunger += 8
        second.hunger += 8

    def _forage(self, agent: Bitzkey) -> bool:
        if agent.species == "plant":
            agent.last_action = "rested"
            return True
        cell = (agent.x, agent.y)
        if self.food.get(cell, 0) > 0:
            self.food[cell] -= 1
            if self.food[cell] == 0:
                del self.food[cell]
            agent.hunger = max(0, agent.hunger - 35)
            agent.last_action = "ate"
            return True
        sight = 2 + int(agent.traits["sight"]) // 2
        targets = [(pos, count) for pos, count in self.food.items()
                   if count and abs(pos[0] - agent.x) + abs(pos[1] - agent.y) <= sight]
        if targets:
            (target_x, target_y), _ = min(targets, key=lambda item: abs(item[0][0] - agent.x)
                                          + abs(item[0][1] - agent.y))
            self._move_toward(agent, target_x, target_y)
            agent.last_action = "foraged"
            return True
        return False

    def _move_toward(self, agent: Bitzkey, x: int, y: int) -> None:
        distance = 1 + int(agent.traits["constitution"]) // 6
        for _ in range(distance):
            if agent.x != x:
                agent.x += 1 if x > agent.x else -1
            elif agent.y != y:
                agent.y += 1 if y > agent.y else -1

    def _move_away(self, agent: Bitzkey, threats: list[Bitzkey]) -> bool:
        if int(agent.traits["aggression"]) >= 12:
            return False
        threat = threats[0]
        dx = 1 if agent.x >= threat.x else -1
        dy = 1 if agent.y >= threat.y else -1
        agent.x = (agent.x + dx) % self.width
        agent.y = (agent.y + dy) % self.height
        return True

    def _wander(self, agent: Bitzkey) -> None:
        distance = 1 + int(agent.traits["constitution"]) // 8
        for _ in range(distance):
            agent.x = (agent.x + self.random.choice((-1, 0, 1))) % self.width
            agent.y = (agent.y + self.random.choice((-1, 0, 1))) % self.height