"""What the Bitzkeys have learned: recent choice statistics and brain probing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from tools.Brain import ACTIONS, choice_odds, make_senses
from tools.Genome import BODY_TRAITS, MIND_TRAITS

if TYPE_CHECKING:
    from tools.Simulation import Bitzkey


ANIMALS = ("grazer", "omnivore", "predator")
GRAZERS = ("grazer", "omnivore")  # the animals that eat grass
TRAITS = MIND_TRAITS + BODY_TRAITS[1:]
DECAY = 0.99
MIN_SAMPLE = 5.0

# Situations used to probe brains: (title, situation senses, actions possible, species it can
# arise for). Each animal is probed with its own current temperament added to the situation.
SCENARIOS = (
    ("Fed and safe, a willing mate beside them",
     {"hunger": 0.2, "health": 1, "mate_near": 1, "mate_closeness": 1, "ready_to_breed": 1, "kin_near": 1 / 3,
      "age": 0.2},
     ("breed", "court", "rest", "explore"), ANIMALS),
    ("Fed and safe, a willing mate in sight",
     {"hunger": 0.2, "health": 1, "ready_to_breed": 1, "mate_closeness": 0.6, "age": 0.2},
     ("court", "rest", "explore"), ANIMALS),
    ("Hungry, grass underfoot, a willing mate beside them",
     {"hunger": 0.6, "health": 1, "grass_here": 1, "mate_near": 1, "mate_closeness": 1, "ready_to_breed": 1,
      "kin_near": 1 / 3, "age": 0.2},
     ("eat", "breed", "court", "rest", "explore"), GRAZERS),
    ("Hungry, grass in sight",
     {"hunger": 0.6, "health": 1, "grass_seen": 1, "grass_closeness": 0.6, "age": 0.2},
     ("forage", "rest", "explore"), GRAZERS),
    ("Hungry, a plant in sight",
     {"hunger": 0.6, "health": 1, "plant_closeness": 0.6, "age": 0.2},
     ("browse", "rest", "explore"), GRAZERS),
    ("Hungry, carrion in sight",
     {"hunger": 0.6, "health": 1, "carrion_closeness": 0.6, "age": 0.2},
     ("scavenge", "rest", "explore"), ANIMALS),
    ("Peckish, prey in sight",
     {"hunger": 0.5, "health": 1, "prey_closeness": 0.6, "age": 0.2},
     ("hunt", "rest", "explore"), ("omnivore", "predator")),
    ("A hunter in sight",
     {"hunger": 0.3, "health": 1, "threat_closeness": 0.6, "age": 0.2},
     ("flee", "rest", "explore"), ANIMALS),
    ("Wounded, a dangerous hunter beside them",
     {"hunger": 0.3, "health": 0.4, "threat_near": 0.25, "threat_danger": 0.4, "threat_closeness": 1,
      "age": 0.2},
     ("flee", "rest", "explore"), ANIMALS),
    ("Wounded, the attacker still beside them",
     {"hunger": 0.3, "health": 0.5, "threat_near": 0.25, "threat_danger": 0.3, "threat_closeness": 1,
      "assailant_near": 1, "counter_frailty": 0.3, "age": 0.2},
     ("defend", "flee", "rest", "explore"), ANIMALS),
    ("Crowded territory, a rival beside them",
     {"hunger": 0.3, "health": 1, "rivalry": 1, "prey_near": 0.25, "prey_frailty": 0.3, "prey_closeness": 1,
      "threat_near": 0.25, "threat_danger": 0.3, "threat_closeness": 1, "age": 0.2},
     ("attack", "flee", "rest", "explore"), ("omnivore", "predator")),
    ("Peckish, frail prey beside them",
     {"hunger": 0.5, "health": 1, "prey_near": 0.25, "prey_frailty": 0.5, "prey_closeness": 1, "age": 0.2},
     ("attack", "hunt", "rest", "explore"), ("omnivore", "predator")),
    ("Nothing around",
     {"hunger": 0.3, "health": 1, "age": 0.2},
     ("rest", "explore"), ANIMALS),
)


def band(value: float) -> str:
    """Trait values 0-7 in three bands: low 0-2, mid 3-4, high 5-7."""
    value = round(value)
    return "low" if value <= 2 else "mid" if value <= 4 else "high"


def group_label(species: str, trait: str | None = None, level: str | None = None) -> str:
    return f"{species}s" if trait is None else f"{level}-{trait} {species}s"


def _groups(species: str, traits: Mapping[str, float]) -> list[tuple[str, str | None, str | None]]:
    return [(species, None, None)] + [(species, trait, band(traits[trait])) for trait in TRAITS]


class Insights:
    """Recent decisions, discounted each turn: how often each kind of animal chose an action
    when it was possible."""

    def __init__(self) -> None:
        self.offered: dict[tuple, float] = {}
        self.chosen: dict[tuple, float] = {}

    def record(self, species: str, traits: Mapping[str, float],
               options: Iterable[str], choice: str) -> None:
        options = tuple(options)
        for group in _groups(species, traits):
            for action in options:
                key = (*group, action)
                self.offered[key] = self.offered.get(key, 0.0) + 1
            key = (*group, choice)
            self.chosen[key] = self.chosen.get(key, 0.0) + 1

    def decay(self) -> None:
        for counts in (self.offered, self.chosen):
            for key in list(counts):
                counts[key] *= DECAY
                if counts[key] < 0.01:
                    del counts[key]

    def rate(self, action: str, species: str, trait: str | None = None,
             level: str | None = None) -> tuple[float | None, float]:
        """Return (share of chances taken, number of recent chances)."""
        key = (species, trait, level, action)
        offered = self.offered.get(key, 0.0)
        if offered < MIN_SAMPLE:
            return None, offered
        return self.chosen.get(key, 0.0) / offered, offered

    def extremes(self, action: str, count: int = 3,
                 lowest: bool = True) -> list[tuple[str, float, float]]:
        """Trait groups that take this action least (or most) often when they can."""
        found = []
        for species, trait, level, offered_action in self.offered:
            if trait is None or offered_action != action:
                continue
            share, offered = self.rate(action, species, trait, level)
            if share is not None:
                found.append((group_label(species, trait, level), share, offered))
        found.sort(key=lambda item: (item[1] if lowest else -item[1], item[0]))
        return found[:count]

    def snapshot(self) -> tuple[dict[tuple, float], dict[tuple, float]]:
        return self.offered.copy(), self.chosen.copy()

    def restore(self, state: tuple[dict[tuple, float], dict[tuple, float]]) -> None:
        self.offered, self.chosen = state[0].copy(), state[1].copy()


def brain_report(turn: int, animals: Iterable[Bitzkey], insights: Insights) -> str:
    """Probe every living brain with standard situations and summarise who prefers what."""
    thinkers = [animal for animal in animals if animal.brain is not None]
    lines = [
        f"BITZKEY BRAIN REPORT - turn {turn:05d}, {len(thinkers)} thinking animals",
        "Figures are the average chance an animal would pick each action in the situation.",
        "Groups smaller than 5 animals are left out.",
    ]
    for title, situation, actions, species_list in SCENARIOS:
        options = [ACTIONS.index(action) for action in actions]
        headline = actions[0]
        per_group: dict[tuple, list[dict[str, float]]] = {}
        for animal in thinkers:
            if animal.species not in species_list:
                continue
            senses = make_senses(**situation, **{trait: animal.feeling(trait) for trait in MIND_TRAITS})
            odds = choice_odds(animal.brain.values(senses), options)
            named = {ACTIONS[option]: chance for option, chance in odds.items()}
            for group in _groups(animal.species, animal.profile):
                per_group.setdefault(group, []).append(named)

        lines += ["", title, "  " + " " * 18 + "".join(f"{action:>9}" for action in actions)]
        for species in species_list:
            samples = per_group.get((species, None, None), [])
            if not samples:
                lines.append(f"  {species:<10} (none)")
                continue
            averages = "".join(f"{_mean(samples, action):>8.0%} " for action in actions)
            lines.append(f"  {species:<10}{f'(n={len(samples)})':>8}{averages}")

        ranked = sorted(
            ((_mean(samples, headline), group_label(*group), len(samples))
             for group, samples in per_group.items()
             if group[1] is not None and len(samples) >= MIN_SAMPLE),
        )
        if ranked:
            lines.append(f"  Least keen to {headline}: " + _join(ranked[:3]))
            lines.append(f"  Most keen to {headline}:  " + _join(ranked[::-1][:3]))

    lines += ["", "Temperament shaped by experience (average shift from inborn values, 0..7 scale)",
              "  " + " " * 10 + "".join(f"{trait[:5]:>7}" for trait in MIND_TRAITS)]
    for species in ANIMALS:
        members = [animal for animal in thinkers if animal.species == species]
        if not members:
            continue
        shifts = "".join(
            f"{sum(animal.mind[trait] - animal.inborn[trait] for animal in members) / len(members):>+7.2f}"
            for trait in MIND_TRAITS)
        lines.append(f"  {species:<10}{shifts}")

    lines += ["", "Observed behaviour, recent turns (share of chances taken)",
              "  " + " " * 10 + "".join(f"{action:>9}" for action in ACTIONS)]
    for species in ANIMALS:
        cells = []
        for action in ACTIONS:
            share, _ = insights.rate(action, species)
            cells.append(f"{'-':>9}" if share is None else f"{share:>8.0%} ")
        lines.append(f"  {species:<10}" + "".join(cells))
    return "\n".join(lines) + "\n"


def _mean(samples: list[dict[str, float]], action: str) -> float:
    return sum(sample[action] for sample in samples) / len(samples)


def _join(ranked: list[tuple[float, str, int]]) -> str:
    return ", ".join(f"{label} {share:.0%} (n={size})" for share, label, size in ranked)
