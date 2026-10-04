"""What the Bitzkeys have learned: recent choice statistics and brain probing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from tools.Brain import ACTIONS, choice_odds, make_senses

if TYPE_CHECKING:
    from tools.Simulation import Bitzkey


ANIMALS = ("grazer", "omnivore", "predator")
TRAITS = ("metabolism", "aggression", "constitution", "fertility", "sight", "social", "learning")
DECAY = 0.99
MIN_SAMPLE = 5.0

# Situations used to probe brains: (title, senses, actions possible, species it can arise for).
SCENARIOS = (
    ("Fed and safe, a willing mate beside them",
     make_senses(hunger=0.2, health=1, mate_near=1, ready_to_breed=1, kin_near=1 / 3, age=0.2),
     ("breed", "rest", "explore"), ANIMALS),
    ("Hungry, food underfoot, a willing mate beside them",
     make_senses(hunger=0.6, health=1, food_here=1, mate_near=1, ready_to_breed=1, kin_near=1 / 3, age=0.2),
     ("eat", "breed", "rest", "explore"), ANIMALS),
    ("Hungry, food in sight",
     make_senses(hunger=0.6, health=1, food_seen=1, food_closeness=0.6, age=0.2),
     ("forage", "rest", "explore"), ANIMALS),
    ("Wounded, a hunter beside them",
     make_senses(hunger=0.3, health=0.4, threat_near=0.25, age=0.2),
     ("flee", "rest", "explore"), ("grazer", "omnivore")),
    ("Peckish, prey beside them",
     make_senses(hunger=0.5, health=1, prey_near=0.25, age=0.2),
     ("attack", "rest", "explore"), ("omnivore", "predator")),
    ("Nothing around",
     make_senses(hunger=0.3, health=1, age=0.2),
     ("rest", "explore"), ANIMALS),
)


def band(value: int) -> str:
    return "low" if value <= 4 else "mid" if value <= 10 else "high"


def group_label(species: str, trait: str | None = None, level: str | None = None) -> str:
    return f"{species}s" if trait is None else f"{level}-{trait} {species}s"


def _groups(species: str, traits: Mapping[str, int | str]) -> list[tuple[str, str | None, str | None]]:
    return [(species, None, None)] + [(species, trait, band(int(traits[trait]))) for trait in TRAITS]


class Insights:
    """Recent decisions, discounted each turn: how often each kind of animal chose an action
    when it was possible."""

    def __init__(self) -> None:
        self.offered: dict[tuple, float] = {}
        self.chosen: dict[tuple, float] = {}

    def record(self, species: str, traits: Mapping[str, int | str],
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
    for title, senses, actions, species_list in SCENARIOS:
        options = [ACTIONS.index(action) for action in actions]
        headline = actions[0]
        per_group: dict[tuple, list[dict[str, float]]] = {}
        for animal in thinkers:
            if animal.species not in species_list:
                continue
            odds = choice_odds(animal.brain.values(senses), options)
            named = {ACTIONS[option]: chance for option, chance in odds.items()}
            for group in _groups(animal.species, animal.traits):
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
