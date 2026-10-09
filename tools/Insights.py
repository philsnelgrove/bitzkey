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
COLUMN = 11  # characters to an action's column in the report: room for the longest name

# Situations used to probe brains: (title, situation senses, actions possible, species it can
# arise for). Each animal is probed with its own current temperament added to the situation.
SCENARIOS = (
    ("Fed and safe, a willing mate beside them",
     {"hunger": 0.2, "health": 1, "mate_near": 1, "mate_closeness": 1, "ready_to_breed": 1, "kin_near": 1 / 3,
      "age": 0.2},
     ("breed", "court", "rest", "go_east", "go_west"), ANIMALS),
    ("Fed and safe, a willing mate in sight",
     {"hunger": 0.2, "health": 1, "ready_to_breed": 1, "mate_closeness": 0.6, "age": 0.2},
     ("court", "rest", "go_east", "go_west"), ANIMALS),
    ("Hungry, grass underfoot, a willing mate beside them",
     {"hunger": 0.6, "health": 1, "grass_here": 1, "mate_near": 1, "mate_closeness": 1, "ready_to_breed": 1,
      "kin_near": 1 / 3, "age": 0.2},
     ("eat", "breed", "court", "rest", "go_east", "go_west"), GRAZERS),
    ("Hungry, grass in sight",
     {"hunger": 0.6, "health": 1, "grass_seen": 1, "grass_closeness": 0.6, "age": 0.2},
     ("forage", "rest", "go_east", "go_west"), GRAZERS),
    ("Hungry, a plant in sight",
     {"hunger": 0.6, "health": 1, "plant_closeness": 0.6, "age": 0.2},
     ("browse", "rest", "go_east", "go_west"), GRAZERS),
    ("Hungry, carrion in sight",
     {"hunger": 0.6, "health": 1, "carrion_closeness": 0.6, "age": 0.2},
     ("scavenge", "rest", "go_east", "go_west"), ("omnivore", "predator")),
    ("Hungry, a fresh kill in sight",
     {"hunger": 0.6, "health": 1, "kill_closeness": 0.6, "age": 0.2},
     ("feast", "rest", "go_east", "go_west"), ("omnivore", "predator")),
    ("Peckish, prey in sight",
     {"hunger": 0.5, "health": 1, "prey_closeness": 0.6, "age": 0.2},
     ("hunt", "rest", "go_east", "go_west"), ("omnivore", "predator")),
    ("A hunter in sight",
     {"hunger": 0.3, "health": 1, "threat_closeness": 0.6, "age": 0.2},
     ("flee", "rest", "go_east", "go_west"), ANIMALS),
    ("Wounded, a dangerous hunter beside them",
     {"hunger": 0.3, "health": 0.4, "threat_near": 0.25, "threat_danger": 0.4, "threat_closeness": 1,
      "age": 0.2},
     ("flee", "rest", "go_east", "go_west"), ANIMALS),
    ("Wounded, the attacker still beside them",
     {"hunger": 0.3, "health": 0.5, "threat_near": 0.25, "threat_danger": 0.3, "threat_closeness": 1,
      "assailant_near": 1, "counter_frailty": 0.3, "age": 0.2},
     ("defend", "flee", "rest", "go_east", "go_west"), ANIMALS),
    ("Crowded territory, a rival beside them",
     {"hunger": 0.3, "health": 1, "rivalry": 1, "prey_near": 0.25, "prey_frailty": 0.3, "prey_closeness": 1,
      "threat_near": 0.25, "threat_danger": 0.3, "threat_closeness": 1, "age": 0.2},
     ("attack", "flee", "rest", "go_east", "go_west"), ("omnivore", "predator")),
    ("Alone, its herd in range",
     {"hunger": 0.3, "health": 1, "herd_closeness": 0.4, "age": 0.2},
     ("herd", "rest", "go_east", "go_west"), ANIMALS),
    ("A herd-mate struck close by",
     {"hunger": 0.3, "health": 1, "alarm": 1, "herd_near": 1 / 3, "threat_closeness": 0.6,
      "counter_frailty": 0.3, "age": 0.2},
     ("defend", "flee", "herd", "rest", "go_east", "go_west"), ANIMALS),
    ("Peckish, frail prey beside them",
     {"hunger": 0.5, "health": 1, "prey_near": 0.25, "prey_frailty": 0.5, "prey_closeness": 1, "age": 0.2},
     ("attack", "hunt", "rest", "go_east", "go_west"), ("omnivore", "predator")),
    # The climate: off home each way, far off, and at home with the weather turning toward it or not,
    # so the report shows whether brains follow the warmth senses or only a habit of one direction.
    ("Too cold, warmer to the south",
     {"hunger": 0.3, "health": 1, "climate_discomfort": 0.5, "comfort_closeness": 0.6,
      "warmth": -0.5, "warmth_north": -1, "latitude": -0.3, "age": 0.2},
     ("go_north", "go_south", "walk_north", "walk_south", "go_east", "go_west", "rest"), ANIMALS),
    ("Too cold, warmer to the north",
     {"hunger": 0.3, "health": 1, "climate_discomfort": 0.5, "comfort_closeness": 0.6,
      "warmth": -0.5, "warmth_south": -1, "latitude": 0.3, "age": 0.2},
     ("go_north", "go_south", "walk_north", "walk_south", "go_east", "go_west", "rest"), ANIMALS),
    ("Too hot, cooler to the north",
     {"hunger": 0.3, "health": 1, "climate_discomfort": 0.5, "comfort_closeness": 0.6,
      "warmth": 0.5, "warmth_south": 0.5, "latitude": 0.1, "age": 0.2},
     ("go_north", "go_south", "walk_north", "walk_south", "go_east", "go_west", "rest"), ANIMALS),
    ("Too hot, cooler to the south",
     {"hunger": 0.3, "health": 1, "climate_discomfort": 0.5, "comfort_closeness": 0.6,
      "warmth": 0.5, "warmth_north": 0.5, "latitude": -0.1, "age": 0.2},
     ("go_north", "go_south", "walk_north", "walk_south", "go_east", "go_west", "rest"), ANIMALS),
    ("Far too cold, warmer to the south",
     {"hunger": 0.3, "health": 1, "climate_discomfort": 1, "comfort_closeness": 0.3,
      "warmth": -1, "warmth_north": -1, "warmth_south": -0.5, "latitude": -0.5, "age": 0.2},
     ("go_north", "go_south", "walk_north", "walk_south", "go_east", "go_west", "rest"), ANIMALS),
    ("At home, cold coming from the north",
     {"hunger": 0.3, "health": 1, "comfort_closeness": 1, "warmth_north": -0.5, "latitude": -0.3, "age": 0.2},
     ("go_north", "go_south", "walk_north", "walk_south", "go_east", "go_west", "rest"), ANIMALS),
    ("At home, cold coming from the south",
     {"hunger": 0.3, "health": 1, "comfort_closeness": 1, "warmth_south": -0.5, "latitude": 0.3, "age": 0.2},
     ("go_north", "go_south", "walk_north", "walk_south", "go_east", "go_west", "rest"), ANIMALS),
    ("At home, the same all around",
     {"hunger": 0.3, "health": 1, "comfort_closeness": 1, "latitude": -0.3, "age": 0.2},
     ("go_north", "go_south", "walk_north", "walk_south", "go_east", "go_west", "rest"), ANIMALS),
    ("Starving, food scarce, a plant beside them",
     {"hunger": 0.85, "health": 1, "plant_closeness": 1, "age": 0.2},
     ("uproot", "browse", "rest", "go_east", "go_west"), ("grazer",)),
    ("Nothing around",
     {"hunger": 0.3, "health": 1, "age": 0.2},
     ("rest", "go_east", "go_west"), ANIMALS),
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
            senses = make_senses(**situation, **{trait: animal.feeling(trait) for trait in MIND_TRAITS},
                                 **animal.genes)
            odds = choice_odds(animal.brain.values(senses), options)
            named = {ACTIONS[option]: chance for option, chance in odds.items()}
            for group in _groups(animal.species, animal.profile):
                per_group.setdefault(group, []).append(named)

        lines += ["", title, "  " + " " * 18 + "".join(f"{action:>{COLUMN}}" for action in actions)]
        for species in species_list:
            samples = per_group.get((species, None, None), [])
            if not samples:
                lines.append(f"  {species:<10} (none)")
                continue
            averages = "".join(f"{_mean(samples, action):>{COLUMN - 1}.0%} " for action in actions)
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
              "  " + " " * 10 + "".join(f"{action:>{COLUMN}}" for action in ACTIONS)]
    for species in ANIMALS:
        cells = []
        for action in ACTIONS:
            share, _ = insights.rate(action, species)
            cells.append(f"{'-':>{COLUMN}}" if share is None else f"{share:>{COLUMN - 1}.0%} ")
        lines.append(f"  {species:<10}" + "".join(cells))
    return "\n".join(lines) + "\n"


def _mean(samples: list[dict[str, float]], action: str) -> float:
    return sum(sample[action] for sample in samples) / len(samples)


def _join(ranked: list[tuple[float, str, int]]) -> str:
    return ", ".join(f"{label} {share:.0%} (n={size})" for share, label, size in ranked)


# The two reports a viewer can open: every death so far by cause and species, and one animal's brain probed
# with the standard situations. Each is laid out as lines of text in a fixed-width font.

SPECIES_ORDER = ("plant", "grazer", "omnivore", "predator")
SHORT_ACTIONS = {"eat": "eat", "forage": "forage", "browse": "browse", "uproot": "uproot", "scavenge": "scaven",
                 "feast": "feast", "rest": "rest", "court": "court", "breed": "breed", "hunt": "hunt",
                 "attack": "attack", "defend": "defend", "flee": "flee", "herd": "herd", "go_north": "go N",
                 "go_south": "go S", "go_east": "go E", "go_west": "go W", "walk_north": "walk N",
                 "walk_south": "walk S", "walk_east": "walk E", "walk_west": "walk W"}
CELL = 7  # characters to a column of figures
SITUATION = 42  # characters to the situations' column


def death_report_lines(death_causes: Mapping[str, Mapping[str, int]]) -> list[str]:
    """Every death so far: a row for each cause, the most common first, a column for each species, and the totals."""
    causes: dict[str, int] = {}
    for counts in death_causes.values():
        for cause, count in counts.items():
            causes[cause] = causes.get(cause, 0) + count
    lines = [f"{'CAUSE':<18}" + "".join(f"{species.upper():>10}" for species in SPECIES_ORDER) + f"{'ALL':>10}"]
    for cause, total in sorted(causes.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"{cause:<18}" + "".join(f"{death_causes.get(species, {}).get(cause, 0):>10}" for species in SPECIES_ORDER)
                     + f"{total:>10}")
    totals = [sum(death_causes.get(species, {}).values()) for species in SPECIES_ORDER]
    lines.append(f"{'ALL':<18}" + "".join(f"{total:>10}" for total in totals) + f"{sum(totals):>10}")
    return lines


def brain_matrix(animal: Bitzkey) -> list[tuple[str, dict[str, float]]]:
    """The animal's chance of choosing each action open to it, in each standard situation its species can meet,
    with its own temperament."""
    rows = []
    for title, situation, actions, species_list in SCENARIOS:
        if animal.species not in species_list:
            continue
        senses = make_senses(**situation, **{trait: animal.feeling(trait) for trait in MIND_TRAITS}, **animal.genes)
        odds = choice_odds(animal.brain.values(senses), [ACTIONS.index(action) for action in actions])
        rows.append((title, {ACTIONS[option]: chance for option, chance in odds.items()}))
    return rows


def brain_matrix_lines(animal: Bitzkey) -> list[str]:
    """The brain matrix as a table: a row for each situation, a column for each action any of them offers, the
    likeliest choice in each row marked; an action not open in a situation shows a dot."""
    rows = brain_matrix(animal)
    used = [action for action in ACTIONS if any(action in chances for _, chances in rows)]
    lines = [f"{'SITUATION':<{SITUATION}}" + "".join(f"{SHORT_ACTIONS[action]:>{CELL}}" for action in used)]
    for title, chances in rows:
        best = max(chances, key=chances.get)
        cells = "".join((f"{chances[action]:.0%}" + ("*" if action == best else " ")).rjust(CELL) if action in chances
                        else "· ".rjust(CELL) for action in used)
        lines.append(f"{title[:SITUATION - 1]:<{SITUATION}}" + cells)
    return lines
