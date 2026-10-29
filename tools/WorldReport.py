"""A whole world's report: everything the probe tells of a world, gathered in one go, and read for what it says of
it. How many of each species live and how that has gone, their condition, what kills them, their brains, their herds
and where those stand against the climate, the climate and grass, and the rules; what stands out pulled together as
findings at the top. As the C++ version's Bitzkey Reports.

A world is read from a save, loaded here, or from a probe port (the C++ app's /probe or the daemon's /api/probe).
A save is read into the same shapes the probe answers in, so both report alike.

A section's lines are text, some of them marked: "# " a heading, "! " a finding, "~" a table's heading row; a table's
cells are parted by tabs."""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from tools import Rules
from tools.Brain import ACTIONS
from tools.Genome import SPECIES, TERRAINS, decode_body
from tools.Insights import ANIMALS, brain_report, death_report_lines
from tools.Simulation import SEASONS, Simulation, season, year, zone_temperature
from tools.WorldFile import load_world

TEMPERATURE_NAMES = ("hot", "medium", "cold", "very cold")
HISTORY_POINTS = 400
DEATHS_READ = 5000
CLIMATE_BANDS = 12
HERDS_LISTED = 12
HERD_REACH = 2
# What stands out: a species this far below its peak, a cause this large a share of a species' deaths, a brain value
# running away in this share of a species, neurons jammed in this share, and hunger or health this far gone.
CRASH = 0.5
CRASH_PEAK = 20  # a species that never reached this many isn't said to crash
DOMINANT_CAUSE = 0.5
RUNAWAY_VALUE = 50
JAMMED = 0.99
RUNAWAY_SHARE = 0.25
JAMMED_SHARE = 0.2
HUNGRY = 0.75  # of a full hunger, or stamina, of 100
FULL = 100
FRAIL = 0.5
OFF_HOME = 0.5  # a herd this far from its home temperature is away from home


@dataclass
class World:
    """What the probe answered, route by route; a route that failed (an older app without it, say) is left None."""
    name: str
    state: dict
    settings: dict | None = None
    rules: dict | None = None
    history: dict | None = None
    deaths: dict | None = None
    insights: dict | None = None
    climate: dict | None = None
    animals: dict | None = None
    minds: dict | None = None
    herds: dict[str, dict | None] = field(default_factory=dict)  # by species
    brain_report: str = ""
    death_report: str = ""
    missing: list[str] = field(default_factory=list)  # the routes that failed, and why
    live: bool = False  # a running world, worth looking at again


# ---- reading a save ----

def _brief(agent) -> dict:
    return {"id": agent.identifier, "species": agent.species, "terrain": agent.terrain, "nocturnal": agent.nocturnal,
            "wet": agent.wet, "x": agent.x, "y": agent.y, "age": agent.age,
            "health": max(0.0, agent.health) / agent.max_health if agent.max_health > 0 else 0.0,
            "hunger": agent.hunger, "stamina": agent.stamina, "action": agent.last_action}


def _across(start: int, to: int, width: int) -> int:
    """Across the map, east and west wrapping round: the nearer way from one column to another."""
    dx = (to - start) % width
    return dx - width if dx > width / 2 else dx


def _mean_column(columns: list[int], width: int) -> float:
    """The mean of columns on a map that wraps east to west."""
    sine = sum(math.sin(2 * math.pi * x / width) for x in columns)
    cosine = sum(math.cos(2 * math.pi * x / width) for x in columns)
    angle = math.atan2(sine, cosine) % (2 * math.pi)
    return angle * width / (2 * math.pi)


def _herds(world: Simulation, species: str) -> dict:
    """The herds of a species as they stand: herd-mates within HERD_REACH steps of one another, and theirs."""
    animals = [agent for agent in world.living if agent.species == species]
    index = {agent.identifier: at for at, agent in enumerate(animals)}
    root = list(range(len(animals)))

    def find(at: int) -> int:
        while root[at] != at:
            root[at] = root[root[at]]
            at = root[at]
        return at

    for agent in animals:
        for other in world._nearby(agent, min(HERD_REACH, agent.sight)):
            found = index.get(other.identifier)
            if found is not None and (world._herd_mate(agent, other) or world._herd_mate(other, agent)):
                root[find(index[agent.identifier])] = find(found)
    groups: dict[int, list] = {}
    for at, agent in enumerate(animals):
        groups.setdefault(find(at), []).append(agent)
    listed, alone = [], 0
    for group in groups.values():
        if len(group) < 2:
            alone += 1
            continue
        x = _mean_column([agent.x for agent in group], world.width)
        y = sum(agent.y for agent in group) / len(group)
        spread = sum(abs(_across(round(x), agent.x, world.width)) + abs(agent.y - y) for agent in group) / len(group)
        listed.append({"size": len(group), "x": x, "y": y, "spread": spread,
                       "home": sum(agent.climate[0] for agent in group) / len(group), "drift": None})
    listed.sort(key=lambda herd: -herd["size"])
    for number, herd in enumerate(listed, 1):
        herd["id"] = number
    return {"turn": world.turn, "species": species, "herds": listed, "alone": alone}


def _minds(world: Simulation) -> dict:
    """Each species' brains at a glance: the farthest value each gives an action for what it last sensed, and its
    neurons jammed near either end of tanh."""
    result = {}
    for species in ANIMALS:
        reaches, thinking, runaway, neurons, jammed, depth, width = [], 0, 0, 0, 0, 0, 0
        for agent in world.living:
            if agent.species != species or agent.brain is None:
                continue
            thinking += 1
            depth += agent.brain.depth
            width += agent.brain.width
            if agent.last_senses is None:
                continue
            reach = max(abs(value) for value in agent.brain.values(agent.last_senses))
            reaches.append(reach)
            runaway += reach > RUNAWAY_VALUE
            for layer in agent.brain._activations(agent.last_senses)[1:]:
                neurons += len(layer)
                jammed += int((abs(layer) > JAMMED).sum())
        reaches.sort()

        def at(share: float) -> float:
            return reaches[int(share * (len(reaches) - 1))] if reaches else 0.0

        count = max(1, thinking)
        result[species] = {"thinking": thinking, "valued": len(reaches), "depth": depth / count, "width": width / count,
                           "value_median": at(0.5), "value_p90": at(0.9), "value_max": at(1.0), "runaway": runaway,
                           "neurons": neurons, "jammed": jammed}
    return {"turn": world.turn, "runaway_value": RUNAWAY_VALUE, "jammed_at": JAMMED, "minds": result}


def _history(world: Simulation) -> dict:
    snapshots = world.census.snapshots
    stride = max(1, -(-len(snapshots) // HISTORY_POINTS))
    series = []
    for turn, identifiers in snapshots[::stride]:
        counts = Counter()
        for identifier in identifiers:
            agent = world.find_agent(identifier)
            if agent is not None:
                counts[agent.species] += 1
            elif identifier in world.lineage:
                counts[decode_body(world.lineage[identifier])["species"]] += 1
        series.append({"turn": turn, "alive": {species: counts[species] for species in SPECIES}})
    return {"turn": world.turn, "points": len(series), "history": series}


def read_simulation(world: Simulation, name: str) -> World:
    """A world as the probe would show it."""
    living = world.living
    alive, infants, sheltered, terrains = Counter(), Counter(), Counter(), {species: Counter() for species in SPECIES}
    for agent in living:
        alive[agent.species] += 1
        terrains[agent.species][agent.terrain] += 1
        if agent.infant:
            infants[agent.species] += 1
            sheltered[agent.species] += world.sheltered(agent)
    state = {"turn": world.turn, "year": year(world.turn), "season": SEASONS[season(world.turn)], "year_length": world.year_length,
             "width": world.width, "height": world.height, "alive": {species: alive[species] for species in SPECIES},
             "by_terrain": {species: {terrain: terrains[species][terrain] for terrain in TERRAINS} for species in SPECIES},
             "infants": {species: {"infants": infants[species], "sheltered": sheltered[species]} for species in SPECIES}}
    report = World(name=name, state=state)
    report.settings = {"turn": world.turn, "width": world.width, "height": world.height, "year_length": world.year_length,
                       "population_cap": world.population_cap, "mind_mutation": world.mind_mutation,
                       "body_mutation": world.body_mutation, "ended": world.ended, "deaths": world.deaths,
                       "next_identifier": world.next_identifier, "grass_regrowth": world.grass_regrowth,
                       "ocean": world.ocean, "sea_squares": sum(world.sea_map)}
    report.rules = {"rules": {rule.key: {"value": Rules.get(rule.key), "group": rule.group, "label": rule.label}
                              for rule in Rules.RULES}}
    report.history = _history(world)
    listed = []
    for death in reversed(world.deaths_so_far()):
        if len(listed) >= DEATHS_READ:
            break
        entry = {"turn": death.turn, "species": death.species, "cause": death.cause}
        if death.killer is not None:
            entry.update(killer=death.killer, killer_species=death.killer_species, activity=death.activity)
        listed.append(entry)
    report.deaths = {"turn": world.turn, "deaths": listed,
                     "since_the_start": {species: dict(world.death_causes.get(species, {})) for species in SPECIES}}
    insights = {}
    for species in ANIMALS:
        actions = {}
        for action in ACTIONS:
            share, offered = world.insights.rate(action, species)
            if offered > 0:
                actions[action] = {"offered": offered, "taken": share}
        insights[species] = actions
    report.insights = {"turn": world.turn, "insights": insights}
    rows = []
    for y in range(world.height):
        rows.append({"y": y, "temperature": world.temperature(y), "zone": zone_temperature(y, world.height),
                     "growth": world._growth[y] if y < len(world._growth) else 0.0,
                     "grass": sum(world.grass.get((x, y), 0) for x in range(world.width)),
                     "sea": sum(world.sea(x, y) for x in range(world.width))})
    report.climate = {"turn": world.turn, "season": state["season"], "rows": rows}
    animals = [_brief(agent) for agent in living if agent.species != "plant"]
    report.animals = {"turn": world.turn, "matched": len(animals), "animals": animals}
    report.minds = _minds(world)
    report.herds = {species: _herds(world, species) for species in ANIMALS}
    report.brain_report = brain_report(world.turn, living, world.insights)
    report.death_report = "\n".join(death_report_lines(world.death_tally))
    return report


def read_save(path: str | Path) -> World:
    """A saved world, loaded and read. Raises what loading raises if it can't be read."""
    return read_simulation(load_world(path), Path(path).name)


# ---- reading a probe ----

class Probe:
    """A probe at "host:port", "port" alone (on this computer) or a whole "http://host:port" address; its routes are
    found under /probe or /api/probe, whichever answers. Raises RuntimeError if neither does."""

    FORM = re.compile(r"^\s*(?:https?://)?(?:([^:/\s]+):)?(\d{1,5})/?\s*$")

    def __init__(self, address: str) -> None:
        match = self.FORM.match(address)
        if not match:
            raise RuntimeError("give the probe as host:port, or a port alone")
        self.host, self.port = match.group(1) or "127.0.0.1", int(match.group(2))
        self.name = f"{self.host}:{self.port}"
        for prefix in ("/probe", "/api/probe"):
            self.prefix = prefix
            try:
                self.get("state")
                return
            except RuntimeError:
                pass
        raise RuntimeError(f"no probe answers at {self.name}")

    def text(self, route: str) -> str:
        url = f"http://{self.host}:{self.port}{self.prefix}/{route}"
        try:
            with urllib.request.urlopen(url, timeout=120) as answer:
                return answer.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")
            try:
                body = json.loads(body).get("error", body)
            except ValueError:
                pass
            raise RuntimeError(f"{route}: {body}") from None
        except (urllib.error.URLError, OSError) as error:
            raise RuntimeError(f"no answer from {self.name} for {route}: {error}") from None

    def get(self, route: str):
        try:
            return json.loads(self.text(route))
        except ValueError:
            raise RuntimeError(f"{route}: not JSON") from None

    def read(self) -> World:
        """Every route a report reads."""
        report = World(name=self.name, state=self.get("state"), live=True)

        def read(route: str):
            try:
                return self.get(route)
            except RuntimeError as error:
                report.missing.append(str(error))
                return None

        report.settings = read("settings")
        report.rules = read("rules")
        report.history = read(f"history?points={HISTORY_POINTS}")
        report.deaths = read(f"deaths?limit={DEATHS_READ}")
        report.insights = read("insights")
        report.climate = read("climate")
        report.animals = read("animals?limit=1000000")
        report.minds = read("minds")
        report.herds = {species: read(f"herds?species={species}") for species in ANIMALS}
        for attribute, route in (("brain_report", "reports/brain"), ("death_report", "reports/deaths")):
            try:
                setattr(report, attribute, self.text(route))
            except RuntimeError as error:
                report.missing.append(str(error))
        return report


# ---- the sections ----

def _percent(share: float) -> str:
    return f"{100 * share:.0f}%"


def _number(value: float, places: int = 1) -> str:
    return f"{value:.{places}f}"


def _capital(word: str) -> str:
    return word[:1].upper() + word[1:]


def _row(*cells) -> str:
    return "\t".join(str(cell) for cell in cells)


def _temperature_at(climate: dict | None, y: float) -> float:
    """A row of the map's temperature now, 0 hot to 3 very cold, as a herd's home is measured."""
    if not climate or not climate.get("rows"):
        return 0.0
    rows = climate["rows"]
    return rows[min(max(round(y), 0), len(rows) - 1)]["temperature"]


def _temperature_name(temperature: float) -> str:
    return TEMPERATURE_NAMES[min(max(round(temperature), 0), 3)]


def _ranked(counts: dict[str, int], total: int, shown: int) -> str:
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:shown]
    return ", ".join(f"{name} {_percent(count / total)}" for name, count in ranked)


def _population(world: World, findings: list[str]) -> tuple[str, list[str]]:
    lines = []
    alive = world.state["alive"]
    series: dict[str, list[tuple[int, int]]] = {species: [] for species in SPECIES}
    for point in (world.history or {}).get("history", []):
        for species in SPECIES:
            series[species].append((point["turn"], point["alive"].get(species, 0)))
    lines += ["# Numbers now and over the run", _row("~SPECIES", "ALIVE", "PEAK", "PEAK TURN", "LOW SINCE", "LAST QUARTER", "SWINGS")]
    for species in SPECIES:
        now = alive.get(species, 0)
        points = series[species]
        if not points:
            lines.append(_row(_capital(species), now, "", "", "", "", ""))
            continue
        peak = max(points, key=lambda point: point[1])
        if now > peak[1]:
            peak = (world.state["turn"], now)  # the history's last point may lag the turn
        low = min([now] + [count for turn, count in points if turn >= peak[0]])
        quarter = points[len(points) * 3 // 4][1]
        change = (now - quarter) / quarter if quarter > 0 else 0.0
        # Swings: how often the numbers cross their mean, over the run, two crossings to a rise and fall.
        mean = sum(count for _, count in points) / len(points)
        crossings = sum((points[at - 1][1] < mean) != (points[at][1] < mean) for at in range(1, len(points)))
        lines.append(_row(_capital(species), now, peak[1], peak[0], low,
                          ("+" if change >= 0 else "") + _percent(change) if quarter > 0 else "", crossings // 2))
        if species == "plant":
            continue
        if now == 0 and peak[1] > 0:
            findings.append(f"{_capital(species)}s have died out: their peak was {peak[1]} at turn {peak[0]}.")
        elif peak[1] >= CRASH_PEAK and now < (1 - CRASH) * peak[1]:
            findings.append(f"{_capital(species)}s are down to {now}, {_percent(1 - now / peak[1])} below their peak of "
                            f"{peak[1]} at turn {peak[0]}.")
        elif quarter >= CRASH_PEAK and change <= -CRASH:
            findings.append(f"{_capital(species)}s are falling fast: {_percent(change)} over the last quarter of the run.")
    prey, predators = alive.get("grazer", 0) + alive.get("omnivore", 0), alive.get("predator", 0)
    lines.append("")
    if predators:
        lines.append(f"Prey to each predator: {prey / predators:.1f} ({prey} grazers and omnivores to {predators} predators).")
    if "infants" in world.state:
        lines += ["", "# The young", _row("~SPECIES", "INFANTS", "SHARE", "BESIDE A PARENT")]
        for species in ANIMALS:
            young = world.state["infants"][species]
            infants, sheltered, count = young["infants"], young["sheltered"], alive.get(species, 0)
            lines.append(_row(_capital(species), infants, _percent(infants / count) if count else "",
                              _percent(sheltered / infants) if infants else ""))
    if "by_terrain" in world.state:
        terrains = list(world.state["by_terrain"]["grazer"])
        lines += ["", "# Where they live", _row("~SPECIES", *(terrain.upper() for terrain in terrains))]
        for species in SPECIES:
            lines.append(_row(_capital(species), *(world.state["by_terrain"][species].get(terrain, 0) for terrain in terrains)))
    return "Population", lines


def _condition(world: World, findings: list[str]) -> tuple[str, list[str]]:
    if not world.animals:
        return "Condition", ["The probe gave no Bitzkeys to read."]
    tallies: dict[str, dict] = {}
    for animal in world.animals["animals"]:
        tally = tallies.setdefault(animal["species"], {"count": 0, "age": 0.0, "health": 0.0, "hunger": 0.0, "stamina": 0.0,
                                                       "nocturnal": 0, "wet": 0, "actions": Counter()})
        tally["count"] += 1
        for key in ("age", "health", "hunger", "stamina"):
            tally[key] += animal.get(key, 0)
        tally["nocturnal"] += bool(animal.get("nocturnal"))
        tally["wet"] += bool(animal.get("wet"))
        tally["actions"][animal.get("action", "")] += 1
    lines = ["# How they are", _row("~SPECIES", "READ", "MEAN AGE", "HEALTH", "HUNGER", "STAMINA", "NOCTURNAL", "WET")]
    for species in ANIMALS:
        tally = tallies.get(species)
        if not tally:
            lines.append(_row(_capital(species), 0, "", "", "", "", "", ""))
            continue
        n = tally["count"]
        lines.append(_row(_capital(species), n, _number(tally["age"] / n, 0), _percent(tally["health"] / n),
                          _percent(tally["hunger"] / n / FULL), _percent(tally["stamina"] / n / FULL),
                          _percent(tally["nocturnal"] / n), _percent(tally["wet"] / n)))
        if n >= 10 and tally["hunger"] / n / FULL > HUNGRY:
            findings.append(f"{_capital(species)}s are hungry: their hunger is {_percent(tally['hunger'] / n / FULL)} on average.")
        if n >= 10 and tally["health"] / n < FRAIL:
            findings.append(f"{_capital(species)}s are in poor health: {_percent(tally['health'] / n)} on average.")
    if world.animals.get("matched", 0) > len(world.animals["animals"]):
        lines.append(f"({len(world.animals['animals'])} of {world.animals['matched']} read)")
    lines += ["", "# What they did last turn"]
    for species in ANIMALS:
        if species in tallies:
            lines.append(f"{_capital(species)}s: {_ranked(tallies[species]['actions'], tallies[species]['count'], 8)}")
    return "Condition", lines


def _deaths(world: World, findings: list[str]) -> tuple[str, list[str]]:
    if not world.deaths:
        return "Deaths", ["The probe gave no deaths to read."]
    totals = world.deaths["since_the_start"]
    all_of = {species: sum(totals.get(species, {}).values()) for species in ANIMALS}
    causes = Counter()
    for species in ANIMALS:
        causes.update(totals.get(species, {}))
    lines = ["# Every death since the start, by cause", _row("~CAUSE", "GRAZER", "", "OMNIVORE", "", "PREDATOR", "")]
    for cause, _ in sorted(causes.items(), key=lambda item: (-item[1], item[0])):
        cells = [cause]
        for species in ANIMALS:
            count = totals.get(species, {}).get(cause, 0)
            cells += [count, _percent(count / all_of[species]) if all_of[species] else ""]
        lines.append(_row(*cells))
    lines.append(_row("all", all_of["grazer"], "", all_of["omnivore"], "", all_of["predator"], ""))
    for species in ANIMALS:
        if all_of[species] < 20:
            continue
        for cause, count in totals.get(species, {}).items():
            if count / all_of[species] > DOMINANT_CAUSE:
                findings.append(f"{_percent(count / all_of[species])} of {species} deaths have been by {cause}.")

    listed = world.deaths["deaths"]  # the latest first
    if not listed:
        return "Deaths", lines
    latest, earliest = max(death["turn"] for death in listed), min(death["turn"] for death in listed)
    span = max(1, latest - earliest + 1)
    recent: dict[str, Counter] = {}
    whom: dict[str, Counter] = {}
    doing: dict[str, Counter] = {}
    for death in listed:
        recent.setdefault(death["species"], Counter())[death["cause"]] += 1
        if death.get("killer_species"):
            whom.setdefault(death["killer_species"], Counter())[death["species"]] += 1
            doing.setdefault(death["killer_species"], Counter())[death.get("activity") or ""] += 1
    lines += ["", f"# Lately: the last {len(listed)} deaths, turns {earliest} to {latest}",
              _row("~SPECIES", "DIED", "A TURN", "A TURN PER 1,000", "CHIEF CAUSES")]
    for species in ANIMALS:
        counts = recent.get(species, Counter())
        died, living = sum(counts.values()), world.state["alive"].get(species, 0)
        rate = died / span
        lines.append(_row(_capital(species), died, _number(rate, 2), _number(1000 * rate / living, 2) if living else "",
                          _ranked(counts, died, 3) if died else ""))
    if whom:
        lines += ["", "# Who killed whom lately (killers in rows)", _row("~KILLER", "PLANTS", "GRAZERS", "OMNIVORES", "PREDATORS", "WHILE")]
        for killer in ANIMALS:
            if killer not in whom:
                continue
            killed = whom[killer]
            lines.append(_row(_capital(killer) + "s", killed["plant"], killed["grazer"], killed["omnivore"], killed["predator"],
                              _ranked(doing[killer], sum(doing[killer].values()), 3)))
        cannibal = whom.get("predator", Counter())
        if cannibal["predator"] > 0 and cannibal["predator"] * 4 > cannibal["grazer"] + cannibal["omnivore"]:
            findings.append(f"Predators are turning on their own kind: {cannibal['predator']} of their recent kills were predators.")
    return "Deaths", lines


def _brains(world: World, findings: list[str]) -> tuple[str, list[str]]:
    lines = []
    if world.minds:
        runaway, jammed_at = world.minds.get("runaway_value", RUNAWAY_VALUE), world.minds.get("jammed_at", JAMMED)
        lines += ["# Their brains at a glance",
                  f"The farthest value from nothing each brain gives an action, for what it last sensed; neurons beyond {jammed_at:.2f} are jammed.",
                  _row("~SPECIES", "THINKING", "DEPTH", "WIDTH", "MEDIAN VALUE", "90TH", "HIGHEST", f"BEYOND {runaway:.0f}", "JAMMED")]
        for species in ANIMALS:
            mind = world.minds["minds"][species]
            valued, neurons = mind["valued"], mind["neurons"]
            runaway_share = mind["runaway"] / valued if valued else 0.0
            jammed_share = mind["jammed"] / neurons if neurons else 0.0
            lines.append(_row(_capital(species), mind["thinking"], _number(mind["depth"]), _number(mind["width"], 0),
                              _number(mind["value_median"]), _number(mind["value_p90"]), _number(mind["value_max"]),
                              _percent(runaway_share) if valued else "", _percent(jammed_share) if neurons else ""))
            if valued >= 10 and runaway_share > RUNAWAY_SHARE:
                findings.append(f"{_percent(runaway_share)} of {species}s value an action beyond {runaway:.0f}: their brains are running away.")
            if neurons >= 50 and jammed_share > JAMMED_SHARE:
                findings.append(f"{_percent(jammed_share)} of {species}s' neurons are jammed at either end.")
        lines.append("")
    else:
        lines += ["(This probe has no minds route: brain values and jammed neurons need a newer Bitzkey.)", ""]
    if world.insights:
        insights = world.insights["insights"]
        actions = sorted({action for species in ANIMALS for action in insights.get(species, {})})
        lines += ["# How often each action is taken when it is open, lately", _row("~ACTION", "GRAZER", "OMNIVORE", "PREDATOR")]
        for action in actions:
            cells = [action]
            for species in ANIMALS:
                rate = insights.get(species, {}).get(action)
                cells.append(_percent(rate["taken"]) if rate and rate.get("taken") is not None else "")
            lines.append(_row(*cells))
    return "Brains", lines


def _herd_section(world: World, findings: list[str]) -> tuple[str, list[str]]:
    lines = ["# The herds as they stand",
             "A herd is herd-mates within two steps of one another, and theirs. Away from home: its middle's row is half a band or more",
             "from the herd's mean home temperature. Drift is since the last look, so it shows from the second look of a running world.",
             _row("~SPECIES", "HERDS", "ALONE", "IN HERDS", "LARGEST", "MEAN SIZE", "MEAN SPREAD", "AWAY FROM HOME", "MOVED", "TOWARD HOME")]
    any_drift = False
    listed = []
    for species in ANIMALS:
        found = world.herds.get(species)
        if not found:
            continue
        herds, alone = found["herds"], found.get("alone", 0)
        members = largest = away_members = moved = toward = 0
        spread = 0.0
        for herd in herds:
            size = herd["size"]
            members += size
            largest = max(largest, size)
            spread += herd["spread"] * size
            off = _temperature_at(world.climate, herd["y"]) - herd["home"]
            away = abs(off) >= OFF_HOME
            away_members += size if away else 0
            if herd.get("drift"):
                any_drift = True
                dy = herd["drift"][1]
                if away and abs(dy) >= 0.5:
                    moved += 1
                    toward += abs(_temperature_at(world.climate, herd["y"] + dy) - herd["home"]) < abs(off)
            listed.append((species, herd))
        count = len(herds)
        lines.append(_row(_capital(species), count, alone, _percent(members / (members + alone)) if members + alone else "", largest,
                          _number(members / count) if count else "", _number(spread / members) if members else "",
                          _percent(away_members / members) if members else "", moved if any_drift else "",
                          _percent(toward / moved) if moved else ""))
        if members >= 20 and away_members / members > 0.5:
            findings.append(f"Most {species}s in herds ({_percent(away_members / members)}) are away from their home temperature.")
        if moved >= 5:
            findings.append(f"Of the {species} herds away from home that moved, {_percent(toward / moved)} moved toward it.")
    if not any_drift:
        lines.append("(No drift yet: it shows on a running world's second look. Press REFRESH, or turn on LIVE.)")
    lines += ["", "# The largest herds", _row("~SPECIES", "HERD", "SIZE", "COLUMN", "ROW", "SPREAD", "HOME", "HERE", "DRIFT", "HISTORY")]
    listed.sort(key=lambda entry: -entry[1]["size"])
    for species, herd in listed[:HERDS_LISTED]:
        drift = f"{herd['drift'][0]:+d}, {herd['drift'][1]:+.1f}" if herd.get("drift") else ""
        past = "; ".join(part for part in (f"split from {herd['split_from']}" if herd.get("split_from") else "",
                                           f"joined by {herd['merged_from']}" if herd.get("merged_from") else "") if part)
        lines.append(_row(_capital(species), f"#{herd['id']}", herd["size"], _number(herd["x"], 0), _number(herd["y"], 0),
                          _number(herd["spread"]), _number(herd["home"], 2), _number(_temperature_at(world.climate, herd["y"]), 2),
                          drift, past))
    return "Herds & migration", lines


def _climate(world: World) -> tuple[str, list[str]]:
    if not world.climate or not world.climate.get("rows"):
        return "Climate & grass", ["The probe gave no climate to read."]
    rows = world.climate["rows"]
    height = len(rows)
    bands = min(CLIMATE_BANDS, height)
    living = [[0, 0, 0] for _ in range(bands)]
    for animal in (world.animals or {}).get("animals", []):
        if animal["species"] in ANIMALS:
            living[min(max(animal["y"] * bands // height, 0), bands - 1)][ANIMALS.index(animal["species"])] += 1
    lines = [f"# The map from north to south, in {bands} bands; the season: {world.climate.get('season', '')}",
             "Temperatures run 0 hot to 3 very cold; usual is the zone's temperature with no season.",
             _row("~ROWS", "NOW", "USUAL", "", "GROWTH", "GRASS", "SEA", "GRAZERS", "OMNIVORES", "PREDATORS")]
    for band in range(bands):
        start, end = band * height // bands, (band + 1) * height // bands
        part = rows[start:end]
        n = max(1, len(part))
        now = sum(line["temperature"] for line in part) / n
        lines.append(_row(f"{start}-{end - 1}", _number(now, 2), _number(sum(line["zone"] for line in part) / n, 2), _temperature_name(now),
                          _number(sum(line["growth"] for line in part) / n, 2), sum(line["grass"] for line in part),
                          sum(line["sea"] for line in part), *living[band]))
    return "Climate & grass", lines


def _rules(world: World) -> tuple[str, list[str]]:
    lines = []
    if world.settings:
        lines.append("# The world's settings")
        lines += [_row(key, value if isinstance(value, str) else json.dumps(value)) for key, value in sorted(world.settings.items())]
        lines.append("")
    if world.rules:
        grouped: dict[str, list] = {}
        for key, rule in world.rules["rules"].items():
            grouped.setdefault(rule.get("group", ""), []).append((rule.get("label", key), rule["value"]))
        for group in sorted(grouped):
            lines.append("# " + group)
            for label, value in grouped[group]:
                lines.append(_row(label, f"{value:.0f}" if float(value).is_integer() else f"{value:g}"))
            lines.append("")
    return "Rules & settings", lines


def _from_text(title: str, text: str) -> tuple[str, list[str]]:
    lines = text.splitlines()
    return title, lines or [f"The probe gave no {title}."]


def analyse(world: World) -> list[tuple[str, list[str]]]:
    """The report's sections, each (title, lines), the summary first."""
    findings: list[str] = []
    sections = [_population(world, findings), _condition(world, findings), _deaths(world, findings), _brains(world, findings),
                _herd_section(world, findings), _climate(world), _rules(world),
                _from_text("Brain report", world.brain_report), _from_text("Death report", world.death_report)]
    state, alive = world.state, world.state["alive"]
    lines = ["# " + world.name,
             f"Turn {state['turn']}, year {state['year']}, season {state['season']}. The map is {state['width']} by {state['height']}."]
    if world.settings and world.settings.get("ended"):
        lines.append("! This world has ended.")
    lines += ["", _row("~", "PLANTS", "GRAZERS", "OMNIVORES", "PREDATORS"),
              _row("Alive", *(alive.get(species, 0) for species in SPECIES)), "", "# What stands out"]
    if not findings:
        lines.append("Nothing stands out: every species is near its usual numbers, and nothing is going wrong.")
    lines += ["! " + finding for finding in findings]
    if world.missing:
        lines += ["", "# Not read", *world.missing]
    return [("Summary", lines)] + sections


def cells_of(line: str) -> list[str]:
    return line.split("\t")


def as_text(sections: list[tuple[str, list[str]]]) -> str:
    """The report as plain text, its tables lined up in spaces: for the clipboard and for a file."""
    out = []
    for title, lines in sections:
        out += [f"== {title} ==", ""]
        at = 0
        while at < len(lines):
            if "\t" not in lines[at]:
                line = lines[at]
                out.append(line[2:] if line.startswith("# ") else line)
                at += 1
                continue
            end = at
            while end < len(lines) and "\t" in lines[end]:
                end += 1
            table = [[cell.lstrip("~") if index == 0 else cell for index, cell in enumerate(cells_of(line))] for line in lines[at:end]]
            widths = [max(len(row[c]) for row in table if c < len(row)) for c in range(max(len(row) for row in table))]
            for row in table:
                out.append("".join(cell.ljust(widths[c]) if c == 0 else "  " + cell.rjust(widths[c])
                                   for c, cell in enumerate(row)).rstrip())
            at = end
        out.append("")
    return "\n".join(out) + "\n"
