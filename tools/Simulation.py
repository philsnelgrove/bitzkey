"""Turn-based Bitzkey world and its deterministic history."""

from __future__ import annotations

import bisect
import math
import random
import shutil
import tempfile
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cache, cached_property, lru_cache
from itertools import accumulate
from pathlib import Path

from tools.Brain import ACTIONS, SENSES, Brain, choice_odds, depth_for, gene_senses, make_senses
from tools.Genome import (GENOME, MIND_TRAITS, SPECIES, TRAIT_MAX, body_word, color_from_word, cross, dot_difference, inherit_mind, ring_difference,
                          BODY_MUTATION_CHANCE, COLD_TOLERANCE, HEAT_TOLERANCE, MIND_MUTATION_CHANCE,
                          TEMPERATURE_SHIFT, TEMPERATURES, decode_body, decode_climate, decode_mind, make_dna, mind_word, share)
from tools.Census import Census
from tools.Exchange import Imported
from tools.Insights import Insights


GRID_SIZE = 100
INITIAL_POPULATION = 90
GRASS_DENSITY = 50      # percent of cells that start with grass
GRASS_REGROWTH = 0.016  # share of the starting grass that grows back each turn
MAX_POPULATION = 20000  # the most Bitzkeys spawned at the start; a world's population cap is a setting
CROWDING_RADIUS = 4    # animals within this many cells count as crowding a would-be parent
CROWDING_LIMIT = 10    # litters shrink as crowding nears this; at it, an animal doesn't breed; 0: no limit
SPAWN_GROUP = (3, 5)  # Bitzkeys spawn in groups of this many sharing one genome
MAX_IMPORTED = 120  # imported Bitzkeys, each spawning a group of its own
MAX_GRID = 1000  # the widest and tallest a world can be, in cells
SPAWN_SPREAD = 2      # a group's members appear within this many cells of each other's centre
USE_GPU = 0  # the C++ version's: 1 thinks on the GPUs; the Python version always thinks on the CPU
SPAWN_AT_HOME = 1     # 1: a group appears in rows whose usual temperature is its home (the nearest, if none); 0: anywhere
# The history is kept in memory: the world at each of the latest HISTORY_TURNS turns, whole, so the map
# can be rewound to any of them at once, with nothing to replay. When a crowded world's brains make them
# large, fewer are kept, so they fit HISTORY_MEMORY gigabytes; the newest turn is always kept. Older turns live on
# only in the colour graph.
HISTORY_TURNS = 100
HISTORY_ALL = 5000  # at this many history turns or more, every turn is kept, so far as the memory allows
HISTORY_MEMORY = 1.0  # GB, near enough
AGENT_BYTES = 1000  # what a Bitzkey takes in the history, besides its brain, near enough
# Earlier versions kept the history on disk, in folders among the temporary files.
HISTORY_PREFIX = "bitzkey-history-"  # their names start so
STALE_HISTORY_HOURS = 24  # one left this long by a run that never closed it is cleared away
CORPSE_FERTILITY = 2   # grass grows this many times as fast on and around a corpse
CORPSE_TURNS = 30
BREED_RESERVE = 0  # hunger a parent must still have spare after paying to breed
# The most a partner's ring and dot colours may differ from an animal's own before it shuns
# them, at no tolerance; full tolerance widens both to 1, accepting any partner of its species.
MATE_RING_DIFFERENCE = 0.05
MATE_DOT_DIFFERENCE = 0.20
LITTER_RANGE = (1, 6)  # young per birth, while there is room around the parent
LITTER_SIZE_WEIGHT = 0.5  # how much of litter size comes from small bodies; the rest from the litter trait
SMALL_LITTERS = 2.0  # the smallest bodies' litters are this many times the usual, falling evenly to the usual for the largest
# Turns an animal carries its young, from mating to birth: from the first for the smallest body to the
# second for the largest, evenly between, rounded half up.
GESTATION = (10, 50)
BIRTH_REACH = 2  # young are born on the free cells nearest their mother, up to this many cells away
# Years a Bitzkey, plant or animal, is expected to live: the smallest bodies, with the fastest
# metabolisms; middling ones; and the largest, with the slowest. Between them, lifespan rises steadily
# with the metabolism slowing.
LIFESPAN_YEARS = (1.0, 5.0, 10.0)
METABOLISM = (1.6, 0.6)  # metabolic rate of the smallest and the largest bodies
STOMACH = (0.5, 2.0)     # food a body holds, relative to the middle, from the smallest to the largest;
                         # hunger is how empty it is, so a meal fills a small stomach more
SIZE_IN_MIGHT = 0.5      # share of fighting strength and toughness that comes from body size
AGEING = 0.7  # from this share of its lifespan, an animal may die of old age

DEATH_PENALTY = 2.0
BREED_JOY = 1.0
FIGHT_JOY = 0.5
COMPANY_JOY = 0.1
CURIOSITY_JOY = 0.15   # pleasure of setting off in a direction, scaled by curiosity
# Moving costs food. A walk, a single cell, saves this share of the turn's burn, as resting does; going
# as far as its speed allows costs this share of a turn's burn more for each cell beyond the first.
WALK_REST = 0.5
GO_EFFORT = 0.25
FEAR_DREAD = 0.3
UNEASE = 0.1           # how much an intolerant animal minds other species close by
CLIMATE_UNEASE = 1.5   # discomfort two or more steps from home, half that one step; less for the hardy
EMPATHY_PAIN = 0.15    # how much an empathetic animal feels another's wound; twice that for a death
# Society: an animal sees what happens to its herd-mates (its pack, for a predator) within its sight. A
# herd-mate's kill is a success, its death (but of old age) a failure; the witness learns from it as
# if it had made the herd-mate's last choice itself, and feels it, by its empathy times its intellect.
SOCIAL_LEARNING = 1.0  # how fast it learns from what it sees, beside learning from its own choices
SOCIAL_MOOD = 0.3      # how much what it sees moves its mood
SOCIAL_MEMORY = 0.8    # the share of a seen success or failure it still senses a turn later
OTHER_KIND = 0.4       # how much of that pain reaches it when the victim is of another species
LEARNING_RATE = 0.045  # at the highest intellect, before its boost
# The clever learn faster. The learning rate rises with intellect, and on top of that by up to
# LEARNING_BOOST times again at the highest; the least intelligent get no boost. And each turn an
# animal re-learns from REPLAY_PER_INTELLECT of its most recent experiences for each point of its
# inborn intellect, so the least replay none.
LEARNING_BOOST = 1.0
REPLAY_PER_INTELLECT = 2
PREDATOR_LEARNING = 2.0  # a predator's brain learns this many times as fast, from its own choices and replays;
# an omnivore's gets half the bonus
# Food that takes a kill is harder to learn than grass: a choice to hunt, attack or feast is learnt from
# KILL_FOOD_LEARNING times as much, by any animal, and one to eat, forage, browse or uproot PLANT_FOOD_LEARNING.
KILL_FOOD_LEARNING = 5.0
PLANT_FOOD_LEARNING = 0.5
KILL_FOOD = frozenset(ACTIONS.index(action) for action in ("hunt", "attack", "feast"))
PLANT_FOOD = frozenset(ACTIONS.index(action) for action in ("eat", "forage", "browse", "uproot"))


def food_learning(choice: int) -> float:
    """How much more, or less, a choice is learnt from for the food it seeks."""
    return KILL_FOOD_LEARNING if choice in KILL_FOOD else PLANT_FOOD_LEARNING if choice in PLANT_FOOD else 1.0
# Primal fear: facing a hunter stronger than itself, an animal may be overcome, each turn, with a chance of
# PRIMAL_FEAR times how lopsided the fight would be (predators PRIMAL_FEAR_PREDATORS as often, and less
# the hungrier, none as they starve). While a
# hunter is in sight it then panics every turn, fleeing, or, with a chance of PRIMAL_FREEZE at the lowest
# intellect and none at the highest, freezes and does nothing at all.
# The instinct to hunt: whenever hunting or attacking is open to a predator, it values each this much more,
# times its hunger as a share, whatever its brain has learnt; the brain's learning adds to it.
HUNTING_INSTINCT = 1.0
MATE_REACH = 2         # a partner within this many steps is close enough to breed with
MATING_CALL = 0.5      # how much more a predator ready to breed values breeding, or else courting a mate it sees
MATING_CALL_RANGE = 6  # a predator ready to breed seeks a mate this far off, beyond its sight
OMNIVORE_MATING_CALL = 0.25  # an omnivore's milder mating call
HUNTER_BREEDING_COST = 0.5  # share of the usual breeding cost a predator or omnivore pays
PACK_MOUTHS = 0.5      # a hunter's pack-mates count as this many mouths each on its food, sharing their kills
BLOODLUST = 1.0        # how much more a predator values feasting whenever it sees a fresh kill, whatever its hunger
CARRION_CRAVING = 1.0  # how much more an animal values scavenging carrion it sees, at the highest carrion trait;
                       # nothing up to the middle of the trait's range
ALERT_JOY = 0.1        # pleasure of alerting its herd: to a hunter, for a grazer; to a fresh kill, for a hunter
ALARM_REACH = 6        # an alarmed animal knows where the hunter is within this many steps
PREDATOR_SIGHT = 3  # predators see prey this many cells further than they otherwise see, up to MAX_PREDATOR_SIGHT
MAX_PREDATOR_SIGHT = 3
HUNT_STEPS = 1      # and chase it this many more steps a turn
PRIMAL_FEAR = 0.5
PREDATOR_INTELLIGENCE = 2.0  # a predator's intellect counts this many times over in all it does, up to the full range
PREDATOR_FEAR = 0.5  # a predator feels this share of its fear's effects: dread, panic, keeping close to its herd
PREDATOR_LOOMS = 2   # sizing up a fight for primal fear, a predator counts as this many steps larger and stronger
PRIMAL_FEAR_PREDATORS = 1 / 3
PRIMAL_FREEZE = 0.5
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
PACK_RIVALRY = 0.0     # how much a predator's pack-mates count as rivals crowding its territory: its pack is no rival
TERRITORY_RAMP = 3     # each rival past that limit adds a third of the species' strength to rivalry
RIVAL_UNEASE = 0.3     # discomfort of a fully overcrowded territory, less for the tolerant
RIVALRY_WEAR = 0.5     # how much an overcrowded territory wears down a hunter's reluctance to kill its own species
RIVAL_KILL_JOY = 0.5   # relief at killing a predator in an overcrowded territory
# Herding: an animal's herd instinct is the lower of its current empathy and patience, 0..1, and
# scales every effect below. Its herd-mates are animals of its species, and for predators its kind,
# that it would accept as mates (tolerance widens that).
HERD_RADIUS = 4        # herd-mates are sought within this many steps
HERD_SPREAD = 3        # how far from its herd's middle a fearless animal is content; the fearful want the middle
HERD_JOY = 0.15        # pleasure of being among three or more herd-mates
HERD_LONELINESS = 0.1  # unease at having no herd-mate in range
HERD_SAFETY = 0.6      # how much three herd-mates close by calm fear and steady morale
HERD_DEFEND_JOY = 0.5  # pleasure of striking back at whatever struck a herd-mate alongside
PACK_BONUS = 0.5       # extra force of a blow at prey that a herd-mate is also alongside
# Predators whose ring colours differ by no more than this are one kind: they hunt together, and breed
# together as readily as other animals do with this much less difference.
KIND_DIFFERENCE = 0.25
# Breeding responsibly: before breeding, an animal weighs the food it eats within FORAGE_RADIUS
# against the mouths that share it, itself included. Below FOOD_NEEDED meals a mouth it doesn't
# breed; between that and twice that, its litter shrinks in proportion.
FORAGE_RADIUS = 6
FOOD_NEEDED = 0.0  # 0: no limit, and litters don't follow food
UPROOT_FOOD = 3.0  # a starving grazer uproots a plant only with fewer meals a mouth than this about it
BREED_COOLDOWN = 0.0  # share of the usual rest after breeding, 20 turns for the least fertile to 5 for the most
PREDATOR_BREED_COOLDOWN = 0.5  # the same for predators: a short rest
PREDATOR_FOOD_NEEDED = 0.0  # the same for predators: none, so the food about them never stops them breeding
PLANT_MEALS = 2.0      # meals a plant is worth to a full-efficiency browser
PREY_MEALS = 10.0      # meals a prey animal is worth: meat fills about ten times what grass does
SHARED_DIET = 0.5      # how much an animal of another species eating some of the same food counts as a mouth
# Prey switching: a hunter goes after a kind of prey only while it makes up at least SWITCH_SHARE
# of the food it weighs within FORAGE_RADIUS (for omnivores, grazers against plants; for predators,
# grazers against omnivores), so a scarce prey is left alone to recover.
SWITCH_SHARE = 0.5
STRIKE_EFFORT = 0.2    # food burned for each point of damage a blow deals
STRIKE_FORCE = 3.5     # scales every blow: the best-armed, mightiest predator kills the largest, best-armoured grazer in three
GRAZER_DEFENCE_WEAPONS = 0.5    # share of its weapons a grazer brings to defence: striking back, and its spines
OMNIVORE_DEFENCE_WEAPONS = 0.75  # the same for an omnivore; a predator defends with all of its weapons
PREDATOR_WEAPONS = 3.0  # a predator's natural weapons count this many times over, striking and stinging;
# an omnivore's three quarters of that and a grazer's half
GRAZER_DEFENCE_STRENGTH = 5  # a grazer can strike back only with strength above this
FEAST_JOY = 1.5        # a predator's pleasure in feasting on a fresh kill
SATED_JOY = 1.0        # pleasure in hunger relieved by eating, hunting, feasting or scavenging, per its full range
ATTACK_JOY = 0.1       # a predator's pleasure in a blow it lives through
KILL_MEAL = 185                # hunger a kill sates, plus up to KILL_MEAL_FOR_SIZE for the largest prey
KILL_MEAL_FOR_SIZE = 200  # meat is about ten times as filling as grass, and five times plants
CARRION_MEAT = 30      # meat a well-fed corpse leaves, plus up to CARRION_MEAT_FOR_SIZE for the largest;
CARRION_MEAT_FOR_SIZE = 50  # a hungrier body leaves less, and a starved one nothing
GRAZER_SPEED = 1.5     # grazers move this many times as far a step, rounded half up
# Separate niches: grazers are the specialists, getting more from grass and plants than anyone;
# omnivores live on meat, getting only three quarters of the food in plants and half of grass.
GRAZING = {"grazer": 1.25, "omnivore": 0.5}   # share of the food in grass each species gets
BROWSING = {"grazer": 1.25, "omnivore": 0.75}  # share of the food in plants
FLEE_RANGE = 3         # a hunter is only worth fleeing within this many steps, unless the herd raised the alarm
HERD_VIGILANCE = 0.5   # how far three herd-mates close by, at full herd instinct, shrink that range
GRASS_OUTLOOK = 150    # judging grass before breeding, an animal counts what will regrow in this many turns
STANDING_GRASS = 0.25  # and this share of the grass standing now
GRASS_MEAL = 18        # hunger a tuft of grass sates: about a tenth of a middle-sized kill
PLANT_MEAL = 37         # hunger a bite of a plant sates, plus up to PLANT_MEAL_FOR_SIZE for the largest:
PLANT_MEAL_FOR_SIZE = 66  # a fifth of a kill of the same size
PLANT_BITE = 15        # health a plant loses to each bite
PLANT_REGROWTH = 0.02  # share of its health pool a plant regrows each turn
PLANT_RECOVERY = 8     # turns after a bite before a plant can be bitten again
# A desperate grazer, starving with food scarce about it, may uproot a plant beside it: it eats this
# many times a bite's meal, and tears this much of the plant's health away, recovering or not.
UPROOT_HUNGER = 80     # starving: hunger at least this
UPROOT_MEALS = 0.5     # share of a feast on a body of the plant's size an uprooting grazer eats
UPROOT_BITE = 45
SEED_CHANCE = (0.03, 0.10)  # a well-fed plant's chance each turn of seeding, from least to most fertile
SEED_COST = 20         # hunger seeding costs a plant
SEED_HUNGER = 40       # a plant hungrier than this doesn't seed
SEEDLING_HUNGER = 50
SEED_REACH = 5         # how far, in cells, a seed may land from its plant
LIGHT_RADIUS = 3       # plants compete for light with the plants within this many cells
LIGHT_LIMIT = 3        # each such plant cuts a plant's chance of seeding by this share, to none at this many
PLANT_CROWDING = 3     # a seed doesn't take where more than this many plants grow within two cells
TEMPERAMENT_RELAPSE = 0.0005  # share of the gap back to the inborn value closed each turn
# Climate. The map is banded into zones across its height: ARCTIC_SHARE at the top and at the
# bottom is arctic, cold; a strip of EQUATOR_SHARE across the middle is the equator, hot; the rest
# is temperate, medium. Every YEAR_LENGTH turns is a year, starting mild. Through its second quarter
# the north, from the top down to and through the equator, is a step colder; through its fourth the
# south, from the bottom up to and through the equator, is. Every Bitzkey has a home temperature,
# and may tolerate heat or cold, suffering HARDINESS of its penalties; plants tolerate nothing, and a plant
# spawned at the start is at home in its zone's usual temperature.
DEFAULT_YEAR_LENGTH = 360
YEAR_LENGTH = DEFAULT_YEAR_LENGTH  # each world sets it from its year length setting
ARCTIC_SHARE = 0.2
EQUATOR_SHARE = 0.2
CLIMATE_HUNGER = 2.0   # an animal a step or more from home, or a plant too hot, needs this many times the food
CLIMATE_DAMAGE = 0.02  # for each step from home, a Bitzkey loses this share of its full health a turn; plants from two
PLANT_CLIMATE_YIELD = 0.0  # a plant out of its temperature gives this share of its food
GRASS_DIEBACK = 20     # where it isn't medium, a cell's grass loses a tuft every this many turns
# The weather turns in waves, one in each climate zone, all setting off together as the season turns,
# so no zone waits on another. In each, a winter washes in from the side toward its hemisphere's edge
# toward the equator, and as it ends ebbs back the way it came, over this share of each season. The wave moves in bands of
# WAVE_BAND rows; at its front, WAVE_FRONT bands stand part way between the old weather and the new,
# in even steps, so the change comes as a staircase. The climate's effects move with it.
SEASON_TURNING = 0.5
WAVE_BAND = 2
WAVE_FRONT = 4
# A dead animal is a fresh kill, not yet carrion, through the turn it died and FRESH_KILL_TURNS more;
# then it starts to rot. A predator or omnivore may eat a fresh kill of what it lives on whole,
# FRESH_KILL_MEALS times the meal of making the kill, and nothing is left.
FRESH_KILL_TURNS = 3
FRESH_KILL_MEALS = 3.0
SEED_VIABILITY = 0.75  # years a seed that falls where it can't grow stays able to germinate
PLANT_GESTATION = 10  # turns a seed lies in the ground before it can sprout
STARTING_TOLERANCE = 0.1  # the chance a random starting genome carries each tolerance flag
HARDINESS = 0.5  # the share of exposure's extra hunger and damage an animal tolerant of it suffers
HOT, MEDIUM, COLD, VERY_COLD = range(len(TEMPERATURES))
SEASONS = ("mild", "northern winter", "mild", "southern winter")  # by quarter of the year

# How experiences reshape temperament, in shares of a trait's full range before scaling by
# plasticity.
EXPERIENCES = {
    "kill": {"aggression": 0.04, "fear": -0.02, "intellect": 0.002},
    "hit": {"aggression": 0.0067},
    "stung": {"aggression": -0.0133, "fear": 0.0133},
    "wounded": {"fear": 0.0333, "aggression": -0.0067},
    "wounded_alone": {"sociability": 0.02},
    "bred": {"libido": 0.02, "sociability": 0.0133, "intellect": 0.002},
    "bred_into_hunger": {"libido": -0.0533},
    "starving": {"appetite": 0.0033, "patience": -0.0033},
    "foraged_meal": {"patience": 0.0067, "intellect": 0.0003},
    "discovery": {"curiosity": 0.01, "intellect": 0.002},
    "ambushed": {"curiosity": -0.02, "fear": 0.0133},
    "defended": {"fear": -0.0133, "aggression": 0.0067, "intellect": 0.002},
    "panicked": {"fear": 0.0067},
    "lesson": {"intellect": 0.0001},  # a choice that turned out well, learnt from
}

# Setting off in a direction: the step in x and y, whether it is a walk of a single cell (else it goes
# as far as its speed allows), and what it then says it did. The top and bottom of the map are
# barriers; east and west wrap round.
HEADINGS = {"go_north": (0, -1, False, "went north"), "go_south": (0, 1, False, "went south"),
            "go_east": (1, 0, False, "went east"), "go_west": (-1, 0, False, "went west"),
            "walk_north": (0, -1, True, "walked north"), "walk_south": (0, 1, True, "walked south"),
            "walk_east": (1, 0, True, "walked east"), "walk_west": (-1, 0, True, "walked west")}
TRAVELLED = {went for *_, went in HEADINGS.values()}
GRASS_SEEN = SENSES.index("grass_seen")
HUNT, ATTACK = ACTIONS.index("hunt"), ACTIONS.index("attack")
FEAST, SCAVENGE, COURT, BREED = (ACTIONS.index("feast"), ACTIONS.index("scavenge"), ACTIONS.index("court"),
                                 ACTIONS.index("breed"))
EAT, UPROOT, FEAST = ACTIONS.index("eat"), ACTIONS.index("uproot"), ACTIONS.index("feast")
MAX_SIGHT = 2 + TRAIT_MAX + MAX_PREDATOR_SIGHT  # the furthest anything is seen: a predator's sight for prey
SIGHT_OFFSETS = sorted(
    ((dx, dy) for dx in range(-MAX_SIGHT, MAX_SIGHT + 1) for dy in range(-MAX_SIGHT, MAX_SIGHT + 1)
     if 0 < abs(dx) + abs(dy) <= MAX_SIGHT),
    key=lambda offset: (abs(offset[0]) + abs(offset[1]), offset),
)
SEEN_OFFSETS = [(0, 0), *SIGHT_OFFSETS]
# The same offsets cut off at each distance, so scans need no distance checks.
SIGHT_WITHIN = [[offset for offset in SIGHT_OFFSETS if abs(offset[0]) + abs(offset[1]) <= reach]
                for reach in range(MAX_SIGHT + 1)]
SEEN_WITHIN = [[(0, 0), *offsets] for offsets in SIGHT_WITHIN]


def season(turn: int) -> int:
    """The quarter of the year a turn falls in, 0 to 3."""
    return turn % YEAR_LENGTH * 4 // YEAR_LENGTH


def year(turn: int) -> int:
    """The year a turn falls in, from 1."""
    return turn // YEAR_LENGTH + 1


def zone_temperature(y: int, height: int) -> int:
    """A row's temperature outside the winters: cold in the arctic, hot on the equator, medium
    between."""
    place = (y + 0.5) / height
    if place < ARCTIC_SHARE or place > 1 - ARCTIC_SHARE:
        return COLD
    return HOT if abs(place - 0.5) < EQUATOR_SHARE / 2 else MEDIUM


def in_winter(y: int, height: int, quarter: int) -> bool:
    """Whether a row is in winter in a quarter of the year: the north and the equator in the second,
    the south and the equator in the fourth."""
    place = (y + 0.5) / height
    equator = abs(place - 0.5) < EQUATOR_SHARE / 2
    return (quarter == 1 and (place < 0.5 or equator)) or (quarter == 3 and (place > 0.5 or equator))


def season_temperature(y: int, height: int, quarter: int) -> int:
    """A row's settled temperature in a quarter of the year: its zone's, a step colder in its
    hemisphere's winter."""
    return zone_temperature(y, height) + in_winter(y, height, quarter)


def zone_rows(y: int, height: int) -> tuple[int, int]:
    """The first and last rows of the climate zone a row lies in."""
    zone = zone_temperature(y, height)
    first, last = y, y
    while first > 0 and zone_temperature(first - 1, height) == zone:
        first -= 1
    while last < height - 1 and zone_temperature(last + 1, height) == zone:
        last += 1
    return first, last


def wave_band(y: int, height: int, quarter: int) -> tuple[int, int]:
    """As the weather turns into a quarter of the year, which band of its zone's wave a row is in,
    counted in the order the wave reaches them, and how many bands the wave crosses. Every zone's wave
    sets off at once. A winter washes in from the side toward its hemisphere's edge toward the
    equator (the second and fourth quarters), and ebbs back the other way (the third and first)."""
    north = quarter in (1, 2)
    first, last = zone_rows(y, height)
    inward = quarter in (1, 3)
    offset = y - first if north == inward else last - y
    return offset // WAVE_BAND, (last - first) // WAVE_BAND + 1


def season_progress(turn: int) -> float:
    """How far the season's wave has come: 0 as a season begins, 1 once SEASON_TURNING of it has
    passed. The world's first season starts settled."""
    quarter = season(turn)
    if (turn < YEAR_LENGTH and quarter == 0) or SEASON_TURNING <= 0:
        return 1.0
    into = (turn % YEAR_LENGTH * 4 - quarter * YEAR_LENGTH) / YEAR_LENGTH  # share of the season gone, 0 to 1
    return min(1.0, into / SEASON_TURNING)


def temperature_at(y: int, height: int, turn: int) -> float:
    """A row's temperature on a turn, from 0 (hot) to 3 (very cold): its season's, though early in
    a season, until the season's wave has passed it, the last season's or part way between the two."""
    quarter = season(turn)
    now = season_temperature(y, height, quarter)
    progress = season_progress(turn)
    if progress >= 1:
        return float(now)
    before = season_temperature(y, height, (quarter - 1) % 4)
    if now == before:
        return float(now)
    band, bands = wave_band(y, height, quarter)
    reach = progress * (bands + WAVE_FRONT)  # how far the front has come, in bands
    turned = min(1.0, max(0.0, (reach - band) / WAVE_FRONT))
    return before + (now - before) * turned


def dieback_threshold(x: int, y: int) -> float:
    """A cell's fixed share, 0 to 1, below which a row's shortfall from medium makes its grass die back."""
    return (x * 7919 + y * 104729) % 1000 / 1000


@cache
def birth_offsets(reach: int) -> tuple[tuple[int, int], ...]:
    """Where young may be born around their mother, nearest first: the sides, then the corners, then
    each ring further out, up to `reach`, its nearest cells first."""
    further = sorted(((dx, dy) for dy in range(-reach, reach + 1) for dx in range(-reach, reach + 1)
                      if max(abs(dx), abs(dy)) > 1),
                     key=lambda offset: (max(abs(offset[0]), abs(offset[1])), offset[0] ** 2 + offset[1] ** 2,
                                         offset[1], offset[0]))
    return ((0, -1), (1, 0), (0, 1), (-1, 0), (1, -1), (1, 1), (-1, 1), (-1, -1), *further)


def thinking_intellect(intellect: int, species: str) -> int:
    """An inborn intellect as it counts for the brain's depth and how much it replays: a predator's
    PREDATOR_INTELLIGENCE times over, up to the full range."""
    return min(TRAIT_MAX, int(intellect * (PREDATOR_INTELLIGENCE if species == "predator" else 1.0)))


def evolved(parent: int, child: int) -> int:
    """A child's genome with its species moved along the chain of evolution, grazer, omnivore, predator,
    by a mutation of the species: a flip landing on predator makes any animal a predator; otherwise a flip
    of the species' lower bit moves it a step up the chain, and of its upper bit a step down. A step off
    the chain is undone, a plant stays a plant and an animal never becomes one; the rest of the genome is
    as it is."""
    shift = 32  # the species trait is the ring's first; the species is it mod 4
    before, after = (parent >> shift) & 0x3, (child >> shift) & 0x3
    plant, predator = 0, 3
    if after == before:
        return child
    if before == plant:
        species = plant
    elif after == predator:
        species = predator
    else:
        species = before + 1 if (before ^ after) & 0x1 else before - 1
        if not plant < species <= predator:
            species = before
    return (child & ~(0x3 << shift)) | (species << shift)


def predator_dna(dna: int) -> int:
    """The genome with its species trait made a predator's (3, or 7 where its top bit was set: the species is
    the trait mod 4), the rest unchanged."""
    shift = 32  # the species trait is the ring's first
    trait = (dna >> shift) & 0x7
    return (dna & ~(0x7 << shift)) | (((trait & 0x4) | 0x3) << shift)


def gestation(size: int) -> int:
    """Turns an animal of this body size carries its young."""
    shortest, longest = GESTATION
    return shortest + int((longest - shortest) * size / TRAIT_MAX + 0.5)


@cache
def nearby_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    """The offsets within `radius` steps, the animal's own cell among them, column by column."""
    return tuple((dx, dy) for dx in range(-radius, radius + 1)
                 for dy in range(-(radius - abs(dx)), radius - abs(dx) + 1))


@lru_cache(maxsize=1 << 16)
def looks_apart(first: int, second: int) -> tuple[float, float]:
    """How far two genomes' ring and dot colours are apart, each from 0 to 1. Groups share
    genomes, so the same pairs come up again and again."""
    return ring_difference(first, second), dot_difference(first, second)
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
    primal_fear: bool = False  # overcome by a stronger hunter: it panics, or freezes, while one is in sight
    rivalry: float = 0.0    # hunters: how overcrowded its territory was with rivals when it last looked
    hunts_grazers: bool = True    # hunters: whether grazers were common enough to hunt when it last looked
    hunts_omnivores: bool = True  # predators: the same for omnivores
    own_rivalry: float = 0.0  # the same, counting only its own species: what wears down its reluctance
    alarm: int | None = None  # what struck a herd-mate close by since it last acted, or a herd-mate alerted it to
    kill_alert: tuple[int, int] | None = None  # where a pack-mate alerted it to a fresh kill since it last acted
    seen_success: float = 0.0  # a herd-mate's success it saw, fading turn by turn
    seen_failure: float = 0.0  # a herd-mate's failure it saw, the same
    herd_assailant: int | None = None  # what struck a herd-mate close by, from alongside it: it may defend against it
    meat: float = 0.0  # a corpse's meat left for scavengers
    rot: float = 0.0   # meat lost to rot each turn
    portions: int = 0  # a body's portions left to eat, feasting or scavenging: one for each size class
    trail: list[tuple[int, int]] = field(default_factory=list)  # the cells it passed through this turn, for the map
    # Its most recent experiences, each (senses, choice, target), newest last, to re-learn from.
    memories: list[tuple] = field(default_factory=list)
    # The young it carries, conceived at mating, each (dna, mind, brain, energy, generation), and the
    # turn they are born.
    womb: list[tuple] = field(default_factory=list)
    due_turn: int | None = None
    # Current temperament: starts at the inborn values and is reshaped by experience.
    mind: dict[str, float] = field(default_factory=dict)
    # The last decision, kept until the next one so the brain can learn from how it turned out.
    last_senses: list[float] | None = None
    last_choice: int = 0
    last_wellbeing: float = 0.0
    reward: float = 0.0
    last_comfort: float = 0.0  # how comfortable it felt at its last decision
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

    def looks_apart(self, other: Bitzkey) -> tuple[float, float]:
        """How far its ring and dot colours are from another's, each from 0 to 1."""
        return looks_apart(self.dna, other.dna)

    @cached_property
    def species(self) -> str:
        return str(self.body["species"])

    @cached_property
    def climate(self) -> tuple[int, bool, bool]:
        """Its home temperature, and whether it tolerates heat and cold."""
        return decode_climate(self.dna)

    def climate_steps(self, here: float) -> float:
        """How many steps a temperature is from home, part steps while the weather turns."""
        return abs(here - self.climate[0])

    def exposure_share(self, here: float) -> float:
        """The share of exposure's penalties it suffers here: HARDINESS for an animal tolerant of
        heat where it is hotter than home, or of cold where it is colder; otherwise all of them.
        Plants tolerate nothing."""
        home, heat, cold = self.climate
        if self.species != "plant" and ((here > home and cold) or (here < home and heat)):
            return HARDINESS
        return 1.0

    @property
    def profile(self) -> dict[str, float]:
        """Current temperament and physical traits, keyed by trait name."""
        return {**self.mind, **{trait: float(value) for trait, value in self.body.items()
                                if trait != "species"}}

    def physique(self, trait: str) -> float:
        """A body trait as a share of its full range."""
        return share(int(self.body[trait]))

    @property
    def intelligence(self) -> float:
        """Its intellect now, as a share of the full range, as it counts in all it does: a predator's
        PREDATOR_INTELLIGENCE times over, up to the full range."""
        return min(1.0, self.feeling("intellect") * (PREDATOR_INTELLIGENCE if self.species == "predator" else 1.0))

    @property
    def fearfulness(self) -> float:
        """Its fear, as its effects feel it: a predator's PREDATOR_FEAR as much, and less the hungrier it is."""
        return self.feeling("fear") * (PREDATOR_FEAR * self.unafraid if self.species == "predator" else 1.0)

    @property
    def unafraid(self) -> float:
        """The share of its fear's effects hunger leaves a predator: all when fed, falling with its hunger
        to none as it starves; any other animal feels them all."""
        return max(0.0, 1 - self.hunger / 100) if self.species == "predator" else 1.0

    @property
    def herd_instinct(self) -> float:
        """How strongly it herds: the lower of its current empathy and patience, 0 to 1."""
        return min(self.feeling("empathy"), self.feeling("patience"))

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
        self.relieve(food / self.stomach)

    def relieve(self, amount: float) -> None:
        """Ease its hunger by an amount, eating, and take SATED_JOY's pleasure in what was eased."""
        eased = min(self.hunger, amount)
        self.hunger -= eased
        self.reward += SATED_JOY * eased / 100

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
        """How much of the food in grass nourishes it: grazers more than all, omnivores half."""
        return GRAZING.get(self.species, 0.0)

    @property
    def browsing(self) -> float:
        """How much of the food in plants nourishes it: grazers more than all, omnivores three quarters."""
        return BROWSING.get(self.species, 0.0)

    @property
    def sight(self) -> int:
        return 2 + int(self.body["sight"])

    @cached_property
    def genes(self) -> dict[str, float]:
        """Its brain's inputs for its genome: its ring's and dot's traits and its climate genes."""
        return gene_senses(self.dna)

    @property
    def lifespan(self) -> float:
        """Expected turns of life, plant or animal: a slower metabolism, that is a larger body,
        lives longer."""
        shortest, middle, longest = LIFESPAN_YEARS
        fastest, slowest = METABOLISM
        slowing = (fastest - self.metabolism) / (fastest - slowest)  # 0 at the fastest metabolism, 1 the slowest
        if slowing <= 0.5:
            years = shortest + (middle - shortest) * (2 * slowing)
        else:
            years = middle + (longest - middle) * (2 * slowing - 1)
        return years * YEAR_LENGTH

    @property
    def breeding_cost(self) -> float:
        """Hunger it pays to breed: less the more fertile it is, and HUNTER_BREEDING_COST of that for a hunter."""
        hunter = HUNTER_BREEDING_COST if self.species in ("omnivore", "predator") else 1.0
        return (25 - 15 * self.physique("fertility")) * hunter

    @property
    def can_breed(self) -> bool:
        """Ready, not carrying young, and able to pay for breeding with energy to spare."""
        return self.cooldown == 0 and not self.womb and self.hunger + self.breeding_cost <= 100 - BREED_RESERVE

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

    @property
    def brain_bytes(self) -> int:
        """The memory its brain, and those of the young it carries, take."""
        return sum(brain.nbytes for brain in (self.brain, *(young[2] for young in self.womb)) if brain is not None)

    def copy(self) -> Bitzkey:
        return replace(self, brain=self.brain.copy() if self.brain else None, mind=self.mind.copy(),
                       trail=list(self.trail), memories=list(self.memories),
                       womb=[(dna, dict(mind), brain.copy() if brain else None, energy, generation)
                             for dna, mind, brain, energy, generation in self.womb])


def litter_size(parents: Sequence[tuple[int, int]]) -> int:
    """Young per birth, before space limits it, from each parent's (litter trait, size): a high
    litter trait and a small body both make for bigger litters, and the smallest bodies have
    SMALL_LITTERS times as many, falling evenly to the usual for the largest. Predators have half
    as many, rounded up."""
    smallest, largest = LITTER_RANGE
    tendency = sum((1 - LITTER_SIZE_WEIGHT) * share(litter) + LITTER_SIZE_WEIGHT * (1 - share(size))
                   for litter, size in parents) / len(parents)
    litter = smallest + int(tendency * (largest - smallest) + 0.5)
    small = 1 - sum(share(size) for _, size in parents) / len(parents)
    litter = int(litter * (1 + (SMALL_LITTERS - 1) * small) + 0.5)
    return litter


def remove_stale_histories() -> None:
    """Clear away history folders that earlier versions left among the temporary files, once they are
    STALE_HISTORY_HOURS old, so they don't go on filling the disk."""
    cutoff = time.time() - STALE_HISTORY_HOURS * 3600
    for folder in Path(tempfile.gettempdir()).glob(HISTORY_PREFIX + "*"):
        try:
            if folder.is_dir() and folder.stat().st_mtime < cutoff:
                shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            pass


class Simulation:
    def __init__(self, width: int = GRID_SIZE, height: int = GRID_SIZE,
                 population: int = INITIAL_POPULATION, seed: int | None = None,
                 grass_density: float = GRASS_DENSITY, imported: Sequence[Imported] = (),
                 population_cap: int | None = None, mind_mutation: float = 100 * MIND_MUTATION_CHANCE,
                 body_mutation: float = 100 * BODY_MUTATION_CHANCE, year_length: int = DEFAULT_YEAR_LENGTH):
        """Spawn `population` random Bitzkeys in groups sharing a genome, plus a group for each
        `imported` Bitzkey, whose members all start with copies of its saved brain if it has one.
        `population_cap` limits living animals; None leaves crowding alone to limit them.
        `mind_mutation` and `body_mutation` are the percent chances that a birth mutates the
        child's mind or body. `year_length` is the turns in a year, for the seasons and lifespans;
        like the advanced rules, it holds for the whole program while the world runs."""
        global YEAR_LENGTH
        if year_length < 4:
            raise ValueError("A year must be at least 4 turns.")
        YEAR_LENGTH = self.year_length = year_length
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
        # Each death: its cause, and the identifier and species of what killed it, when something did.
        self.death_records: dict[int, tuple[str, int | None, str | None]] = {}
        self.turn = 0
        self.deaths = 0
        self.death_causes: dict[str, dict[str, int]] = {}  # every death so far, by species and then cause
        self.next_identifier = 0
        self.agents: list[Bitzkey] = []
        self._cells: dict[tuple[int, int], list[Bitzkey]] = {}
        self._carrion: dict[tuple[int, int], list[Bitzkey]] = {}
        self._kills: dict[tuple[int, int], list[Bitzkey]] = {}  # fresh kills, by cell
        self._population = 0
        self.population_cap = population_cap
        self.mind_mutation = mind_mutation / 100
        self.body_mutation = body_mutation / 100
        self._plants = 0
        # Seeds waiting where they fell for the temperature they can grow in: (x, y, genome,
        # generation, turn sown).
        self.seeds: list[tuple[int, int, int, int, int]] = []
        self._climate: list[float] = []
        self._growth: list[float] = []      # each row's share of medium, where grass grows
        self._grass_rows: list[int] = []    # the rows where grass grows at all
        self._grass_reach: list[float] = []  # their growth added up, row by row, for sowing
        self._update_climate()
        self.grass: dict[tuple[int, int], int] = {}
        initial_grass = round(width * height * grass_density / 100)
        self.grass_regrowth = initial_grass * GRASS_REGROWTH
        self._sow(initial_grass)
        sizes = self._group_sizes(population)
        species = [self.random.randrange(len(SPECIES)) for _ in sizes]
        homes = self._plant_homes(species)
        for group, size in enumerate(sizes):
            self._spawn_group(make_dna(species[group], self.random), size, native=True, rows=homes.get(group))
        for bitzkey in imported:
            self._spawn_group(bitzkey.dna, self.random.randint(*SPAWN_GROUP), bitzkey.brain, bitzkey.mind)
        # The world at every turn kept, oldest first, each a whole copy.
        self.history: deque[dict] = deque()  # each {"turn": the turn, "state": the world, "bytes": its size}
        self._history_bytes = 0  # the size of them all, near enough
        self.frontier = 0  # the latest turn simulated; seeking never goes past it
        # What herd-mates saw this turn, to learn from at its end: (observer, senses, choice, outcome, rate).
        self._witnessed: list[tuple] = []
        self._take_census()
        self._record()

    @property
    def living(self) -> list[Bitzkey]:
        return [agent for agent in self.agents if agent.alive]

    def temperature(self, y: int) -> float:
        """A row's temperature now, part way between two while the weather turns."""
        return self._climate[y]

    @staticmethod
    def _exposure(agent: Bitzkey, here: int) -> str:
        """How the climate killed it: colder than home, it froze; hotter, it overheated."""
        return "froze" if here > agent.climate[0] else "overheated"

    def _time_of_year(self) -> dict[str, float]:
        """The brain's sense of the time of year: two hands of a clock a quarter of a year apart."""
        day = self.turn % YEAR_LENGTH / YEAR_LENGTH  # 0 at the year's start, nearly 1 at its end
        return {"time_of_year_a": abs(2 * day - 1), "time_of_year_b": abs(2 * ((day + 0.25) % 1.0) - 1)}

    def _update_climate(self) -> None:
        """Each row's temperature this turn, and how well grass grows in it: fully where it is
        medium, less the further it has turned from medium, and not at all a step away."""
        self._climate = [temperature_at(y, self.height, self.turn) for y in range(self.height)]
        self._growth = [max(0.0, 1 - abs(temperature - MEDIUM)) for temperature in self._climate]
        self._grass_rows = [y for y, growth in enumerate(self._growth) if growth > 0]
        self._grass_reach = list(accumulate(self._growth[y] for y in self._grass_rows))

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

    def _zones(self) -> list[list[int]]:
        """The climate zones from top to bottom, each the rows it covers outside the winters."""
        zones: list[list[int]] = []
        for y in range(self.height):
            if zones and zone_temperature(zones[-1][0], self.height) == zone_temperature(y, self.height):
                zones[-1].append(y)
            else:
                zones.append([y])
        return zones

    def _plant_homes(self, species: list[int]) -> dict[int, list[int]]:
        """So that every zone starts with plants, the starting groups' plant groups take a zone each,
        top to bottom, until every zone has one; if there are too few, the last animal groups become
        plants, though one animal group is always kept. Returns the rows each such group's centre may
        fall in, by group; the rest fall anywhere. `species` is changed in place."""
        zones = self._zones()
        plants = [group for group, kind in enumerate(species) if kind == 0]
        animals = [group for group, kind in enumerate(species) if kind != 0]
        while len(plants) < len(zones) and len(animals) > 1:
            group = animals.pop()
            species[group] = 0
            plants.append(group)
        lowest, highest = SPAWN_SPREAD, self.height - SPAWN_SPREAD - 1  # where a group's centre may be
        homes = {}
        for group, rows in zip(sorted(plants), zones):
            centres = [y for y in rows if lowest <= y <= highest]
            homes[group] = centres or [min(highest, max(lowest, rows[len(rows) // 2]))]
        return homes

    def _spawn_group(self, dna: int, size: int, brain: Brain | None = None,
                     mind: Mapping[str, float] | None = None, native: bool = False,
                     rows: Sequence[int] | None = None) -> None:
        """Place `size` Bitzkeys of one genome around a random centre, kept clear of the edges
        so the group is never split across them, and in one of `rows` if given; otherwise, with
        SPAWN_AT_HOME, in its home zone. Each gets its own
        copy of `brain` and starts with the temperament in `mind`, if given. `native` groups have
        random starting genomes: each carries a tolerance flag only with STARTING_TOLERANCE, and
        plants take the usual temperature of their centre's zone as their home."""
        if rows is None and SPAWN_AT_HOME:
            rows = self._home_rows(decode_climate(dna)[0])
        x = self.random.randrange(SPAWN_SPREAD, self.width - SPAWN_SPREAD)
        y = (self.random.randrange(SPAWN_SPREAD, self.height - SPAWN_SPREAD) if rows is None
             else rows[self.random.randrange(len(rows))])
        if native:  # tolerance is rare at the start
            for flag in (HEAT_TOLERANCE, COLD_TOLERANCE):
                dna = dna | flag if self.random.random() < STARTING_TOLERANCE else dna & ~flag
        if native and decode_body(dna)["species"] == "plant":
            dna = dna & ~(0x3 << TEMPERATURE_SHIFT) | zone_temperature(y, self.height) << TEMPERATURE_SHIFT
        for _ in range(size):
            self._add_agent(dna, x + self.random.randint(-SPAWN_SPREAD, SPAWN_SPREAD),
                            y + self.random.randint(-SPAWN_SPREAD, SPAWN_SPREAD),
                            brain=brain.copy() if brain else None, mind=mind)

    def _home_rows(self, home: int) -> list[int]:
        """The rows a group's centre may fall in whose usual temperature is `home`, or nearest it:
        a very cold home has no zone of its own outside the winters, so it takes the arctic."""
        centres = range(SPAWN_SPREAD, self.height - SPAWN_SPREAD)
        nearest = min(abs(zone_temperature(y, self.height) - home) for y in centres)
        return [y for y in centres if abs(zone_temperature(y, self.height) - home) == nearest]

    def _add_agent(self, dna: int, x: int, y: int, brain: Brain | None = None,
                   mind: Mapping[str, float] | None = None) -> Bitzkey:
        agent = Bitzkey(self.next_identifier, dna & GENOME, x, y)
        agent.health = agent.max_health
        if mind:
            agent.mind.update({trait: mind[trait] for trait in MIND_TRAITS if trait in mind})
        if agent.species != "plant":  # plants need no brain
            depth = depth_for(thinking_intellect(agent.inborn["intellect"], agent.species))  # twice its inborn intellect, at least two
            agent.brain = Brain.random(self.random, depth) if brain is None else brain.resized(depth)
        self.next_identifier += 1
        self.agents.append(agent)
        self.lineage[agent.identifier] = agent.dna
        self._cells.setdefault((x, y), []).append(agent)
        return agent

    def close(self) -> None:
        """Let the history go. The world can't be rewound after; it is closed when left."""
        self.history = deque()
        self._history_bytes = 0

    @property
    def earliest(self) -> int:
        """The oldest turn the history still keeps."""
        return self.history[0]["turn"] if self.history else self.turn

    def _record(self) -> None:
        """Keep a copy of the newest turn, then let the oldest go while the history holds more than
        HISTORY_TURNS turns or HISTORY_MEMORY gigabytes."""
        self.frontier = self.turn
        if self.history and self.history[-1]["turn"] == self.turn:  # recorded again, as it now is
            self._history_bytes -= self.history.pop()["bytes"]
        agents = [agent.copy() for agent in self.agents]
        size = sum(AGENT_BYTES + agent.brain_bytes for agent in agents)
        self.history.append({"turn": self.turn, "bytes": size, "state": {
            "turn": self.turn, "deaths": self.deaths, "next_identifier": self.next_identifier, "agents": agents,
            "death_causes": {species: dict(causes) for species, causes in self.death_causes.items()},
            "grass": self.grass.copy(), "seeds": list(self.seeds), "random_state": self.random.getstate(),
            "insights": self.insights.snapshot()}})
        self._history_bytes += size
        most = HISTORY_MEMORY * (1 << 30)
        turns = float("inf") if HISTORY_TURNS >= HISTORY_ALL else HISTORY_TURNS
        while len(self.history) > 1 and (len(self.history) > turns or self._history_bytes > most):
            self._history_bytes -= self.history.popleft()["bytes"]

    def _restore(self, entry: dict) -> None:
        """Bring back a turn the history keeps, as a copy, so the history's stays as it was."""
        snapshot = entry["state"]
        self.turn = snapshot["turn"]
        self.deaths = snapshot["deaths"]
        self.death_causes = {species: dict(causes) for species, causes in snapshot["death_causes"].items()}
        self.next_identifier = snapshot["next_identifier"]
        self.agents = [agent.copy() for agent in snapshot["agents"]]
        self.grass = snapshot["grass"].copy()
        self.seeds = list(snapshot["seeds"])
        self.random.setstate(snapshot["random_state"])
        self.insights.restore(snapshot["insights"])
        self._update_climate()
        self._index_cells()

    def seek(self, target: int) -> None:
        """Move the world to any turn the history keeps, from the earliest to the latest simulated,
        by bringing back its copy."""
        target = max(self.earliest, min(self.frontier, target))
        if target != self.turn:
            self._restore(self.history[target - self.earliest])

    def rewind(self) -> bool:
        if self.turn <= self.earliest:
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

    def _take_turn(self) -> None:
        """One turn. Every Bitzkey first lives through it, in a shuffled order: it ages, hungers,
        weathers the climate, perhaps dies, and plants grow and seed. Then every animal left thinks
        at once, sensing the world as it stands and deciding, and they act in the same order, each
        finding the world as those before it left it."""
        self.turn += 1
        self._update_climate()
        self._index_cells()
        for agent in self.agents:  # a body some cost starved, its death not yet recorded: recorded now
            agent.trail.clear()  # last turn's way is forgotten
            if agent.died_turn is None and not agent.alive:
                self._die(agent, "starved" if agent.hunger >= 100 else "killed")
        order = self.living
        self._plants = sum(agent.species == "plant" for agent in order)
        self._population = len(order) - self._plants  # animals; the population cap is theirs alone
        self.random.shuffle(order)
        thinkers = []
        for agent in order:
            if not agent.alive:
                continue
            agent.age += 1
            plant = agent.species == "plant"
            here = self._climate[agent.y]
            steps = agent.climate_steps(here)
            hungrier = steps > 0 and (not plant or here < agent.climate[0])  # out of its temperature; a plant, too hot
            share = agent.exposure_share(here)  # less for the hardy
            # The extra food it needs rises with the weather's turning, to the full amount a step away.
            agent.hunger += agent.burn * (1 + (CLIMATE_HUNGER - 1) * share * min(1.0, steps) if hungrier else 1)
            if agent.cooldown:
                agent.cooldown -= 1
            if agent.health < agent.max_health and agent.hunger < 65:
                agent.heal(0.005)
            # A tenth of its health a step, part steps in part; a plant takes one step in its stride,
            # and is hurt more and more as the weather turns toward a second, to full harm two away.
            felt = max(0.0, min(steps, 2 * (steps - 1))) if plant else steps
            if felt > 0:
                agent.health -= CLIMATE_DAMAGE * felt * share * agent.max_health
            if agent.hunger >= 100:  # hungrier out of its temperature, it died of exposure
                self._die(agent, self._exposure(agent, here) if hungrier else "starved")
                continue
            if agent.health <= 0:
                self._die(agent, self._exposure(agent, here))
                continue
            if self._dies_of_age(agent):
                self._die(agent, "old age")
                continue
            if plant:
                agent.hunger = max(0, agent.hunger - 1.5)
                if steps >= 0.5:  # nearer another temperature than its own, a plant neither grows nor seeds
                    agent.last_action = "withering"
                    continue
                agent.health = min(agent.max_health, agent.health + agent.max_health * PLANT_REGROWTH)
                agent.last_action = "photosynthesized"
                self._seed(agent)
                continue
            if agent.due_turn is not None and agent.due_turn <= self.turn:
                self._give_birth(agent)
            agent.seen_success *= SOCIAL_MEMORY
            agent.seen_failure *= SOCIAL_MEMORY
            self._settle(agent)
            if agent.hunger >= 80:
                self._experience(agent, "starving")
            thinkers.append(agent)
        self._think_all(thinkers)

        self._germinate()
        self._wither_grass()
        whole, fraction = divmod(self.grass_regrowth, 1)
        self._sow(int(whole) + (1 if fraction and self.random.random() < fraction else 0))
        for agent in self.agents:
            if agent.died_turn is not None and self.turn - agent.died_turn >= FRESH_KILL_TURNS:
                agent.meat = max(0.0, agent.meat - agent.rot)  # no longer fresh by next turn: it rots
        self.agents = [agent for agent in self.agents
                       if agent.died_turn is None or self.turn - agent.died_turn < CORPSE_TURNS]
        self.insights.decay()
        self._take_census()

    def _seed(self, plant: Bitzkey) -> None:
        """A well-fed plant may sow a seed of its own kind on a free cell up to SEED_REACH away, unless
        plants already crowd that spot; it lies PLANT_GESTATION turns in the ground before it can sprout. Plants compete for light: the more plants grow
        within LIGHT_RADIUS, the less likely a plant is to seed, and with LIGHT_LIMIT it doesn't."""
        if plant.hunger > SEED_HUNGER:
            return
        least, most = SEED_CHANCE
        roll = self.random.random()
        chance = least + (most - least) * plant.physique("fertility")
        if roll >= chance:
            return
        shade = sum(other.species == "plant" for other in self._nearby(plant, LIGHT_RADIUS))
        if roll >= chance * max(0.0, 1 - shade / LIGHT_LIMIT):
            return
        dx, dy = SIGHT_OFFSETS[self.random.randrange(SEED_OFFSETS)]
        if not 0 <= plant.y + dy < self.height:  # off the top or bottom of the map: the seed is lost
            return
        cell = ((plant.x + dx) % self.width, plant.y + dy)
        if not self._room_for_seedling(cell):
            return
        dna = evolved(plant.dna, cross(plant.dna, plant.dna, self.random, None, self.mind_mutation, self.body_mutation))
        plant.hunger += SEED_COST
        plant.children += 1
        plant.last_action = "seeded"
        if not plant.alive:  # it spent the last of its food on the seed
            self._die(plant, "seeded to death")
        self.seeds.append((*cell, dna, plant.generation + 1, self.turn))

    def _wither_grass(self) -> None:
        """Grass dies back where it isn't medium: a tuft a cell every GRASS_DIEBACK turns, the cells
        taking their turns in step. While a row turns away from medium, only some of its cells die
        back, more as it turns further: each cell whose fixed threshold its shortfall passes."""
        for (x, y), tufts in list(self.grass.items()):
            if (1 - self._growth[y] > dieback_threshold(x, y)
                    and (x + y * self.width + self.turn) % GRASS_DIEBACK == 0):
                if tufts > 1:
                    self.grass[(x, y)] = tufts - 1
                else:
                    del self.grass[(x, y)]

    def _room_for_seedling(self, cell: tuple[int, int]) -> bool:
        """Whether a seedling can take on a cell: it is free, and plants don't crowd it."""
        if any(other.alive for other in self._cells.get(cell, ())):
            return False
        spot = Bitzkey(-1, 0, *cell)
        return sum(other.species == "plant" for other in self._nearby(spot, 2)) <= PLANT_CROWDING

    def _sprout(self, dna: int, cell: tuple[int, int], generation: int) -> None:
        seedling = self._add_agent(dna, *cell)
        seedling.hunger = SEEDLING_HUNGER
        seedling.generation = generation
        self._plants += 1

    def _germinate(self) -> None:
        """Seeds sprout once they have lain PLANT_GESTATION turns, their cell is at their temperature
        and there is room, unless they have waited SEED_VIABILITY of a year, when they are dead."""
        viable = SEED_VIABILITY * YEAR_LENGTH
        waiting = []
        for seed in self.seeds:
            x, y, dna, generation, sown = seed
            if self.turn - sown >= viable:
                continue
            if self.turn - sown >= PLANT_GESTATION and abs(self._climate[y] - decode_climate(dna)[0]) < 0.5 and self._room_for_seedling((x, y)):
                self._sprout(dna, (x, y), generation)
            else:
                waiting.append(seed)
        self.seeds = waiting

    def _sow(self, tufts: int) -> None:
        """Scatter new grass at random where it is medium, the only temperature it grows in, and less
        in a row turning to or from medium, by how medium it is; on and around a corpse, it grows
        CORPSE_FERTILITY times as fast."""
        rows, reach = self._grass_rows, self._grass_reach
        if not rows:
            return
        fertile = {((corpse.x + dx) % self.width, corpse.y + dy)
                   for corpse in self.agents if corpse.died_turn is not None
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1) if 0 <= corpse.y + dy < self.height}
        for _ in range(tufts):
            x = self.random.randrange(self.width)
            at = bisect.bisect_right(reach, self.random.random() * reach[-1])
            cell = (x, rows[min(at, len(rows) - 1)])
            self.grass[cell] = self.grass.get(cell, 0) + (CORPSE_FERTILITY if cell in fertile else 1)

    def _take_census(self) -> None:
        self.census.record(self.turn, (agent.identifier for agent in self.living))

    # Temperament

    def _experience(self, agent: Bitzkey, event: str) -> None:
        """Shift temperament after an experience; intellect sets how far."""
        plasticity = 0.2 + 0.8 * agent.intelligence
        for trait, change in EXPERIENCES[event].items():
            agent.mind[trait] = max(0.0, min(float(TRAIT_MAX), agent.mind[trait] + change * TRAIT_MAX * plasticity))

    @staticmethod
    def _settle(agent: Bitzkey) -> None:
        """Let learned temperament drift slowly back toward the inborn values."""
        for trait in MIND_TRAITS:
            agent.mind[trait] += (agent.inborn[trait] - agent.mind[trait]) * TEMPERAMENT_RELAPSE

    # Decisions and learning

    def _think(self, agent: Bitzkey) -> None:
        """One animal senses, learns from its previous decision's outcome, then chooses and acts."""
        self._think_all([agent])

    def _think_all(self, agents: list[Bitzkey]) -> None:
        """Animals think at once, in phases, so that their brains can be worked as one batch: all
        sense the world as it stands; all their brains value the actions open to them; all learn
        from their previous decisions; all choose; then each acts in turn, finding the world as
        those before it left it. What herd-mates learn from watching is learnt last, all together."""
        # Every hunter first surveys its territory, so that all then sense the same world whatever
        # their order (and the C++ version can sense on every core at once).
        for agent in agents:
            self._survey(agent)
        plans = []
        for agent in agents:  # all sense the world as it stands
            senses, options, situation = self._sense(agent, surveyed=True)
            if agent.last_action in TRAVELLED:
                if situation["threat"]:
                    self._experience(agent, "ambushed")
                elif (situation["grass"] or situation["grass_here"]) and not agent.last_senses[GRASS_SEEN]:
                    self._experience(agent, "discovery")
            plans.append((agent, senses, options, situation))
        values = [agent.brain.values(senses) for agent, senses, _, _ in plans]  # one batch of brains
        comforts = [self._comfort(agent, situation) for agent, _, _, situation in plans]
        for (agent, _, options, _), valued, comfort in zip(plans, values, comforts):
            if agent.last_senses is not None:
                # What came of its last choice: whatever raised its health (fullness counting with it) or
                # its mood was a success, whatever lowered them a failure, by how much; both may happen.
                reward = (agent.reward + self._wellbeing(agent) - agent.last_wellbeing
                          + comfort - agent.last_comfort)
                self._learn(agent, reward + self._discount(agent) * max(valued[option] for option in options),
                            replay=True)
                if reward > 0:  # a lesson learnt from a choice that turned out well
                    self._experience(agent, "lesson")
        actions = []
        for (agent, senses, options, situation), valued, comfort in zip(plans, values, comforts):
            if self._primal_fear(agent, situation["hunter"]):
                # Overcome: it panics, fleeing the hunter, or freezes where it stands, whatever its brain wants.
                if self.random.random() < PRIMAL_FREEZE * (1 - agent.intelligence):
                    situation["frozen"] = True
                    options = [ACTIONS.index("rest")]
                else:
                    situation["panic"] = True
                    options = [ACTIONS.index("flee")]
            elif agent.panicked and situation["assailant"]:
                # Fight or flight, decided by fear: it breaks off and flees its assailant.
                situation["hunter"] = situation["assailant"]
                situation["panic"] = True
                options = [ACTIONS.index("flee")]
            if agent.species == "predator":  # the instinct to hunt, on top of what its brain has learnt
                urge = HUNTING_INSTINCT * min(1.0, agent.hunger / 100)
                valued = [value + urge if action in (HUNT, ATTACK) else value for action, value in enumerate(valued)]
                valued[FEAST] += BLOODLUST  # bloodlust: a fresh kill in sight is hard to pass by
            call = MATING_CALL if agent.species == "predator" else OMNIVORE_MATING_CALL if agent.species == "omnivore" else 0.0
            if call and agent.can_breed:  # the mating call: to breed if it can, else to court a mate it seeks
                valued[BREED if BREED in options else COURT] += call
            valued[SCAVENGE] += CARRION_CRAVING * max(0.0, 2 * agent.physique("carrion") - 1)  # a carrion-eater's craving
            odds = choice_odds(valued, options)
            roll = self.random.random()
            choice = options[-1]
            for option in options:
                roll -= odds[option]
                if roll < 0:
                    choice = option
                    break
            action = ACTIONS[choice]
            if not situation.get("panic") and not situation.get("frozen"):  # a panicked flight, or freezing, was no choice
                self.insights.record(agent.species, agent.profile, (ACTIONS[option] for option in options), action)
            agent.odds = {ACTIONS[option]: chance for option, chance in odds.items()}
            agent.last_senses = senses
            agent.last_choice = choice
            agent.last_wellbeing = self._wellbeing(agent)
            agent.last_comfort = comfort
            agent.reward = 0.0
            agent.assailant = None  # the chance to strike back, or need to flee, passes with this turn's choice
            agent.panicked = False
            agent.alarm = agent.herd_assailant = agent.kill_alert = None
            actions.append((agent, action, situation))
        for agent, action, situation in actions:  # each in turn, in the world as those before it left it
            if agent.alive:
                self._act(agent, action, situation)
        self._learn_witnessed()

    def _survey(self, agent: Bitzkey) -> None:
        """A hunter's view of its territory: how crowded with rivals, and whether its prey are common
        enough to hunt."""
        if agent.species in TERRITORY:
            agent.rivalry, agent.own_rivalry = self._territory(agent)
            agent.hunts_grazers, agent.hunts_omnivores = self._prey_switching(agent)

    def _sense(self, agent: Bitzkey, surveyed: bool = False) -> tuple[list[float], list[int], dict]:
        """What an animal senses, the actions open to it, and its situation, surveying its territory
        first unless it already has this turn."""
        if not surveyed:
            self._survey(agent)
        neighbors = self._nearby(agent, 1)
        mate = next((other for other in self._nearby(agent, MATE_REACH) if self._compatible(agent, other) and other.can_breed),
                    None)  # close enough to breed with
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
        carrion = self._nearest_carrion(agent, sight) if agent.species != "grazer" else None  # grazers don't scavenge
        kill = self._nearest_kill(agent, sight) if agent.species in ("omnivore", "predator") else None
        prey_sight = sight + (PREDATOR_SIGHT if agent.species == "predator" else 0)
        quarry, hunter, suitor, flora = self._look_around(agent, sight, prey_sight)
        if suitor is None and agent.species == "predator" and agent.can_breed:  # the mating call reaches further
            suitor = next((other for other in self._nearby(agent, MATING_CALL_RANGE)
                           if self._compatible(agent, other) and other.can_breed), None)
        assailant = next((other for other in neighbors if other.identifier == agent.assailant), None)
        herd_assailant = next((other for other in neighbors if other.identifier == agent.herd_assailant), None)
        defend_target = assailant or herd_assailant
        herd, herd_near, herd_middle = self._herd(agent)
        here = self._climate[agent.y]
        discomfort = agent.climate_steps(here)
        comfort = self._comfort_row(agent, sight) if discomfort else None
        seen_hunter = hunter  # what it sees itself, to alert its herd to
        alarm_source = None
        if agent.alarm is not None:
            alarm_source = next((other for other in self._nearby(agent, ALARM_REACH) if other.identifier == agent.alarm), None)
            if hunter is None and alarm_source is not None:
                hunter = alarm_source  # the herd's alarm: flight from whatever struck a herd-mate, or it was alerted to
        seen_kill = kill
        alerted_kill = None
        if agent.kill_alert is not None:  # a pack-mate's alert: a fresh kill it may not see itself
            alerted_kill = next((corpse for corpse in self._kills.get(agent.kill_alert, ()) if self._feeds_on(agent, corpse)),
                                None)
            if kill is None:
                kill = alerted_kill

        available = {"rest", "go_east", "go_west", "walk_east", "walk_west"}
        if grass_here:
            available.add("eat")
        if grass:
            available.add("forage")
        if flora:
            available.add("browse")
        if carrion:
            available.add("scavenge")
        if kill:
            available.add("feast")
        if suitor:
            available.add("court")
        if (mate and agent.can_breed and nurseries and self._room() > 0
                and (not CROWDING_LIMIT or self._crowding(agent, mate, own_kind=True) < CROWDING_LIMIT)  # its own kind crowds it
                and (not self._food_needed(agent) or self._food_supply(agent) >= self._food_needed(agent))):
            available.add("breed")
        if quarry:
            available.add("hunt")
        if target:
            available.add("attack")
        if hunter and (alarm_source is not None or abs(hunter.x - agent.x) + abs(hunter.y - agent.y)
                       <= self._flee_range(agent, herd_near)):
            available.add("flee")  # only from a hunter close by, unless the herd raised the alarm
        if defend_target and (agent.species != "grazer" or agent.body["strength"] > GRAZER_DEFENCE_STRENGTH):
            available.add("defend")  # a grazer only when strong enough
        if herd_middle is not None and (abs(herd_middle[0] - agent.x) + abs(herd_middle[1] - agent.y)
                                        > HERD_SPREAD * (1 - agent.fearfulness)):
            # The more fearful it is, the closer it keeps to the herd's middle, but not where the
            # middle is further from its home temperature than here.
            if agent.climate_steps(self._climate[herd_middle[1]]) <= discomfort:
                available.add("herd")
        uprootable = None
        if agent.species == "grazer" and agent.hunger >= UPROOT_HUNGER:
            uprootable = next((other for other in neighbors if other.species == "plant"), None)
            if uprootable is not None and self._food_supply(agent) < UPROOT_FOOD:
                available.add("uproot")  # starving, with food scarce: it may tear into a plant beside it
        alert_to = None
        if herd:  # with a herd to tell: a grazer of a hunter it sees, a hunter of a fresh kill it sees
            if agent.species == "grazer" and seen_hunter is not None:
                alert_to = seen_hunter
            elif agent.species in ("omnivore", "predator") and seen_kill is not None:
                alert_to = seen_kill
        if alert_to is not None:
            available.add("alert")
        if agent.y > 0:
            available.update(("go_north", "walk_north"))
        if agent.y < self.height - 1:
            available.update(("go_south", "walk_south"))

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
            prey_closeness=self._closeness(agent, quarry and (quarry.x, quarry.y), prey_sight),
            prey_pregnant=float(quarry is not None and bool(quarry.womb)),
            threat_near=min(4, len(threats)) / 4,
            threat_danger=0.0 if threat is None else min(1.0, self._damage(threat, agent) / agent.health),
            threat_closeness=self._closeness(agent, hunter and (hunter.x, hunter.y), sight),
            kin_near=min(3, kin) / 3,
            others_near=min(3, others) / 3,
            carrion_closeness=self._closeness(agent, carrion and (carrion.x, carrion.y), sight),
            kill_closeness=self._closeness(agent, kill and (kill.x, kill.y), sight),
            plant_closeness=self._closeness(agent, flora and (flora.x, flora.y), sight),
            assailant_near=float(assailant is not None),
            rivalry=agent.rivalry,
            counter_frailty=(0.0 if defend_target is None
                             else min(1.0, self._damage(agent, defend_target) / defend_target.health)),
            herd_near=min(3, herd_near) / 3,
            herd_closeness=(0.0 if herd_middle is None else
                            max(0.0, 1 - (abs(herd_middle[0] - agent.x) + abs(herd_middle[1] - agent.y)) / HERD_RADIUS)),
            alarm=float(agent.alarm is not None),
            kill_alert=float(alerted_kill is not None),
            herd_success=agent.seen_success,
            herd_failure=agent.seen_failure,
            climate_discomfort=min(2, discomfort) / 2,
            comfort_closeness=1.0 if not discomfort else self._closeness(agent, comfort, sight),
            age=min(1.0, agent.age / 400),
            **self._time_of_year(),
            latitude=2 * (agent.y + 0.5) / self.height - 1,
            warmth=self._warmth(agent, agent.y),
            warmth_north=self._warmth(agent, max(0, agent.y - sight)),
            warmth_south=self._warmth(agent, min(self.height - 1, agent.y + sight)),
            **{trait: agent.feeling(trait) for trait in MIND_TRAITS},
            **agent.genes,
        )
        situation = {"mate": mate, "nurseries": nurseries, "prey": target, "threat": threat,
                     "threats": len(threats), "quarry": quarry, "hunter": hunter, "suitor": suitor,
                     "grass": grass, "grass_here": grass_here, "carrion": carrion, "kill": kill, "flora": flora, "assailant": assailant, "rivalry": agent.rivalry, "kin": kin,
                     "others": others, "defend_target": defend_target, "for_herd": assailant is None,
                     "herd": herd, "herd_near": herd_near, "herd_middle": herd_middle,
                     "exposure": min(2, discomfort) / 2 * agent.exposure_share(here), "uprootable": uprootable,
                     "alert_to": alert_to}
        return senses, [ACTIONS.index(action) for action in ACTIONS if action in available], situation

    def _wellbeing(self, agent: Bitzkey) -> float:
        """How comfortable an animal feels; appetite sets how much hunger hurts."""
        appetite = 0.5 + agent.feeling("appetite")
        return (-appetite * min(100.0, agent.hunger) / 100
                - (1 - max(0.0, agent.health) / agent.max_health))

    @staticmethod
    def _comfort(agent: Bitzkey, situation: dict) -> float:
        """Company pleases the sociable, and the tolerant count other species as company.
        Hunters close by frighten the fearful, less so the tolerant and those among their herd; the
        intolerant are uneasy around anything unlike them. Herd animals enjoy their herd's company
        and are uneasy with none of it in range, the more so the more fearful they are. Out of its
        home temperature, an animal is uneasy, the more so the further, and less if hardy to it."""
        tolerance = agent.feeling("tolerance")
        instinct = agent.herd_instinct
        herd_near = min(3, situation.get("herd_near", 0)) / 3
        company_around = min(3.0, situation["kin"] + tolerance * situation["others"])
        company = COMPANY_JOY * agent.feeling("sociability") * company_around / 3
        dread = (FEAR_DREAD * agent.fearfulness * (1 - 0.5 * tolerance) * min(2, situation["threats"]) / 2
                 * (1 - HERD_SAFETY * instinct * herd_near))
        unease = UNEASE * (1 - tolerance) * min(3, situation["others"]) / 3
        crowded = RIVAL_UNEASE * (1 - 0.5 * tolerance) * situation.get("rivalry", 0.0)
        herding = HERD_JOY * instinct * herd_near
        if not situation.get("herd", True):
            herding -= HERD_LONELINESS * instinct
        herding *= 0.5 + agent.fearfulness  # the fearful feel their herd, and its absence, most
        exposed = CLIMATE_UNEASE * situation.get("exposure", 0.0)
        return company - dread - unease - crowded + herding - exposed

    @staticmethod
    def _discount(agent: Bitzkey) -> float:
        """How much the future counts against the present: more for the patient."""
        return 0.7 + 0.27 * agent.feeling("patience")

    @staticmethod
    def _learning_rate(agent: Bitzkey) -> float:
        """How fast its brain learns: with its intellect, boosted further the cleverer it is, and a
        predator's PREDATOR_LEARNING times as fast, an omnivore's with half that bonus."""
        sharpness = agent.intelligence
        hunter = {"predator": PREDATOR_LEARNING, "omnivore": 1 + (PREDATOR_LEARNING - 1) / 2}.get(agent.species, 1.0)
        return LEARNING_RATE * sharpness * (1 + LEARNING_BOOST * sharpness) * hunter

    def _learn(self, agent: Bitzkey, target: float, replay: bool = False) -> None:
        """Learn from its own last choice, by TD(lambda), so the choices before it share in what came of it.
        With `replay`, it then re-learns from its most recent experiences, as many as its intellect
        allows, and remembers this one."""
        rate = self._learning_rate(agent)
        agent.brain.learn_traced(agent.last_senses, agent.last_choice, target, rate * food_learning(agent.last_choice),
                                 self._discount(agent), agent.intelligence)
        if not replay:
            return
        count = REPLAY_PER_INTELLECT * thinking_intellect(agent.inborn["intellect"], agent.species)
        for senses, choice, remembered in agent.memories[-count:] if count else ():  # the latest, up to count
            agent.brain.learn(senses, choice, remembered, rate * food_learning(choice))
        if count:
            agent.memories.append((agent.last_senses, agent.last_choice, target))
            del agent.memories[:-count]

    def _dies_of_age(self, agent: Bitzkey) -> bool:
        """Past AGEING of its lifespan, the chance of dying each turn grows with age, so that about
        half live to their lifespan and almost none beyond 1.3 times it."""
        onset = AGEING * agent.lifespan
        if agent.age <= onset:
            return False
        steepness = 2 * math.log(2) / (agent.lifespan - onset) ** 2
        return self.random.random() < steepness * (agent.age - onset)

    def _die(self, agent: Bitzkey, cause: str, killer: Bitzkey | None = None) -> None:
        """Record a death. The body becomes carrion holding the energy it had left: a starved body
        leaves nothing, and a killed one half, as the killer has already eaten. Dying of old age
        is not held against the brain's last choice."""
        self.death_records[agent.identifier] = (cause, None if killer is None else killer.identifier,
                                                None if killer is None else killer.species)
        if agent.brain is not None and agent.last_senses is not None:
            penalty = 0.0 if cause == "old age" else DEATH_PENALTY
            self._learn(agent, agent.reward + self._wellbeing(agent) - agent.last_wellbeing - penalty)
            if cause != "old age":  # its herd sees it fail
                self._witness(agent, -DEATH_PENALTY)
        agent.brain = None
        agent.womb, agent.due_turn = [], None  # the young it carried die with it
        agent.health = min(agent.health, 0.0)  # whatever the cause, a recorded death leaves it dead
        agent.died_turn = self.turn
        agent.last_action = cause
        nourished = 0.0 if agent.species == "plant" else max(0.0, 1 - agent.hunger / 100)
        agent.meat = ((CARRION_MEAT + CARRION_MEAT_FOR_SIZE * agent.physique("size")) * nourished
                      * (0.5 if cause == "killed" else 1))
        agent.rot = agent.meat / max(1, CORPSE_TURNS - FRESH_KILL_TURNS)  # it rots once no longer fresh
        agent.portions = 0 if agent.species == "plant" else 1 + int(agent.body["size"])
        if agent.species != "plant":
            self._kills.setdefault((agent.x, agent.y), []).append(agent)
        self.deaths += 1
        causes = self.death_causes.setdefault(agent.species, {})
        causes[cause] = causes.get(cause, 0) + 1
        if agent.species == "plant":
            self._plants -= 1
        else:
            self._population -= 1

    # Actions

    def _act(self, agent: Bitzkey, action: str, situation: dict) -> None:
        if action == "eat":
            cell = (agent.x, agent.y)
            if not self.grass.get(cell):  # another got here first
                agent.last_action = "found it eaten"
                return
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
        elif action == "uproot":
            self._uproot(agent, situation["uprootable"])
        elif action == "feast":
            self._feast(agent, situation["kill"])
        elif action == "scavenge":
            self._scavenge(agent, situation["carrion"])
        elif action == "rest" and situation.get("frozen"):
            agent.last_action = "froze"  # paralysed by fear: nothing done, nothing gained
        elif action == "rest":
            agent.hunger = max(0, agent.hunger - agent.burn / 2)  # burns less, but eases no hunger
            agent.heal(0.015)
            agent.last_action = "rested"
        elif action == "court":
            suitor = situation["suitor"]
            self._move_toward(agent, suitor.x, suitor.y)
            agent.last_action = "courted"
        elif action == "breed":
            mate = situation["mate"]
            if not (mate.alive and mate.can_breed and agent.can_breed
                    and abs(mate.x - agent.x) + abs(mate.y - agent.y) <= MATE_REACH):
                agent.last_action = "was spurned"  # its mate died, moved off or bred with another first
                return
            agent.last_action = "mated"  # before breeding, so a parent that dies of it says so
            self._breed(agent, mate)
        elif action == "hunt":
            quarry = situation["quarry"]
            self._move_toward(agent, quarry.x, quarry.y,
                              agent.steps + (HUNT_STEPS if agent.species == "predator" else 0))  # predators chase further
            agent.last_action = "hunted"
            if quarry.alive and abs(quarry.x - agent.x) + abs(quarry.y - agent.y) <= 1:
                self._attack(agent, quarry)
                if agent.alive:
                    agent.last_action = "pounced"
        elif action == "attack":
            prey = situation["prey"]
            if not (prey.alive and self._beside(agent, prey)):
                agent.last_action = "missed"  # it fell or got away first
                return
            self._attack(agent, prey)
        elif action == "defend":
            target = situation["defend_target"]
            if not (target.alive and self._beside(agent, target)):
                agent.last_action = "missed"
                return
            self._defend(agent, target, for_herd=situation["for_herd"])
        elif action == "herd":
            self._move_toward(agent, *situation["herd_middle"])
            agent.last_action = "herded"
        elif action == "alert":
            self._alert(agent, situation["alert_to"])
        elif action in HEADINGS:
            dx, dy, walk, went = HEADINGS[action]
            cells = 1 if walk else agent.steps
            y = self._within(agent.y + dy * cells)
            moved = abs(y - agent.y) if dy else cells  # north and south stop at the map's edge
            self._move(agent, (agent.x + dx * cells) % self.width, y)
            if walk:
                agent.hunger = max(0, agent.hunger - agent.burn * WALK_REST)  # burns less, but eases no hunger
            else:
                agent.hunger += agent.burn * GO_EFFORT * max(0, moved - 1)
                self._exhausted(agent)
            agent.reward += CURIOSITY_JOY * agent.feeling("curiosity")
            agent.last_action = went
        elif action == "flee":
            self._move_away(agent, situation["hunter"])
            agent.last_action = "panicked" if situation.get("panic") else "fled"

    def _browse(self, agent: Bitzkey, plant: Bitzkey) -> None:
        """Go to the plant and, once alongside, take a bite: far more filling than grass, and more
        so from a bigger plant. Each bite wounds the plant, and enough of them kill it. A bitten
        plant can't be bitten again until it has recovered."""
        self._move_toward(agent, plant.x, plant.y)
        if not plant.alive or abs(plant.x - agent.x) + abs(plant.y - agent.y) > 1 or plant.cooldown:
            agent.last_action = "sought plants"
            return
        agent.nourish((PLANT_MEAL + PLANT_MEAL_FOR_SIZE * plant.physique("size")) * agent.browsing * self._plant_yield(plant))
        plant.health -= PLANT_BITE
        plant.cooldown = PLANT_RECOVERY
        agent.last_action = "browsed"
        if not plant.alive:
            self._die(plant, "eaten", agent)
        elif plant.physique("weapons") > 0:
            self._defend(plant, agent)  # a plant can't flee, but an armed one strikes back

    def _uproot(self, agent: Bitzkey, plant: Bitzkey) -> None:
        """A starving grazer, with food scarce about it, tears into a plant beside it, recovering from
        a bite or not: it eats UPROOT_MEALS of a feast on a body of the plant's size, and the plant
        loses UPROOT_BITE health, often all it has. An armed plant still strikes back."""
        if not plant.alive:  # another got to it first
            agent.last_action = "sought plants"
            return
        feast = FRESH_KILL_MEALS * (KILL_MEAL + KILL_MEAL_FOR_SIZE * plant.physique("size"))
        agent.nourish(UPROOT_MEALS * feast * self._plant_yield(plant))
        plant.health -= UPROOT_BITE
        plant.cooldown = PLANT_RECOVERY
        agent.last_action = "uprooted"
        if not plant.alive:
            self._die(plant, "uprooted", agent)
        elif plant.physique("weapons") > 0:
            self._defend(plant, agent)

    def _alert(self, agent: Bitzkey, about: Bitzkey) -> None:
        """Alert its herd within HERD_RADIUS: a grazer to the hunter it sees, so they may flee it from
        further than they would; a hunter to the fresh kill it sees, so they may come to feast on it."""
        for other in self._nearby(agent, HERD_RADIUS):
            if self._herd_mate(agent, other):
                if agent.species == "grazer":
                    if other.alarm is None:
                        other.alarm = about.identifier
                else:
                    other.kill_alert = (about.x, about.y)
        agent.reward += ALERT_JOY
        agent.last_action = "alerted"

    def _scavenge(self, agent: Bitzkey, corpse: Bitzkey) -> None:
        """Go to the carrion and, once alongside, eat a portion, its share of the meat left; a
        carrion-eater's gut gets the most from it. The last portion eaten, the body is gone."""
        self._move_toward(agent, corpse.x, corpse.y)
        if abs(corpse.x - agent.x) + abs(corpse.y - agent.y) > 1 or corpse.portions <= 0:
            agent.last_action = "sought carrion"
            return
        bite = corpse.meat / corpse.portions
        corpse.meat -= bite
        agent.nourish(bite * agent.digestion)
        agent.last_action = "scavenged"
        self._eat_portion(corpse)

    def _feast(self, agent: Bitzkey, corpse: Bitzkey) -> None:
        """Go to the fresh kill and, once alongside, eat a portion of it: a whole feast, FRESH_KILL_MEALS
        times the meal of making the kill, so a large body is many feasts. It isn't scavenging; the carrion
        left shrinks by the portion's share. The last portion eaten, the body is gone."""
        self._move_toward(agent, corpse.x, corpse.y)
        if (abs(corpse.x - agent.x) + abs(corpse.y - agent.y) > 1 or not self._fresh(corpse) or corpse.meat <= 0
                or corpse.portions <= 0):
            agent.last_action = "sought a kill"  # out of reach, or already eaten
            return
        agent.nourish(FRESH_KILL_MEALS * (KILL_MEAL + KILL_MEAL_FOR_SIZE * corpse.physique("size")))
        if agent.species == "predator":
            agent.reward += FEAST_JOY
        corpse.meat -= corpse.meat / corpse.portions
        agent.last_action = "feasted"
        self._eat_portion(corpse)

    def _eat_portion(self, corpse: Bitzkey) -> None:
        """A portion of a body eaten: once the last is, the body is gone at the turn's end."""
        corpse.portions -= 1
        if corpse.portions > 0:
            return
        corpse.meat = 0.0
        corpse.died_turn = self.turn - CORPSE_TURNS
        for found in (self._kills, self._carrion):
            bodies = found.get((corpse.x, corpse.y))
            if bodies and corpse in bodies:
                bodies.remove(corpse)

    def _fresh(self, corpse: Bitzkey) -> bool:
        """Whether a body is still a fresh kill, not yet carrion."""
        return corpse.died_turn is not None and self.turn - corpse.died_turn <= FRESH_KILL_TURNS

    @staticmethod
    def _damage(attacker: Bitzkey, defender: Bitzkey, defending: bool = False) -> float:
        """One blow: the attacker's might, sharpened by weapons, blunted by the defender's toughness;
        size counts strongly in both. A plant's thorns are mostly its weapons, a little its size,
        and owe nothing to strength. Struck in defence, a grazer's or omnivore's weapons count for less."""
        if attacker.species == "plant":
            thorns = THORN_SIZE_SHARE * attacker.physique("size") + (1 - THORN_SIZE_SHARE) * attacker.physique("weapons")
            return THORN_FORCE * thorns * (1 - 0.6 * defender.toughness)
        force = STRIKE_FORCE * (2 + 9 * attacker.might)
        armed = Simulation._armed(attacker) * (Simulation._defence_share(attacker) if defending else 1.0)
        return force * (1 + 0.75 * armed) * (1 - 0.6 * defender.toughness)

    @staticmethod
    def _defence_share(animal: Bitzkey) -> float:
        """The share of its weapons an animal brings to defence."""
        return {"grazer": GRAZER_DEFENCE_WEAPONS, "omnivore": OMNIVORE_DEFENCE_WEAPONS}.get(animal.species, 1.0)

    @staticmethod
    def _armed(animal: Bitzkey) -> float:
        """Its natural weapons as a share of their full range: a predator's count PREDATOR_WEAPONS times over,
        an omnivore's three quarters and a grazer's half as many times."""
        share = {"predator": 1.0, "omnivore": 0.75, "grazer": 0.5}.get(animal.species)
        return animal.physique("weapons") * (PREDATOR_WEAPONS * share if share else 1.0)

    def _attack(self, agent: Bitzkey, victim: Bitzkey) -> None:
        """A blow. The intolerant enjoy fighting other species most; the empathetic feel their
        victim's pain, and so does any empathetic animal watching, most for its own kind."""
        empathy = agent.feeling("empathy")
        joy = FIGHT_JOY * agent.feeling("aggression") * (1 - 0.6 * empathy) * (1 - 0.75 * agent.feeling("tolerance"))
        damage = self._damage(agent, victim)
        if any(self._herd_mate(agent, other) for other in self._nearby(victim, 1) if other is not agent):
            damage *= 1 + PACK_BONUS * agent.herd_instinct  # hunting as a pack
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
            meal = KILL_MEAL + KILL_MEAL_FOR_SIZE * victim.physique("size")
            agent.nourish(meal)
            agent.reward += 2 * joy
            self._witness(agent, meal / agent.stomach / 100)  # its pack sees it succeed: the fullness it gained
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
            self._die(victim, "killed", agent)

        sting = 4.5 * self._armed(victim) * self._defence_share(victim) * (1 - 0.6 * agent.toughness)
        if sting > 0:
            agent.health -= sting
            self._experience(agent, "stung")
            if not agent.alive:
                self._die(agent, "killed", victim)  # its victim's spines
        self._exhausted(agent)
        if agent.species == "predator" and agent.alive:  # a blow it lived through
            agent.reward += ATTACK_JOY

    def _exhausted(self, agent: Bitzkey) -> None:
        """A blow's effort, or a long way gone, can leave an animal starved: it dies there, rather than
        lingering as a body whose death was never recorded."""
        if not agent.alive and agent.died_turn is None:
            self._die(agent, "exhausted")

    def _defend(self, agent: Bitzkey, assailant: Bitzkey, for_herd: bool = False) -> None:
        """Strike back at the animal that just struck it, or a herd-mate alongside it, or, for a
        plant, browsed it, with the usual force of might and weapons against the assailant's
        toughness. The blow costs effort like any other, and feeds the defender nothing, but it
        can kill, and a struck assailant must check its own morale. Defending its herd pleases a
        herd animal."""
        damage = self._damage(agent, assailant, defending=True)
        assailant.health -= damage
        agent.hunger += STRIKE_EFFORT * damage / agent.stomach
        self._experience(assailant, "stung")
        if agent.species != "plant":
            agent.reward += FIGHT_JOY * agent.feeling("aggression") * (1 - 0.6 * agent.feeling("empathy"))
            if for_herd:
                agent.reward += HERD_DEFEND_JOY * agent.herd_instinct
            agent.last_action = "defended"
            self._experience(agent, "defended")
        if not assailant.alive:
            self._die(assailant, "killed", agent)
        elif assailant.species != "plant":
            self._struck(assailant, agent)
        self._exhausted(agent)

    def _struck(self, victim: Bitzkey, striker: Bitzkey) -> None:
        """A struck animal may strike back on its next turn, unless it fails a morale check
        against its current fear, more likely the more hurt it is and less among its herd, and so
        must flee instead. Its herd-mates close by are alarmed, and those alongside the striker may
        defend it."""
        victim.assailant = striker.identifier
        herd_near = 0
        for other in self._nearby(victim, 2):
            if other is striker or other.brain is None:
                continue
            herd_near += self._herd_mate(victim, other)
            if self._herd_mate(other, victim):
                other.alarm = striker.identifier
                if abs(other.x - striker.x) + abs(other.y - striker.y) <= 1:
                    other.herd_assailant = striker.identifier
        unhurt, dying = PANIC
        hurt = 1 - max(0.0, victim.health) / victim.max_health
        steadied = 1 - HERD_SAFETY * victim.herd_instinct * min(3, herd_near) / 3
        if self.random.random() < victim.fearfulness * (unhurt + (dying - unhurt) * hurt) * steadied:
            victim.panicked = True
            self._experience(victim, "panicked")

    def _primal_fear(self, agent: Bitzkey, hunter: Bitzkey | None) -> bool:
        """Whether an animal is in primal fear this turn. With no hunter in sight it is calm again. Facing
        one stronger than itself, it may be overcome, more likely the more lopsided the fight; a predator a
        third as often. Once overcome, it stays so while a hunter is in sight."""
        if hunter is None or agent.species == "plant":
            agent.primal_fear = False
            return False
        if not agent.primal_fear:
            disparity = self.disparity(agent, hunter)
            if disparity > 0:
                chance = (PRIMAL_FEAR * disparity * (PRIMAL_FEAR_PREDATORS if agent.species == "predator" else 1.0)
                          * agent.unafraid)
                if self.random.random() < chance:
                    agent.primal_fear = True
                    self._experience(agent, "panicked")
        return agent.primal_fear

    @classmethod
    def disparity(cls, agent: Bitzkey, opponent: Bitzkey) -> float:
        """How much stronger an opponent is, 0 (no stronger) to 1 (overwhelmingly): weighing the share of
        each other's health one blow takes, with every trait that counts in a fight."""
        mine = cls._looming_damage(agent, opponent) / max(1.0, cls._looming_health(opponent))
        theirs = cls._looming_damage(opponent, agent) / max(1.0, cls._looming_health(agent))
        return 0.0 if theirs <= mine else 1 - mine / theirs

    @staticmethod
    def _looming(animal: Bitzkey) -> tuple[float, float]:
        """Its size and strength, as shares of their full range, as they count in sizing up a fight: a
        predator's PREDATOR_LOOMS steps more of each (past the full range, for the largest and strongest)."""
        more = PREDATOR_LOOMS / TRAIT_MAX if animal.species == "predator" else 0.0
        return animal.physique("size") + more, animal.physique("strength") + more

    @classmethod
    def _looming_damage(cls, attacker: Bitzkey, defender: Bitzkey) -> float:
        """One blow, as _damage, with each side's size and strength as they count in sizing up a fight."""
        size, strength = cls._looming(attacker)
        defender_size, _ = cls._looming(defender)
        might = (1 - SIZE_IN_MIGHT) * strength + SIZE_IN_MIGHT * size
        toughness = (1 - SIZE_IN_MIGHT) * defender.physique("armour") + SIZE_IN_MIGHT * defender_size
        return STRIKE_FORCE * (2 + 9 * might) * (1 + 0.75 * cls._armed(attacker)) * (1 - 0.6 * toughness)

    @classmethod
    def _looming_health(cls, animal: Bitzkey) -> float:
        """Its health, as it counts in sizing up a fight: as large a share of the larger health a larger
        body has."""
        size, _ = cls._looming(animal)
        return animal.health * (60.0 + 90 * size) / animal.max_health

    def _breed(self, first: Bitzkey, second: Bitzkey) -> None:
        """Fertile parents breed cheaply and often. The parent that chose to breed conceives a
        litter, LITTER_RANGE young (bigger for a high litter trait and small bodies, smaller where
        animals crowd around), and carries it for its gestation before giving birth. The young
        share the energy both parents give up now. Animals only choose to breed when both can
        afford it, but a parent can only give what it has left. Each child inherits a mix of its
        parents' brains, and starts its temperament at the average of its parents' current
        temperaments, which its dot then shows."""
        litter = litter_size([(int(parent.body["litter"]), int(parent.body["size"])) for parent in (first, second)])
        if CROWDING_LIMIT:
            litter = max(1, int(litter * (1 - self._crowding(first, second, own_kind=True) / CROWDING_LIMIT) + 0.5))  # its own kind
        needed = self._food_needed(first)
        if needed:  # litters follow food only while food limits breeding
            litter = max(1, int(litter * min(1.0, self._food_supply(first) / (2 * needed)) + 0.5))
        litter = max(1, min(litter, self._room()))
        costs = [parent.breeding_cost for parent in (first, second)]
        # Energy in absolute terms: a share of each parent's stomach, refilled in the young's.
        given = sum(min(cost, max(0.0, 100 - parent.hunger)) * parent.stomach
                    for cost, parent in zip(costs, (first, second)))
        generation = max(first.generation, second.generation) + 1
        for _ in range(litter):
            # The child's mind starts at the average of its parents' current minds, mutated, and
            # its dot is written from that mind, so experience is inherited and shows. Its brain,
            # a mix of its parents', is as deep as its own inborn intellect makes it.
            mind = inherit_mind(first.mind, second.mind, self.random, self.mind_mutation)
            dna = evolved(first.dna, cross(first.dna, second.dna, self.random, mind, body_chance=self.body_mutation))
            if "predator" in (first.species, second.species):
                dna = predator_dna(dna)  # a predator's young are predators, whatever crossing over or a mutation made
            brain = Brain.inherit(first.brain, second.brain, self.random,
                                  depth_for(thinking_intellect(decode_mind(dna)["intellect"], decode_body(dna)["species"])))
            if decode_body(dna)["species"] == "predator" and first.species != "predator":
                # Born a predator of other parents: what they learnt of eating and uprooting it brings to
                # hunting and feasting.
                for learnt, now in ((EAT, HUNT), (UPROOT, FEAST)):
                    brain.output_weights[now] = brain.output_weights[learnt].copy()
                    brain.output_biases[now] = brain.output_biases[learnt]
            first.womb.append((dna, mind, brain, given / litter, generation))
        first.due_turn = self.turn + gestation(int(first.body["size"]))
        for parent, cost in zip((first, second), costs, strict=True):
            parent.hunger += cost
            rest = PREDATOR_BREED_COOLDOWN if parent.species == "predator" else BREED_COOLDOWN
            parent.cooldown = round((20 - 15 * parent.physique("fertility")) * rest)
            parent.children += litter
            parent.reward += BREED_JOY * parent.feeling("libido")
            if not parent.alive:
                self._die(parent, "bred to death")
            else:
                self._experience(parent, "bred_into_hunger" if parent.hunger >= 75 else "bred")

    # Space

    def _index_cells(self) -> None:
        self._cells = {}
        self._carrion = {}
        self._kills = {}
        for agent in self.agents:
            if agent.alive:
                self._cells.setdefault((agent.x, agent.y), []).append(agent)
            elif self._fresh(agent):
                if agent.species != "plant":
                    self._kills.setdefault((agent.x, agent.y), []).append(agent)
            elif agent.meat > 0:
                self._carrion.setdefault((agent.x, agent.y), []).append(agent)

    def _nearby(self, agent: Bitzkey, radius: int) -> list[Bitzkey]:
        found = []
        cells, x, y = self._cells, agent.x, agent.y
        for dx, dy in nearby_offsets(radius):
            for other in cells.get((x + dx, y + dy), ()):
                if other is not agent and other.alive:
                    found.append(other)
        return found

    def _room(self) -> float:
        """How many more animals the population cap allows; unlimited without a cap."""
        return float("inf") if self.population_cap is None else self.population_cap - self._population

    def _crowding(self, agent: Bitzkey, partner: Bitzkey, own_kind: bool = False) -> int:
        """How many other animals, besides its partner, live within CROWDING_RADIUS cells; with `own_kind`,
        only those of its own kind."""
        return sum(other.species != "plant" and other is not partner and (not own_kind or self._same_kind(agent, other))
                   for other in self._nearby(agent, CROWDING_RADIUS))

    def _give_birth(self, mother: Bitzkey) -> None:
        """The young it carries are born on the free cells nearest it, as many as there are cells
        and the population cap allows; the rest are lost. Each starts with its share of the energy
        its parents gave."""
        cells = self._free_cells_near(mother)
        born = int(min(len(mother.womb), len(cells), max(0, self._room())))
        for (dna, mind, brain, energy, generation), cell in zip(mother.womb[:born], cells):
            child = self._add_agent(dna, *cell, brain=brain)
            child.mind = mind
            child.hunger = max(0.0, 100 - energy / child.stomach)
            child.generation = generation
            self._population += 1
        mother.womb, mother.due_turn = [], None

    def _free_cells_near(self, agent: Bitzkey) -> list[tuple[int, int]]:
        """The empty cells around an animal, up to BIRTH_REACH away and nearest first, where young
        can be born."""
        cells = []
        for dx, dy in birth_offsets(BIRTH_REACH):
            if not 0 <= agent.y + dy < self.height:  # beyond the top or bottom of the map
                continue
            cell = ((agent.x + dx) % self.width, agent.y + dy)
            if not any(other.alive for other in self._cells.get(cell, ())):
                cells.append(cell)
        return cells

    def _nearest_grass(self, agent: Bitzkey, sight: int) -> tuple[int, int] | None:
        """The nearest grass in sight, but none in a row further from its home temperature than here."""
        grass, climate, x, y = self.grass, self._climate, agent.x, agent.y
        worst = agent.climate_steps(climate[y])
        for dx, dy in SIGHT_WITHIN[sight]:
            cell = (x + dx, y + dy)
            if grass.get(cell, 0) > 0 and agent.climate_steps(climate[y + dy]) <= worst:
                return cell
        return None

    def _warmth(self, agent: Bitzkey, y: int) -> float:
        """How much warmer than home a row feels, as a sense: 0.5 a step, from -1 (two or more steps
        too cold) to 1 (two or more too hot)."""
        return max(-2, min(2, agent.climate[0] - self._climate[y])) / 2

    def _nearest_carrion(self, agent: Bitzkey, sight: int) -> Bitzkey | None:
        if not self._carrion:
            return None
        carrion, x, y = self._carrion, agent.x, agent.y
        for dx, dy in SEEN_WITHIN[sight]:
            for corpse in carrion.get((x + dx, y + dy), ()):
                if corpse.meat > 0:
                    return corpse
        return None

    def _nearest_kill(self, agent: Bitzkey, sight: int) -> Bitzkey | None:
        """The nearest fresh kill in sight of something it lives on."""
        if not self._kills:
            return None
        kills, x, y = self._kills, agent.x, agent.y
        for dx, dy in SEEN_WITHIN[sight]:
            for corpse in kills.get((x + dx, y + dy), ()):
                if self._feeds_on(agent, corpse):
                    return corpse
        return None

    def _look_around(self, agent: Bitzkey, sight: int,
                     prey_sight: int | None = None) -> tuple[Bitzkey | None, Bitzkey | None, Bitzkey | None, Bitzkey | None]:
        """The nearest prey, hunter, willing mate and edible plant in sight, for the roles this
        animal can have; prey as far as `prey_sight`, when it sees prey further."""
        prey_sight = sight if prey_sight is None else prey_sight
        quarry = hunter = suitor = flora = None
        hunts = agent.species in ("omnivore", "predator")
        hunted = agent.species != "plant"
        browses = agent.species in ("grazer", "omnivore")
        courts = agent.can_breed
        cells, x, y = self._cells, agent.x, agent.y
        for dx, dy in SEEN_WITHIN[max(sight, prey_sight)]:
            far = abs(dx) + abs(dy) > sight  # beyond its sight but for prey
            if ((quarry or not hunts) and (far or ((hunter or not hunted)
                                                   and (suitor or not courts) and (flora or not browses)))):
                break
            for other in cells.get((x + dx, y + dy), ()):
                if other is agent or not other.alive:
                    continue
                if quarry is None and self._is_prey(agent, other):
                    quarry = other
                elif far:
                    continue
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

    def _within(self, y: int) -> int:
        """A row kept on the map: the top and bottom edges are barriers, though east and west wrap
        round."""
        return max(0, min(self.height - 1, y))

    def _move(self, agent: Bitzkey, x: int, y: int) -> None:
        # Each cell left behind on the way, the shorter way round east or west; a move goes straight
        # or diagonally, so every cell on the way is a whole one.
        dx, dy = x - agent.x, y - agent.y
        if dx > self.width // 2:
            dx -= self.width
        elif dx < -(self.width // 2):
            dx += self.width
        cells = max(abs(dx), abs(dy))
        agent.trail.extend(((agent.x + dx * step // cells) % self.width, agent.y + dy * step // cells)
                           for step in range(cells))
        self._cells[(agent.x, agent.y)].remove(agent)
        agent.x, agent.y = x, y
        self._cells.setdefault((x, y), []).append(agent)

    def _move_toward(self, agent: Bitzkey, x: int, y: int, steps: int | None = None) -> None:
        for _ in range(agent.steps if steps is None else steps):
            if agent.x != x:
                self._move(agent, agent.x + (1 if x > agent.x else -1), agent.y)
            elif agent.y != y:
                self._move(agent, agent.x, agent.y + (1 if y > agent.y else -1))

    def _move_away(self, agent: Bitzkey, threat: Bitzkey) -> None:
        for _ in range(agent.steps):
            dx = 1 if agent.x >= threat.x else -1
            dy = 1 if agent.y >= threat.y else -1
            self._move(agent, (agent.x + dx) % self.width, self._within(agent.y + dy))

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
        ring, dot = agent.looks_apart(other)
        accepted = KIND_DIFFERENCE if agent.species == "predator" else MATE_RING_DIFFERENCE  # a predator its kind
        return (ring <= accepted + (1 - accepted) * tolerance
                and dot <= MATE_DOT_DIFFERENCE + (1 - MATE_DOT_DIFFERENCE) * tolerance)

    @staticmethod
    def _same_kind(first: Bitzkey, second: Bitzkey) -> bool:
        """Of one kind: the same species, and, for predators, ring colours within KIND_DIFFERENCE."""
        return first.species == second.species and (
            first.species != "predator" or first.looks_apart(second)[0] <= KIND_DIFFERENCE)

    @staticmethod
    def cannibal_reluctance(predator: Bitzkey) -> float:
        """How hungry, as a share, a predator must be before it will hunt its own kind: the
        empathetic feel for them, the tolerant bear them, and the patient can wait for other prey."""
        return (predator.feeling("empathy") + predator.feeling("tolerance") + predator.feeling("patience")) / 3

    @staticmethod
    def _rival_weight(hunter: Bitzkey, other: Bitzkey) -> float:
        """How much another animal counts as a rival for a hunter's feeding ground: only its own
        species, predators for a predator and omnivores for an omnivore. Other species are prey."""
        return 1.0 if hunter.species in ("predator", "omnivore") and other.species == hunter.species else 0.0

    @staticmethod
    def _feeds_on(eater: Bitzkey, other: Bitzkey) -> bool:
        """Whether an animal counts another as a food source when judging whether to breed:
        predators live on grazers and omnivores, and omnivores on grazers. Other predators, of
        any kind, are rivals for that prey rather than a supply of food, though they may be hunted."""
        if eater.species == "predator":
            return other.species in ("grazer", "omnivore")
        return eater.species == "omnivore" and other.species == "grazer"

    @classmethod
    def _diet_share(cls, eater: Bitzkey, other: Bitzkey) -> float:
        """How much another animal counts as a mouth on an animal's food: fully for its own species
        (every predator, whatever its kind), by half for another species eating some of the same food."""
        if other.species == "plant" or cls._feeds_on(eater, other):
            return 0.0
        if other.species == eater.species:
            return 1.0
        grazers = ("grazer", "omnivore")
        if eater.species in grazers and other.species in grazers:
            return SHARED_DIET  # both eat grass and plants
        if eater.species == "omnivore" and other.species == "predator":
            return SHARED_DIET  # both hunt grazers
        return 0.0

    def _plant_yield(self, plant: Bitzkey) -> float:
        """The share of its food a plant gives: less out of its temperature."""
        return 1 - (1 - PLANT_CLIMATE_YIELD) * min(1.0, plant.climate_steps(self._climate[plant.y]))

    def _comfort_row(self, agent: Bitzkey, sight: int) -> tuple[int, int] | None:
        """The nearest cell straight up or down, within sight, where it would feel at home: north
        before south at the same distance."""
        for distance in range(1, sight + 1):
            for y in (agent.y - distance, agent.y + distance):
                if 0 <= y < self.height and agent.climate_steps(self._climate[y]) < 0.5:
                    return agent.x, y
        return None

    @staticmethod
    def _food_needed(agent: Bitzkey) -> float:
        """The meals a mouth an animal needs about it to breed: 0, none."""
        return PREDATOR_FOOD_NEEDED if agent.species == "predator" else FOOD_NEEDED

    def _food_supply(self, agent: Bitzkey) -> float:
        """The meals within FORAGE_RADIUS of an animal per mouth sharing them, itself included:
        grass and plants for those that graze, as well as they digest them, and prey for hunters.
        Grass is judged by what will grow back over GRASS_OUTLOOK turns where it grows now, and a
        share of what stands now, so a crowd can't breed on a stock it is about to eat."""
        food = 0.0
        if agent.grazing:
            grass, x, y = self.grass, agent.x, agent.y
            growth = self._growth
            cells = 0.0
            tufts = 0
            for dx, dy in nearby_offsets(FORAGE_RADIUS):
                if 0 <= x + dx < self.width and 0 <= y + dy < self.height:
                    cells += growth[y + dy]
                    tufts += grass.get((x + dx, y + dy), 0)
            growing = self.width * self._grass_reach[-1] if self._grass_rows else 0.0
            regrowth = cells * self.grass_regrowth / growing * GRASS_OUTLOOK if growing else 0.0
            food += agent.grazing * (STANDING_GRASS * tufts + regrowth)
        mouths = 1.0
        for other in self._nearby(agent, FORAGE_RADIUS):
            if other.species == "plant":
                food += PLANT_MEALS * agent.browsing * self._plant_yield(other)
            elif self._feeds_on(agent, other):
                food += PREY_MEALS
            else:
                share = self._diet_share(agent, other)
                if agent.species in ("omnivore", "predator") and self._herd_mate(agent, other):
                    share *= PACK_MOUTHS  # a pack shares its kills
                mouths += share
        return food / mouths

    @staticmethod
    def _flee_range(agent: Bitzkey, herd_near: int) -> float:
        """How close a hunter must be before fleeing it is worthwhile: FLEE_RANGE, less among a
        herd keeping watch together."""
        return FLEE_RANGE * (1 - HERD_VIGILANCE * agent.herd_instinct * min(3, herd_near) / 3)

    def _witness(self, actor: Bitzkey, outcome: float) -> None:
        """The herd-mates within sight of an animal's success (an outcome above 0) or failure learn
        from it as if they had made its last choice themselves, feel it, and sense it for a while;
        how much they take it in is their empathy times their intellect."""
        if actor.last_senses is None:
            return
        for observer in self._nearby(actor, MAX_SIGHT):
            if (observer.brain is None or not self._society(observer, actor)
                    or abs(observer.x - actor.x) + abs(observer.y - actor.y) > observer.sight):
                continue
            heed = observer.feeling("empathy") * observer.intelligence
            self._witnessed.append((observer, actor.last_senses, actor.last_choice, outcome,
                                    LEARNING_RATE * heed * SOCIAL_LEARNING * food_learning(actor.last_choice)))  # at the end
            observer.reward += SOCIAL_MOOD * heed * outcome
            if outcome > 0:
                observer.seen_success = 1.0
            else:
                observer.seen_failure = 1.0

    def _learn_witnessed(self) -> None:
        """Herd-mates learn from what they saw this turn, each brain in the order it saw it."""
        for observer, senses, choice, outcome, rate in self._witnessed:
            if observer.brain is not None:  # not since dead
                observer.brain.learn(senses, choice, outcome, rate)
        self._witnessed = []

    @staticmethod
    def _beside(agent: Bitzkey, other: Bitzkey) -> bool:
        """Whether two Bitzkeys are on the same cell or side by side, as sensing finds a mate, prey or
        a threat beside an animal."""
        return abs(agent.x - other.x) + abs(agent.y - other.y) <= 1

    def _society(self, observer: Bitzkey, actor: Bitzkey) -> bool:
        """Whether an animal counts another, alive or not, as one of its herd (or pack): as for
        _herd_mate."""
        return (observer is not actor and observer.species != "plant" and observer.species == actor.species
                and (observer.species != "predator" or self._same_kind(observer, actor)) and self._willing(observer, actor))

    def _herd_mate(self, agent: Bitzkey, other: Bitzkey) -> bool:
        """Whether an animal counts another as one of its herd: its species, and its kind for a
        predator, alive, and close enough in looks for it to accept as a mate."""
        return (other is not agent and other.alive and agent.species != "plant" and other.species == agent.species
                and (agent.species != "predator" or self._same_kind(agent, other)) and self._willing(agent, other))

    def _herd(self, agent: Bitzkey) -> tuple[bool, int, tuple[int, int] | None]:
        """Its herd: whether any herd-mate is within HERD_RADIUS, how many are within two
        cells, and the cell at the middle of those in range."""
        mates = [other for other in self._nearby(agent, HERD_RADIUS) if self._herd_mate(agent, other)]
        if not mates:
            return False, 0, None
        close = sum(abs(other.x - agent.x) + abs(other.y - agent.y) <= 2 for other in mates)
        middle = (round(sum(other.x for other in mates) / len(mates)), round(sum(other.y for other in mates) / len(mates)))
        return True, close, middle

    def _territory(self, hunter: Bitzkey) -> tuple[float, float]:
        """How overcrowded a hunter's territory is, with all its rivals and with its own species
        alone: nothing up to the rivals it will tolerate, fewer the less empathetic it is, then
        rising over TERRITORY_RAMP more to its species' strength of territorial feeling. A predator's pack-mates
        count only PACK_RIVALRY each: its pack mentality overcomes rivalry."""
        (fewest, most), strength = TERRITORY[hunter.species]
        tolerated = fewest + (most - fewest) * hunter.feeling("empathy")
        rivals = own = 0.0
        for other in self._nearby(hunter, TERRITORY_RADIUS):
            pack = PACK_RIVALRY if hunter.species == "predator" and self._herd_mate(hunter, other) else 1.0
            rivals += self._rival_weight(hunter, other) * pack
            own += (other.species == hunter.species) * pack
        return tuple(strength * min(1.0, max(0.0, count - tolerated) / TERRITORY_RAMP) for count in (rivals, own))

    def _prey_switching(self, hunter: Bitzkey) -> tuple[bool, bool]:
        """Which prey a hunter goes after, grazers and omnivores. A predator always hunts both. An omnivore
        hunts grazers only while they make up at least SWITCH_SHARE of the food weighed within
        FORAGE_RADIUS, against plants, or while none of that food is near at all."""
        if hunter.species == "predator":
            return True, True  # grazers are always prey to a predator, and omnivores it would beat
        grazers = omnivores = plants = 0
        for other in self._nearby(hunter, FORAGE_RADIUS):
            grazers += other.species == "grazer"
            omnivores += other.species == "omnivore"
            plants += other.species == "plant"
        total = grazers + plants
        return (total == 0 or grazers / total >= SWITCH_SHARE), False

    @classmethod
    def would_win(cls, agent: Bitzkey, opponent: Bitzkey) -> bool:
        """Whether an animal's blows would bring the opponent down as fast as the opponent's would bring
        it down, each as it stands now."""
        return (cls._damage(agent, opponent) / max(1.0, opponent.health)
                >= cls._damage(opponent, agent) / max(1.0, agent.health))

    @classmethod
    def _is_prey(cls, hunter: Bitzkey, other: Bitzkey) -> bool:
        if hunter.species == "omnivore":
            # Grazers while they are common enough to switch to, and other omnivores only in an
            # overcrowded territory, once hungry past its reluctance as that rivalry wears it down.
            # Predators only when it would win, blow for blow.
            return ((other.species == "grazer" and hunter.hunts_grazers)
                    or (other.species == "predator" and cls.would_win(hunter, other)) or (
                other.species == "omnivore" and hunter.own_rivalry > 0
                and hunter.hunger / 100 > cls.cannibal_reluctance(hunter) * (1 - RIVALRY_WEAR * hunter.own_rivalry)))
        if hunter.species != "predator":
            return False
        if other.species == "grazer":
            return hunter.hunts_grazers
        if other.species == "omnivore":  # only one it would beat, blow for blow
            return hunter.hunts_omnivores and cls.would_win(hunter, other)
        if other.species != "predator" or cls._same_kind(hunter, other):  # never its own kind
            return False
        # Other kinds of predator only when hungry past its reluctance, which a territory overcrowded
        # with predators wears away.
        return hunter.hunger / 100 > cls.cannibal_reluctance(hunter) * (1 - RIVALRY_WEAR * hunter.own_rivalry)
