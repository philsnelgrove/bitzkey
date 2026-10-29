"""Turn-based Bitzkey world and its deterministic history."""

from __future__ import annotations

import bisect
import math
import random
import shutil
import tempfile
import time
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from typing import NamedTuple
from dataclasses import dataclass, field, replace
from functools import cache, cached_property, lru_cache
from itertools import accumulate
from pathlib import Path

import numpy as np

import tools.Brain as brain_rules  # its rules, as they now stand
from tools.Brain import (ACTIONS, ALERT_CALL, CALLS, FIRST_CALL, MEMORY_NEURONS, OUTPUTS, SENSES, SILENT, Brain, BrainBatch, call_output,
                         choice_odds, depth_for, gene_senses, make_senses, memory_after, width_for)
from tools.Genome import (GENOME, SPECIES, TEMPERAMENT, TRAIT_MAX, body_word, color_from_word, cross, decode_nocturnal, dot_difference,
                          inherit_mind,
                          ring_difference, social_difference, social_word, BODY_MUTATION_CHANCE, COLD_TOLERANCE,
                          HEAT_TOLERANCE, MIND_MUTATION_CHANCE, TEMPERATURE_SHIFT, TEMPERATURES, decode_body, decode_climate,
                          decode_temperament, decode_terrain, fits, make_dna, mind_word, share, swims, walks, with_terrain)
from tools.Census import Census
from tools.Exchange import Imported
from tools.Insights import DeathTally, Insights


GRID_SIZE = 100
INITIAL_POPULATION = 90
GRASS_DENSITY = 50      # percent of cells that start with grass
GRASS_REGROWTH = 0.032  # share of the starting grass that grows back each turn
MAX_POPULATION = 20000  # the most Bitzkeys spawned at the start; a world's population cap is a setting
CROWDING_LIMIT = 10    # litters shrink as crowding nears this; at it, an animal doesn't breed; 0: no limit
SPAWN_GROUP = (3, 5)  # Bitzkeys spawn in groups of this many sharing one genome
MAX_IMPORTED = 120  # imported Bitzkeys, each spawning a group of its own
MAX_GRID = 1000  # the widest and tallest a world can be, in cells
# The sea: a new world's share of the map under it, in percent, and how broad its land and sea are, as a share of the
# map's width (smaller, and they break into more and smaller islands and seas). East and west, the coasts wrap round.
DEFAULT_OCEAN = 200 / 3
CONTINENT_BREADTH = 0.25
# Terrain. An avian animal's size, weapons and armour count AVIAN_BODY of its genes' (light enough to fly), and it burns
# FLIGHT_HUNGER more food; the terrestrial and aquatic can hunt or attack it only while it rests. An amphibious animal
# goes AMPHIBIOUS_SPEED of its usual steps on land, and rests AMPHIBIOUS_COOLDOWN times as long after breeding. An avian
# plant's seeds fly AVIAN_SEED_REACH times as far.
AVIAN_BODY = 0.5
FLIGHT_HUNGER = 0.25
AMPHIBIOUS_SPEED = 0.5
AMPHIBIOUS_COOLDOWN = 2.0
AVIAN_SEED_REACH = 2.0
# Stamina, 0 to 100: every action but rest tires an animal, and resting restores it. Its pace follows the day: of the
# half of each day an animal is awake, the average one acts ACTIVE_SHARE and rests the rest, spending its stamina in
# the one and getting it back in the other, however many turns a day is (see stamina_pace and rest_recovery). Before
# that pace is applied, any action but rest costs ACTION_FATIGUE, each square it moves STEP_FATIGUE more (a flyer's
# FLIGHT_FATIGUE times that), each blow it strikes STRIKE_FATIGUE and mating BREED_FATIGUE (each parent); a flyer over
# the sea, where it can't rest, tires HOVER_FATIGUE a turn more; an average turn of acting costs TYPICAL_FATIGUE. An
# animal with no stamina left can only rest, and a flyer exhausted over the sea drowns. An amphibious animal tires
# AMPHIBIOUS_FATIGUE times as fast at everything.
ACTIVE_SHARE = 2 / 3
TYPICAL_FATIGUE = 20.0  # an action and two squares' going
ACTION_FATIGUE = 10.0
STEP_FATIGUE = 5.0
FLIGHT_FATIGUE = 2.0
STRIKE_FATIGUE = 50.0
BREED_FATIGUE = 75.0
HOVER_FATIGUE = 5.0
AMPHIBIOUS_FATIGUE = 2.0
FULL_STAMINA = 100
SPAWN_SPREAD = 2      # a group's members appear within this many cells of each other's centre
# Biomes: each stretch of land or sea within one climate zone, joined square to square (east and west wrapping round),
# of BIOME_SIZE squares or more. A world started with any Bitzkeys has BIOME_GROUPS more groups in every biome, on
# top of its population, each at home there: the zone's usual temperature its home, and a terrain that suits it, the
# first the surface's own (terrestrial on land, aquatic at sea) and the rest any that may live there.
BIOME_GROUPS = 2
# Each death is kept whole, for the graph's tooltips, for this many turns; the death report counts every death since
# the world began as it happens (see DeathTally), and the tally lets go of the killers gone, but the deadliest, every
# KILLERS_LET_GO turns.
RECENT_DEATH_TURNS = 1000
KILLERS_LET_GO = 100
BIOME_SIZE = 25
# A square holds CELL_SPACE spaces, and every Bitzkey, plant or animal, takes 1 + its size of them. Bitzkeys share
# a square, and meet there as they would alongside, but none is born, moves or spawns into one without room for it.
CELL_SPACE = 10
# Full hunger doesn't kill at once: a starving Bitzkey loses this share of its full health each turn.
STARVING_DAMAGE = 0.1
# What a large body costs, against all it gains, so that size isn't simply better: it burns 1 + SIZE_HUNGER times its
# size squared as much (a share of the size range: twice as much at the largest, by default); it wastes away
# 1 + SIZE_STARVING times its size as fast starving; a hunter's blow misses a small animal outright with a chance of
# SMALL_HIDING times its smallness squared (40% at the smallest, none at the largest); and the smallest are infants
# SMALL_MATURITY as long as their lifespans would make them, rising to the whole of it at the largest.
SIZE_HUNGER = 1.0
SIZE_STARVING = 1.0
SMALL_HIDING = 0.4
SMALL_MATURITY = 0.1  # the smallest an infant some 4 days, the largest 360
# Meat lasts: the share of its fullness that is meat empties at this share of the usual rate.
# Meat lasts seven times as long as plant food: a meal of meat a week to a meal of plants a day. An omnivore's mixed
# fullness empties at the mix of the two rates.
MEAT_DIGESTION = 1 / 7
# Feeble: an animal in its infancy (the first FEEBLE_YOUTH of its lifespan) or its old age (past AGEING of it)
# has every body trait, and its health, at FEEBLE_SHARE of its own.
FEEBLE_YOUTH = 1 / 30  # 360 days for the largest; the smallest SMALL_MATURITY of their 36
PREDATOR_INFANCY = 0.5  # a predator is an infant this share as long
FEEBLE_SHARE = 0.5
DEPLOY_GROUP = 5      # Bitzkeys deployed from the catalog onto a running world, of each chosen, a click
USE_GPU = 0  # the C++ version's: 1 thinks on the GPUs; the Python version always thinks on the CPU
SPAWN_AT_HOME = 1     # 1: a group appears in rows whose usual temperature is its home (the nearest, if none); 0: anywhere
# Earlier versions kept the history on disk, in folders among the temporary files.
HISTORY_PREFIX = "bitzkey-history-"  # their names start so
STALE_HISTORY_HOURS = 24  # one left this long by a run that never closed it is cleared away
CORPSE_FERTILITY = 2   # grass grows this many times as fast on and around a corpse
CORPSE_TURNS = 90  # under four days
CORPSE_PLANT = 0.1     # chance an animal's body that rots away entirely leaves the seed of a new plant where it lay
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
GESTATION = (30, 150)  # a day and a quarter to six and a quarter days
# Heat: an animal is fertile only while in heat, a stretch of the turns before its next winter, the winter of the row
# it stands in. Heat ends HEAT_CLOSEST of a year before that winter, and HEAT_FORESIGHT more for the most patient, who
# breed further from it; it begins HEAT_LENGTH of a year before it ends. No animal is in heat through its own winter,
# and one where no winter comes, at the polar caps or the equator's heart, always is. By species.
HEAT_CLOSEST = {"grazer": 0.1, "omnivore": 0.1, "predator": 0.1}
HEAT_LENGTH = {"grazer": 0.5, "omnivore": 0.5, "predator": 0.5}
HEAT_FORESIGHT = 0.15
BIRTH_REACH = 2  # young are born on the free cells nearest their mother, up to this many cells away
# Years a Bitzkey, plant or animal, is expected to live: the smallest bodies, with the fastest
# metabolisms; middling ones; and the largest, with the slowest. Between them, lifespan rises steadily
# with the metabolism slowing.
LIFESPAN_YEARS = (3.0, 15.0, 30.0)  # long enough for even the smallest to see its winter come round again
METABOLISM = (1.6, 0.6)  # metabolic rate of the smallest and the largest bodies
STOMACH = (0.5, 2.0)     # food a body holds, relative to the middle, from the smallest to the largest;
                         # hunger is how empty it is, so a meal fills a small stomach more
SIZE_IN_MIGHT = 0.5      # share of fighting strength and toughness that comes from body size
AGEING = 0.7  # from this share of its lifespan, an animal may die of old age

DEATH_PENALTY = 2.0
SLOW_DEATHS = frozenset(("old age", "starved", "froze", "overheated"))  # deaths no last choice is blamed for
# Breeding is the greatest pleasure there is, and courting a mate into reach the next: each at three quarters
# of its full strength, and the rest by libido, so even the coldest animal learns to seek a mate. Courting is
# rewarded only once it brings a ready, willing partner within MATE_REACH; neither costs anything when it fails.
BREED_JOY = 4.0
COURT_JOY = 2.5
LIBIDO_IN_JOY = 0.25  # the share of those pleasures that libido sets
FIGHT_JOY = 0.5
COMPANY_JOY = 0.1
CURIOSITY_JOY = 0.15   # pleasure of setting off in a direction, scaled by curiosity
CURIOUS_EXPLORATION = 0.1  # the chance, times its curiosity, that an animal tries any of its options at random rather
                            # than as its brain values them, so no option goes untried for good
RESTLESS_CURIOSITY = 0.5  # the chance, times a half and its curiosity, that an animal whose brain chose to rest (not made
                          # to) does any of its other options instead; in peril, neither, but as its brain chooses
# Moving costs food. A walk, a single cell, saves this share of the turn's burn, as resting does; going
# as far as its speed allows costs this share of a turn's burn more for each cell beyond the first.
WALK_REST = 0.5
GO_EFFORT = 0.25
FEAR_DREAD = 0.3
UNEASE = 0.1           # how much an intolerant animal minds other species close by
CLIMATE_UNEASE = 2.0   # discomfort two or more steps from home, half that one step; less for the hardy
PATIENT_DISCOUNT = 0.99  # gamma at the highest patience: how far ahead the patient look
IMPATIENT_DISCOUNT = 0.9  # and at the lowest: even the least patient look some ten turns ahead
RESTING_UNEASE = 1.0  # unease, a turn, of having chosen to rest out of its temperature, two or more steps from home (half one step; less if hardy)
HOMEWARD_UNEASE = 3.0  # more discomfort away from home, rising with the distance to home to this at half the map's height
EMPATHY_PAIN = 0.15    # how much an empathetic animal feels another's wound; twice that for a death
# Parental attachment: a parent is glad of its own young, still infants, within its sight, up to PARENT_JOY with
# three or more, and grieves PARENT_GRIEF when one of them dies, wherever it is; both by half again and up to one
# and a half times as much with its empathy. Neither tells it what to do: what keeps its young alive is for its
# brain to learn.
PARENT_JOY = 0.3
FEEDING_JOY = 1.0  # how glad a parent is of feeding a starving infant of its own, and in proportion for one less hungry; more if empathetic
FEED_HUNGER = 25.0  # a parent may feed its infants within its sight once one of them is this hungry
PARENT_GRIEF = 2.0
# A hunter that kills a mother carrying young feels remorse, by half again and up to one and a half times as much
# with its empathy: a lesson, not a law.
PREGNANT_REMORSE = 1.0
# Society: an animal sees what happens to its herd-mates (its pack, for a predator) within its sight. A
# herd-mate's kill is a success, its death (but of old age) a failure; the witness learns from it as
# if it had made the herd-mate's last choice itself, and feels it, by its empathy times its intellect.
FATAL_LESSON = 5.0  # how many times as hard a herd-mate's death teaches those who see it
SOCIAL_LEARNING = 1.0  # how fast it learns from what it sees, beside learning from its own choices
SOCIAL_MOOD = 0.3      # how much what it sees moves its mood
SOCIAL_MEMORY = 0.93   # the share of a seen success or failure it still senses a turn later (0.8 every three hours)
OTHER_KIND = 0.4       # how much of that pain reaches it when the victim is of another species
# The social background's traits. Loyalty: whenever it can herd, an animal values keeping to its herd this much
# more, times its loyalty. Dominance: a ready animal breeds with the most dominant willing partner in reach, and
# a territory's crowding troubles the dominant less and the submissive more, by up to this share.
LOYALTY_PULL = 0.25
PARENTAL_INSTINCT = 2.0  # how much of the gap between feeding its young and its best other option a parent closes, times
                         # the hungriest's hunger and a half and its empathy: at 2, half hungry, level
INFANT_HERDING = 1.0  # how much more an infant values keeping to its kind, and it herds once a cell from their middle
NURSING_REACH = 2     # each turn, a grown animal feeds the hungriest infant of its kind this close, a turn's burn of food,
NURSING_LIMIT = 80    # while it stays below this hunger itself
DOMINANCE_EASE = 0.5
# Trust: how much more, or less, the most and least trusting take in what they see of their herd-mates, by
# this share; and an alert reaches it from a herd-mate this many times its trust times its sight away.
TRUST_HEED = 0.5
TRUST_REACH = 2.0
# Kinship: the most a herd-mate's or partner's social background may differ from its own, at full kinship and
# no tolerance, is 1 less this; less kinship, or more tolerance, accepts more.
KINSHIP_SHUN = 0.8
# The mind's new traits. Memory: an animal remembers where it last fed, and where it was last struck, for up to
# MEMORY_TURNS at the best memory, fewer the poorer. Resilience: the most resilient get over a herd-mate's failure
# faster, and fear a bad experience less, by up to this share, the least resilient slower and more. Moodiness:
# experience moves the moodiest temperament by up to this share more, the steadiest by this share less.
MEMORY_TURNS = 300
RESILIENCE = 0.5
MOODINESS = 0.5
LEARNING_RATE = 0.027  # at the highest intellect, before its boost
# The clever learn faster. The learning rate rises with intellect, and on top of that by up to
# LEARNING_BOOST times again at the highest; the least intelligent get no boost. And each turn an
# animal re-learns from REPLAY_PER_INTELLECT of its most recent experiences for each point of its
# inborn intellect, so the least replay none.
LEARNING_BOOST = 1.0
LEARNING_FLOOR = 0.25  # the share of LEARNING_RATE even the dullest learns at, intellect adding to it
REPLAY_PER_INTELLECT = 2
PREDATOR_LEARNING = 1.0  # a predator's brain learns this many times as fast, from its own choices and replays;
# an omnivore's gets half the bonus
# Food that takes a kill is harder to learn than grass: a choice to hunt, attack or feast is learnt from
# KILL_FOOD_LEARNING times as much, by any animal, and one to eat, forage, browse or uproot PLANT_FOOD_LEARNING.
# Breeding and courting may be learnt from more than other choices, BREED_LEARNING and COURT_LEARNING times as much;
# by default no more, as learning them harder unsettles the brain for all else (their pleasures teach them instead).
KILL_FOOD_LEARNING = 2.0
PLANT_FOOD_LEARNING = 0.5
BREED_LEARNING = 1.0
COURT_LEARNING = 1.0
# Forced lessons: 1 learns from every turn, even one with no choice in it (resting through its night or exhausted,
# panicking or frozen); 0 learns only from the turns it chose what to do.
FORCED_LESSONS = 0
KILL_FOOD = frozenset(ACTIONS.index(action) for action in ("hunt", "attack", "feast"))
PLANT_FOOD = frozenset(ACTIONS.index(action) for action in ("eat", "forage", "browse", "uproot"))


def lesson_weight(choice: int, outcome: float) -> float:
    """How much a lesson from a choice counts: a success (an outcome above 0) by the choice's learning weight, a
    failure as any other, so that a weight makes success more, or less, memorable without making failure so."""
    return choice_learning(choice) if outcome > 0 else 1.0


def choice_learning(choice: int) -> float:
    """How much more, or less, a success from a choice is learnt from: for the food it seeks, or the mate."""
    if choice == ACTIONS.index("breed"):
        return BREED_LEARNING
    if choice == ACTIONS.index("court"):
        return COURT_LEARNING
    return KILL_FOOD_LEARNING if choice in KILL_FOOD else PLANT_FOOD_LEARNING if choice in PLANT_FOOD else 1.0


def mating_joy(agent: Bitzkey, joy: float) -> float:
    """A pleasure of mating as an animal feels it: LIBIDO_IN_JOY of it by its libido, the rest whatever it is."""
    return joy * (1 - LIBIDO_IN_JOY + LIBIDO_IN_JOY * agent.feeling("libido"))
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
MATING_CALL_RANGE = 2.0  # a predator ready to breed that sees no mate seeks one this many times its sight off
OMNIVORE_MATING_CALL = 0.25  # an omnivore's milder mating call
HUNTER_BREEDING_COST = 0.5  # share of the usual breeding cost a predator or omnivore pays
GRAZER_BREEDING_COST = 0.5  # share of the usual breeding cost a grazer pays
PREDATOR_STING = 0.5   # share of a victim's spines a predator striking it takes
# A species grown scarce breeds more cheaply: below SCARCE_POPULATION living, its breeding cost falls evenly, to
# RARE_BREEDING_COST of the usual at RARE_POPULATION or fewer.
SCARCE_POPULATION = 100
RARE_POPULATION = 10
RARE_BREEDING_COST = 1 / 3
GRAZER_SCARCE_POPULATION = 300  # grazers' cheap breeding begins below this many,
GRAZER_RARE_BREEDING_COST = 1 / 9  # and falls to this share of the usual cost at RARE_POPULATION or fewer
PREDATOR_FULL_LITTER_FOOD = 8.0  # a predator's full litter needs this many times the prey a predator needs to breed
# Predators glutting the world turn on one another: above PREDATOR_GLUT alive, their reluctance to hunt other kinds of
# predator wears away, to none at twice it; past that their own kind become prey too, fully at three times it. So do
# predators whose prey are dying out: below twice the rare-prey line, other kinds, to none at it; below that, their own
# kind too, fully with none left.
PREDATOR_GLUT = 600
FULL_LITTER_FOOD = 4.0  # a full litter needs this many times the home food a mouth needs to breed; at the need, 1/this
RARE_PREY = 1800        # a hunter doesn't breed while its prey species (all together) number under this percentage of its own
HUNTERS_PREY = {"predator": ("grazer", "omnivore"), "omnivore": ("grazer", "predator")}  # the species each hunter lives on


def scarcity_share(population: int, species: str | None = None) -> float:
    """The share of the usual breeding cost a species of this many living pays: all of it at SCARCE_POPULATION or
    more, falling evenly to RARE_BREEDING_COST at RARE_POPULATION or fewer; for grazers, from GRAZER_SCARCE_POPULATION
    to GRAZER_RARE_BREEDING_COST."""
    grazer = species == "grazer"
    scarce = GRAZER_SCARCE_POPULATION if grazer else SCARCE_POPULATION
    rare_cost = GRAZER_RARE_BREEDING_COST if grazer else RARE_BREEDING_COST
    if population >= scarce:
        return 1.0
    if population <= RARE_POPULATION or scarce <= RARE_POPULATION:
        return rare_cost
    return rare_cost + (1 - rare_cost) * (population - RARE_POPULATION) / (scarce - RARE_POPULATION)
NEWBORN_FOOD = 15.0    # parents conceive only as many young as can each start with this much of a stomach of food; 0: no limit
BREEDING_COST_MOST = 25.0  # the hunger the least fertile pay to breed: the most a parent could give its young
PACK_MOUTHS = 0.5      # a hunter's pack-mates count as this many mouths each on its food, sharing their kills
BLOODLUST = 1.0        # how much more a predator values feasting whenever it sees a fresh kill, whatever its hunger
CARRION_CRAVING = 1.0  # how much more an animal values scavenging carrion it sees, at the highest carrion trait;
                       # nothing up to the middle of the trait's range
CARRION_AVERSION = 1.0  # how much less a carrion-eater values hunting and attacking, at the highest carrion trait, likewise
CARRION_COWARDICE = 1.0  # a carrion-eater's least chance of breaking when struck, at the highest carrion trait, likewise
ALERT_JOY = 0.1        # pleasure of alerting its herd: to a hunter, for a grazer; to a fresh kill, for a hunter
ALARM_REACH = 6        # an alarmed animal knows where the hunter is within this many steps
# Calls (Brain.py): one carries this many steps times a half and its caller's size (a share of its range), at least
# one, and costs its caller this much hunger, as any call does, the alert among them.
CALL_REACH = 4
CALL_COST = 0.2
# Memory (Brain.py): how many choices back a lesson's error is carried to the memory gates that wrote what the brain
# read, 0 for none (what it keeps is then only what it was born to), up to MAX_MEMORY_STEPS; and how fast they learn,
# times the brain's learning rate.
MEMORY_STEPS = 1
MEMORY_LEARNING = 1.0
# The mental map: an animal remembers up to Map places of the MAP_PATCH-square patches it has been in (none at 0), the
# one it was last in longest ago forgotten for a new one; each visit moves what it remembers of a place in that season
# toward what it finds there by Map learning (the first sets it). Among places as good, it heeds the nearer, by
# MAP_NEARNESS a patch away. As the C++ version's.
MAP_PATCH = 8
MAP_PLACES = 16
MAP_LEARNING = 0.2
MAP_NEARNESS = 0.01
# Those it knows: an animal remembers up to Acquaintances others (none at 0) by what they did to it, the one it dealt
# with longest ago forgotten for a new one; each deed moves its regard for the doer toward the deed's worth by Regard
# learning (the first sets it): a blow -1, being fed or nursed as an infant 1, an alert it heeded or a mating a half.
# As the C++ version's.
ACQUAINTANCES = 8
REGARD_LEARNING = 0.3
BLOW_DEED, FEEDING_DEED, ALERT_DEED, MATING_DEED = -1.0, 1.0, 0.5, 0.5


@dataclass
class Acquaintance:
    """One it knows: who, how well it regards them, -1 to 1, and the turn they last dealt with it."""
    identifier: int
    regard: float
    last: int


@dataclass
class MapPlace:
    """A patch of the mental map: which (its column and row of patches), the turn it was last there, and, for each
    season, whether it has been there then and what it found: food (grass and plants for a grazer, prey, kills and
    carrion for a predator, either for an omnivore, by how close), danger (a hunter close, or a blow), and comfort (the
    weather's), 0..1."""
    column: int
    row: int
    last: int
    seen: list[bool] = field(default_factory=lambda: [False] * 4)
    food: list[float] = field(default_factory=lambda: [0.0] * 4)
    danger: list[float] = field(default_factory=lambda: [0.0] * 4)
    comfort: list[float] = field(default_factory=lambda: [0.0] * 4)
MAX_CALL_REACH = 20  # the most Call reach may be
SIGHT_PER_TRAIT = 3  # each step of the sight trait lets an animal see this many cells further
PREDATOR_SIGHT = 3  # predators see prey this many cells further than they otherwise see, up to MAX_PREDATOR_SIGHT
MAX_PREDATOR_SIGHT = 3
HUNT_STEPS = 1      # and chase it this many more steps a turn
PRIMAL_FEAR = 1.0
PREDATOR_INTELLIGENCE = 2.0  # a predator's intellect counts this many times over in all it does, up to the full range
BRAIN_COST = 0.15  # an animal's hunger burns this much faster at the highest inborn intellect, in proportion below
PREDATOR_BRAIN_COST = 0.5  # the share of that brain cost a predator pays: meat affords a big brain
PREDATOR_FEAR = 0.5  # a predator feels this share of its fear's effects: dread, panic, keeping close to its herd
PREDATOR_LOOMS = 2   # sizing up a fight for primal fear, a predator counts as this many steps larger and stronger
PRIMAL_FEAR_PREDATORS = 0.0  # predators are never overcome
PRIMAL_FREEZE = 1.0
PANIC = (0.3, 1.0)     # chance a fully fearful animal breaks when struck, unhurt and near death;
                       # its current fear scales it, so the fearless never break
COWARDLY_OMNIVORES = 1  # 1: an omnivore struck always breaks and flees, whatever its fear; 0: it checks its morale
THORN_FORCE = 15       # a plant's counterblow at full weapons and size, before the defender's toughness
THORN_SIZE_SHARE = 0.2  # share of that from the plant's size; the rest is its weapons, and strength counts for nothing
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
HERD_SPREAD = 3        # how far from its herd's middle a fearless animal is content; the fearful want the middle
HERD_JOY = 0.15        # pleasure of being among three or more herd-mates
HERD_LONELINESS = 0.1  # unease at having no herd-mate in range
SOLITARY_OMNIVORES = 1  # 1: omnivores live alone, in neither herd nor pack; 0: they herd with their own kind


def solitary(species: str) -> bool:
    """Whether a species lives alone, with no herd or pack: plants, and omnivores while SOLITARY_OMNIVORES."""
    return species == "plant" or (species == "omnivore" and bool(SOLITARY_OMNIVORES))
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
BREED_COOLDOWN = 1.0  # grazers' and omnivores' share of the usual rest after breeding, 20 turns for the least fertile to 5 for the most
PREDATOR_BREED_COOLDOWN = 0.5  # the same for predators: a short rest
PREDATOR_FOOD_NEEDED = 0.0  # the same for predators: none, so the food about them never stops them breeding
# Plants each grazer needs in its home band, the climate band of its home temperature nearest it, shared
# among the grazers at home there, to breed; between this and twice this its litter shrinks. 0: no limit.
PLANTS_PER_GRAZER = 0.1
# The same for predators: the prey in its home band (the grazers and omnivores there, what it lives on), shared among
# the predators at home there. Below this a predator doesn't breed; between this and twice this its litter shrinks.
PREY_PER_PREDATOR = 2.0
PLANT_MEALS = 2.0      # meals a plant is worth to a full-efficiency browser
PREY_MEALS = 10.0      # meals a prey animal is worth: meat fills about ten times what grass does
SHARED_DIET = 0.5      # how much an animal of another species eating some of the same food counts as a mouth
# Prey switching: a hunter goes after a kind of prey only while it makes up at least SWITCH_SHARE
# of the food it weighs within FORAGE_RADIUS (for omnivores, grazers against plants; for predators,
# grazers against omnivores), so a scarce prey is left alone to recover.
SWITCH_SHARE = 0.5
OMNIVORE_PREDATOR_FOOD = 0  # an omnivore hunts predators only while no more of its other food (grazers, plants and
                            # grassy cells) is within FORAGE_RADIUS; a predator's infant with a grown one of its kind
                            # within NURSING_REACH is guarded, and no prey to anyone
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
GRASS_OUTLOOK = 450    # judging grass before breeding, an animal counts what will regrow in this many turns
STANDING_GRASS = 0.25  # and this share of the grass standing now
GRASS_MEAL = 18        # hunger a tuft of grass sates: about a tenth of a middle-sized kill
# A plant is eaten in portions, a bite each: every bite sates PLANT_MEAL hunger (and PLANT_MEAL_FOR_SIZE more for the
# largest, none by default) and takes PLANT_BITE of its health, so a plant's food is in proportion to its size.
PLANT_MEAL = 70
PLANT_MEAL_FOR_SIZE = 0
PLANT_BITE = 15        # health a plant loses to each bite
PLANT_REGROWTH = 0.02  # share of its health pool a plant regrows each turn
PLANT_RECOVERY = 24    # turns after a bite before a plant can be bitten again: a day
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
# south, from the bottom up to and through the equator, is; but the arctic's outermost POLAR_SHARE, the polar caps,
# is very cold all year, and a strip of WINTER_EQUATOR_SHARE across the equator's middle stays hot through both
# winters. Every Bitzkey has a home temperature,
# and may tolerate heat or cold, suffering HARDINESS of its penalties, and FRAILTY of the other's; plants tolerate nothing, and a plant
# spawned at the start is at home in its zone's usual temperature.
# Days: every DAY_LENGTH turns is a day, and a new world's year is 360 of them. Night covers half the map, its
# columns from the top to the bottom, and moves west round it once a day. With SLEEP, an animal rests through its
# night, the dark for the diurnal and the light for the nocturnal (see the activity gene): it may only rest, but for
# fleeing a hunter or striking back at what struck it, startled awake, and a flyer over the sea, which can't rest.
DAY_LENGTH = 24  # a turn is an hour
FORESIGHT_DAY = 8  # the day the Mind rules' discounts and trace decays are set for (see Simulation._discount)
SLEEP = 1
SHELTER_REACH = 2  # an infant within this many squares of a living parent is sheltered: the climate's extra hunger and harm pass it by
DAYS_IN_YEAR = 360
DEFAULT_YEAR_LENGTH = 24 * DAYS_IN_YEAR  # a new world's, unless its settings say otherwise
YEAR_LENGTH = DEFAULT_YEAR_LENGTH  # each world sets it from its year length setting
ARCTIC_SHARE = 0.2
EQUATOR_SHARE = 0.2
POLAR_SHARE = 0.025  # at the top and again at the bottom: 5% of the map in all
WINTER_EQUATOR_SHARE = 0.05
CLIMATE_HUNGER = 2.0   # an animal a step or more from home, or a plant too hot, needs this many times the food
RESTING_EXPOSURE = 2.0  # times the weather's harm to an animal that chose to rest out of its temperature
WEATHER_GRACE = 4.0  # days an animal endures out of its temperature, uneasy but unharmed, before the weather harms it; half as long for the least resilient, half as long again for the most; twice as long weather it is hardy to
CLIMATE_DAMAGE = 0.025  # for each step from home, a Bitzkey loses this share of its full health a turn; plants from two
PLANT_CLIMATE_YIELD = 0.0  # a plant out of its temperature gives this share of its food
GRASS_DIEBACK = 60     # where it isn't medium, a cell's grass loses a tuft every this many turns
# The weather turns in waves, one in each climate zone, all setting off together, so no zone waits on
# another. In each, a winter washes in from the side toward its hemisphere's edge toward the equator
# over the first SEASON_TURNING of its season, stays through the middle, and ebbs back the way it came
# over the last SEASON_TURNING (at most half each). The wave moves in bands of WAVE_BAND rows; at its
# front, WAVE_FRONT bands stand part way between the old weather and the new, in even steps, so the
# change comes as a staircase. The climate's effects move with it.
SEASON_TURNING = 0.4
WAVE_BAND = 2
WAVE_FRONT = 4
# A dead animal is a fresh kill, not yet carrion, through the turn it died and FRESH_KILL_TURNS more;
# then it starts to rot. A predator or omnivore may eat a fresh kill of what it lives on whole,
# FRESH_KILL_MEALS times the meal of making the kill, and nothing is left.
FRESH_KILL_TURNS = 9
FRESH_KILL_MEALS = 3.0
SEED_VIABILITY = 0.75  # years a seed that falls where it can't grow stays able to germinate
PLANT_GESTATION = 30  # turns a seed lies in the ground before it can sprout
# The spans above as they were counted in turns when a day was eight of them, for the older tests' figures.
EIGHT_TURN_DAYS = dict(CORPSE_TURNS=30, GESTATION=(10, 50), FRESH_KILL_TURNS=3, PLANT_GESTATION=10, PLANT_RECOVERY=8,
                       GRASS_DIEBACK=20, GRASS_OUTLOOK=150, MEMORY_TURNS=100, SOCIAL_MEMORY=0.8,
                       WEATHER_GRACE=0.0, LEARNING_FLOOR=0.0)  # and the weather harming at once, the dullest learning nothing, as when they were written
STARTING_TOLERANCE = 0.1  # the chance a random starting genome carries each tolerance flag
HARDINESS = 0.5  # the share of exposure's extra hunger and damage an animal tolerant of it suffers
FRAILTY = 2.0    # the share an animal tolerant of the other extreme suffers
HOT, MEDIUM, COLD, VERY_COLD = range(len(TEMPERATURES))
SEASONS = ("mild", "northern winter", "mild", "southern winter")  # by quarter of the year

# How experiences reshape temperament, in shares of a trait's full range before scaling by
# plasticity.
EXPERIENCES = {
    "kill": {"aggression": 0.04, "fear": -0.02, "intellect": 0.004},
    "hit": {"aggression": 0.0067},
    "stung": {"aggression": -0.0133, "fear": 0.0133},
    "wounded": {"fear": 0.0333, "aggression": -0.0067, "intellect": -0.002},
    "wounded_alone": {"sociability": 0.02},
    "bred": {"libido": 0.02, "sociability": 0.0133, "intellect": 0.002},
    "bred_into_hunger": {"libido": -0.0533},
    "starving": {"appetite": 0.0033, "patience": -0.0033, "intellect": -0.0005},
    "foraged_meal": {"patience": 0.0067},
    "discovery": {"curiosity": 0.01, "intellect": 0.002},
    "ambushed": {"curiosity": -0.02, "fear": 0.0133},
    "defended": {"fear": -0.0133, "aggression": 0.0067, "intellect": 0.002},
    "panicked": {"fear": 0.0067, "intellect": -0.001},
    "lesson": {},  # a choice that turned out well, learnt from: it no longer sharpens the mind
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
HERD = ACTIONS.index("herd")
FEED_YOUNG = ACTIONS.index("feed_young")
FED = {"ate", "browsed", "uprooted", "feasted", "scavenged"}  # what it did when it fed, to remember where
MAX_SIGHT = 2 + SIGHT_PER_TRAIT * TRAIT_MAX + MAX_PREDATOR_SIGHT  # the furthest anything is seen: a predator's sight for prey
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


def month(turn: int) -> int:
    """The month of its year a turn falls in, 1 to 12."""
    return turn % YEAR_LENGTH * 12 // YEAR_LENGTH + 1


def zone_temperature(y: int, height: int) -> int:
    """A row's temperature outside the winters: cold in the arctic, hot on the equator, medium
    between."""
    place = (y + 0.5) / height
    if place < POLAR_SHARE or place > 1 - POLAR_SHARE:
        return VERY_COLD
    if place < ARCTIC_SHARE or place > 1 - ARCTIC_SHARE:
        return COLD
    return HOT if abs(place - 0.5) < EQUATOR_SHARE / 2 else MEDIUM


def in_winter(y: int, height: int, quarter: int) -> bool:
    """Whether a row is in winter in a quarter of the year: the north and the equator in the second,
    the south and the equator in the fourth."""
    place = (y + 0.5) / height
    if place < POLAR_SHARE or place > 1 - POLAR_SHARE:
        return False  # the polar caps are very cold all year
    equator = abs(place - 0.5) < EQUATOR_SHARE / 2
    if equator and abs(place - 0.5) < WINTER_EQUATOR_SHARE / 2:
        return False  # the equator's heart stays hot
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


def _season_into(turn: int) -> float:
    """The share of its season gone, 0 to 1."""
    return (turn % YEAR_LENGTH * 4 - season(turn) * YEAR_LENGTH) / YEAR_LENGTH


def _turning() -> float:
    """SEASON_TURNING, at most half, so the wave in and the ebb never overlap."""
    return min(0.5, SEASON_TURNING)


def season_progress(turn: int) -> float:
    """How far a winter's wave has washed in: 0 as the winter begins, 1 once SEASON_TURNING of it has
    passed; always 1 in the mild seasons."""
    if season(turn) % 2 == 0 or _turning() <= 0:
        return 1.0
    return min(1.0, _season_into(turn) / _turning())


def temperature_at(y: int, height: int, turn: int) -> float:
    """A row's temperature on a turn, from 0 (hot) to 3 (very cold): its mild one, its winter's, or part
    way between the two while a winter's wave washes in or ebbs out across it."""
    quarter = season(turn)
    mild = zone_temperature(y, height)
    if not in_winter(y, height, quarter):
        return float(mild)
    winter = mild + 1
    into, turning = _season_into(turn), _turning()
    if turning <= 0 or turning <= into < 1 - turning:
        return float(winter)
    # Washing in, from the edge toward the equator, or ebbing back the way it came.
    ebbing = into >= 1 - turning
    progress = (into - (1 - turning)) / turning if ebbing else into / turning
    band, bands = wave_band(y, height, (quarter + 1) % 4 if ebbing else quarter)
    reach = progress * (bands + WAVE_FRONT)  # how far the front has come, in bands
    turned = min(1.0, max(0.0, (reach - band) / WAVE_FRONT))
    return winter - turned if ebbing else mild + turned


def per_turn(rate: float) -> float:
    """A rate set for a turn of a FORESIGHT_DAY-turn day, for a turn of this one: the same over the same span of days."""
    return rate ** (FORESIGHT_DAY / DAY_LENGTH)


def stamina_pace() -> float:
    """The share of the listed fatigue an action costs: the average animal spends its stamina over its waking half
    day's ACTIVE_SHARE and gets it back over the rest."""
    return FULL_STAMINA / (TYPICAL_FATIGUE * max(1.0, DAY_LENGTH / 2 * ACTIVE_SHARE))


def rest_recovery() -> float:
    """Stamina a turn's rest restores."""
    return FULL_STAMINA / max(1.0, DAY_LENGTH / 2 * (1 - ACTIVE_SHARE))


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
    """An inborn intellect as it counts for the brain's shape and how much it replays: a predator's
    PREDATOR_INTELLIGENCE times over, up to the full range."""
    return min(TRAIT_MAX, int(intellect * (PREDATOR_INTELLIGENCE if species == "predator" else 1.0)))


def brain_shape(dna: int) -> tuple[int, int]:
    """The depth and width of a genome's brain, by its thinking intellect: deeper up to MAX_DEPTH, then wider."""
    intellect = thinking_intellect(decode_temperament(dna)["intellect"], decode_body(dna)["species"])
    return depth_for(intellect), width_for(intellect)


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


def call_reach(caller: Bitzkey) -> int:
    """How many steps its calls carry (Call reach)."""
    return min(MAX_CALL_REACH, max(1, round(CALL_REACH * (0.5 + caller.physique("size")))))


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
def looks_apart(first: int, second: int) -> tuple[float, float, float]:
    """How far two genomes' ring, dot and social background colours are apart, each from 0 to 1.
    Groups share genomes, so the same pairs come up again and again."""
    return ring_difference(first, second), dot_difference(first, second), social_difference(first, second)
SEED_OFFSETS = sum(abs(dx) + abs(dy) <= SEED_REACH for dx, dy in SIGHT_OFFSETS)  # the first offsets: those a seed reaches


@cache
def seed_offsets(reach: int) -> int:
    """How many of the first sight offsets a seed reaches from this far."""
    return sum(abs(dx) + abs(dy) <= reach for dx, dy in SIGHT_OFFSETS)


def _metabolism(size: float) -> float:
    """Metabolic rate for a size, as a share of its range: fast in small bodies, slow in large ones."""
    fastest, slowest = METABOLISM
    return fastest + (slowest - fastest) * size


@dataclass
class Bitzkey:
    identifier: int
    dna: int
    x: int
    y: int
    hunger: float = 0.0
    health: float = 100.0
    stamina: float = FULL_STAMINA  # spent by acting, restored by resting (see ACTIVE_SHARE)
    wet: bool = False  # in the sea, as the world keeps it while it moves
    age: int = 0
    cooldown: int = 0
    last_action: str = "new"
    brain: Brain | None = None
    generation: int = 0
    children: int = 0
    in_heat: bool = True  # in heat now, and so fertile: kept turn by turn (see Simulation.heat)
    mother: int = -1  # its parents' identifiers; -1 for one spawned, imported or deployed
    father: int = -1
    died_turn: int | None = None
    assailant: int | None = None  # the animal that struck it since it last acted, which it may defend against
    panicked: bool = False  # failed a morale check when struck, so it will flee its assailant
    primal_fear: bool = False  # overcome by a stronger hunter: it panics, or freezes, while one is in sight
    rivalry: float = 0.0    # hunters: how overcrowded its territory was with rivals when it last looked
    hunts_grazers: bool = True    # hunters: whether grazers were common enough to hunt when it last looked
    hunts_omnivores: bool = True  # predators: the same for omnivores
    hunts_predators: bool = True  # omnivores: whether its other food was scarce enough to hunt predators
    guarded: bool = False  # a predator's infant: whether a grown one of its kind was close by as the turn began
    own_rivalry: float = 0.0  # the same, counting only its own species: what wears down its reluctance
    alarm: int | None = None  # what struck a herd-mate close by since it last acted, or a herd-mate alerted it to
    kill_alert: tuple[int, int] | None = None  # where a pack-mate alerted it to a fresh kill since it last acted
    call: int = SILENT  # what it called alongside its last action (Brain.py's CALLS), heard the turn after
    call_turn: int = -1  # the turn it called it
    last_call: int = -1  # its last choice of call, to learn from with its action; -1 for none of its own
    held: list[float] = field(default_factory=lambda: [0.0] * MEMORY_NEURONS)  # what its memory neurons hold (Brain.py)
    # Its senses at the choices before its last, most recent first, back to the first its memory wasn't written at, and
    # no more than Memory steps: what learning back in time goes over. Whether its memory was written at its last choice.
    past: list[list[float]] = field(default_factory=list)
    last_wrote: bool = False
    places: list[MapPlace] = field(default_factory=list)  # its mental map, in the order it first came to each place
    known: list[Acquaintance] = field(default_factory=list)  # those it knows, in the order it came to know them
    seen_success: float = 0.0  # a herd-mate's success it saw, fading turn by turn
    seen_failure: float = 0.0  # a herd-mate's failure it saw, the same
    weathered: float = 0.0  # turns endured out of its temperature, one regained each turn back in it; the weather harms it once past its grace
    herd_assailant: int | None = None  # what struck a herd-mate close by, from alongside it: it may defend against it
    food_memory: tuple[int, int, int] | None = None    # where it last fed, and the turn: (x, y, turn)
    danger_memory: tuple[int, int, int] | None = None  # where it was last struck, and the turn
    meat: float = 0.0  # a corpse's meat left for scavengers
    meat_fill: float = 0.0  # how much of its fullness is meat, which it digests more slowly
    population: int = 0  # how many of its species live, as of the turn's start: its awareness of its own numbers
    cost_share: float = 1.0  # the share of its breeding cost its species' scarcity leaves it, as of the turn's start
    glut: float = 0.0  # for a predator, what drives it to hunt other predators: 1 for other kinds fully, 2 its own kind
    rot: float = 0.0   # meat lost to rot each turn
    portions: int = 0  # a body's portions left to eat, feasting or scavenging: one for each size class
    trail: list[tuple[int, int]] = field(default_factory=list)  # the cells it passed through this turn, for the map
    # Its most recent experiences, each (senses, choice, target), newest last, to re-learn from.
    memories: list[tuple] = field(default_factory=list)
    # The young it carries, conceived at mating, each (dna, mind, brain, energy, generation), and the
    # turn they are born.
    womb: list[tuple] = field(default_factory=list)
    due_turn: int | None = None
    # Current temperament, mind and social traits: starts at the inborn values and is reshaped by experience.
    mind: dict[str, float] = field(default_factory=dict)
    # The last decision, kept until the next one so the brain can learn from how it turned out.
    last_senses: list[float] | None = None
    last_choice: int = 0
    last_forced: bool = False  # whether it had no choice but last_choice
    last_panicked: bool = False  # whether that was a panicked flight, which teaches though it was no choice
    last_wellbeing: float = 0.0
    reward: float = 0.0
    last_comfort: float = 0.0  # how comfortable it felt at its last decision
    odds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.mind:
            self.mind = {trait: float(value) for trait, value in self.inborn.items()}

    @cached_property
    def inborn(self) -> dict[str, int]:
        return decode_temperament(self.dna)

    @cached_property
    def body(self) -> dict[str, int | str]:
        return decode_body(self.dna)

    def looks_apart(self, other: Bitzkey) -> tuple[float, float, float]:
        """How far its ring, dot and social background colours are from another's, each from 0 to 1."""
        return looks_apart(self.dna, other.dna)

    @cached_property
    def species(self) -> str:
        return str(self.body["species"])

    @cached_property
    def terrain(self) -> str:
        return decode_terrain(self.dna)

    @cached_property
    def nocturnal(self) -> bool:
        """Active by night, resting by day; else the other way round."""
        return decode_nocturnal(self.dna)

    def can_be(self, sea: bool) -> bool:
        """Whether it may be in the sea, or on the land, by its terrain."""
        return fits(self.terrain, sea)

    @property
    def exhausted(self) -> bool:
        """With no stamina left, it can only rest."""
        return self.species != "plant" and self.stamina <= 0

    @property
    def aloft_at_sea(self) -> bool:
        """A flyer over the sea, where it can't rest."""
        return self.terrain == "avian" and self.species != "plant" and self.wet

    def fatigue(self, cost: float) -> float:
        """What tiring by `cost` takes from it: at the day's pace, and an amphibious animal's AMPHIBIOUS_FATIGUE
        times as much."""
        cost *= stamina_pace()
        return cost * AMPHIBIOUS_FATIGUE if self.terrain == "amphibious" else cost

    @cached_property
    def climate(self) -> tuple[int, bool, bool]:
        """Its home temperature, and whether it tolerates heat and cold."""
        return decode_climate(self.dna)

    def climate_steps(self, here: float) -> float:
        """How many steps a temperature is from home, part steps while the weather turns."""
        return abs(here - self.climate[0])

    def exposure_share(self, here: float) -> float:
        """The share of exposure's penalties it suffers here: HARDINESS for an animal tolerant of
        heat where it is hotter than home, or of cold where it is colder; FRAILTY for one tolerant
        of the other; otherwise all of them. Plants tolerate nothing."""
        home, heat, cold = self.climate
        if self.species == "plant" or here == home:
            return 1.0
        if (cold if here > home else heat):
            return HARDINESS
        if (heat if here > home else cold):
            return FRAILTY
        return 1.0

    @property
    def profile(self) -> dict[str, float]:
        """Current temperament and physical traits, keyed by trait name."""
        return {**self.mind, **{trait: float(value) for trait, value in self.body.items()
                                if trait != "species"}}

    @property
    def feeble(self) -> bool:
        """In its infancy, the first FEEBLE_YOUTH of its lifespan, or its old age, past AGEING of it, an animal
        is feeble: its body traits and health are at FEEBLE_SHARE. Plants never are."""
        if self.species == "plant":
            return False
        return self.age < self.infancy or self.age > AGEING * self.lifespan

    @property
    def infancy(self) -> float:
        """How long its infancy lasts, in turns: FEEBLE_YOUTH of its lifespan (a predator's PREDATOR_INFANCY as long),
        the smallest SMALL_MATURITY of that, the larger more of it."""
        maturity = SMALL_MATURITY + (1 - SMALL_MATURITY) * share(int(self.body["size"]))
        return FEEBLE_YOUTH * (PREDATOR_INFANCY if self.species == "predator" else 1.0) * maturity * self.lifespan

    @property
    def infant(self) -> bool:
        """Whether it is an animal in its infancy: the first FEEBLE_YOUTH of its lifespan, a predator's PREDATOR_INFANCY
        as long."""
        if self.species == "plant":
            return False
        return self.age < self.infancy

    def trait(self, name: str) -> int:
        """A body trait as it is now: its genes' value, or FEEBLE_SHARE of it while feeble; an avian animal's size,
        weapons and armour AVIAN_BODY of that."""
        value = int(self.body[name])
        if name == "species":
            return value
        value = int(value * FEEBLE_SHARE) if self.feeble else value
        if self.terrain == "avian" and self.species != "plant" and name in ("size", "weapons", "armour"):
            value = int(value * AVIAN_BODY)
        return value

    def physique(self, trait: str) -> float:
        """A body trait, as it is now, as a share of its full range."""
        return share(self.trait(trait))

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
    def scavenger(self) -> float:
        """How much a carrion-eater it is: 0 up to the middle of its carrion trait's range, rising to 1 at the top."""
        return max(0.0, 2 * self.physique("carrion") - 1)

    @property
    def herd_instinct(self) -> float:
        """How strongly it herds: the lower of its current empathy and patience, 0 to 1."""
        return min(self.feeling("empathy"), self.feeling("patience"))

    def feeling(self, trait: str) -> float:
        """A temperament trait, as it is now, as a share of its full range."""
        return share(self.mind[trait])

    @property
    def max_health(self) -> float:
        """Its full health: by its size, and FEEBLE_SHARE of that while feeble."""
        return (60.0 + 90 * self.physique("size")) * (FEEBLE_SHARE if self.feeble else 1.0)

    @property
    def metabolism(self) -> float:
        """Metabolic rate: fast in small bodies, slow in large ones."""
        return _metabolism(self.physique("size"))

    @property
    def stomach(self) -> float:
        """How much food the body holds: hunger is a share of it, so food needs grow with size."""
        smallest, largest = STOMACH
        return smallest + (largest - smallest) * self.physique("size")

    @property
    def burn(self) -> float:
        """Hunger gained each turn, as a share of its stomach: a faster metabolism, and a stronger,
        faster, better-armed body, empty it sooner, and so does a cleverer animal's brain (Brain cost)."""
        body = self.metabolism * (0.45 + 0.18 * self.physique("strength") + 0.24 * self.physique("speed")
                                  + 0.12 * (self.physique("weapons") + self.physique("armour")))
        if self.species == "plant":
            return body
        body *= 1 + SIZE_HUNGER * self.physique("size") ** 2  # a large body is hungry work (Size hunger)
        cost = BRAIN_COST * (PREDATOR_BRAIN_COST if self.species == "predator" else 1.0)
        burn = body * (1 + cost * self.inborn["intellect"] / TRAIT_MAX)
        return burn * (1 + FLIGHT_HUNGER) if self.terrain == "avian" else burn  # flight is hungry work

    @property
    def might(self) -> float:
        """Fighting strength: its strength trait and, strongly, its size."""
        return (1 - SIZE_IN_MIGHT) * self.physique("strength") + SIZE_IN_MIGHT * self.physique("size")

    @property
    def toughness(self) -> float:
        """How well its body turns blows: its armour trait and, strongly, its size."""
        return (1 - SIZE_IN_MIGHT) * self.physique("armour") + SIZE_IN_MIGHT * self.physique("size")

    def nourish(self, food: float, meat: bool = False) -> None:
        """Eat: the same food fills a small stomach more than a large one. Meat is kept track of, as it
        lasts longer."""
        eased = self.relieve(food / self.stomach)
        if meat:
            self.meat_fill += eased

    def relieve(self, amount: float) -> float:
        """Ease its hunger by an amount, eating, and take SATED_JOY's pleasure in what was eased. Returns
        what was eased."""
        eased = min(self.hunger, amount)
        self.hunger -= eased
        self.reward += SATED_JOY * eased / 100
        return eased

    def digest(self, burn: float) -> float:
        """The hunger a turn's burn brings: the share of its fullness that is meat empties at MEAT_DIGESTION of
        the rate, so a meal of meat keeps it fed longer."""
        fullness = 100 - self.hunger
        self.meat_fill = min(self.meat_fill, max(0.0, fullness))
        meat = self.meat_fill / fullness if fullness > 0 else 0.0
        self.meat_fill = max(0.0, self.meat_fill - burn * meat * MEAT_DIGESTION)
        return burn * (1 - meat * (1 - MEAT_DIGESTION))

    @property
    def space(self) -> int:
        """The spaces of a square it takes: 1 + its size."""
        return 1 + self.trait("size")

    def heal(self, share: float) -> None:
        """Heal a share of the health pool, faster with a faster metabolism."""
        self.health = min(self.max_health, self.health + self.max_health * share * self.metabolism)

    @property
    def steps(self) -> int:
        """Cells moved a step from where it stands: 1 to 4 by speed, and half as many again for grazers; for the
        amphibious on land, AMPHIBIOUS_SPEED of that, at least one."""
        steps = 1 + self.trait("speed") * 3 // TRAIT_MAX
        steps = int(steps * GRAZER_SPEED + 0.5) if self.species == "grazer" else steps
        if self.terrain == "amphibious" and self.species != "plant" and not self.wet:
            steps = max(1, int(steps * AMPHIBIOUS_SPEED + 0.5))
        return steps

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
        return 2 + SIGHT_PER_TRAIT * self.trait("sight")

    @cached_property
    def genes(self) -> dict[str, float]:
        """Its brain's inputs for its genome: its ring's and dot's traits and its climate genes."""
        return gene_senses(self.dna)

    @property
    def lifespan(self) -> float:
        """Expected turns of life, plant or animal: a slower metabolism, that is a larger body,
        lives longer. It is its genes' body that counts, feeble or not."""
        shortest, middle, longest = LIFESPAN_YEARS
        fastest, slowest = METABOLISM
        slowing = (fastest - _metabolism(share(int(self.body["size"])))) / (fastest - slowest)  # 0 at the fastest metabolism, 1 the slowest
        if slowing <= 0.5:
            years = shortest + (middle - shortest) * (2 * slowing)
        else:
            years = middle + (longest - middle) * (2 * slowing - 1)
        return years * YEAR_LENGTH

    @property
    def breeding_cost(self) -> float:
        """Hunger it pays to breed: less the more fertile it is, and HUNTER_BREEDING_COST of that for a hunter,
        GRAZER_BREEDING_COST for a grazer; less again while its species is scarce (cost_share)."""
        hunter = (HUNTER_BREEDING_COST if self.species in ("omnivore", "predator")
                  else GRAZER_BREEDING_COST if self.species == "grazer" else 1.0)
        return (BREEDING_COST_MOST - 15 * self.physique("fertility")) * hunter * self.cost_share

    @property
    def can_breed(self) -> bool:
        """In heat, rested, not carrying young, and able to pay for breeding with energy to spare."""
        return (self.in_heat and self.cooldown == 0 and not self.womb
                and self.hunger + self.breeding_cost <= 100 - BREED_RESERVE)

    @property
    def digestion(self) -> float:
        """How much of the meat in carrion nourishes it."""
        return 0.2 + 0.8 * self.physique("carrion")

    @property
    def alive(self) -> bool:
        """Full hunger doesn't kill at once: it wears its health away (STARVING_DAMAGE)."""
        return self.health > 0

    @property
    def dot_color(self) -> tuple[int, int, int]:
        return color_from_word(mind_word(self.dna))

    @property
    def ring_color(self) -> tuple[int, int, int]:
        return color_from_word(body_word(self.dna))

    @property
    def social_color(self) -> tuple[int, int, int]:
        return color_from_word(social_word(self.dna))

    @property
    def brain_bytes(self) -> int:
        """The memory its brain, and those of the young it carries, take."""
        return sum(brain.nbytes for brain in (self.brain, *(young[2] for young in self.womb)) if brain is not None)

    def copy(self) -> Bitzkey:
        return replace(self, brain=self.brain.copy() if self.brain else None, mind=self.mind.copy(),
                       trail=list(self.trail), memories=list(self.memories),
                       womb=[(dna, dict(mind), brain.copy() if brain else None, energy, generation, father)
                             for dna, mind, brain, energy, generation, father in self.womb])


def litter_size(parents: Sequence[tuple[int, int]]) -> int:
    """Young per birth, before space limits it, from each parent's (litter trait, size): a high
    litter trait and a small body both make for bigger litters, and the smallest bodies have
    SMALL_LITTERS times as many, falling evenly to the usual for the largest."""
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


class Death(NamedTuple):
    """A death: its cause, the dead Bitzkey's species and the turn, and, when something killed it, the killer's
    identifier and species and what the killer was doing (hunting, attacking, defending, browsing, uprooting, or
    spines: struck, its spines killed its attacker)."""
    cause: str
    species: str
    turn: int
    killer: int | None = None
    killer_species: str | None = None
    activity: str | None = None
    age_days: int = -1  # how many days it lived; -1 unknown


def add_instincts(agent: Bitzkey, valued: list[float], options: list[int], hunger: float, hungriest_young: float) -> list[float]:
    """Its instincts, on top of what its brain values its `options` at: to hunt and feast, the mating call, a scavenger's
    craving, loyalty to its herd, an infant's to its kind, and a parent's to feed its young (the hungriest that hungry).
    As the C++ version's."""
    valued = list(valued)
    if agent.species == "predator":  # the instinct to hunt, on top of what its brain has learnt
        urge = HUNTING_INSTINCT * min(1.0, hunger / 100)
        valued = [value + urge if action in (HUNT, ATTACK) else value for action, value in enumerate(valued)]
        valued[FEAST] += BLOODLUST  # bloodlust: a fresh kill in sight is hard to pass by
    call = MATING_CALL if agent.species == "predator" else OMNIVORE_MATING_CALL if agent.species == "omnivore" else 0.0
    if call and agent.can_breed:  # the mating call: to breed if it can, else to court a mate it seeks
        valued[BREED if BREED in options else COURT] += call
    scavenger = agent.scavenger
    valued[SCAVENGE] += CARRION_CRAVING * scavenger  # a carrion-eater's craving
    valued[HUNT] -= CARRION_AVERSION * scavenger  # and its aversion to killing for itself
    valued[ATTACK] -= CARRION_AVERSION * scavenger
    valued[HERD] += LOYALTY_PULL * agent.feeling("loyalty")  # loyalty: keeping to its herd
    if agent.infant:  # an infant keeps close to its kind
        valued[HERD] += INFANT_HERDING
    if FEED_YOUNG in options:
        # The instinct to feed its hungry young: closing on the best of its other options, however much those are worth.
        others = [valued[option] for option in options if option != FEED_YOUNG]
        gap = max(0.0, max(others) - valued[FEED_YOUNG]) if others else 0.0
        valued[FEED_YOUNG] += PARENTAL_INSTINCT * (0.5 + agent.feeling("empathy")) * hungriest_young / 100 * gap
    return valued


class Simulation:
    def __init__(self, width: int = GRID_SIZE, height: int = GRID_SIZE,
                 population: int = INITIAL_POPULATION, seed: int | None = None,
                 grass_density: float = GRASS_DENSITY, imported: Sequence[Imported] = (),
                 population_cap: int | None = None, mind_mutation: float = 100 * MIND_MUTATION_CHANCE,
                 body_mutation: float = 100 * BODY_MUTATION_CHANCE, year_length: int | None = None,
                 ocean: float = 0.0):
        """Spawn `population` random Bitzkeys in groups sharing a genome, plus a group for each
        `imported` Bitzkey, whose members all start with copies of its saved brain if it has one.
        `population_cap` limits living animals; None leaves crowding alone to limit them.
        `mind_mutation` and `body_mutation` are the percent chances that a birth mutates the
        child's mind or body. `year_length` is the turns in a year, for the seasons and lifespans;
        like the advanced rules, it holds for the whole program while the world runs. `ocean` is the percent of the
        map under the sea, laid out by the seed; none unless asked for (a new world's setup offers DEFAULT_OCEAN)."""
        global YEAR_LENGTH
        year_length = DEFAULT_YEAR_LENGTH if year_length is None else year_length
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
        if not 0 <= ocean <= 100:
            raise ValueError("The sea must cover between 0 and 100 percent.")
        self.width = width
        self.height = height
        self.random = random.Random(seed)
        self.ocean = ocean
        self.sea_map = bytearray()  # 1 for each square under the sea, row by row; empty without sea
        if ocean > 0:
            self._make_sea(ocean, random.SystemRandom().getrandbits(64) if seed is None else seed)
        self.insights = Insights()
        # Who was alive, over the last GRAPH_TURNS turns, for the colour graph.
        self.census = Census()
        # The genome of every Bitzkey spawned from lineage_first on, by identifier, so the graph can
        # colour and export Bitzkeys long gone; those before it are off the graph and forgotten.
        self.lineage: dict[int, int] = {}
        self.lineage_first = 0
        # Each death, by the dead Bitzkey's identifier: back to the graph's start, for the graph and the death report.
        self.death_records: dict[int, Death] = {}  # each death of the last RECENT_DEATH_TURNS turns
        self.death_tally = DeathTally()  # every death since the world began, counted
        self._death_order: deque[tuple[int, int]] = deque()  # (turn, identifier) of each death, oldest first
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
        self._band_of_row: list[int] = []   # the climate band each row lies in, top to bottom
        self._home_band: list[list[int]] = []  # by home temperature and row: the home band nearest
        self._band_plants: list[int] = []   # the plants in each band, and the grazers at home there,
        self._band_grazers: list[int] = []  # as the cells were last indexed
        self._band_prey: list[int] = []       # the prey in each band, and the predators at home there, likewise
        self._band_predators: list[int] = []
        self._band_omnivores: list[int] = []       # the omnivores at home in each band, and what they hunt there:
        self._band_omnivore_prey: list[int] = []   # grazers and predators
        self._living: Counter[str] = Counter()  # the living of each species, as the cells were last indexed
        self._grass_rows: list[int] = []    # the rows where grass grows at all
        self._grass_reach: list[float] = []  # their growth added up, row by row, for sowing
        self._update_climate()
        self.grass: dict[tuple[int, int], int] = {}
        initial_grass = round(width * height * grass_density / 100)
        self._initial_grass = initial_grass  # what grows back each turn is a share of it
        self.grass_regrowth = initial_grass * GRASS_REGROWTH
        self._sow(initial_grass)
        sizes = self._group_sizes(population)
        species = [self.random.randrange(len(SPECIES)) for _ in sizes]
        homes = self._plant_homes(species)
        for group, size in enumerate(sizes):
            self._spawn_group(make_dna(species[group], self.random), size, native=True, rows=homes.get(group))
        if population > 0:
            self._spawn_biomes()
        for bitzkey in imported:
            self._spawn_group(bitzkey.dna, self.random.randint(*SPAWN_GROUP), bitzkey.brain, bitzkey.mind)
        # What herd-mates saw this turn, to learn from at its end: (observer, senses, choice, outcome, rate).
        self._witnessed: list[tuple] = []
        self._take_census()

    def _make_sea(self, ocean: float, seed: int) -> None:
        """Lay out the sea over `ocean` percent of the map, from its own stream of `seed`, so the world's own draws
        are left as they were: value noise in octaves, each twice as fine and half as strong as the last, wrapping
        round east and west, and the lowest of it sea."""
        rng = random.Random(seed ^ 0x5EA5EA5EA5EA5EA5)
        width, height = self.width, self.height
        level = [0.0] * (width * height)
        columns = max(2, int(1 / CONTINENT_BREADTH + 0.5))
        rows = max(1, int(columns * height / width + 0.5))
        strength = 1.0
        for octave in range(4):
            across, down = columns << octave, rows << octave
            lattice = [rng.random() for _ in range(across * (down + 1))]
            for y in range(height):
                fy = (y + 0.5) / height * down
                j = min(down - 1, int(fy))
                t = fy - j
                ty = t * t * (3 - 2 * t)
                for x in range(width):
                    fx = (x + 0.5) / width * across
                    i = int(fx)
                    t = fx - i
                    tx = t * t * (3 - 2 * t)
                    a, b = lattice[j * across + i % across], lattice[j * across + (i + 1) % across]
                    c, d = lattice[(j + 1) * across + i % across], lattice[(j + 1) * across + (i + 1) % across]
                    top = a + (b - a) * tx
                    bottom = c + (d - c) * tx
                    level[y * width + x] += strength * (top + (bottom - top) * ty)
            strength /= 2
        order = sorted(range(width * height), key=lambda index: (level[index], index))
        self.sea_map = bytearray(width * height)
        for index in order[:min(round(width * height * ocean / 100), width * height)]:
            self.sea_map[index] = 1

    def night_start(self) -> int:
        """The westernmost column of the night, which runs east half the map's width from it."""
        return int(-(self.turn % DAY_LENGTH) * self.width / DAY_LENGTH) % self.width  # as the C++ version divides

    def night(self, x: int) -> bool:
        """Whether a column lies in the night now: half the map, moving west a whole turn round each day."""
        return (x - self.night_start()) % self.width < self.width // 2

    def _known_senses(self, agent: Bitzkey, caller: Bitzkey | None, sight: int) -> dict[str, float]:
        """Those it knows: the nearest within its sight, and whether it knows the loudest caller, as the C++ version."""
        found: dict[str, float] = {}
        nearest, regard = None, 0.0
        for known in agent.known:
            other = self.find_agent(known.identifier)
            if caller is not None and known.identifier == caller.identifier:
                found.update(caller_known=1.0, caller_regard=known.regard)
            if other is None or not other.alive or abs(other.x - agent.x) + abs(other.y - agent.y) > sight:
                continue
            if nearest is None or (abs(other.x - agent.x) + abs(other.y - agent.y)
                                   < abs(nearest.x - agent.x) + abs(nearest.y - agent.y)):
                nearest, regard = other, known.regard
        if nearest is not None:
            found.update(known_closeness=self._closeness(agent, (nearest.x, nearest.y), sight), known_regard=regard,
                         known_east=max(-1.0, min(1.0, (nearest.x - agent.x) / sight)),
                         known_south=max(-1.0, min(1.0, (nearest.y - agent.y) / sight)))
        return found

    def _note_deed(self, agent: Bitzkey, other: Bitzkey, deed: float) -> None:
        """What `other` did to `agent`, worth `deed`, -1 to 1, noted in its regard for them."""
        known = agent.known
        while len(known) > max(0, ACQUAINTANCES):
            known.remove(min(known, key=lambda one: one.last))
        if ACQUAINTANCES <= 0 or agent is other:
            return
        found = next((one for one in known if one.identifier == other.identifier), None)
        if found is None:
            if len(known) >= ACQUAINTANCES:
                known.remove(min(known, key=lambda one: one.last))
            known.append(Acquaintance(other.identifier, deed, self.turn))
            return
        found.regard += REGARD_LEARNING * (deed - found.regard)
        found.last = self.turn

    def _map_senses(self, agent: Bitzkey) -> dict[str, float]:
        """Its mental map: the best place it remembers for food this season, the worst for danger, the best for comfort
        next season, as the C++ version."""
        quarter = season(self.turn)
        found: dict[str, float] = {}
        for kind, in_season in (("food", quarter), ("danger", quarter), ("comfort", (quarter + 1) % 4)):
            best = None
            for place in agent.places:
                if not place.seen[in_season]:
                    continue
                east = (place.column * MAP_PATCH + MAP_PATCH // 2 - agent.x + self.width // 2) % self.width - self.width // 2
                south = place.row * MAP_PATCH + MAP_PATCH // 2 - agent.y
                score = getattr(place, kind)[in_season] - MAP_NEARNESS * (abs(east) + abs(south)) / MAP_PATCH
                if best is None or score > best[0]:
                    best = (score, place, east, south)
            if best is not None:
                _, place, east, south = best
                found[f"map_{kind}"] = getattr(place, kind)[in_season]
                found[f"map_{kind}_east"] = max(-1.0, min(1.0, east / (4.0 * MAP_PATCH)))
                found[f"map_{kind}_south"] = max(-1.0, min(1.0, south / (4.0 * MAP_PATCH)))
        return found

    def _note_place(self, agent: Bitzkey, senses: Sequence[float]) -> None:
        """What an animal finds where it stands, as it senses it, noted on its mental map."""
        places = agent.places
        while len(places) > max(0, MAP_PLACES):  # fewer held since: the longest ago let go
            places.remove(min(places, key=lambda place: place.last))
        if MAP_PLACES <= 0 or not (0 <= agent.x < self.width and 0 <= agent.y < self.height):
            return
        column, row = agent.x // MAP_PATCH, agent.y // MAP_PATCH
        place = next((place for place in places if (place.column, place.row) == (column, row)), None)
        if place is None:
            if len(places) >= MAP_PLACES:
                places.remove(min(places, key=lambda place: place.last))
            place = MapPlace(column, row, self.turn)
            places.append(place)
        sense = dict(zip(SENSES, senses))
        food = 0.0
        if agent.species != "predator":
            food = max(sense["grass_here"], sense["grass_closeness"], sense["plant_closeness"])
        if agent.species != "grazer":
            food = max(food, sense["prey_closeness"], sense["kill_closeness"], sense["carrion_closeness"])
        found = {"food": food, "danger": max(sense["threat_closeness"], sense["assailant_near"]),
                 "comfort": 1 - sense["climate_discomfort"]}
        quarter = season(self.turn)
        for kind, value in found.items():
            kept = getattr(place, kind)
            kept[quarter] = value if not place.seen[quarter] else kept[quarter] + MAP_LEARNING * (value - kept[quarter])
        place.seen[quarter] = True
        place.last = self.turn

    def _heard(self, agent: Bitzkey) -> tuple[dict[str, float], Bitzkey | None]:
        """The calls made about it last turn, by any kind, each as its loudest caller made it, and the loudest it heard of
        all, as the C++ version."""
        heard: dict[str, float] = {}
        loudest = [0.0] * (len(CALLS) - FIRST_CALL)
        caller, loudest_of_all = None, 0.0
        furthest = min(MAX_CALL_REACH, math.ceil(CALL_REACH * 1.5))  # the largest caller's
        for other in self._nearby(agent, furthest):
            if other.call < FIRST_CALL or other.call_turn != self.turn - 1:
                continue
            reach = call_reach(other)
            away = abs(other.x - agent.x) + abs(other.y - agent.y)
            if away > reach:
                continue
            loudness = 1 - away / (reach + 1)
            if loudness > loudest_of_all:
                caller, loudest_of_all = other, loudness
            if loudness <= loudest[other.call - FIRST_CALL]:
                continue
            loudest[other.call - FIRST_CALL] = loudness
            name = f"heard_call_{other.call - FIRST_CALL + 1}"
            east = (other.x - agent.x + self.width // 2) % self.width - self.width // 2
            heard[name] = loudness
            heard[f"{name}_east"] = max(-1.0, min(1.0, east / reach))
            heard[f"{name}_south"] = max(-1.0, min(1.0, (other.y - agent.y) / reach))
            heard[f"{name}_kin"] = float(other.species == agent.species)
        return heard, caller

    def _parent_senses(self, agent: Bitzkey, sight: int) -> dict[str, float]:
        """An infant's nearest living parent within its sight: how close, and which way it lies, as the C++ version."""
        nearest, reach = None, None
        if agent.infant:
            for parent in (agent.mother, agent.father):
                found = self.find_agent(parent) if parent >= 0 else None
                if found is not None and found.alive:
                    gap = abs(found.x - agent.x) + abs(found.y - agent.y)
                    if gap <= sight and (reach is None or gap < reach):
                        nearest, reach = found, gap
        if nearest is None:
            return {"parent_closeness": 0.0, "parent_east": 0.0, "parent_south": 0.0}
        return {"parent_closeness": self._closeness(agent, (nearest.x, nearest.y), sight),
                "parent_east": max(-1.0, min(1.0, (nearest.x - agent.x) / sight)),
                "parent_south": max(-1.0, min(1.0, (nearest.y - agent.y) / sight))}

    def sheltered(self, agent: Bitzkey) -> bool:
        """Whether an infant is sheltered: a living parent within SHELTER_REACH of it. It is spared the climate."""
        if not agent.infant:
            return False
        for parent in (agent.mother, agent.father):
            found = self.find_agent(parent) if parent >= 0 else None
            if found is not None and found.alive and abs(found.x - agent.x) + abs(found.y - agent.y) <= SHELTER_REACH:
                return True
        return False

    def resting_time(self, agent: Bitzkey) -> bool:
        """Whether an animal is in its night where it stands: dark for the diurnal, light for the nocturnal."""
        return agent.nocturnal != self.night(agent.x)

    def sea(self, x: int, y: int) -> bool:
        """Whether a square lies under the sea: none off the map, nor in a world without sea."""
        return bool(self.sea_map) and 0 <= x < self.width and 0 <= y < self.height and bool(self.sea_map[y * self.width + x])

    @property
    def has_sea(self) -> bool:
        return bool(self.sea_map)

    def suits(self, terrain: str, cell: tuple[int, int]) -> bool:
        """Whether a Bitzkey of this terrain may be in a square, sea or land."""
        return fits(terrain, self.sea(*cell))

    def _can_enter(self, agent: Bitzkey, cell: tuple[int, int]) -> bool:
        return fits(agent.terrain, self.sea(*cell))

    def deploy(self, bitzkeys: Sequence[Imported], x: int, y: int) -> list[Bitzkey]:
        """Put DEPLOY_GROUP of each of these Bitzkeys, each with its own copy of its saved brain and
        temperament, within SPAWN_SPREAD cells of a cell of the running world, between its turns. Returns those
        deployed."""
        deployed = []
        for bitzkey in bitzkeys:
            space = 1 + int(decode_body(bitzkey.dna)["size"])
            for _ in range(DEPLOY_GROUP):
                spot = self._spot_near((x + self.random.randint(-SPAWN_SPREAD, SPAWN_SPREAD),
                                        y + self.random.randint(-SPAWN_SPREAD, SPAWN_SPREAD)), space, decode_terrain(bitzkey.dna))
                if spot is not None:  # only where there is room
                    deployed.append(self._add_agent(bitzkey.dna, *spot, brain=bitzkey.brain.copy() if bitzkey.brain else None,
                                                    mind=bitzkey.mind))
        self.census.forget_from(self.turn)  # this turn again, as it now is
        self._take_census()
        return deployed

    def rules_changed(self) -> None:
        """The advanced rules were changed while the world runs: work out afresh what the world
        settled from them as it began, the grass that grows back each turn and its climate bands.
        Everything else reads the rules as it goes."""
        self.grass_regrowth = self._initial_grass * GRASS_REGROWTH
        self._band_of_row = []  # the bands are worked out again as the cells are next indexed
        self._update_climate()

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
                     rows: Sequence[int] | None = None, biome: Sequence[tuple[int, int]] | None = None) -> None:
        """Place `size` Bitzkeys of one genome around a random centre, kept clear of the edges
        so the group is never split across them, and in one of `rows` if given; otherwise, with
        SPAWN_AT_HOME, in its home zone. Each gets its own
        copy of `brain` and starts with the temperament in `mind`, if given. `native` groups have
        random starting genomes: each carries a tolerance flag only with STARTING_TOLERANCE, and
        plants take the usual temperature of their centre's zone as their home. It spawns on land or sea as its
        terrain allows; a world without sea starts with nothing but terrestrial Bitzkeys. Given a biome, it is centred
        on one of its squares, and each member that would fall outside it at the centre."""
        if native and not self.has_sea:
            dna = with_terrain(dna, "terrestrial")
        if rows is None and SPAWN_AT_HOME:
            rows = self._home_rows(decode_climate(dna)[0])
        terrain = decode_terrain(dna)
        centre = biome[self.random.randrange(len(biome))] if biome is not None else self._group_centre(terrain, rows)
        if centre is None:  # nowhere it could live
            return
        x, y = centre
        if native:  # tolerance is rare at the start
            for flag in (HEAT_TOLERANCE, COLD_TOLERANCE):
                dna = dna | flag if self.random.random() < STARTING_TOLERANCE else dna & ~flag
        if native and decode_body(dna)["species"] == "plant":
            dna = dna & ~(0x3 << TEMPERATURE_SHIFT) | zone_temperature(y, self.height) << TEMPERATURE_SHIFT
        space = 1 + int(decode_body(dna)["size"])
        squares = set(biome) if biome is not None else None
        for _ in range(size):
            dx, dy = self.random.randint(-SPAWN_SPREAD, SPAWN_SPREAD), self.random.randint(-SPAWN_SPREAD, SPAWN_SPREAD)
            aim = ((x + dx) % self.width, y + dy)
            if squares is not None and aim not in squares:
                aim = (x, y)  # kept in its biome
            spot = self._spot_near(aim, space, terrain)
            if spot is not None:  # only where there is room
                self._add_agent(dna, *spot, brain=brain.copy() if brain else None, mind=mind)

    def _group_centre(self, terrain: str, rows: Sequence[int] | None) -> tuple[int, int] | None:
        """A square for a group's centre of this terrain, in one of `rows` if given, drawn until one suits it; None if
        none does."""
        for _ in range(1000):
            x = self.random.randrange(SPAWN_SPREAD, self.width - SPAWN_SPREAD)
            y = (self.random.randrange(SPAWN_SPREAD, self.height - SPAWN_SPREAD) if rows is None
                 else rows[self.random.randrange(len(rows))])
            if self.suits(terrain, (x, y)):
                return x, y
        return None

    def biomes(self) -> list[list[tuple[int, int]]]:
        """The biomes, each its squares, in the order they are first met row by row (see BIOME_GROUPS); those smaller
        than BIOME_SIZE left out."""
        band = [0] * self.height
        for index, rows in enumerate(self._zones()):
            for y in rows:
                band[y] = index
        width, height = self.width, self.height
        seen = bytearray(width * height)
        found = []
        for y in range(height):
            for x in range(width):
                if seen[y * width + x]:
                    continue
                seen[y * width + x] = 1
                key = (band[y], self.sea(x, y))
                biome = [(x, y)]  # gathered breadth first, from its first square
                next_ = 0
                while next_ < len(biome):
                    here_x, here_y = biome[next_]
                    next_ += 1
                    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                        nx, ny = (here_x + dx) % width, here_y + dy
                        if not 0 <= ny < height or seen[ny * width + nx] or (band[ny], self.sea(nx, ny)) != key:
                            continue
                        seen[ny * width + nx] = 1
                        biome.append((nx, ny))
                if len(biome) >= BIOME_SIZE:
                    found.append(biome)
        return found

    def _spawn_biomes(self) -> None:
        """BIOME_GROUPS starting groups at home in every biome."""
        for biome in self.biomes():
            wet = self.sea(*biome[0])
            home = zone_temperature(biome[0][1], self.height)
            fitting = ("aquatic", "avian", "amphibious") if wet else ("terrestrial", "avian", "amphibious")
            for group in range(BIOME_GROUPS):
                dna = make_dna(self.random.randrange(len(SPECIES)), self.random)
                dna = dna & ~(0x3 << TEMPERATURE_SHIFT) | home << TEMPERATURE_SHIFT
                # The first the surface's own; the rest any terrain that may live there.
                dna = with_terrain(dna, fitting[0] if group == 0 else fitting[self.random.randrange(3)])
                self._spawn_group(dna, self.random.randint(*SPAWN_GROUP), native=True, biome=biome)

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
        agent.wet = self.sea(x, y)
        if mind:
            agent.mind.update({trait: mind[trait] for trait in TEMPERAMENT if trait in mind})
        if agent.species != "plant":  # plants need no brain
            shape = brain_shape(agent.dna)
            agent.brain = Brain.random(self.random, *shape) if brain is None else brain.resized(*shape)
        self.next_identifier += 1
        self.agents.append(agent)
        self.lineage[agent.identifier] = agent.dna
        self._cells.setdefault((x, y), []).append(agent)
        return agent

    def advance(self) -> None:
        """Step one turn, simulating it, unless the simulation has ended."""
        if self.end_reason:
            return
        self._take_turn()

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
        """At the end of the run: with an end reason."""
        return self.end_reason is not None

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
            full = agent.max_health
            share_before = max(0.0, agent.health) / full
            agent.age += 1
            agent.health = min(agent.health, agent.max_health)  # growing old, it grows feeble
            if agent.max_health != full:  # grown out of infancy, or into old age: the change in its health's share
                agent.last_wellbeing += max(0.0, agent.health) / agent.max_health - share_before  # is no lesson
            plant = agent.species == "plant"
            here = self._climate[agent.y]
            steps = 0.0 if self.sheltered(agent) else agent.climate_steps(here)  # a sheltered infant feels no weather
            hungrier = steps > 0 and (not plant or here < agent.climate[0])  # out of its temperature; a plant, too hot
            share = agent.exposure_share(here)  # less for the hardy
            # The extra food it needs rises with the weather's turning, to the full amount a step away.
            burn = agent.burn * (1 + (CLIMATE_HUNGER - 1) * share * min(1.0, steps) if hungrier else 1)
            agent.hunger = min(100.0, agent.hunger + agent.digest(burn))  # meat lasts; hunger stops at full
            if agent.cooldown:
                agent.cooldown -= 1
            if agent.health < agent.max_health and agent.hunger < 65:
                agent.heal(0.005)
            # A tenth of its health a step, part steps in part; a plant takes one step in its stride,
            # and is hurt more and more as the weather turns toward a second, to full harm two away.
            felt = max(0.0, min(steps, 2 * (steps - 1))) if plant else steps
            # An animal endures the weather awhile, uneasy but unharmed, the more resilient the longer, and twice as
            # long weather it is hardy to.
            home, heat, cold = agent.climate
            hardy = here != home and (cold if here > home else heat)
            grace = 0.0 if plant else WEATHER_GRACE * DAY_LENGTH * (0.5 + agent.feeling("resilience")) * (2 if hardy else 1)
            agent.weathered = agent.weathered + 1 if felt > 0 else max(0.0, agent.weathered - 1)
            # Choosing to sit the weather out hurts the more (Resting exposure); resting asleep or spent was no choice.
            resting = (RESTING_EXPOSURE if agent.last_senses is not None and ACTIONS[agent.last_choice] == "rest"
                       and not agent.last_forced else 1.0)
            if felt > 0 and agent.weathered > grace:
                agent.health -= CLIMATE_DAMAGE * felt * share * resting * agent.max_health
            starving = agent.hunger >= 100
            if starving:  # it wastes away, the larger the faster
                agent.health -= STARVING_DAMAGE * (1 + SIZE_STARVING * agent.physique("size")) * agent.max_health
            if agent.health <= 0:  # starving, and hungrier out of its temperature, it died of exposure
                self._die(agent, "starved" if starving and not hungrier else self._exposure(agent, here))
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
            agent.in_heat = self.heat(agent)
            if agent.due_turn is not None and agent.due_turn <= self.turn:
                self._give_birth(agent)
            agent.seen_success *= SOCIAL_MEMORY
            agent.seen_failure *= SOCIAL_MEMORY ** (1 + RESILIENCE * (2 * agent.feeling("resilience") - 1))  # got over sooner
            self._settle(agent)
            self._nurse(agent)
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
        for agent in self.agents:  # bodies rotted away, not eaten to the last portion, may feed a new plant
            if (agent.died_turn is not None and self.turn - agent.died_turn >= CORPSE_TURNS and agent.species != "plant"
                    and agent.portions > 0 and CORPSE_PLANT and self.random.random() < CORPSE_PLANT):
                self._seed_corpse(agent)
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
        reach = min(MAX_SIGHT, int(SEED_REACH * AVIAN_SEED_REACH)) if plant.terrain == "avian" else SEED_REACH
        dx, dy = SIGHT_OFFSETS[self.random.randrange(seed_offsets(reach))]
        if not 0 <= plant.y + dy < self.height:  # off the top or bottom of the map: the seed is lost
            return
        cell = ((plant.x + dx) % self.width, plant.y + dy)
        if not self._room_for_seedling(cell):
            return
        dna = evolved(plant.dna, cross(plant.dna, plant.dna, self.random, None, self.mind_mutation, self.body_mutation,
                                       self.has_sea))
        if not self.suits(decode_terrain(dna), cell):  # fallen in the sea, or on the land, where it can't grow
            return
        plant.hunger += SEED_COST
        plant.children += 1
        plant.last_action = "seeded"
        if not plant.alive:  # it spent the last of its food on the seed
            self._die(plant, "seeded to death")
        self.seeds.append((*cell, dna, plant.generation + 1, self.turn))

    def _nurse(self, agent: Bitzkey) -> None:
        """A grown animal feeds the hungriest infant of its kind within NURSING_REACH: a turn's burn of food, so the
        infant grows no hungrier, paid from its own stomach, while it stays below NURSING_LIMIT hunger itself."""
        if agent.species == "plant" or agent.infant or not agent.alive:
            return
        infant = None
        for other in self._nearby(agent, NURSING_REACH):
            if (other is not agent and other.alive and other.species == agent.species and other.infant and other.hunger > 0
                    and (infant is None or other.hunger > infant.hunger)):
                infant = other
        if infant is None:
            return
        fed = min(infant.hunger, infant.burn)
        cost = fed * infant.stomach / agent.stomach
        if agent.hunger + cost >= NURSING_LIMIT:
            return
        agent.hunger += cost
        infant.hunger -= fed
        self._note_deed(infant, agent, FEEDING_DEED)

    def _seed_corpse(self, corpse: Bitzkey) -> None:
        """A body rotted away leaves the seed of a new plant where it lay: a random plant genome, at home in the
        temperature there. It sprouts as any seed does, once it has lain PLANT_GESTATION turns and there is room."""
        dna = make_dna(SPECIES.index("plant"), self.random)
        dna = dna & ~(0x3 << TEMPERATURE_SHIFT) | zone_temperature(corpse.y, self.height) << TEMPERATURE_SHIFT
        dna = with_terrain(dna, "aquatic" if self.sea(corpse.x, corpse.y) else "terrestrial")  # in its sea, or on its land
        self.seeds.append((corpse.x, corpse.y, dna, 0, self.turn))

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
            if (self.turn - sown >= PLANT_GESTATION and abs(self._climate[y] - decode_climate(dna)[0]) < 0.5
                    and self.suits(decode_terrain(dna), (x, y)) and self._room_for_seedling((x, y))):
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
        if self.census.record(self.turn, (agent.identifier for agent in self.living)):
            self._forget_before(self.census.snapshots[0][0])
        self._forget_old_deaths()

    def _forget_old_deaths(self) -> None:
        """Forget the deaths kept whole from before the last RECENT_DEATH_TURNS turns, and now and then the killers
        gone."""
        order, start = self._death_order, self.turn - RECENT_DEATH_TURNS
        while order and order[0][0] < start:  # oldest first, so only those forgotten are looked at
            self._forget_death(*order.popleft())
        if self.turn % KILLERS_LET_GO == 0:
            self.death_tally.let_go(self._alive)

    def _alive(self, identifier: int) -> bool:
        agent = self.find_agent(identifier)
        return agent is not None and agent.alive

    def _forget_before(self, start: int) -> None:
        """Forget the deaths before the graph's start, and the genomes of the Bitzkeys neither on
        the graph nor in the world."""
        order = self._death_order
        while order and order[0][0] < start:  # oldest first, so only those forgotten are looked at
            self._forget_death(*order.popleft())
        # Any kept death before the graph's start was alive then, and in its first snapshot, unless it
        # died that very turn; any born since has a later identifier.
        dying_then = []
        for turn, identifier in order:
            if turn > start:
                break
            dying_then.append(identifier)
        first = min((*self.census.snapshots[0][1], *dying_then, *(agent.identifier for agent in self.agents)),
                    default=self.next_identifier)
        for identifier in range(self.lineage_first, first):
            self.lineage.pop(identifier, None)
        self.lineage_first = max(self.lineage_first, first)

    def _forget_death(self, turn: int, identifier: int) -> None:
        """Forget a death, unless another has since been recorded under its identifier."""
        death = self.death_records.get(identifier)
        if death is not None and death.turn == turn:
            del self.death_records[identifier]

    def _order_deaths(self) -> None:
        """The deaths' order, oldest first, from the records alone, as a loaded world has them."""
        self._death_order = deque(sorted((death.turn, identifier) for identifier, death in self.death_records.items()))

    # Temperament

    def _experience(self, agent: Bitzkey, event: str) -> None:
        """Shift temperament after an experience; intellect sets how far, more for the moody and less
        for the steady. The resilient take fright less, and the fragile more."""
        plasticity = (0.2 + 0.8 * agent.intelligence) * (1 + MOODINESS * (2 * agent.feeling("moodiness") - 1))
        for trait, change in EXPERIENCES[event].items():
            if trait == "fear" and change > 0:
                change *= 1 + RESILIENCE * (1 - 2 * agent.feeling("resilience"))
            agent.mind[trait] = max(0.0, min(float(TRAIT_MAX), agent.mind[trait] + change * TRAIT_MAX * plasticity))

    @staticmethod
    def _settle(agent: Bitzkey) -> None:
        """Let learned temperament drift slowly back toward the inborn values."""
        for trait in TEMPERAMENT:
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
            self._note_place(agent, senses)
            if agent.last_action in TRAVELLED:
                if situation["threat"]:
                    self._experience(agent, "ambushed")
                elif (situation["grass"] or situation["grass_here"]) and not agent.last_senses[GRASS_SEEN]:
                    self._experience(agent, "discovery")
            plans.append((agent, senses, options, situation))
        # A brain idle this turn isn't asked: one that can only rest (asleep with nothing to startle it, or spent),
        # with no lesson waiting on what it values now, its last turn having been as forced. Nothing it would say is used.
        startling = (ACTIONS.index("flee"), ACTIONS.index("defend"))
        idle = [(agent.exhausted or (SLEEP and self.resting_time(agent) and not agent.aloft_at_sea
                                     and not any(option in startling for option in options)))
                and not (agent.last_senses is not None and (FORCED_LESSONS or not agent.last_forced or agent.last_panicked))
                for agent, _, options, _ in plans]
        # The other brains, of a shape, are worked as one batch: valued together, then taught together.
        groups: dict[tuple[int, int], list[int]] = {}
        for index, (agent, *_) in enumerate(plans):
            if not idle[index]:
                groups.setdefault((agent.brain.depth, agent.brain.width), []).append(index)
        batches = []
        values: list[list[float]] = [[0.0] * len(OUTPUTS) for _ in plans]
        for members in groups.values():
            batch = BrainBatch([plans[index][0].brain for index in members])
            for row, valued in zip(members, batch.values(np.array([plans[index][1] for index in members])).tolist()):
                values[row] = valued
            batches.append((batch, members))
        comforts = [self._comfort(agent, situation) for agent, _, _, situation in plans]
        lessons: dict[int, tuple] = {}  # by plan: what came of its last choice, to learn from
        for index, ((agent, _, options, _), valued, comfort) in enumerate(zip(plans, values, comforts)):
            if agent.last_senses is not None and (FORCED_LESSONS or not agent.last_forced or agent.last_panicked):  # a panicked flight teaches
                # What came of its last choice: whatever raised its health (fullness counting with it) or
                # its mood was a success, whatever lowered them a failure, by how much; both may happen.
                reward = (agent.reward + self._wellbeing(agent) - agent.last_wellbeing
                          + comfort - agent.last_comfort)
                # Having chosen to sit the weather out, it feels it the more (Resting unease); asleep or spent, it had
                # no choice.
                if ACTIONS[agent.last_choice] == "rest" and not agent.last_forced:
                    reward -= RESTING_UNEASE * plans[index][3].get("exposure", 0.0)
                best_call = max(valued[call_output(call)] for call in (SILENT, *plans[index][3]["calls"]))
                lessons[index] = (reward + self._discount(agent) * max(valued[option] for option in options), reward,
                                  reward + self._discount(agent) * best_call)
                if reward > 0:  # a lesson learnt from a choice that turned out well
                    self._experience(agent, "lesson")
        for batch, members in batches:
            self._learn_batch(batch, [(plans[index][0], *lessons[index]) if index in lessons else None for index in members])
        actions = []
        for index, ((agent, senses, options, situation), valued, comfort) in enumerate(zip(plans, values, comforts)):
            calls_valued = valued
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
            if agent.exhausted:  # spent: it can do nothing but rest, whatever it fears
                situation["panic"] = situation["frozen"] = False
                options = [ACTIONS.index("rest")]
            elif (SLEEP and self.resting_time(agent) and not agent.aloft_at_sea and not situation.get("panic")
                  and not situation.get("frozen")):  # its night: it rests, unless startled awake by a hunter or a blow
                options = [ACTIONS.index("rest")] + [option for option in options if ACTIONS[option] in ("defend", "flee")]
            hungriest = max((min(100.0, child.hunger) for child in situation.get("young", ())), default=0.0)  # of its infants about it
            valued = add_instincts(agent, valued, options, agent.hunger, hungriest)
            odds = choice_odds(valued, options)
            roll = self.random.random()
            explore = self.random.random()
            restless = self.random.random()
            call_roll = self.random.random()
            # In peril, with a hunter or threat near, struck, or starving, it heeds its brain alone: curiosity pales.
            peril = bool(situation["hunter"] or situation["threats"] or situation["assailant"] or agent.hunger >= 100)
            curiosity = agent.feeling("curiosity")
            choice = options[-1]
            if not peril and len(options) > 1 and explore < CURIOUS_EXPLORATION * curiosity:
                # Curious, it tries any of its options, whatever its brain makes of them.
                choice = options[min(len(options) - 1, int(roll * len(options)))]
            else:
                for option in options:
                    roll -= odds[option]
                    if roll < 0:
                        choice = option
                        break
                restlessness = RESTLESS_CURIOSITY * (0.5 + curiosity)
                if not peril and ACTIONS[choice] == "rest" and len(options) > 1 and restless < restlessness:
                    # Restless, it does any of its other options instead of resting.
                    others = [option for option in options if ACTIONS[option] != "rest"]
                    choice = others[min(len(others) - 1, int(restless / restlessness * len(others)))]
            action = ACTIONS[choice]
            if not situation.get("panic") and not situation.get("frozen"):  # a panicked flight, or freezing, was no choice
                self.insights.record(agent.species, agent.profile, (ACTIONS[option] for option in options), action)
            agent.odds = {ACTIONS[option]: chance for option, chance in odds.items()}
            # And what it calls as it does it: by its brain alone, one asleep, spent or frozen with fear keeping silent.
            agent.call, agent.last_call = SILENT, -1
            if not idle[index] and not situation.get("frozen"):
                for output, chance in choice_odds(calls_valued, [call_output(call) for call in situation["calls"]]).items():
                    call_roll -= chance
                    agent.call = output - len(ACTIONS)
                    if call_roll < 0:
                        break
                agent.last_call = agent.call
                agent.call_turn = self.turn
            # What its memory now holds, written as it chose, unless its brain wasn't asked (asleep or spent, it keeps what
            # it held); and the senses it chose by before, while each choice since wrote its memory, to learn back in time.
            if agent.last_senses is not None and agent.last_wrote and MEMORY_STEPS > 0:
                agent.past = [agent.last_senses, *agent.past][:MEMORY_STEPS]
            else:
                agent.past = []
            agent.last_wrote = not idle[index]
            if not idle[index]:
                agent.held = memory_after(calls_valued, agent.held)
            agent.last_senses = senses
            agent.last_choice = choice
            agent.last_forced = len(options) == 1
            agent.last_panicked = bool(situation.get("panic"))
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
            if agent.species == "omnivore":
                agent.hunts_predators = self._other_food(agent) <= OMNIVORE_PREDATOR_FOOD

    def _sense(self, agent: Bitzkey, surveyed: bool = False) -> tuple[list[float], list[int], dict]:
        """What an animal senses, the actions open to it, and its situation, surveying its territory
        first unless it already has this turn."""
        if not surveyed:
            self._survey(agent)
        neighbors = self._nearby(agent, 1)
        # Close enough to breed with: the most dominant willing partner, the first found on a tie.
        mate = max((other for other in self._nearby(agent, MATE_REACH) if self._compatible(agent, other) and other.can_breed),
                   key=lambda other: other.mind["dominance"], default=None)
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
            suitor = next((other for other in self._nearby(agent, int(MATING_CALL_RANGE * sight))
                           if self._compatible(agent, other) and other.can_breed), None)
        assailant = next((other for other in neighbors if other.identifier == agent.assailant), None)
        herd_assailant = next((other for other in neighbors if other.identifier == agent.herd_assailant), None)
        defend_target = assailant or herd_assailant
        herd, herd_near, herd_middle = self._herd(agent)
        own_young = [] if not agent.children else [  # its own young, still infants, within its sight
            other for other in self._nearby(agent, sight)
            if agent.identifier in (other.mother, other.father) and other.infant]
        young = len(own_young)
        here = self._climate[agent.y]
        discomfort = agent.climate_steps(here)
        # The nearest row it would feel at home in, however far, and the same only if within sight.
        home = self._comfort_row(agent, self.height) if discomfort else None
        comfort = home if home is not None and abs(home[1] - agent.y) <= sight else None
        home_distance = (0.0 if not discomfort else 1.0 if home is None
                         else min(1.0, abs(home[1] - agent.y) / (0.5 * self.height)))
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

        available = {"go_east", "go_west", "walk_east", "walk_west"}
        if not agent.aloft_at_sea:  # a flyer can't rest over the sea
            available.add("rest")
        shores = [0.0] * 4  # north, south, east and west: the nearest shore it can't cross
        if self.has_sea and agent.terrain not in ("avian", "amphibious"):
            for heading, (dx, dy) in enumerate(((0, -1), (0, 1), (1, 0), (-1, 0))):
                for cells in range(1, sight + 1):
                    y = agent.y + dy * cells
                    if not 0 <= y < self.height:
                        break
                    if not self._can_enter(agent, ((agent.x + dx * cells) % self.width, y)):
                        shores[heading] = 1 - (cells - 1) / sight
                        break
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
        # A pack keeps to its temperature as a herd does: a predator courts and hunts only toward a row no further
        # from its home temperature than where it stands.
        pack = agent.species == "predator"
        if suitor and self._may_breed(agent, suitor) and (not pack or self._homeward(agent, suitor.y, discomfort)):
            available.add("court")  # courting only a partner it could then breed with
        if mate and nurseries and self._may_breed(agent, mate):
            available.add("breed")
        if quarry and (not pack or self._homeward(agent, quarry.y, discomfort)):
            available.add("hunt")
        if target:
            available.add("attack")
        if hunter and (alarm_source is not None or abs(hunter.x - agent.x) + abs(hunter.y - agent.y)
                       <= self._flee_range(agent, herd_near)):
            available.add("flee")  # only from a hunter close by, unless the herd raised the alarm
        if defend_target and (agent.species != "grazer" or agent.trait("strength") > GRAZER_DEFENCE_STRENGTH):
            available.add("defend")  # a grazer only when strong enough
        spread = 1 if agent.infant else HERD_SPREAD * (1 - agent.fearfulness)  # an infant wants to be right among them
        if herd_middle is not None and abs(herd_middle[0] - agent.x) + abs(herd_middle[1] - agent.y) > spread:
            # The more fearful it is, the closer it keeps to the herd's middle, but not where the
            # middle is further from its home temperature than here.
            if self._homeward(agent, herd_middle[1], discomfort):
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
        fed_at = self._remembered(agent, agent.food_memory)
        if fed_at and agent.food_memory[:2] != (agent.x, agent.y):
            available.add("revisit")  # back to where it last fed, while it remembers
        if not agent.infant and any(child.hunger >= FEED_HUNGER for child in own_young):
            available.add("feed_young")  # once one of its infants about it is hungry, it may feed them
        if agent.y > 0:
            available.update(("go_north", "walk_north"))
        if agent.y < self.height - 1:
            available.update(("go_south", "walk_south"))

        heard, caller = self._heard(agent)
        senses = make_senses(
            hunger=min(1.0, agent.hunger / 100),
            health=max(0.0, agent.health / agent.max_health),
            grass_here=float(grass_here),
            grass_seen=float(grass is not None),
            grass_closeness=self._closeness(agent, grass, sight),
            mate_near=float(mate is not None),
            mate_closeness=self._closeness(agent, suitor and (suitor.x, suitor.y), sight),
            ready_to_breed=float(agent.cooldown == 0 and agent.in_heat),  # rested, and in heat
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
                            max(0.0, 1 - (abs(herd_middle[0] - agent.x) + abs(herd_middle[1] - agent.y)) / agent.sight)),
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
            food_remembered=fed_at,
            danger_remembered=(self._remembered(agent, agent.danger_memory)
                               * max(0.0, self._closeness(agent, agent.danger_memory and agent.danger_memory[:2], sight))),
            home_food=self._home_food_sense(agent),
            home_direction=0.0 if home is None else (1.0 if home[1] > agent.y else -1.0),
            home_distance=home_distance,
            young_near=min(3, young) / 3,
            sea_here=float(agent.wet),
            shore_north=shores[0], shore_south=shores[1], shore_east=shores[2], shore_west=shores[3],
            stamina=agent.stamina / FULL_STAMINA,
            night=float(self.night(agent.x)),
            **self._parent_senses(agent, sight),
            **heard,
            **{f"memory_{neuron + 1}": held for neuron, held in enumerate(agent.held)},
            **self._map_senses(agent),
            **self._known_senses(agent, caller, sight),
            **{trait: agent.feeling(trait) for trait in TEMPERAMENT},
            **agent.genes,
        )
        situation = {"mate": mate, "nurseries": nurseries, "young_near": young, "young": own_young, "prey": target, "threat": threat,
                     "threats": len(threats), "quarry": quarry, "hunter": hunter, "suitor": suitor, "suitor_at": suitor and (suitor.x, suitor.y),
                     "grass": grass, "grass_here": grass_here, "carrion": carrion, "kill": kill, "flora": flora, "assailant": assailant, "rivalry": agent.rivalry, "kin": kin,
                     "others": others, "defend_target": defend_target, "for_herd": assailant is None,
                     "herd": herd, "herd_near": herd_near, "herd_middle": herd_middle,
                     "exposure": min(2, discomfort) / 2 * agent.exposure_share(here), "uprootable": uprootable,
                     "home_distance": home_distance * agent.exposure_share(here),
                     "alert_to": alert_to, "revisit": agent.food_memory and agent.food_memory[:2],
                     "calls": [SILENT, *([ALERT_CALL] if alert_to is not None else []), *range(FIRST_CALL, len(CALLS))]}
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
        crowded = (RIVAL_UNEASE * (1 - 0.5 * tolerance) * situation.get("rivalry", 0.0)
                   * (1 + DOMINANCE_EASE * (1 - 2 * agent.feeling("dominance"))))  # the submissive feel it most
        herding = HERD_JOY * instinct * herd_near
        if not situation.get("herd", True) and not solitary(agent.species):  # a loner misses no herd
            herding -= HERD_LONELINESS * instinct
        herding *= 0.5 + agent.fearfulness  # the fearful feel their herd, and its absence, most
        # and more the further home lies, so every step toward it is felt, not only the last.
        exposed = CLIMATE_UNEASE * situation.get("exposure", 0.0) + HOMEWARD_UNEASE * situation.get("home_distance", 0.0)
        # A parent is glad of its own young close by, the more so the more empathetic.
        parental = PARENT_JOY * (0.5 + agent.feeling("empathy")) * min(3, situation.get("young_near", 0)) / 3
        return company - dread - unease - crowded + herding - exposed + parental

    @staticmethod
    def _discount(agent: Bitzkey) -> float:
        """How much the future counts against the present, a turn: more for the patient. The discounts and trace
        decays are set for a turn of a FORESIGHT_DAY-turn day, and kept to the same span of days whatever the day's
        length."""
        return per_turn(IMPATIENT_DISCOUNT + (PATIENT_DISCOUNT - IMPATIENT_DISCOUNT) * agent.feeling("patience"))

    @staticmethod
    def _trace_decay(agent: Bitzkey) -> float:
        """How much of a choice's credit carries on to the next, a turn, beside the discount: more for the clever."""
        return per_turn(brain_rules.TRACE_DECAY + (brain_rules.SMART_TRACE_DECAY - brain_rules.TRACE_DECAY) * agent.intelligence)

    @staticmethod
    def _learning_rate(agent: Bitzkey) -> float:
        """How fast its brain learns: LEARNING_FLOOR at the least, and more with its intellect, boosted further the
        cleverer it is; a predator's PREDATOR_LEARNING times as fast, an omnivore's with half that bonus."""
        sharpness = agent.intelligence
        hunter = {"predator": PREDATOR_LEARNING, "omnivore": 1 + (PREDATOR_LEARNING - 1) / 2}.get(agent.species, 1.0)
        return LEARNING_RATE * (LEARNING_FLOOR + sharpness * (1 + LEARNING_BOOST * sharpness)) * hunter

    def _learn(self, agent: Bitzkey, target: float, outcome: float, replay: bool = False) -> None:
        """Learn from its own last choice, by TD(lambda), so the choices before it share in what came of it; its
        `outcome` (what came of it, before the future's worth) sets how much it counts. With `replay`, it then
        re-learns from its most recent experiences, as many as its intellect allows, and remembers this one."""
        rate = self._learning_rate(agent)
        weight = lesson_weight(agent.last_choice, outcome)
        agent.brain.learn_traced(agent.last_senses, agent.last_choice, target, rate * weight,
                                 self._discount(agent), decay=self._discount(agent) * self._trace_decay(agent))
        if not replay:
            return
        count = REPLAY_PER_INTELLECT * thinking_intellect(agent.inborn["intellect"], agent.species)
        for senses, choice, remembered, counted in agent.memories[-count:] if count else ():  # the latest, up to count
            agent.brain.learn(senses, choice, remembered, rate * counted)
        if count:
            agent.memories.append((agent.last_senses, agent.last_choice, target, weight))
            del agent.memories[:-count]

    def _learn_batch(self, batch: BrainBatch, lessons: list[tuple | None]) -> None:
        """_learn, with replays, for every brain of a batch at once, each (agent, target, outcome, call target) or None
        for a brain with nothing to learn this turn; then the weights go back to the brains. Last, each learns from its
        call, if it chose one, as from an action untraced: what came of the turn, and the best call open now."""
        size = len(lessons)
        senses = np.zeros((size, len(SENSES)))
        actions, targets = np.zeros(size, dtype=int), np.zeros(size)
        rates, decays, taking = np.zeros(size), np.zeros(size), np.zeros(size, dtype=bool)
        replays: list[list[tuple]] = [[] for _ in range(size)]
        calls: list[tuple | None] = [None] * size
        for row, lesson in enumerate(lessons):
            if lesson is None:
                continue
            agent, target, outcome, call_target = lesson
            rate = self._learning_rate(agent)
            if agent.last_call >= 0:
                calls[row] = (agent.last_senses, call_output(agent.last_call), call_target, rate)
            weight = lesson_weight(agent.last_choice, outcome)
            senses[row], actions[row], targets[row] = agent.last_senses, agent.last_choice, target
            rates[row] = rate * weight
            decays[row] = self._discount(agent) * self._trace_decay(agent)
            taking[row] = rates[row] > 0
            count = REPLAY_PER_INTELLECT * thinking_intellect(agent.inborn["intellect"], agent.species)
            if count:
                replays[row] = [(remembered_senses, choice, remembered, rate * counted)
                                for remembered_senses, choice, remembered, counted in agent.memories[-count:]]
                agent.memories.append((agent.last_senses, agent.last_choice, target, weight))
                del agent.memories[:-count]
        if taking.any():
            batch.learn_traced(senses, actions, targets, rates, decays, taking)
        for turn in range(max(map(len, replays), default=0)):  # every brain's first replay, then its second, and so on
            senses, rates = np.zeros((size, len(SENSES))), np.zeros(size)
            for row, replayed in enumerate(replays):
                if turn < len(replayed):
                    senses[row], actions[row], targets[row], rates[row] = replayed[turn]
            batch.learn(senses, actions, targets, rates)
        if any(call is not None for call in calls):
            senses, rates = np.zeros((size, len(SENSES))), np.zeros(size)
            for row, call in enumerate(calls):
                if call is not None:
                    senses[row], actions[row], targets[row], rates[row] = call
            batch.learn(senses, actions, targets, rates)
        batch.write_back()
        for lesson in lessons:  # and back in time: what its memory held, which the choice read, taught to the gates that wrote it
            if lesson is None:
                continue
            agent, target = lesson[0], lesson[1]
            steps = min(len(agent.past), MEMORY_STEPS)
            if steps > 0 and MEMORY_LEARNING > 0:
                agent.brain.learn_memory(agent.last_senses, agent.past[:steps], agent.last_choice, target,
                                         self._learning_rate(agent) * MEMORY_LEARNING)

    def _dies_of_age(self, agent: Bitzkey) -> bool:
        """Past AGEING of its lifespan, the chance of dying each turn grows with age, so that about
        half live to their lifespan and almost none beyond 1.3 times it."""
        onset = AGEING * agent.lifespan
        if agent.age <= onset:
            return False
        steepness = 2 * math.log(2) / (agent.lifespan - onset) ** 2
        return self.random.random() < steepness * (agent.age - onset)

    def deaths_so_far(self) -> list[Death]:
        """Every death of the last RECENT_DEATH_TURNS turns."""
        return list(self.death_records.values())

    def _die(self, agent: Bitzkey, cause: str, killer: Bitzkey | None = None, activity: str | None = None) -> None:
        """Record a death, with what killed it and what the killer was doing. The body becomes carrion holding
        the energy it had left: a starved body leaves nothing, and a killed one half, as the killer has already
        eaten. Dying of old age is not held against the brain's last choice."""
        self.death_records[agent.identifier] = Death(cause, agent.species, self.turn,
                                                     None if killer is None else killer.identifier,
                                                     None if killer is None else killer.species, activity,
                                                     agent.age // DAY_LENGTH)
        self._death_order.append((self.turn, agent.identifier))
        self.death_tally.add(self.death_records[agent.identifier])
        if killer is not None and agent.womb and killer.species in ("predator", "omnivore"):
            # A hunter that killed a mother carrying young feels remorse for it.
            if killer.alive and killer.brain is not None:
                killer.reward -= PREGNANT_REMORSE * (0.5 + killer.feeling("empathy"))
        if agent.infant:  # its parents grieve, wherever they are, and learn from it what came before
            for identifier in (agent.mother, agent.father):
                parent = self.find_agent(identifier) if identifier >= 0 else None
                if parent is not None and parent.alive and parent.brain is not None:
                    parent.reward -= PARENT_GRIEF * (0.5 + parent.feeling("empathy"))
        # Its herd sees a sudden death fail; a slow one (old age, starving or the cold or heat) is not its last choice's
        # doing. Its own brain learns nothing from it, as it goes with it.
        if agent.brain is not None and agent.last_senses is not None and cause not in SLOW_DEATHS:
            self._witness(agent, -DEATH_PENALTY)
        agent.brain = None
        agent.memories = []
        agent.womb, agent.due_turn = [], None  # the young it carried die with it
        agent.health = min(agent.health, 0.0)  # whatever the cause, a recorded death leaves it dead
        agent.died_turn = self.turn
        agent.last_action = cause
        nourished = 0.0 if agent.species == "plant" else max(0.0, 1 - agent.hunger / 100)
        agent.meat = ((CARRION_MEAT + CARRION_MEAT_FOR_SIZE * agent.physique("size")) * nourished
                      * (0.5 if cause == "killed" else 1))
        agent.rot = agent.meat / max(1, CORPSE_TURNS - FRESH_KILL_TURNS)  # it rots once no longer fresh
        agent.portions = 0 if agent.species == "plant" else 1 + agent.trait("size")
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
        """Call, act, remember where it fed, and tire (see ACTIVE_SHARE)."""
        if agent.call != SILENT and agent.call_turn == self.turn:  # its call, as it sets about its action
            agent.hunger = min(100.0, agent.hunger + CALL_COST)
            if agent.call == ALERT_CALL and situation["alert_to"] is not None:
                self._alert(agent, situation["alert_to"])
        passed = len(agent.trail)
        self._do(agent, action, situation)
        if agent.last_action in FED:
            agent.food_memory = (agent.x, agent.y, self.turn)
        if not agent.alive:
            return
        if agent.last_action == "rested":
            agent.stamina = min(FULL_STAMINA, agent.stamina + rest_recovery())
            return
        # Tired by what it did: each square it went (flying, the more so), each blow struck, and keeping aloft over the sea.
        cost = ACTION_FATIGUE + (len(agent.trail) - passed) * STEP_FATIGUE * (FLIGHT_FATIGUE if agent.terrain == "avian" else 1.0)
        if agent.last_action in ("attacked", "pounced", "defended"):
            cost += STRIKE_FATIGUE
        if agent.aloft_at_sea:
            cost += HOVER_FATIGUE
        agent.stamina = max(0.0, agent.stamina - agent.fatigue(cost))
        if agent.aloft_at_sea and agent.exhausted:
            agent.health = 0
            self._die(agent, "drowned")  # too tired to stay aloft, with nowhere to land

    def _do(self, agent: Bitzkey, action: str, situation: dict) -> None:
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
            # Toward where it saw its suitor; the suitor itself is looked at only if it was seen within sight (the mating
            # call reaches further), as one further off can't come within reach this turn. As the C++ version.
            suitor = situation["suitor"]
            seen = situation.get("suitor_at") or (suitor.x, suitor.y)
            near = abs(seen[0] - agent.x) + abs(seen[1] - agent.y) <= MAX_SIGHT
            self._move_toward(agent, *seen)
            agent.last_action = "courted"
            if (near and suitor.alive and abs(suitor.x - agent.x) + abs(suitor.y - agent.y) <= MATE_REACH
                    and self._may_breed(agent, suitor)):
                agent.reward += mating_joy(agent, COURT_JOY)  # courted into reach, to breed
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
                self._attack(agent, quarry, "hunting")
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
        elif action == "revisit":
            self._move_toward(agent, *situation["revisit"])
            agent.last_action = "revisited"
        elif action == "feed_young":
            self._feed_young(agent, situation["young"])
        elif action in HEADINGS:
            dx, dy, walk, went = HEADINGS[action]
            cells = 1 if walk else agent.steps
            moved = 0  # as far as it goes before the map's edge (north and south) or a full square
            for step in range(1, cells + 1):
                x, y = (agent.x + dx * step) % self.width, agent.y + dy * step
                if not 0 <= y < self.height or not self._can_enter(agent, (x, y)) or self._room_in((x, y), agent) < agent.space:
                    break
                moved = step
            if moved:
                self._move(agent, (agent.x + dx * moved) % self.width, agent.y + dy * moved)
            if walk:
                agent.hunger = max(0, agent.hunger - agent.burn * WALK_REST)  # burns less, but eases no hunger
            else:
                self._effort(agent, moved)
                self._exhausted(agent)
            if moved:  # setting off pleases the curious; walking into a wall doesn't
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
            self._die(plant, "eaten", agent, "browsing")
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
            self._die(plant, "uprooted", agent, "uprooting")
        elif plant.physique("weapons") > 0:
            self._defend(plant, agent)

    def _feed_young(self, parent: Bitzkey, young: list[Bitzkey]) -> None:
        """Its infants within its sight fed full, at no cost to it, and it glad of it, the more the hungrier they were."""
        parent.last_action = "fed its young"
        for child in young:
            if not child.alive:
                continue
            parent.reward += FEEDING_JOY * (0.5 + parent.feeling("empathy")) * min(100.0, child.hunger) / 100
            child.hunger = 0
            self._note_deed(child, parent, FEEDING_DEED)

    def _alert(self, agent: Bitzkey, about: Bitzkey) -> None:
        """Alert its herd within its sight: a grazer to the hunter it sees, so they may flee it from
        further than they would; a hunter to the fresh kill it sees, so they may come to feast on it. A
        herd-mate heeds it only from TRUST_REACH times its trust times its own sight away, or nearer."""
        heeded = False
        for other in self._nearby(agent, agent.sight):
            if (self._herd_mate(agent, other) and abs(other.x - agent.x) + abs(other.y - agent.y)
                    <= TRUST_REACH * other.feeling("trust") * other.sight):
                heeded = True
                self._note_deed(other, agent, ALERT_DEED)
                if agent.species == "grazer":
                    if other.alarm is None:
                        other.alarm = about.identifier
                else:
                    other.kill_alert = (about.x, about.y)
        if heeded:  # a pleasure only when a herd-mate took it in
            agent.reward += ALERT_JOY

    def _scavenge(self, agent: Bitzkey, corpse: Bitzkey) -> None:
        """Go to the carrion and, once alongside, eat a portion, its share of the meat left; a
        carrion-eater's gut gets the most from it. The last portion eaten, the body is gone."""
        self._move_toward(agent, corpse.x, corpse.y)
        if abs(corpse.x - agent.x) + abs(corpse.y - agent.y) > 1 or corpse.portions <= 0:
            agent.last_action = "sought carrion"
            return
        bite = corpse.meat / corpse.portions
        corpse.meat -= bite
        agent.nourish(bite * agent.digestion, meat=True)
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
        agent.nourish(FRESH_KILL_MEALS * (KILL_MEAL + KILL_MEAL_FOR_SIZE * corpse.physique("size")), meat=True)
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

    def _attack(self, agent: Bitzkey, victim: Bitzkey, activity: str = "attacking") -> None:
        """A blow, struck hunting or attacking (`activity`). The intolerant enjoy fighting other species most; the empathetic feel their
        victim's pain, and so does any empathetic animal watching, most for its own kind. A small animal may slip away."""
        if victim.species != "plant" and SMALL_HIDING > 0 and int(victim.body["size"]) < TRAIT_MAX:
            smallness = 1 - victim.physique("size")
            if self.random.random() < SMALL_HIDING * smallness * smallness:
                agent.last_action = "missed"
                self._exhausted(agent)
                return
        empathy = agent.feeling("empathy")
        joy = FIGHT_JOY * agent.feeling("aggression") * (1 - 0.6 * empathy) * (1 - 0.75 * agent.feeling("tolerance"))
        damage = self._damage(agent, victim)
        if any(self._herd_mate(agent, other) for other in self._nearby(victim, 1) if other is not agent):
            damage *= 1 + PACK_BONUS * agent.herd_instinct  # hunting as a pack
        victim.health -= damage
        agent.nourish(5, meat=True)
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
            agent.nourish(meal, meat=True)
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
            self._die(victim, "killed", agent, activity)

        sting = ((0.0 if victim.species == "omnivore"  # an omnivore's weapons are for fighting, not spines
                  else 4.5 * self._armed(victim) * self._defence_share(victim) * (1 - 0.6 * agent.toughness))
                 * (PREDATOR_STING if agent.species == "predator" else 1.0))  # a predator takes spines half as hard
        stung = max(0.0, sting)
        if sting > 0:
            agent.health -= sting
            self._experience(agent, "stung")
            if not agent.alive:
                self._die(agent, "killed", victim, "spines")  # its victim's spines
        self._exhausted(agent)
        if agent.species == "predator" and agent.alive and damage > stung:  # a blow that went its way
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
            agent.reward += (FIGHT_JOY * agent.feeling("aggression") * (1 - 0.6 * agent.feeling("empathy"))
                             * (1 - 0.75 * agent.feeling("tolerance")))  # as attacking is
            if for_herd:
                agent.reward += HERD_DEFEND_JOY * agent.herd_instinct
            agent.last_action = "defended"
            self._experience(agent, "defended")
        if not assailant.alive:
            self._die(assailant, "killed", agent, "defending")
        elif assailant.species != "plant":
            self._struck(assailant, agent)
        self._exhausted(agent)

    def _struck(self, victim: Bitzkey, striker: Bitzkey) -> None:
        """A struck animal may strike back on its next turn, unless it fails a morale check
        against its current fear, more likely the more hurt it is and less among its herd, and so
        must flee instead. Its herd-mates close by are alarmed, and those alongside the striker may
        defend it."""
        victim.assailant = striker.identifier
        self._note_deed(victim, striker, BLOW_DEED)
        victim.danger_memory = (victim.x, victim.y, self.turn)
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
        coward = victim.species == "omnivore" and COWARDLY_OMNIVORES  # an omnivore struck always breaks
        chance = max(victim.fearfulness * (unhurt + (dying - unhurt) * hurt) * steadied,
                     CARRION_COWARDICE * victim.scavenger)  # and a carrion-eater, the more so the more it is one
        if coward or self.random.random() < chance:
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
        animals crowd around or, for a grazer, plants are few in its home band, and no more than the energy they give can start on NEWBORN_FOOD each), and carries it for its gestation before giving birth. The young
        share the energy both parents give up now. Animals only choose to breed when both can
        afford it, but a parent can only give what it has left. Each child inherits a mix of its
        parents' brains, and starts its temperament at the average of its parents' current
        temperaments, which its dot then shows."""
        self._note_deed(first, second, MATING_DEED)
        self._note_deed(second, first, MATING_DEED)
        litter = litter_size([(parent.trait("litter"), parent.trait("size")) for parent in (first, second)])
        if CROWDING_LIMIT:
            litter = max(1, int(litter * (1 - self._crowding(first, second, own_kind=True) / CROWDING_LIMIT) + 0.5))  # its own kind
        needed = self._food_needed(first)
        if needed:  # litters follow food only while food limits breeding
            litter = max(1, int(litter * min(1.0, self._food_supply(first) / (2 * needed)) + 0.5))
        # A grazer's litters follow the plants in its home band, a predator's the prey, an omnivore's both, half each.
        litter = max(1, int(litter * self._home_fed(first) + 0.5))
        if NEWBORN_FOOD:  # only as many young as the parents could feed, rather than many born starving
            # What the parents could give, BREEDING_COST_MOST of each stomach (or what each has left), whatever they
            # pay: neither a fertile parent's cheap breeding nor a hunter's halved cost shrinks its litters.
            could_give = sum(min(BREEDING_COST_MOST, max(0.0, 100 - parent.hunger)) * parent.stomach for parent in (first, second))
            stomach = (first.stomach + second.stomach) / 2
            litter = max(1, min(litter, int(could_give / (NEWBORN_FOOD * stomach))))
        litter = max(1, min(litter, self._room()))
        # Each parent pays its breeding cost, and at least its half of what its young need, NEWBORN_FOOD of a stomach
        # each, so a large litter costs what it is worth rather than being born starving.
        costs = [max(parent.breeding_cost, litter * NEWBORN_FOOD / 2) for parent in (first, second)]
        # Energy in absolute terms: a share of each parent's stomach, refilled in the young's.
        given = sum(min(cost, max(0.0, 100 - parent.hunger)) * parent.stomach
                    for cost, parent in zip(costs, (first, second)))
        generation = max(first.generation, second.generation) + 1
        for _ in range(litter):
            # The child's mind starts at the average of its parents' current minds, mutated, and
            # its dot is written from that mind, so experience is inherited and shows. Its brain,
            # a mix of its parents', is as deep and wide as its own inborn intellect makes it.
            mind = inherit_mind(first.mind, second.mind, self.random, self.mind_mutation)
            dna = evolved(first.dna, cross(first.dna, second.dna, self.random, mind, body_chance=self.body_mutation,
                                           terrain_mutation=self.has_sea))
            if "predator" in (first.species, second.species):
                dna = predator_dna(dna)  # a predator's young are predators, whatever crossing over or a mutation made
            brain = Brain.inherit(first.brain, second.brain, self.random, *brain_shape(dna))
            if decode_body(dna)["species"] == "predator" and first.species != "predator":
                # Born a predator of other parents: what they learnt of eating and uprooting it brings to
                # hunting and feasting.
                for learnt, now in ((EAT, HUNT), (UPROOT, FEAST)):
                    brain.output_weights[now] = brain.output_weights[learnt].copy()
                    brain.output_biases[now] = brain.output_biases[learnt]
            first.womb.append((dna, mind, brain, given / litter, generation, second.identifier))
        first.due_turn = self.turn + gestation(first.trait("size"))
        for parent, cost in zip((first, second), costs, strict=True):
            parent.hunger += cost
            rest = PREDATOR_BREED_COOLDOWN if parent.species == "predator" else BREED_COOLDOWN
            parent.cooldown = round((60 - 45 * parent.physique("fertility")) * rest
                                    * (AMPHIBIOUS_COOLDOWN if parent.terrain == "amphibious" else 1.0))
            parent.children += litter
            parent.stamina = max(0.0, parent.stamina - parent.fatigue(BREED_FATIGUE))
            parent.reward += mating_joy(parent, BREED_JOY)
            if not parent.alive:
                self._die(parent, "bred to death")
            else:
                self._experience(parent, "bred_into_hunger" if parent.hunger >= 75 else "bred")

    # Space

    def _index_bands(self) -> None:
        """The climate bands: the band each row lies in, and for each home temperature and row, the
        home band nearest: of the bands whose usual temperature is nearest home, the one nearest the
        row, the upper on a tie."""
        bands = self._zones()
        self._band_of_row = [band for band, rows in enumerate(bands) for _ in rows]
        temperatures = [zone_temperature(rows[0], self.height) for rows in bands]
        self._home_band = []
        for home in range(len(TEMPERATURES)):
            nearest = min(abs(temperature - home) for temperature in temperatures)
            homes = [band for band, temperature in enumerate(temperatures) if abs(temperature - home) == nearest]
            self._home_band.append([min(homes, key=lambda band: (max(0, bands[band][0] - y, y - bands[band][-1]), band))
                                    for y in range(self.height)])

    def _index_cells(self) -> None:
        self._cells = {}
        self._carrion = {}
        self._kills = {}
        if len(self._band_of_row) != self.height:
            self._index_bands()
        plants = [0] * (self._band_of_row[-1] + 1 if self._band_of_row else 0)
        grazers, prey, predators = [0] * len(plants), [0] * len(plants), [0] * len(plants)
        omnivores, omnivore_prey = [0] * len(plants), [0] * len(plants)
        for agent in self.agents:
            agent.wet = self.sea(agent.x, agent.y)
            if agent.alive:
                self._cells.setdefault((agent.x, agent.y), []).append(agent)
                if agent.species == "plant":
                    plants[self._band_of_row[agent.y]] += 1
                else:
                    if agent.species == "grazer":
                        grazers[self._home_band[agent.climate[0]][agent.y]] += 1
                    elif agent.species == "predator":
                        predators[self._home_band[agent.climate[0]][agent.y]] += 1
                    else:
                        omnivores[self._home_band[agent.climate[0]][agent.y]] += 1
                    if agent.species in ("grazer", "omnivore"):  # what predators live on
                        prey[self._band_of_row[agent.y]] += 1
                    if agent.species in ("grazer", "predator"):  # what omnivores hunt
                        omnivore_prey[self._band_of_row[agent.y]] += 1
            elif self._fresh(agent):
                if agent.species != "plant":
                    self._kills.setdefault((agent.x, agent.y), []).append(agent)
            elif agent.meat > 0:
                self._carrion.setdefault((agent.x, agent.y), []).append(agent)
        self._band_plants, self._band_grazers = plants, grazers
        self._band_prey, self._band_predators = prey, predators
        self._band_omnivores, self._band_omnivore_prey = omnivores, omnivore_prey
        self._living = Counter(agent.species for agent in self.agents if agent.alive)
        shares = {species: scarcity_share(count, species) for species, count in self._living.items()}
        glut = max(0.0, (self._living["predator"] - PREDATOR_GLUT) / PREDATOR_GLUT) if PREDATOR_GLUT > 0 else 0.0
        famine_line = 2 * RARE_PREY / 100 * self._living["predator"]  # and prey dying out: from twice the rare line, other kinds by it, its own by none
        prey = sum(self._living[species] for species in HUNTERS_PREY["predator"])
        if famine_line > 0:
            glut = max(glut, min(2.0, 2 * (famine_line - prey) / famine_line))
        for agent in self.agents:  # a scarce species breeds more cheaply; a glut of predators turns them on one another
            if agent.alive and agent.species != "plant":
                agent.population = self._living[agent.species]
                agent.cost_share = shares[agent.species]
                if agent.species == "predator":
                    agent.glut = glut
        for agent in self.agents:  # a predator's infant is guarded while a grown one of its kind is close by
            if agent.alive and agent.species == "predator":
                agent.guarded = agent.infant and any(
                    other.species == "predator" and not other.infant and self._same_kind(agent, other)
                    for other in self._nearby(agent, NURSING_REACH))

    def _home_plants(self, grazer: Bitzkey) -> float:
        """The plants in a grazer's home band per grazer at home there, itself among them; none
        before the cells are first indexed."""
        if not self._band_plants:
            return 0.0
        band = self._home_band[grazer.climate[0]][grazer.y]
        return self._band_plants[band] / max(1, self._band_grazers[band])

    def _home_prey(self, predator: Bitzkey) -> float:
        """The prey in a predator's home band per predator at home there, itself among them; none before the
        cells are first indexed."""
        if not self._band_prey:
            return 0.0
        band = self._home_band[predator.climate[0]][predator.y]
        return self._band_prey[band] / max(1, self._band_predators[band])

    def _homeward(self, agent: Bitzkey, y: int, discomfort: float) -> bool:
        """Whether heading for row `y` leads no further from the animal's home temperature than where it stands
        (`discomfort` steps from it)."""
        return agent.climate_steps(self._climate[y]) <= discomfort

    def _may_breed(self, agent: Bitzkey, partner: Bitzkey) -> bool:
        """Whether an animal may breed with a partner, all but being close enough and finding a nursery: both
        ready, room in the world, not crowded by its own kind, food enough about it where that limits breeding,
        and, for a grazer, plants enough in its home band. A predator's prey at home shrinks its litters instead."""
        return (agent.can_breed and partner.can_breed and self._room() > 0
                and (not CROWDING_LIMIT or self._crowding(agent, partner, own_kind=True) < CROWDING_LIMIT)  # its own kind
                and (not self._food_needed(agent) or self._food_supply(agent) >= self._food_needed(agent))
                and self._well_fed(agent) and not self._prey_rare(agent))

    def _well_fed(self, agent: Bitzkey) -> bool:
        """Whether its home band holds the food a mouth needs to breed: plants for a grazer, prey for a predator, both
        together, half each, for an omnivore."""
        return self._home_fed(agent) >= 1 / self._full_litter(agent)

    @staticmethod
    def _full_litter(agent: Bitzkey) -> float:
        """How many times the home food it needs to breed a full litter needs: PREDATOR_FULL_LITTER_FOOD for a
        predator, FULL_LITTER_FOOD for any other."""
        return PREDATOR_FULL_LITTER_FOOD if agent.species == "predator" else FULL_LITTER_FOOD

    def _prey_rare(self, agent: Bitzkey) -> bool:
        """Whether a hunter's prey species, all together, have grown so few (under RARE_PREY percent of its own
        species' numbers) that it spares them by not breeding."""
        prey = HUNTERS_PREY.get(agent.species)
        return prey is not None and sum(self._living[species] for species in prey) < RARE_PREY / 100 * agent.population

    def _home_food(self, agent: Bitzkey) -> float:
        """The food in its home band per mouth at home there: plants for a grazer, prey for a predator; 0 for
        an omnivore, which no home food limits."""
        if agent.species == "grazer":
            return self._home_plants(agent)
        return self._home_prey(agent) if agent.species == "predator" else 0.0

    @staticmethod
    def _home_food_needed(agent: Bitzkey) -> float:
        """The home food a mouth needs to breed: PLANTS_PER_GRAZER, PREY_PER_PREDATOR, or none."""
        return {"grazer": PLANTS_PER_GRAZER, "predator": PREY_PER_PREDATOR}.get(agent.species, 0.0)

    def _home_food_sense(self, agent: Bitzkey) -> float:
        """How well fed its home band is, as a sense: the share of a full litter it would feed."""
        return self._home_fed(agent)

    def _home_fed(self, agent: Bitzkey) -> float:
        """The share of a full litter an animal's home band feeds: 1 at twice the food a mouth needs to breed, or
        more, falling to 0 with none, and 1 where no home food limits it. A grazer's is its plants, a predator's its
        prey, and an omnivore's both, half each: half the share its plants would give a grazer, and half the share
        its prey (grazers and predators) would give a predator, each per omnivore at home there."""
        if agent.species == "omnivore":
            if not self._band_omnivores:
                return 0.0
            band = self._home_band[agent.climate[0]][agent.y]
            mouths = max(1, self._band_omnivores[band])
            return (0.5 * self._fed_share(self._band_plants[band] / mouths, PLANTS_PER_GRAZER)
                    + 0.5 * self._fed_share(self._band_omnivore_prey[band] / mouths, PREY_PER_PREDATOR))
        if agent.species == "plant":
            return 0.0
        return self._fed_share(self._home_food(agent), self._home_food_needed(agent), self._full_litter(agent))

    @staticmethod
    def _fed_share(food: float, needed: float, full: float | None = None) -> float:
        """The share of a full litter `food` a mouth feeds where `needed` is the least to breed: full at `full` times
        it (FULL_LITTER_FOOD unless given)."""
        return min(1.0, food / ((FULL_LITTER_FOOD if full is None else full) * needed)) if needed else 1.0

    def heat(self, agent: Bitzkey) -> bool:
        """Whether an animal is in heat now, where it stands (see HEAT_CLOSEST)."""
        if agent.species == "plant" or not 0 <= agent.y < self.height:
            return True
        # Turns until the next winter where it stands: of its hemisphere, or the nearer of the two on the equator.
        day = self.turn % YEAR_LENGTH
        until = None
        for quarter in (1, 3):
            if in_winter(agent.y, self.height, quarter):
                turns = (quarter * YEAR_LENGTH // 4 - day) % YEAR_LENGTH
                until = turns if until is None else min(until, turns)
        if until is None:  # no winter ever comes here
            return True
        end = HEAT_CLOSEST[agent.species] + HEAT_FORESIGHT * agent.feeling("patience")
        begin = min(end + HEAT_LENGTH[agent.species], 0.75)  # never through its own winter, a quarter of the year
        return end * YEAR_LENGTH < until <= begin * YEAR_LENGTH

    def find_agent(self, identifier: int) -> Bitzkey | None:
        """The Bitzkey of this identifier, living or dead, while the world still holds it: the agents stay in the
        order they were made, so it is found by halving."""
        at = bisect.bisect_left(self.agents, identifier, key=lambda agent: agent.identifier)
        return self.agents[at] if at < len(self.agents) and self.agents[at].identifier == identifier else None

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
        """How many other animals, besides its partner, live within its sight; with `own_kind`,
        only those of its own kind."""
        return sum(other.species != "plant" and other is not partner and (not own_kind or self._same_kind(agent, other))
                   for other in self._nearby(agent, agent.sight))

    def _give_birth(self, mother: Bitzkey) -> None:
        """The young it carries are born in the squares nearest it with room for each, its own first and several
        to a square while there is room, as many as the population cap allows; the rest are lost. Each starts
        with its share of the energy its parents gave."""
        born = 0
        for dna, mind, brain, energy, generation, father in mother.womb:
            if born >= self._room():
                break
            space, terrain = 1 + int(decode_body(dna)["size"]), decode_terrain(dna)  # where it may live, of land or sea
            cell = next((cell for cell in self._squares_near(mother) if self.suits(terrain, cell) and self._room_in(cell) >= space), None)
            if cell is None:  # no room left near it
                continue
            child = self._add_agent(dna, *cell, brain=brain)
            child.mother, child.father = mother.identifier, father
            child.mind = mind
            child.hunger = max(0.0, 100 - energy / child.stomach)
            child.generation = generation
            self._population += 1
            born += 1
        mother.womb, mother.due_turn = [], None

    def _squares_near(self, agent: Bitzkey) -> list[tuple[int, int]]:
        """Its own square, then those around it up to BIRTH_REACH away, nearest first, on the map."""
        cells = [(agent.x, agent.y)]
        for dx, dy in birth_offsets(BIRTH_REACH):
            if 0 <= agent.y + dy < self.height:  # not beyond the top or bottom of the map
                cells.append(((agent.x + dx) % self.width, agent.y + dy))
        return cells

    def _free_cells_near(self, agent: Bitzkey) -> list[tuple[int, int]]:
        """The squares around an animal, its own among them, up to BIRTH_REACH away and nearest first, with any
        room for young to be born."""
        return [cell for cell in self._squares_near(agent) if self._can_enter(agent, cell) and self._room_in(cell) > 0]

    def _room_in(self, cell: tuple[int, int], leaving: Bitzkey | None = None) -> int:
        """The spaces of a square that are free, as though `leaving` were not there."""
        return CELL_SPACE - sum(other.space for other in self._cells.get(cell, ()) if other.alive and other is not leaving)

    def _spot_near(self, cell: tuple[int, int], space: int, terrain: str = "terrestrial") -> tuple[int, int] | None:
        """A square with room for a Bitzkey taking `space`, of a terrain it may be in: this one, or else the nearest
        within BIRTH_REACH; None if there is none."""
        x, y = cell[0] % self.width, self._within(cell[1])
        for dx, dy in ((0, 0), *birth_offsets(BIRTH_REACH)):
            spot = ((x + dx) % self.width, y + dy)
            if 0 <= y + dy < self.height and self.suits(terrain, spot) and self._room_in(spot) >= space:
                return (x + dx) % self.width, y + dy
        return None

    def tie_side(self, agent: Bitzkey) -> tuple[int, int]:
        """Which way an animal breaks ties between targets equally near, or flees one straight in line: east (1) or
        west (-1), south (1) or north (-1), turning with its number and the turn, so no way is favoured."""
        turns = agent.identifier + self.turn
        return -1 if turns & 1 else 1, -1 if turns & 2 else 1

    def _nearest_grass(self, agent: Bitzkey, sight: int) -> tuple[int, int] | None:
        """The nearest grass in sight, but none in a row further from its home temperature than here."""
        grass, climate, x, y = self.grass, self._climate, agent.x, agent.y
        worst = agent.climate_steps(climate[y])
        sx, sy = self.tie_side(agent)
        for dx, dy in SIGHT_WITHIN[sight]:
            cell = (x + sx * dx, y + sy * dy)
            if grass.get(cell, 0) > 0 and self._can_enter(agent, cell) and agent.climate_steps(climate[cell[1]]) <= worst:
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
        sx, sy = self.tie_side(agent)
        for dx, dy in SEEN_WITHIN[sight]:
            for corpse in carrion.get((x + sx * dx, y + sy * dy), ()):
                if corpse.meat > 0 and agent.can_be(corpse.wet):
                    return corpse
        return None

    def _nearest_kill(self, agent: Bitzkey, sight: int) -> Bitzkey | None:
        """The nearest fresh kill in sight of something it lives on."""
        if not self._kills:
            return None
        kills, x, y = self._kills, agent.x, agent.y
        sx, sy = self.tie_side(agent)
        for dx, dy in SEEN_WITHIN[sight]:
            for corpse in kills.get((x + sx * dx, y + sy * dy), ()):
                if self._feeds_on(agent, corpse) and agent.can_be(corpse.wet):
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
        sx, sy = self.tie_side(agent)
        for dx, dy in SEEN_WITHIN[max(sight, prey_sight)]:
            far = abs(dx) + abs(dy) > sight  # beyond its sight but for prey
            if ((quarry or not hunts) and (far or ((hunter or not hunted)
                                                   and (suitor or not courts) and (flora or not browses)))):
                break
            for other in cells.get((x + sx * dx, y + sy * dy), ()):
                if other is agent or not other.alive:
                    continue
                if quarry is None and self._is_prey(agent, other):
                    quarry = other
                elif far:
                    continue
                elif hunter is None and self._is_prey(other, agent):
                    hunter = other
                elif browses and flora is None and other.species == "plant" and not other.cooldown and agent.can_be(other.wet):
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
        agent.wet = self.sea(x, y)
        self._cells.setdefault((x, y), []).append(agent)

    def _step(self, agent: Bitzkey, x: int, y: int) -> bool:
        """A step into a square, if it has room for it; whether it took it."""
        if (x, y) != (agent.x, agent.y) and (not self._can_enter(agent, (x, y)) or self._room_in((x, y), agent) < agent.space):
            return False  # full, or sea (or land) it can't go into: it stops short
        self._move(agent, x, y)
        return True

    @staticmethod
    def _effort(agent: Bitzkey, cells: int) -> None:
        """Going further costs: GO_EFFORT of a turn's burn for each cell past the first, however it moves."""
        agent.hunger += agent.burn * GO_EFFORT * max(0, cells - 1)

    def _move_toward(self, agent: Bitzkey, x: int, y: int, steps: int | None = None) -> None:
        cells = 0
        for _ in range(agent.steps if steps is None else steps):
            if agent.x != x:
                moved = self._step(agent, agent.x + (1 if x > agent.x else -1), agent.y)
            elif agent.y != y:
                moved = self._step(agent, agent.x, agent.y + (1 if y > agent.y else -1))
            else:
                break
            if not moved:
                break
            cells += 1
        self._effort(agent, cells)

    def _move_away(self, agent: Bitzkey, threat: Bitzkey) -> None:
        cells = 0
        side = self.tie_side(agent)  # straight in line with it, either way
        for _ in range(agent.steps):
            dx = side[0] if agent.x == threat.x else 1 if agent.x > threat.x else -1
            dy = side[1] if agent.y == threat.y else 1 if agent.y > threat.y else -1
            if not self._step(agent, (agent.x + dx) % self.width, self._within(agent.y + dy)):
                break
            cells += 1
        self._effort(agent, cells)

    @staticmethod
    def _compatible(first: Bitzkey, second: Bitzkey) -> bool:
        """Two animals of one species that are each willing to breed with the other."""
        return (first.species != "plant" and first.species == second.species
                and Simulation._willing(first, second) and Simulation._willing(second, first))

    @staticmethod
    def _willing(agent: Bitzkey, other: Bitzkey) -> bool:
        """Whether an animal would breed with another, or count it among its herd: it shuns a partner
        whose ring or dot colour differs from its own by more than it accepts, which its tolerance widens,
        or whose social background differs by more than its kinship accepts."""
        tolerance = agent.feeling("tolerance")
        ring, dot, social = agent.looks_apart(other)
        accepted = KIND_DIFFERENCE if agent.species == "predator" else MATE_RING_DIFFERENCE  # a predator its kind
        return (ring <= accepted + (1 - accepted) * tolerance
                and dot <= MATE_DOT_DIFFERENCE + (1 - MATE_DOT_DIFFERENCE) * tolerance
                and social <= 1 - KINSHIP_SHUN * agent.feeling("kinship") * (1 - tolerance))

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

    def _remembered(self, agent: Bitzkey, memory: tuple[int, int, int] | None) -> float:
        """How clearly it remembers a place: 1 the turn it happened, fading to nothing over MEMORY_TURNS
        times its memory; 0 when it has nothing to remember."""
        span = MEMORY_TURNS * agent.feeling("memory")
        if memory is None or span <= 0:
            return 0.0
        return max(0.0, 1 - (self.turn - memory[2]) / span)

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
                if 0 <= x + dx < self.width and 0 <= y + dy < self.height and self._can_enter(agent, (x + dx, y + dy)):
                    cells += growth[y + dy]
                    tufts += grass.get((x + dx, y + dy), 0)
            growing = self.width * self._grass_reach[-1] if self._grass_rows else 0.0
            regrowth = cells * self.grass_regrowth / growing * GRASS_OUTLOOK if growing else 0.0
            food += agent.grazing * (STANDING_GRASS * tufts + regrowth)
        mouths = 1.0
        for other in self._nearby(agent, FORAGE_RADIUS):
            if other.species == "plant":
                if agent.can_be(other.wet):
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
            heed = (observer.feeling("empathy") * observer.intelligence
                    * (1 + TRUST_HEED * (2 * observer.feeling("trust") - 1)))  # the trusting take in more
            heft = FATAL_LESSON if outcome <= -DEATH_PENALTY else 1.0  # a death seen teaches hard
            self._witnessed.append((observer, actor.last_senses, actor.last_choice, outcome,
                                    LEARNING_RATE * heed * SOCIAL_LEARNING * heft * lesson_weight(actor.last_choice, outcome)))  # at the end
            observer.reward += SOCIAL_MOOD * heed * outcome
            if outcome > 0:
                observer.seen_success = 1.0
            else:
                observer.seen_failure = 1.0

    def _learn_witnessed(self) -> None:
        """Herd-mates learn from what they saw this turn, each brain in the order it saw it."""
        seen: dict[int, list[tuple]] = {}  # by brain, in the order it saw them
        brains: dict[int, Brain] = {}
        for observer, senses, choice, outcome, rate in self._witnessed:
            if observer.brain is not None:  # not since dead
                seen.setdefault(id(observer.brain), []).append((senses, choice, outcome, rate))
                brains[id(observer.brain)] = observer.brain
        self._witnessed = []
        groups: dict[tuple[int, int], list[int]] = {}
        for key, brain in brains.items():
            groups.setdefault((brain.depth, brain.width), []).append(key)
        for keys in groups.values():  # a batch of a shape: every brain's first lesson, then its second, and so on
            batch = BrainBatch([brains[key] for key in keys])
            lessons = [seen[key] for key in keys]
            for turn in range(max(map(len, lessons))):
                senses, actions = np.zeros((len(keys), len(SENSES))), np.zeros(len(keys), dtype=int)
                targets, rates = np.zeros(len(keys)), np.zeros(len(keys))
                for row, learnt in enumerate(lessons):
                    if turn < len(learnt):
                        senses[row], actions[row], targets[row], rates[row] = learnt[turn]
                batch.learn(senses, actions, targets, rates)
            batch.write_back()

    @staticmethod
    def _beside(agent: Bitzkey, other: Bitzkey) -> bool:
        """Whether two Bitzkeys are on the same cell or side by side, as sensing finds a mate, prey or
        a threat beside an animal."""
        return abs(agent.x - other.x) + abs(agent.y - other.y) <= 1

    def _society(self, observer: Bitzkey, actor: Bitzkey) -> bool:
        """Whether an animal counts another, alive or not, as one of its herd (or pack): as for
        _herd_mate."""
        return (observer is not actor and not solitary(observer.species) and observer.species == actor.species
                and (observer.species != "predator" or self._same_kind(observer, actor)) and self._willing(observer, actor))

    def _herd_mate(self, agent: Bitzkey, other: Bitzkey) -> bool:
        """Whether an animal counts another as one of its herd: its species, and its kind for a
        predator, alive, and close enough in looks for it to accept as a mate."""
        return (other is not agent and other.alive and (not solitary(agent.species) or agent.infant)  # a loner's infant herds
                and other.species == agent.species
                and (agent.species != "predator" or self._same_kind(agent, other)) and self._willing(agent, other))

    def _herd(self, agent: Bitzkey) -> tuple[bool, int, tuple[int, int] | None]:
        """Its herd: whether any herd-mate is within its sight, how many are within two
        cells, and the cell at the middle of those in range."""
        mates = [other for other in self._nearby(agent, agent.sight) if self._herd_mate(agent, other)]
        if not mates:
            return False, 0, None
        close = sum(abs(other.x - agent.x) + abs(other.y - agent.y) <= 2 for other in mates)
        middle = (round(sum(other.x for other in mates) / len(mates)), round(sum(other.y for other in mates) / len(mates)))
        return True, close, middle

    def _territory(self, hunter: Bitzkey) -> tuple[float, float]:
        """How overcrowded a hunter's territory, the cells within its sight, is, with all its rivals and with its own species
        alone: nothing up to the rivals it will tolerate, fewer the less empathetic it is, then
        rising over TERRITORY_RAMP more to its species' strength of territorial feeling. A predator's pack-mates
        count only PACK_RIVALRY each: its pack mentality overcomes rivalry."""
        (fewest, most), strength = TERRITORY[hunter.species]
        tolerated = fewest + (most - fewest) * hunter.feeling("empathy")
        rivals = own = 0.0
        for other in self._nearby(hunter, hunter.sight):
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

    def _other_food(self, omnivore: Bitzkey) -> int:
        """The food an omnivore has besides predators within FORAGE_RADIUS: grazers, plants and grassy cells."""
        food = sum(other.species in ("grazer", "plant") for other in self._nearby(omnivore, FORAGE_RADIUS))
        grass, x, y = self.grass, omnivore.x, omnivore.y
        return food + sum(grass.get((x + dx, y + dy), 0) > 0 and self._can_enter(omnivore, (x + dx, y + dy))
                          for dx, dy in nearby_offsets(FORAGE_RADIUS))

    @classmethod
    def would_win(cls, agent: Bitzkey, opponent: Bitzkey) -> bool:
        """Whether an animal's blows would bring the opponent down as fast as the opponent's would bring
        it down, each as it stands now."""
        return (cls._damage(agent, opponent) / max(1.0, opponent.health)
                >= cls._damage(opponent, agent) / max(1.0, agent.health))

    @staticmethod
    def within_reach(hunter: Bitzkey, other: Bitzkey) -> bool:
        """Whether a hunter could hunt or attack another, wherever they stand: it may go where the other is, sea or
        land, and the other is no flyer out of its reach, avian and not resting, while it is terrestrial or aquatic."""
        if not hunter.can_be(other.wet):
            return False
        grounded = hunter.terrain in ("terrestrial", "aquatic")
        return not (grounded and other.terrain == "avian" and other.species != "plant" and other.last_action != "rested")

    @classmethod
    def _is_prey(cls, hunter: Bitzkey, other: Bitzkey) -> bool:
        if other.guarded:  # a predator's infant, guarded by a grown one of its kind
            return False
        if not cls.within_reach(hunter, other):
            return False
        if hunter.species == "omnivore":
            # Grazers while they are common enough to switch to, and other omnivores only in an
            # overcrowded territory, once hungry past its reluctance as that rivalry wears it down.
            # Predators only when it would win, blow for blow.
            return ((other.species == "grazer" and hunter.hunts_grazers)
                    or (other.species == "predator" and hunter.hunts_predators and cls.would_win(hunter, other)) or (
                other.species == "omnivore" and hunter.own_rivalry > 0
                and hunter.hunger / 100 > cls.cannibal_reluctance(hunter) * (1 - RIVALRY_WEAR * hunter.own_rivalry)))
        if hunter.species != "predator":
            return False
        if other.species == "grazer":
            return hunter.hunts_grazers
        if other.species == "omnivore":  # only one it would beat, blow for blow
            return hunter.hunts_omnivores and cls.would_win(hunter, other)
        if other.species != "predator":
            return False
        # Other kinds of predator only when hungry past its reluctance, which a territory overcrowded with predators
        # wears away, and so does a world glutted with them or running out of prey; its own kind only once that is
        # twice as bad.
        reluctance = cls.cannibal_reluctance(hunter) * (1 - RIVALRY_WEAR * hunter.own_rivalry)
        if cls._same_kind(hunter, other):
            own = min(1.0, max(0.0, hunter.glut - 1))
            return own > 0 and hunter.hunger / 100 > reluctance * (1 - own)
        return hunter.hunger / 100 > reluctance * (1 - min(1.0, hunter.glut))
