"""The simulation's tunable rules: every constant that shapes how the world plays out, so the
advanced panel can change them before a world starts.

Each rule names a module constant, or one entry of a tuple or table constant, by a key such as
FORAGE_RADIUS, ANIMAL_LIFESPAN.0, GRAZING.grazer or EXPERIENCES.kill.aggression. The C++ port lists
the same rules with the same keys, limits and steps (`bitzkey_cli --rules`). Rules are module-wide:
`apply` sets them all, and a world plays by whatever was set when it runs. Settings the
configuration panel already covers (world size, population, grass density, the population cap and
mutation rates) are not repeated here, nor are numbers that fix the genome's or the brain's layout,
or how starting groups are sized.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tools import Brain as brain
from tools import Census as census
from tools import Simulation as world


@dataclass(frozen=True)
class Rule:
    key: str
    group: str
    label: str
    hint: str
    minimum: float
    maximum: float
    step: float

    @property
    def integer(self) -> bool:
        """Whole numbers only: the rule counts cells, turns or animals."""
        return isinstance(self.step, int)

    def clamp(self, value: float) -> float:
        value = max(self.minimum, min(self.maximum, value))
        return int(round(value)) if self.integer else float(value)


_MODULES = {"TEMPERATURE": brain, "LESSON_LIMIT": brain, "FLAT_SPOT": brain, "MUTATION": brain, "TRACE_DECAY": brain, "SMART_TRACE_DECAY": brain,
            "HEADING_RELUCTANCE": brain, "GRAPH_TURNS": census}

# (key, label, hint, minimum, maximum, step) by group. A step that is an int makes a whole-number rule.
_TABLE: dict[str, list[tuple]] = {
    "World": [
        ("GRASS_REGROWTH", "Grass regrowth", "share of the starting grass that grows back each turn", 0, 0.1, 0.001),
        ("CORPSE_FERTILITY", "Corpse fertility", "grass grows this many times as fast on and around a corpse", 1, 10, 1),
        ("WATERED_REACH", "Watered land", "steps from water within which hot land grows grass and can be tilled; 0: none", 0, 20, 1),
        ("OFF_HOURS_SIGHT", "Off-hours sight", "how far an animal awake out of its own hours sees, in the dark if active by day, in the light if active by night, as a share of its sight; calls carry as far as ever", 0.1, 1.0, 0.05),
        ("CORPSE_TURNS", "Corpse turns", "turns a corpse lasts while its meat rots away", 1, 1500, 5),
        ("CORPSE_PLANT", "Corpse plant", "chance an animal's body that rots away entirely leaves the seed of a new plant", 0.0, 1.0, 0.05),
        ("CELL_SPACE", "Square space", "spaces in a square; a Bitzkey takes 1 + its size, and none is born, moves or spawns "
         "into a square without room", 1, 80, 1),
        ("SPAWN_SPREAD", "Spawn spread", "a starting group appears within this many cells of its centre", 0, 4, 1),
        ("USE_GPU", "Think on the GPU", "C++ only: 1 thinks and learns on every GPU found, in floats; 0 on the CPU, exactly", 0, 1, 1),
        ("DAY_LENGTH", "Day length", "turns in a day; night covers half the map and moves round it once a day", 2, 100, 1),
        ("SLEEP", "Sleep", "1: an animal rests through its night (the dark if diurnal, the light if nocturnal) unless startled "
         "awake; 0: never forced to", 0, 1, 1),
        ("BIOME_GROUPS", "Biome groups", "a world started with any Bitzkeys has this many groups of each species native to "
         "every biome, each stretch of land or sea in a climate zone, on top of its population; 0: none", 0, 10, 1),
        ("BIOME_SIZE", "Biome size", "the fewest squares a stretch of land or sea in a climate zone needs to count as a biome",
         1, 1000, 5),
        ("SPAWN_AT_HOME", "Spawn at home", "1: starting and imported groups appear where their home temperature is usual; 0: anywhere", 0, 1, 1),
        ("GRAPH_TURNS", "Graph turns", "turns of the past the graph and death report cover; deaths and genomes older than that are forgotten",
         1000, 2_000_000, 1000),
    ],
    "Bodies": [
        ("LIFESPAN_YEARS.0", "Lifespan, shortest", "years of life expected for the smallest bodies, plant or animal",
         0.5, 30.0, 0.5),
        ("LIFESPAN_YEARS.1", "Lifespan, middle", "years of life expected for middling bodies", 0.5, 30.0, 0.5),
        ("LIFESPAN_YEARS.2", "Lifespan, longest", "years of life expected for the largest bodies", 0.5, 30.0, 0.5),
        ("STARVING_DAMAGE", "Starving damage", "share of its full health a Bitzkey at full hunger loses each turn",
         0.0, 1.0, 0.01),
        ("MEAT_DIGESTION", "Meat digestion", "the share of its fullness that is meat empties at this share of the usual rate "
         "(1/7: meat lasts a week to plants' day)", 0.01, 1.0, 0.001),
        ("AGEING", "Ageing", "from this share of its lifespan, an animal may die of old age, and is feeble", 0.0, 0.95, 0.05),
        ("SIZE_HUNGER", "Size hunger", "a large body burns 1 + this times its size squared as much food (size a share of its "
         "range: twice as much at the largest)", 0.0, 5.0, 0.1),
        ("SIZE_STARVING", "Size starving", "starving, a body wastes away 1 + this times its size as fast (twice as fast at the "
         "largest)", 0.0, 5.0, 0.1),
        ("SMALL_HIDING", "Small hiding", "a hunter's blow misses a small animal outright with this times its smallness squared "
         "as a chance (none at the largest)", 0.0, 1.0, 0.05),
        ("SMALL_MATURITY", "Small maturity", "the smallest are infants this share as long as their lifespans would make them, "
         "the larger more, the largest the whole", 0.01, 1.0, 0.01),
        ("FEEBLE_YOUTH", "Feeble youth", "an animal is feeble, an infant, for this share of its lifespan (1/30: 36 days for the "
         "smallest)", 0.0, 0.5, 0.001),
        ("PREDATOR_INFANCY", "Predator infancy", "a predator is an infant this share as long as other animals", 0.0, 1.0, 0.05),
        ("OMNIVORE_INFANCY", "Omnivore infancy", "an omnivore is an infant this share as long as other animals", 0.0, 1.0, 0.05),
        ("GRAZER_INFANCY", "Grazer infancy", "a grazer is an infant this share as long as other animals", 0.0, 1.0, 0.05),
        ("GRAZER_GESTATION", "Grazer gestation", "a grazer carries its young this share as long as its size would have it", 0.05, 1.0, 0.05),
        ("FEEBLE_SHARE", "Feeble share", "a feeble animal's body traits and health are at this share of its own",
         0.1, 1.0, 0.05),
        ("METABOLISM.0", "Metabolism, smallest", "metabolic rate of the smallest bodies", 0.1, 5.0, 0.1),
        ("METABOLISM.1", "Metabolism, largest", "metabolic rate of the largest bodies", 0.1, 5.0, 0.1),
        ("STOMACH.0", "Stomach, smallest", "food the smallest body holds, relative to the middle", 0.1, 5.0, 0.1),
        ("STOMACH.1", "Stomach, largest", "food the largest body holds, relative to the middle", 0.1, 5.0, 0.1),
        ("SIZE_IN_MIGHT", "Size in might", "share of strength and toughness that comes from body size", 0.0, 1.0, 0.05),
        ("GRAZER_SPEED", "Grazer speed", "grazers move this many times as far a step, rounded half up", 0.5, 4.0, 0.25),
        ("AVIAN_BODY", "Avian body", "an avian animal's size, weapons and armour count this share of its genes'", 0.1, 1.0, 0.05),
        ("FLIGHT_HUNGER", "Flight hunger", "an avian animal burns this share more food", 0.0, 2.0, 0.05),
        ("AMPHIBIOUS_SPEED", "Amphibious speed", "an amphibious animal goes this share of its usual steps on land, rounded half up, "
         "at least one", 0.1, 1.0, 0.05),
        ("AMPHIBIOUS_COOLDOWN", "Amphibious cooldown", "an amphibious animal rests this many times as long after breeding",
         1.0, 5.0, 0.25),
        ("ACTIVE_SHARE", "Active share", "share of its waking half day the average animal's stamina lets it act, resting the rest; the fatigues below scale with the day's length to keep it", 0.1, 0.9, 0.05),
        ("ACTION_FATIGUE", "Action fatigue", "stamina any action but rest costs, relative to the others (an average turn's acting costs 20)", 0.0, 50.0, 0.5),
        ("STEP_FATIGUE", "Step fatigue", "stamina each square moved costs, more", 0.0, 50.0, 0.5),
        ("FLIGHT_FATIGUE", "Flight fatigue", "a flyer's squares tire it this many times as much", 1.0, 5.0, 0.25),
        ("STRIKE_FATIGUE", "Strike fatigue", "stamina each blow struck costs, more", 0.0, 100.0, 1.0),
        ("BREED_FATIGUE", "Breed fatigue", "stamina mating costs each parent", 0.0, 100.0, 1.0),
        ("HOVER_FATIGUE", "Hover fatigue", "stamina a flyer over the sea, unable to rest, loses a turn more; exhausted there, "
         "it drowns", 0.0, 50.0, 0.5),
        ("AMPHIBIOUS_FATIGUE", "Amphibious fatigue", "an amphibious animal tires this many times as fast at everything",
         1.0, 5.0, 0.25),
        ("WALK_REST", "Walk rest", "share of the turn's burn a one-cell walk saves, as resting does", 0.0, 1.0, 0.05),
        ("GO_EFFORT", "Go effort", "share of a turn's burn each cell a go_ moves beyond the first costs", 0.0, 2.0, 0.05),
    ],
    "Breeding": [
        ("BREED_RESERVE", "Breeding reserve", "hunger a parent must still have spare after paying to breed", 0, 100, 5),
        ("FEED_HUNGER", "Feeding hunger", "how hungry one of its infants within its sight must be for a parent to feed them all, "
         "at no cost to it", 0.0, 100.0, 5.0),
        ("MATE_RING_DIFFERENCE", "Mate ring difference", "most a partner's ring colour may differ, at no tolerance",
         0.0, 1.0, 0.01),
        ("MATE_DOT_DIFFERENCE", "Mate dot difference", "most a partner's dot colour may differ, at no tolerance",
         0.0, 1.0, 0.01),
        ("LITTER_RANGE.0", "Smallest litter", "fewest young per birth, while there is room", 1, 20, 1),
        ("LITTER_RANGE.1", "Largest litter", "most young per birth, while there is room", 1, 20, 1),
        ("LITTER_SIZE_WEIGHT", "Small-body litters", "share of litter size from small bodies; the rest from the trait",
         0.0, 1.0, 0.05),
        ("SMALL_LITTERS", "Small-body litter boost",
         "the smallest bodies' litters are this many times the usual, falling to the usual for the largest", 1.0, 4.0, 0.25),
        ("GESTATION.0", "Shortest gestation", "turns the smallest animals carry their young", 1, 1500, 1),
        ("GESTATION.1", "Longest gestation", "turns the largest animals carry their young", 1, 1500, 1),
        ("BIRTH_REACH", "Birth reach", "young are born on free cells up to this many cells from their mother", 1, 5, 1),
        ("PLANT_GESTATION", "Plant gestation", "turns a seed lies in the ground before it can sprout", 0, 1500, 1),
        ("CROWDING_LIMIT", "Crowding limit", "litters shrink as crowding nears this; at it, no breeding; 0: no limit", 0, 100, 1),
        ("FORAGE_RADIUS", "Forage radius", "food and mouths are weighed within this many cells", 1, 12, 1),
        ("FOOD_NEEDED", "Food needed", "meals each mouth needs nearby to breed; twice this for a full litter; 0: no limit",
         0.0, 20.0, 0.5),
        ("UPROOT_FOOD", "Uproot food", "a starving grazer uproots a plant only with fewer meals a mouth than this about it",
         0.0, 20.0, 0.5),
        ("BREED_COOLDOWN", "Breeding cooldown", "grazers' and omnivores' share of the usual rest after breeding (15 to 60 turns "
         "by fertility); 0: none", 0.0, 2.0, 0.05),
        ("HEAT_CLOSEST.grazer", "Heat ends, grazers", "share of a year before its next winter a grazer's heat ends; more for "
         "the patient", 0.0, 0.75, 0.01),
        ("HEAT_CLOSEST.omnivore", "Heat ends, omnivores", "share of a year before its next winter an omnivore's heat ends; more "
         "for the patient", 0.0, 0.75, 0.01),
        ("HEAT_CLOSEST.predator", "Heat ends, predators", "share of a year before its next winter a predator's heat ends; more "
         "for the patient", 0.0, 0.75, 0.01),
        ("HEAT_LENGTH.grazer", "Heat, grazers", "share of a year a grazer is in heat, before its heat ends; 0.75: always but in "
         "its winter", 0.0, 0.75, 0.01),
        ("HEAT_LENGTH.omnivore", "Heat, omnivores", "share of a year an omnivore is in heat, before its heat ends", 0.0, 0.75, 0.01),
        ("HEAT_LENGTH.predator", "Heat, predators", "share of a year a predator is in heat, before its heat ends", 0.0, 0.75, 0.01),
        ("HEAT_FORESIGHT", "Heat foresight", "share of a year further from winter the most patient end their heat", 0.0, 0.75, 0.01),
        ("PREDATOR_BREED_COOLDOWN", "Breeding cooldown, predators", "share of the usual rest after breeding for predators; 0: none",
         0.0, 2.0, 0.05),
        ("PREDATOR_FOOD_NEEDED", "Food needed, predators", "meals each mouth needs nearby for a predator to breed; 0: no limit",
         0.0, 20.0, 0.5),
        ("PLANTS_PER_GRAZER", "Plants per grazer", "plants each grazer needs in its home climate band, shared among the grazers at home there, "
         "to breed; twice this for a full litter; 0: no limit", 0.0, 10.0, 0.05),
        ("PREY_PER_PREDATOR", "Prey per predator", "grazers and omnivores each predator needs in its home climate band, shared among the "
         "predators at home there, to breed; twice this for a full litter; 0: no limit", 0.0, 20.0, 0.1),
        ("PLANT_MEALS", "Plant meals", "meals a plant is worth to a full-efficiency browser", 0.0, 20.0, 0.5),
        ("PREY_MEALS", "Prey meals", "meals a prey animal is worth", 0.0, 20.0, 0.5),
        ("SHARED_DIET", "Shared diet", "how much another species eating the same food counts as a mouth",
         0.0, 1.0, 0.05),
        ("GRASS_OUTLOOK", "Grass outlook", "grass is judged by what regrows in this many turns", 0, 3000, 10),
        ("STANDING_GRASS", "Standing grass", "plus this share of the grass standing now", 0.0, 1.0, 0.05),
    ],
    "Feelings": [
        ("DEATH_PENALTY", "Death penalty", "how sharply a brain learns from the choice that killed it", 0.0, 10.0, 0.1),
        ("BREED_JOY", "Breeding joy", "pleasure of breeding, for each parent: the greatest there is", 0.0, 10.0, 0.05),
        ("COURT_JOY", "Courting joy", "pleasure of courting a ready, willing partner into reach to breed", 0.0, 10.0, 0.05),
        ("LIBIDO_IN_JOY", "Libido in joy", "share of breeding's and courting's pleasure that libido sets; the rest every animal feels",
         0.0, 1.0, 0.05),
        ("FIGHT_JOY", "Fighting joy", "pleasure of a blow, scaled by aggression", 0.0, 5.0, 0.05),
        ("FEAST_JOY", "Feasting joy", "a predator's pleasure in feasting on a fresh kill", 0.0, 5.0, 0.05),
        ("SATED_JOY", "Sated joy", "pleasure in hunger relieved by eating, hunting, feasting or scavenging, for the full range", 0.0, 5.0, 0.05),
        ("ATTACK_JOY", "Attacking joy", "a predator's pleasure in a blow it lives through", 0.0, 5.0, 0.05),
        ("COMPANY_JOY", "Company joy", "pleasure of company, scaled by sociability", 0.0, 5.0, 0.05),
        ("CURIOSITY_JOY", "Curiosity joy", "pleasure of setting off in a direction, scaled by curiosity", 0.0, 5.0, 0.05),
        ("FEAR_DREAD", "Fear dread", "dread of hunters nearby, scaled by fear", 0.0, 5.0, 0.05),
        ("UNEASE", "Unease", "how much an intolerant animal minds other species close by", 0.0, 5.0, 0.05),
        ("CLIMATE_UNEASE", "Climate unease", "discomfort two steps from home, half that one step; less if hardy",
         0.0, 5.0, 0.05),
        ("FEEDING_JOY", "Feeding joy", "how glad a parent is of feeding a starving infant of its own, in proportion for one "
         "less hungry; more if empathetic", 0.0, 5.0, 0.05),
        ("PARENT_JOY", "Parent joy", "how glad a parent is of three or more of its own infants within its sight; "
         "more if empathetic", 0.0, 5.0, 0.05),
        ("PREGNANT_REMORSE", "Pregnant remorse", "how much a hunter regrets killing a mother carrying young; more if "
         "empathetic", 0.0, 10.0, 0.1),
        ("HUNTER_RESTRAINT", "Hunter's restraint", "unease of a kill where few others of its prey are left in its sight: in full with none, none with four or more", 0.0, 10.0, 0.1),
        ("LAST_BITE", "Last-bite unease", "unease of eating grass or a plant and leaving none of either in its sight", 0.0, 10.0, 0.1),
        ("SOWER_PRIDE", "Sower's pride", "gladness of seeing a seed it sowed sprout, within its sight", 0.0, 10.0, 0.1),
        ("FAMINE_LESSON", "Famine lesson", "how much of a sudden death's lesson a herd-mate's starving teaches those who see it; 0 for none", 0.0, 1.0, 0.05),
        ("PARENT_GRIEF", "Parent grief", "how much a parent grieves when one of its infants dies, wherever; more if "
         "empathetic", 0.0, 10.0, 0.1),
        ("RESTING_UNEASE", "Resting unease", "unease, each turn, of having chosen to rest out of its temperature, two steps "
         "from home; half one step, less if hardy, none asleep or spent", 0.0, 5.0, 0.05),
        ("FATIGUE_UNEASE", "Fatigue unease", "unease of being tired, little while fresh and rising steeply towards exhaustion (as the cube of the stamina spent), from half this for the most resilient to one and a half times it for the least", 0.0, 5.0, 0.05),
        ("HOMEWARD_UNEASE", "Homeward unease", "more discomfort the further home lies, this much half the map away; "
         "each step home eases it", 0.0, 5.0, 0.05),
        ("EMPATHY_PAIN", "Empathy pain", "how much an empathetic animal feels another's wound", 0.0, 5.0, 0.05),
        ("OTHER_KIND", "Other kind", "how much of that pain reaches it for another species", 0.0, 1.0, 0.05),
    ],
    "Society": [
        ("SOCIAL_LEARNING", "Social learning", "how much a herd-mate's success or failure seen teaches, by empathy and intellect",
         0.0, 3.0, 0.05),
        ("SOCIAL_MOOD", "Social mood", "how much a herd-mate's success or failure seen moves the mood", 0.0, 3.0, 0.05),
        ("SOCIAL_MEMORY", "Social memory", "share of a seen success or failure still sensed a turn later", 0.0, 1.0, 0.05),
        ("PARENTAL_INSTINCT", "Parental instinct", "how much of the gap between feeding its young and its best other option a "
         "parent closes, times the hungriest one's hunger and a half and its empathy (at 2, its young half hungry, level with "
         "the best)", 0.0, 5.0, 0.1),
        ("LOYALTY_PULL", "Loyalty pull", "how much more an animal values keeping to its herd, times its loyalty", 0.0, 5.0, 0.05),
        ("LOYALTY_MARGIN", "Loyalty margin", "how much more esteem, times its loyalty, a follower must hold another in before it may leave its leader for it; past that, the likelier the further, and surer the less loyal", 0.0, 1.0, 0.05),
        ("CONVICTION_LEVEL", "Conviction level", "share of its full range at which empathy or tolerance becomes a conviction kept even starving: an empathetic grazer won't uproot, an empathetic predator spares the young and those carrying young, a tolerant animal strikes no foe and drives none out", 0.0, 1.0, 0.05),
        ("SOCIABLE_TRADE", "Sociable trade", "how much more an animal values making an offer, times its sociability", 0.0, 5.0, 0.05),
        ("SHARE_JOY", "Share joy", "pleasure of bringing food to its herd's store, times its generosity: the strongest of its patience, sociability and empathy, less the strongest of its dominance, aggression and appetite; the greedy are uneasy", 0.0, 5.0, 0.1),
        ("SELFISH_JOY", "Selfish unease", "unease at eating fruit off a plant, a kill where it lies or what it carries with a herd to share with, times its generosity; the greedy are gratified", 0.0, 5.0, 0.05),
        ("KIN_SHARE_JOY", "Kin sharing joy", "pleasure of bringing food to its herd's store with a friend in sight there, times its kinship, whatever its generosity", 0.0, 5.0, 0.1),
        ("EMPATHY_SCRUPLE", "Empathy scruple", "how much less an animal values uprooting a plant, times its empathy, and a predator hunting, striking or feasting on the young or those carrying young", 0.0, 5.0, 0.05),
        ("TOLERANCE_SCRUPLE", "Tolerance scruple", "how much less an animal values striking a foe or driving one out, times its tolerance; hunting to eat is untouched", 0.0, 5.0, 0.05),
        ("DOMINANCE_EASE", "Dominance ease", "share less a crowded territory troubles the most dominant, and more the most submissive",
         0.0, 1.0, 0.05),
        ("TRUST_HEED", "Trust heed", "share more the most trusting take in what herd-mates do, and less the least", 0.0, 1.0, 0.05),
        ("STUDENT_INTELLECT", "Student intellect", "the least inborn intellect an animal needs to be taught", 0, 7, 1),
        ("STUDENT_AGE", "Student age", "the oldest a student may be, as a share of its elder's age", 0.0, 1.0, 0.05),
        ("STUDENT_TRUST", "Student trust", "how trusting, 0 to 1, an animal must be to be willing to be taught", 0.0, 1.0, 0.05),
        ("REGARD_COMFORT", "Regard comfort", "ease near a friend, more if trusting, and unease near an enemy, more if fearful", 0.0, 2.0, 0.05),
        ("GRUDGE_JOY", "Grudge joy", "gladness in harming an enemy, more if aggressive, and pain in harming a friend, more if empathetic", 0.0, 5.0, 0.1),
        ("FRIEND_JOY", "Friend joy", "extra gladness in feeding or teaching a friend; more if empathetic", 0.0, 5.0, 0.1),
        ("FACTIONS", "Factions", "how many body colours an animal keeps a reputation for; none at 0", 0, 16, 1),
        ("FACTION_SHARE", "Faction share", "how much of a deed's weight also falls on the doer's whole colour", 0.0, 1.0, 0.05),
        ("FACTION_WIDTH", "Faction width", "how far apart on the body ring two animals may be and count as one faction", 0.0, 0.5, 0.01),
        ("SKILL_WITNESS", "Skill witness", "how much of a deed's weight a craft, lesson or fire seen done has on how skilled the doer is held; no less than Witness share", 0.0, 1.0, 0.05),
        ("EAVESDROP_SHARE", "Eavesdropping", "how much a witness's view of one is moved by seeing one it holds a friend treat that one, well or ill (a gift, a blow, a theft, a defeat), times its regard for that friend; no word passed, only seen", 0.0, 1.0, 0.05),
        ("WITNESS_SHARE", "Witness share", "how much of a deed's weight it has on regard when only seen done, not done to the animal", 0.0, 1.0, 0.05),
        ("TEACHING_JOY", "Teaching joy", "how glad an elder is of teaching a student; more if empathetic", 0.0, 5.0, 0.05),
        ("TRUST_REACH", "Trust reach", "times its trust times its sight away a herd-mate's alert is heeded", 0.0, 5.0, 0.25),
        ("KINSHIP_SHUN", "Kinship shun", "at full kinship and no tolerance, a herd-mate's or partner's social background may "
         "differ by no more than 1 less this", 0.0, 1.0, 0.05),
    ],
    "Mind": [
        ("LEARNING_RATE", "Learning rate", "how fast a brain learns, at the highest intellect", 0.0, 0.5, 0.005),
        ("PATIENT_DISCOUNT", "Discount, most patient", "how much the most patient value what comes later (gamma)", 0.5, 0.999,
         0.001),
        ("IMPATIENT_DISCOUNT", "Discount, least patient", "how much the least patient value what comes later (gamma): their "
         "foresight, at the least", 0.5, 0.999, 0.001),
        ("TRACE_DECAY", "Trace decay", "how far back a choice's outcome reaches earlier choices (lambda)",
         0.0, 1.0, 0.05),
        ("SMART_TRACE_DECAY", "Trace decay, smartest", "trace decay at the highest intellect; the lowest keeps the one above",
         0.0, 1.0, 0.01),
        ("LEARNING_FLOOR", "Learning floor", "the share of the learning rate even the dullest animal learns at; its intellect "
         "adds to it", 0.0, 2.0, 0.05),
        ("LEARNING_BOOST", "Learning boost", "the highest intellect learns this much faster again; the lowest, no faster",
         0.0, 5.0, 0.1),
        ("VALUE_LIMIT", "Value limit", "the most, and least, an animal learns a choice is worth, so its brain's overestimates can't run away; 0: no limit",
         0.0, 500.0, 5.0),
        ("REPLAY_PER_INTELLECT", "Replays per intellect", "recent experiences re-learnt each turn per point of intellect",
         0, 4, 1),
        ("BREED_LEARNING", "Breeding learning", "times as much any animal learns from breeding; more unsettles its brain for all else", 0.0, 20.0, 0.1),
        ("COURT_LEARNING", "Courting learning", "times as much any animal learns from courting; more unsettles its brain for all else", 0.0, 20.0, 0.1),
        ("KILL_FOOD_LEARNING", "Kill food learning", "times as much any animal learns from hunting, attacking and feasting",
         0.0, 10.0, 0.1),
        ("PLANT_FOOD_LEARNING", "Plant food learning", "times as much any animal learns from eating, foraging, browsing and uprooting",
         0.0, 10.0, 0.1),
        ("FORCED_LESSONS", "Forced lessons", "1: an animal learns from every turn, even one it had no choice in (asleep, exhausted, panicking or frozen); 0: only from the turns it chose",
         0, 1, 1),
        ("PREDATOR_LEARNING", "Predator learning", "times as fast a predator's brain learns from its own choices; an omnivore gets half the bonus",
         1.0, 5.0, 0.1),
        ("CURIOUS_EXPLORATION", "Curious exploration", "the chance, times its curiosity, that an animal tries any of its "
         "options at random rather than as its brain values them", 0.0, 1.0, 0.01),
        ("FATAL_LESSON", "Fatal lesson", "how many times as hard a herd-mate's death teaches those who see it what it last did",
         1.0, 20.0, 0.5),
        ("BOREDOM_RISE", "Boredom rise", "how much boredom a turn of doing what it did lately adds, out of 1", 0.0, 0.2, 0.005),
        ("BOREDOM_RELIEF", "Boredom relief", "how much boredom doing something long undone takes away", 0.0, 1.0, 0.05),
        ("BOREDOM_DRIVE", "Boredom drive", "the chance, at full boredom, that an animal not in peril does another option instead, the longest undone the likeliest", 0.0, 1.0, 0.05),
        ("BOREDOM_UNEASE", "Boredom unease", "how unpleasant a turn of full boredom is", 0.0, 2.0, 0.05),
        ("UNTRIED_HOPE", "Untried hope", "how highly an animal values an option it has never taken in its life, as a share of the average of those it has: 1 as well as the average, so that what it has never tried gets tried; 0 as its brain alone does", 0.0, 1.5, 0.05),
        ("FORESIGHT_LEARNING", "Foresight learning", "how fast a brain learns what its next turn will do to its hunger, health, stamina and food, as a share of its learning rate; 0 stops it, and surprise with it", 0.0, 2.0, 0.05),
        ("SURPRISE_JOY", "Surprise joy", "how pleasant a turn wholly unlike what it foresaw is, times a half and its curiosity; 0 turns the feeling off", 0.0, 2.0, 0.05),
        ("IMAGINATION", "Imagination", "how much what it imagines would come of its likeliest options sways its choice: the better it foresees an option leaving it, the likelier; 0 stops it imagining, and surprise with it", 0.0, 5.0, 0.1),
        ("RESTLESS_CURIOSITY", "Restless curiosity", "the chance, times a half and its curiosity, that an animal whose brain "
         "chose to rest (not made to) does any of its other options instead; none in peril (a hunter or threat near, struck, "
         "or starving)", 0.0, 1.0, 0.05),
        ("FLAT_SPOT", "Flat spot", "how much learning a saturated neuron keeps: added to each hidden neuron's slope, so one pinned at either end of its range still learns a little; 0 is the plain slope, which lets a jammed neuron stall", 0.0, 0.5, 0.01),
        ("LESSON_LIMIT", "Lesson limit", "the most one lesson's error counts for: higher, and a rare but large surprise teaches "
         "faster, but learning runs jumpier", 0.5, 20.0, 0.5),
        ("TEMPERATURE", "Choice temperature", "how loosely a brain picks among its options", 0.01, 5.0, 0.05),
        ("MUTATION", "Brain mutation", "how much a newborn's brain varies from its parents'", 0.0, 1.0, 0.01),
        ("HEADING_RELUCTANCE", "Heading reluctance", "how much less a new brain values setting off in a direction",
         0.0, 2.0, 0.05),
        ("TEMPERAMENT_RELAPSE", "Temperament relapse", "share of the gap back to the inborn value closed each turn",
         0.0, 0.1, 0.001),
        ("MEMORY_TURNS", "Memory turns", "turns the best memory recalls where it last fed or was struck; fewer the poorer", 0, 3000, 10),
        ("RESILIENCE", "Resilience", "share faster the most resilient get over a seen failure, and less they fear a bad experience",
         0.0, 1.0, 0.05),
        ("MOODINESS", "Moodiness", "share more experience moves the moodiest temperament, and less the steadiest", 0.0, 1.0, 0.05),
        ("PANIC.0", "Panic, unhurt", "chance a fully fearful animal breaks when struck, unhurt", 0.0, 1.0, 0.05),
        ("PANIC.1", "Panic, near death", "chance a fully fearful animal breaks when struck, near death",
         0.0, 1.0, 0.05),
        ("COWARDLY_OMNIVORES", "Cowardly omnivores", "1: an omnivore struck always breaks and flees, whatever its fear; 0: it checks its morale",
         0, 1, 1),
        ("MATE_REACH", "Mate reach", "steps within which a partner is close enough to breed with", 1, 5, 1),
        ("MATING_CALL", "Mating call", "how much more a predator ready to breed values breeding, or else courting a mate it sees", 0.0, 5.0, 0.05),
        ("MATING_CALL_RANGE", "Mating call range", "times its sight a predator ready to breed seeks a mate, when it sees none", 0.0, 3.0, 0.25),
        ("OMNIVORE_MATING_CALL", "Mating call, omnivores", "how much more an omnivore ready to breed values breeding, or else courting", 0.0, 5.0, 0.05),
        ("OMNIVORE_MATING_CALL_RANGE", "Mating call range, omnivores", "times its sight an omnivore ready to breed seeks a mate, when it sees none", 0.0, 3.0, 0.25),
        ("PREGROWTH_TURNS", "Plant head start", "turns a new world's plants grow alone, uncounted, before its animals spawn; 0: none",
         0, 20000, 100),
        ("HUNTERS_APART", "Hunters apart", "squares a new world's hunters start from its grazers, at the least, where there is room; 0: anywhere",
         0, 60, 1),
        ("GRAZER_SPAWN_GROUP", "Grazer group", "grazers in each group a new world starts with (other species start in groups of 3 to 5)", 1, 500, 5),
        ("FIRST_BREEDING_INSTINCT", "First breeding instinct", "how much more one of a world's first Bitzkeys, its brain untrained, values breeding when ready, or else courting",
         0.0, 10.0, 0.25),
        ("HUNTER_BREEDING_COST", "Breeding cost, hunters", "share of the usual breeding cost a predator or omnivore pays", 0.0, 1.0, 0.05),
        ("GRAZER_BREEDING_COST", "Breeding cost, grazers", "share of the usual breeding cost a grazer pays", 0.0, 1.0, 0.05),
        ("SCARCE_POPULATION", "Scarce population", "below this many living, a species' breeding cost begins to fall", 1, 2000, 10),
        ("RARE_POPULATION", "Rare population", "at this many living or fewer, a species pays the rare breeding cost", 0, 1000, 1),
        ("RARE_BREEDING_COST", "Rare breeding cost", "share of the usual breeding cost a rare species pays", 0.0, 1.0, 0.05),
        ("FULL_LITTER_FOOD", "Full litter food", "times the home food a mouth needs to breed that a full litter needs", 1.0, 10.0, 0.5),
        ("PREDATOR_FULL_LITTER_FOOD", "Predator full litter", "times the prey a predator needs to breed that its full litter needs", 1.0, 20.0, 0.5),
        ("PREDATOR_GLUT", "Predator glut", "above this many predators alive, they turn on other kinds, fully at twice it, and their own kind past that", 0, 5000, 10),
        ("GRAZER_SCARCE_POPULATION", "Grazer scarce population", "below this many living, grazers' breeding cost begins to fall", 1, 3000, 10),
        ("GRAZER_RARE_BREEDING_COST", "Grazer rare breeding cost", "share of the usual breeding cost rare grazers pay", 0.0, 1.0, 0.01),
        ("RARE_PREY", "Rare prey", "a hunter doesn't breed while its prey species, all together, number under this percentage of its own", 0, 5000, 10),
        ("PREDATOR_STING", "Predator sting", "share of a victim's spines a predator striking it takes", 0.0, 1.0, 0.05),
        ("NEWBORN_FOOD", "Newborn food", "food, as a share of a stomach, each young must start with: parents conceive no more young than they can give this; 0: no limit",
         0.0, 100.0, 5.0),
        ("PACK_MOUTHS", "Pack mouths", "mouths each pack-mate counts as on a hunter's food, sharing its kills", 0.0, 1.0, 0.05),
        ("BLOODLUST", "Bloodlust", "how much more a predator values feasting whenever it sees a fresh kill", 0.0, 5.0, 0.05),
        ("CARRION_CRAVING", "Carrion craving", "how much more the highest carrion trait values scavenging carrion it sees; none up to the middle",
         0.0, 5.0, 0.05),
        ("CARRION_AVERSION", "Carrion aversion", "how much less the highest carrion trait values hunting and attacking; none up to the middle",
         0.0, 5.0, 0.05),
        ("CARRION_COWARDICE", "Carrion cowardice", "the highest carrion trait's least chance of breaking when struck; none up to the middle",
         0.0, 1.0, 0.05),
        ("ALERT_JOY", "Alert joy", "pleasure of alerting its herd to a hunter (grazers) or a fresh kill (hunters)", 0.0, 5.0, 0.05),
        ("ALARM_REACH", "Alarm reach", "steps within which an alarmed animal knows where the hunter is", 1, 20, 1),
        ("CALL_REACH", "Call reach", "steps a call carries, times a half and its caller's size (as a share of its range): the larger, the further",
         1, 30, 1),
        ("CALL_COST", "Call cost", "hunger each sound of a call costs its caller, the alert among them", 0.0, 20.0, 0.1),
        ("CALL_EFFORT", "Call effort", "how much less each call is taught for the effort of making it, beside being understood; felt, not paid in the body; 0 for none", 0.0, 2.0, 0.05),
        ("LEARN_GAIN", "Learn gain", "spoken to, one that says the word back and points at what the speaker pointed at teaches both their words this many times what it understood (at least the learn match); 0 for no learning", 0.0, 10.0, 0.5),
        ("LEARN_MATCH", "Learn match", "the least a right guess counts as understood, before the learn gain, so a right guess always teaches something", 0.0, 1.0, 0.05),
        ("LEARN_CORRECTED", "Learn corrected", "a wrong guess, corrected by the speaker saying its word and pointing again, is worth this share of a right one, taught as the right pairing; 0 for nothing", 0.0, 1.0, 0.05),
        ("LEARN_INSIGHT", "Learn insight", "chance, times its intellect, that one saying a word back reads what the speaker meant and points at it at once: the cleverest always do; 0 for guessing alone", 0.0, 1.0, 0.05),
        ("LEXICON_SIZE", "Lexicon size", "words an animal can hold in its lexicon for each point of intellect, and one more; words settled by accord, a right guess or a corrected one; 0 for no lexicon", 0, 20, 1),
        ("LEXICON_SPEAK", "Lexicon speech", "chance, times its sociability, that pointing at a thing it has a word for, it says that word", 0.0, 1.0, 0.05),
        ("LEXICON_FAMILY", "Lexicon family", "the regard a word settled with family counts as at least: a word is changed only by a friend or family member regarded more than whoever settled it", 0.0, 1.0, 0.05),
        ("LEXICON_WITNESS", "Lexicon witness", "chance that one of its kind who sees a word agreed takes it into its own lexicon, under the same rule", 0.0, 1.0, 0.05),
        ("LEXICON_INHERIT", "Lexicon inheritance", "share of each parent's words, the most used first, that a newborn starts with, as if taught by family", 0.0, 1.0, 0.05),
        ("LEXICON_HEED", "Lexicon heed", "joy for one that heard a word it holds and by the next turn has come nearer the thing it means", 0.0, 2.0, 0.05),
        ("LEXICON_ACT", "Lexicon fitting deed", "joy for each way what one did on hearing a word it holds was fitting: it went nearer the thing if its own life found that good, or further if bad; and it did what the speaker did", 0.0, 2.0, 0.05),
        ("LEXICON_WORTH_RATE", "Lexicon worth rate", "how fast an animal learns from its own life how it fares nearer each thing, against further", 0.0, 1.0, 0.01),
        ("DUTY_SWAY", "Duty sway", "how strongly a sense of duty sways one told by a word from family or one it regards well: the teller's deed is its duty to this power times as likely (duty 1 is chance, 0 shuns it)", 0.0, 3.0, 0.1),
        ("DUTY_FELT", "Duty felt", "how much a sense of duty colours the joy of doing as told by family or one it regards well: times 1 plus this times its duty less 1, unease for a rebel", 0.0, 2.0, 0.05),
        ("NAMES_SIZE", "Names size", "names an animal can hold for each point of intellect, and one more: words for one animal each, given by its mother at birth and settled by accord, pointing at that one; 0 for no names", 0, 20, 1),
        ("DEEDS_SIZE", "Deed-words size", "deed-words an animal can hold for each point of intellect, and one more: words for what one does, said pointing at oneself and learnt by saying the word back while doing a guess; 0 for none", 0, 20, 1),
        ("CHATTER_FADE", "Chatter fade", "how much of an animal's chatter, the count of its calls of late, is left each turn; it makes calling dearer and less heeded", 0.0, 0.99, 0.05),
        ("HOARSE", "Hoarseness", "a call costs 1 plus this times its caller's chatter as much: calling now and then is cheap, calling on and on tires; 0 for none", 0.0, 5.0, 0.1),
        ("TUNE_OUT", "Tuning out", "hearers hear a call, and feel its words, names and deed-words, 1 over 1 plus this times its caller's chatter as much; 0 for none", 0.0, 5.0, 0.1),
        ("ANNOYANCE", "Annoyance", "how far one of its kind that hears a caller say again what it last heard it say holds it less well, as a deed's weight; 0 for none", 0.0, 1.0, 0.01),
        ("HUNTERS_HEAR", "Hunters hear", "1: a hunter that sees no prey goes after the nearest prey it heard call last turn; 0: calls draw no hunters", 0, 1, 1),
        ("BABBLE", "Babbling", "chance, times its sociability and its youth, that an animal whose brain chose silence calls anyway, the call its brain liked best but silence; and times its sociability again, that it goes up to one it may address; 0 for none", 0.0, 1.0, 0.05),
        ("BABBLE_AGE", "Babbling age", "turns of age by which the urge to babble has faded to nothing, from its full strength at birth", 1, 2000, 1),
        ("ANSWERED_JOY", "Joy of being answered", "joy, times its sociability, a caller feels for each hearer (at most three a turn) that answers it: calls back, says its word back, or does as its word or name meant; whatever was said; 0 for none", 0.0, 2.0, 0.05),
        ("HELP_WINDOW", "Help window", "turns after it called while its goal was stuck in which the goal coming free counts as helped", 1, 48, 1),
        ("HELPED_JOY", "Joy of being helped", "relief, times its sociability, an animal feels when one that heard it call while stuck did something for it (gave, dropped beside it, fought for it, or led it) and its stuck goal came free the next turn; the call it made is taught it; 0 for none", 0.0, 2.0, 0.05),
        ("HELPING_JOY", "Joy of helping", "joy, times its sociability, an animal feels when what it did for one it heard call while stuck freed that one's goal: the deed it did is taught it, against what it sensed, sounds heard and all; 0 for none", 0.0, 2.0, 0.05),
        ("VALUE_MUTATION", "Value mutation", "how far each of a newborn's inherited values may move from its parents' mean, either way, out of 0 to 3", 0.0, 1.0, 0.01),
        ("VALUE_INHERIT", "Value inheritance", "share of its mother's taught values a newborn starts with", 0.0, 1.0, 0.05),
        ("VALUE_TEACH", "Value teaching", "confiding or tutoring, the share of the way the hearer's taught values move to make its own the teller's, times its trust and teachable value; 0 for none", 0.0, 1.0, 0.01),
        ("LIKING", "Liking", "joy felt as a cue wakes its wanting of a thing, companion or deed the more, times that rise and its liking value; on the rise only, and never learnt to want; 0 for none", 0.0, 5.0, 0.05),
        ("HAWK_WINDOW", "Hawk window", "turns within which a caller holding goods, and one that heard it from afar and moved, must make a deal for it to be hawked", 1, 200, 1),
        ("HAWK_JOY", "Hawk joy", "a hawked deal teaches the owner's call at this times what it truly gained and its sociability, and the buyer's approach at this times its gain and curiosity; 0 for none", 0.0, 5.0, 0.05),
        ("HAWK_BUYER", "Hawk buyer", "1: a hawked deal also teaches the buyer the move it made on hearing the call; 0: only the seller's call is taught", 0, 1, 1),
        ("WOO_WINDOW", "Woo window", "turns within which one that called ready to breed, and one ready that heard it from afar and moved, must breed for it to be wooed", 1, 200, 1),
        ("WOO_JOY", "Woo joy", "a wooed mating teaches the caller's call, and the comer's approach, at this as a pleasure of mating, by libido; 0 for none", 0.0, 5.0, 0.05),
        ("WOO_COMER", "Woo comer", "1: a wooed mating also teaches the comer the move it made on hearing the call; 0: only the caller's call is taught", 0, 1, 1),
        ("CONVERSE_WINDOW", "Conversation window", "turns within which a reply to one that answered it counts as an exchange", 1, 50, 1),
        ("CONVERSE_JOY", "Conversation joy", "an exchange gives each side this times its sociability and talk value, whatever was said; 0 for none", 0.0, 5.0, 0.05),
        ("CONCEPTS_SIZE", "Concept-words", "words for ideas (project aims and values) held for each point of intellect and one more, settled in confiding and teaching; 0 for none", 0, 10, 1),
        ("INFLUENCE_JOY", "Influence joy", "joy a caller feels for how far its call moved what its hearers chose, times its sociability and talk value; 0 for none", 0.0, 5.0, 0.05),
        ("INFLUENCE_LESSON", "Influence lesson", "how much of its call's influence on its hearers is added to that call's own lesson", 0.0, 5.0, 0.05),
        ("INFLUENCE_HEARERS", "Influence hearers", "the most hearers whose influence a call is credited with", 1, 20, 1),
        ("INFLUENCE_SAMPLE", "Influence sample", "share of hearers thought a second time, deaf, to measure a call's influence; less is faster", 0.05, 1.0, 0.05),
        ("INFLUENCE_USEFUL", "Influence only useful", "1: a call's influence counts only from a hearer whose turn went at least as well as it expected", 0, 1, 1),
        ("DEAF", "Deaf", "1: no animal hears another's call, to judge whether words matter (for copies of worlds)", 0, 1, 1),
        ("MESSAGE_UNITS", "Message units", "the most units (sounds, or words said whole) a call may go on with after its words, each chosen knowing what came before; 0 for none", 0, 4, 1),
        ("ORDER_SAMPLE", "Order sample", "share of hearers of two units or more thought again with the units reversed, to measure whether order matters", 0.0, 1.0, 0.05),
        ("ORDER_JOY", "Order joy", "joy a caller feels when its hearers heard its words in the order they are used to hearing (and pain when not), times its sociability and talk value; which order is not said; 0 for none", 0.0, 5.0, 0.05),
        ("ORDER_LESSON", "Order lesson", "how much of its hearers' agreement with its word order is added to that call's own lesson", 0.0, 5.0, 0.05),
        ("HELP_WITNESS", "Help witnessed", "share of the joy of helping that each who saw useful help feels, beside esteeming the helper; 0 for none", 0.0, 1.0, 0.05),
        ("PROJECT_RATE", "Project learning", "how fast an animal learns, from its own life, which long aims rose (or fell) before it felt good, sampled every few turns; 0 for none", 0.0, 1.0, 0.01),
        ("PROJECT_EVERY", "Project sampling", "turns between an animal's samples of the world's long aims, learnt against how good it felt between them", 1, 480, 1),
        ("PROJECT_MIN", "Project threshold", "the least an aim must be worth to it, times a half and the mean of its patience and curiosity, to take it up as its project", 0.0, 1.0, 0.001),
        ("PROJECT_CONSISTENCY", "Project consistency", "how surely an aim must have moved its way whenever it felt good (0 never, 1 always), as well as being worth enough, to become its project; lest chance swings be taken for its doing", 0.0, 1.0, 0.05),
        ("PROJECT_PROGRESS", "Project progress", "joy each turn its project's aim moves the way it wants", 0.0, 2.0, 0.05),
        ("PROJECT_PATIENCE", "Project patience", "turns, times a half and its patience, a project may go without progress before it is given up", 1, 20000, 10),
        ("PROJECT_PULL", "Project pull", "how strongly its project draws it, times a half and its patience, against its needs and desires", 0.0, 2.0, 0.05),
        ("PROJECT_RESOLVE", "Project resolve", "how much more its patience holds its project above a need: a need must pull 1 plus this times its patience as hard, rising with each turn it waits", 0.0, 10.0, 0.1),
        ("PROJECT_INHERIT", "Project inheritance", "share of what its parents learnt of the long aims that a newborn starts with", 0.0, 1.0, 0.05),
        ("INSTRUCTED_PULL", "Instruction pull", "how strongly its leader's instruction, shown or said, draws a follower, times its regard for the one instructing", 0.0, 2.0, 0.05),
        ("CONFIDE_SHARE", "Confided share", "share of the one confiding's worth for its aim that one confided in gains, times its regard for it; and how strongly a plan confided draws it", 0.0, 1.0, 0.05),
        ("GOSSIP", "Gossip", "hearing a name of another, an animal takes on this share of the caller's regard for the one named, times its trust in the caller, the caller's urgency and its own sociability; hearsay of one alone, never of its colour, and never checked; 0 for none", 0.0, 1.0, 0.05),
        ("CONFIDE_STEPS", "Confided steps", "1: one that takes up a plan confided also takes up what the one confiding learnt of the deeds and things that move its aim, so it knows the steps; 0: it starts knowing none", 0, 1, 1),
        ("CONFIDE_WORDS", "Confided words", "1: confiding a plan also shares the words the one confiding holds for what its plan and goal need, the deeds and things it learnt move its aim and its goal's deed or thing, taken up as words settled are; 0: plans pass without words", 0, 1, 1),
        ("VENDETTA", "Vendetta", "how strongly a grudge against an enemy that harmed it or its own pulls it to go after that one, times its aggression and a half; 0 for none", 0.0, 5.0, 0.1),
        ("VENDETTA_RATE", "Grudge learning", "how fast its grudge against an enemy follows the joy it felt harming that one", 0.0, 1.0, 0.01),
        ("VENDETTA_MEMORY", "Grudge memory", "turns over which a grudge fades away, and an enemy's last-seen place is remembered", 1, 20000, 10),
        ("ENEMY_STRUCK", "Struck grudge", "the grudge an animal starts with against one that struck it", 0.0, 1.0, 0.05),
        ("ENEMY_KIN_KILLED", "Kin-killer grudge", "the grudge an animal starts with against one it saw kill its kin", 0.0, 1.0, 0.05),
        ("ENEMY_GROUP_KILLED", "Group-killer grudge", "the grudge an animal starts with against one it saw kill one of its own colour group", 0.0, 1.0, 0.05),
        ("CHALLENGE_GLORY", "Challenge glory", "joy, times a half and its dominance, the winner of a challenge for leadership feels", 0.0, 5.0, 0.1),
        ("CHALLENGE_LOSS", "Challenge loss", "unease the loser of a challenge for leadership feels", 0.0, 10.0, 0.1),
        ("CHALLENGE_ESTEEM", "Challenge esteem", "how much more each of its group who saw it esteems the winner of a challenge, in deeds' worth", 0.0, 5.0, 0.1),
        ("CHALLENGE_SHAME", "Challenge shame", "how much less each of its group who saw it esteems the loser of a challenge, in deeds' worth", 0.0, 5.0, 0.1),
        ("CHALLENGE_HUMBLED", "Humbled", "turns a loser of a challenge may not challenge again", 0, 20000, 10),
        ("RELIEF_RATE", "Relief learning", "how fast an animal learns, from its own life, how much each need eased after it came nearer each thing or companion, or did each deed; 0 for none", 0.0, 1.0, 0.01),
        ("RELIEF_INHERIT", "Relief inheritance", "share of what eased its parents' needs that a newborn starts with, as if it had learnt it", 0.0, 1.0, 0.05),
        ("GOAL_MIN_RELIEF", "Goal threshold", "the least an animal must have found a thing, companion or deed eases its worst need to make it its goal", 0.0, 1.0, 0.005),
        ("GOAL_RISE", "Goal urgency rise", "how much a goal's urgency rises each turn it is unmet, times the need it serves", 0.0, 1.0, 0.01),
        ("GOAL_STUBBORNNESS", "Goal stubbornness", "how much more, times twice its patience, another goal must be worth to take the place of the one it holds", 0.0, 5.0, 0.1),
        ("GOAL_PATIENCE", "Goal patience", "turns a goal is held unmet, times a half and its patience, before it is given up", 1, 1000, 1),
        ("GOAL_UNEASE", "Goal unease", "unease each turn for each unit of a goal's urgency while it is unmet", 0.0, 2.0, 0.05),
        ("GOAL_JOY", "Goal joy", "joy at reaching a goal (beside what it sought, or the deed done), for each unit of its urgency", 0.0, 5.0, 0.1),
        ("GOAL_GIVE_UP", "Giving up", "unease at giving up a goal held too long", 0.0, 2.0, 0.05),
        ("GOAL_PROGRESS", "Goal progress", "comfort at each step nearer what it seeks, and as much unease at each step away", 0.0, 1.0, 0.01),
        ("DESIRE", "Desire", "how strongly a cue to what an animal has liked pulls it, against its worst need, to make it its goal; 0 for none", 0.0, 5.0, 0.1),
        ("WANTING_RATE", "Liking learning", "how fast an animal learns, from its own life, how much good it felt after it came nearer each thing or companion, or did each deed; 0 for none", 0.0, 1.0, 0.01),
        ("WANTING_INHERIT", "Liking inheritance", "share of what its parents liked that a newborn starts with, as if it had learnt it", 0.0, 1.0, 0.05),
        ("HEARD_CUE", "Heard cue", "how strongly a word, name or deed-word heard for what it likes wakes desire, against seeing it close by", 0.0, 1.0, 0.05),
        ("DESIRE_LINGER", "Desire lingering", "share of a desire's spark kept each turn after its cue has gone", 0.0, 1.0, 0.05),
        ("DESIRE_MIN", "Desire threshold", "the least spark of desire that can make a goal", 0.0, 1.0, 0.005),
        ("PLEASURE_FADE", "Pleasure fading", "share of what is left of its pleasure in a thing, companion or deed that fades each time it reaches it as a desire", 0.0, 1.0, 0.05),
        ("REMEMBERED_CUE", "Remembered cue", "how strongly a remembered place, out of sight, wakes desire for what was there, against seeing it close by; fading to nothing over the place memory", 0.0, 1.0, 0.05),
        ("PLACE_MEMORY", "Place memory", "turns an animal remembers where it last found a thing close by, or last saw a companion", 1, 5000, 10),
        ("OPENS_RATE", "Opening learning", "how fast an animal learns that doing a deed, or coming near a thing, opens to it a deed or thing shut before; 0 for none", 0.0, 1.0, 0.01),
        ("OPENS_INHERIT", "Opening inheritance", "share of what its parents learnt opens what that a newborn starts with", 0.0, 1.0, 0.05),
        ("OPENS_MIN", "Opening threshold", "the least a link must be worth, its strength times a half and the mean of patience and curiosity, to be followed as a step toward a goal shut to it", 0.0, 1.0, 0.01),
        ("PLAN_DEPTH", "Planning depth", "how many steps beyond one an animal will chain toward a goal shut to it, times the mean of its patience and curiosity", 0.0, 8.0, 0.5),
        ("PLEASURE_RETURN", "Pleasure returning", "share of faded pleasure that returns each turn", 0.0, 1.0, 0.005),
        ("CALL_WORDS", "Call words", "the most words a call may hold: 2 lets a call go on to a second word after its first, chosen sound by sound; a call holding a name is for the one so named; 1 a single word", 1, 2, 1),
        ("MAP_BLIND", "Map blind", "for experiments: 1 withholds the mental map senses and the way back to the last meal, so only its own memory can hold where things are", 0, 1, 1),
        ("ADDRESS_RETRIES", "Address retries", "how many times one that spoke to another alone, and was not answered with a right guess, may say the same word and point again; the more sociable it is, the likelier it must; 0 for none", 0, 10, 1),
        ("CALL_FATIGUE", "Call fatigue", "stamina each sound of a call costs its caller, the alert among them, as an action's tiredness is counted (an action alone costs 10)", 0.0, 50.0, 0.5),
        ("TUTOR_COST", "Tutor cost", "hunger a lesson costs the one who tutors", 0.0, 5.0, 0.05),
        ("CALL_LENGTH", "Call length", "the most sounds a call may run to, in order: 1 single sounds only, 3 pairs and threes too", 1, 3, 1),
        ("SILENCE_LESSON", "Silence lesson", "share of its learning by which each call also teaches what keeping quiet would have brought: the same turn without the call's cost or pleasures; 0 for none",
         0.0, 2.0, 0.1),
        ("CALL_HABIT", "Call habit", "pleasure of a call that matches those it made where it sensed most alike, more than its calls at large; which call goes with what is its own; 0 for none",
         0.0, 2.0, 0.1),
        ("CALL_EMPATHY", "Call empathy", "how much a caller learns, by its empathy, from whether those of its kind who heard its call then fared better or worse than they expected, kin most; 0 for none",
         0.0, 5.0, 0.1),
        ("CALL_BASELINE", "Call baseline", "how far a call learns only its own part, beyond keeping quiet: its own pleasures, habit and empathy, less its cost; 0: the whole turn, as any choice",
         0.0, 1.0, 0.1),
        ("CALL_TEMPERATURE", "Call temperature", "how loosely a brain picks among silence and calls, and each later sound", 0.01, 5.0, 0.01),
        ("OUTSIDE_FORESIGHT", "Outside foresight", "how fast a brain learns how near it will sense a hunter, prey and meat next turn, times its foresight learning; no reward in it; 0 for none",
         0.0, 20.0, 0.5),
        ("SPEAK_AS_HEARD", "Speak as heard", "how much a call is taught by how much nearer the caller's own foresight, hearing that call, comes to the hunter, prey or meat it saw than on hearing nothing or any one sound; 0 for none",
         0.0, 20.0, 0.5),
        ("ECHO", "Echo", "how much readier the sounds an animal last heard just before something surprised it come, when that same thing next surprises it: it changes only which sound, never whether it calls or how long; 0 for none", 0.0, 10.0, 0.1),
        ("INTENT", "Intent", "how much a call is taught by the share of those who heard it who then did what its caller most wished of them (come nearer, go off, flee, rest, eat, trade or fight), its wishes learnt from how others' deeds went for it; 0 for none", 0.0, 10.0, 0.1),
        ("WISH_LEARNING", "Wish learning", "how much a caller learns what it wants of those who hear it: each deed they then did, valued by how its own turn went against what it expected, as a share of its learning; 0 for none", 0.0, 2.0, 0.05),
        ("POINTING", "Pointing", "how much a caller learns what to point at as it calls: those who hear it sense how close the thing it points at is to them, not what it is, and each thing is valued by how much less surprised by it they then were, as a share of its learning; 0 for none", 0.0, 2.0, 0.05),
        ("UNDERSTOOD", "Understood", "how much a call is taught by how much less surprised by the thing its caller pointed at its hearers then were, so sounds that tell what is pointed at are favoured; 0 for none", 0.0, 20.0, 0.5),
        ("ATTENTION", "Attention", "a finger is seen in two steps: hearing a call, or seeing one point, an animal chooses whether to attend to that one, and only the turn after, attending, sees its finger; while it attends, its sight of all else outside is dimmed by this share; 0 for no price", 0.0, 1.0, 0.05),
        ("ATTEND_LEARNING", "Attend learning", "how much an animal learns whether to attend to a caller or one pointing, by what came of its turn, as a share of its learning; 0 for none", 0.0, 2.0, 0.05),
        ("ATTEND_CURIOSITY", "Attend curiosity", "how much curiosity leans an animal to attend to a caller or one pointing: the most curious lean to attend by this much, the least to ignore; 0 for none", 0.0, 1.0, 0.01),
        ("SHOWN_LEARNING", "Shown learning", "how much an attending animal learns of the thing it was shown, as a share of its foresight's learning; this and whether to attend are learnt the more the more sociable and clever it is; 0 for none", 0.0, 5.0, 0.1),
        ("WATCHED_REPEAT", "Watched repeat", "a caller that knows it is attended to repeats its call and point until none watches, one is less surprised by the thing, or it flees or cries out; 0 for off", 0.0, 1.0, 1.0),
        ("KNOWN_ATTEND", "Known attend", "when one it knows (an acquaintance, a parent or its young) calls or points, an animal attends unbidden, with this chance times its sociability; strangers it still chooses whether to heed; 0 for off", 0.0, 1.0, 0.01),
        ("ARENA_THINGS", "Arena things", "for experiments: animals sense only this many of the outside things, grass, plant and herd first, then hunter, then the rest; the others as none; 0 for all", 0, 21, 1),
        ("ARENA_BLIND", "Arena blind", "for experiments: chance an animal that heard a call senses none of the outside things that turn, so the call is all it has to go by; 0 for off", 0.0, 1.0, 0.05),
        ("ARENA_PAYOFF", "Arena payoff", "for experiments: how a blinded listener's turn went against what it expected is taught to the call it heard, this much, whole, not weighed by care or empathy; 0 for off", 0.0, 20.0, 0.5),
        ("HABITUATION", "Habituation", "how much an animal gets used to things: each outside thing is sensed by how much nearer it is than it has lately been, this much, and by its plain nearness for the rest; 0 for off", 0.0, 1.0, 0.05),
        ("HABITUATION_RATE", "Habituation rate", "how fast an animal gets used to what is about it: the share of the way its sense of how near each thing has lately been moves each turn", 0.0, 1.0, 0.01),
        ("CALL_HARM", "Harm after calling", "struck soon after it called, an animal's call is taught the harm, this far below nothing, against what it sensed as it called; whether calling draws hunters is left to them; 0 for none", 0.0, 10.0, 0.1),
        ("CALL_HARM_TURNS", "Harm after calling: turns", "how many turns after calling a strike still teaches the call its harm", 1, 20, 1),
        ("PARENT_HUSH", "Hushing", "a young one that calls with a parent in sight is taught, of that call, how much its parent would now rather keep quiet than call, this much; only a parent that has come to prefer quiet hushes; 0 for none", 0.0, 10.0, 0.1),
        ("CUES", "Cues", "1: each action has a cue, one of the five sounds, first by chance and then taken on from others seen calling as they act; pain and panic cry out; by its intellect an animal voices its cue by instinct or chooses its call; 0: calls by the brain alone", 0, 1, 1),
        ("CUE_URGE", "Cue urge", "with cues, the chance that an animal acting by instinct voices the cue of what it does, rather than keep silent", 0.0, 1.0, 0.05),
        ("CUE_FAVOUR", "Cue favour", "with cues, how much more an animal choosing its call values the cue of what it does", 0.0, 1.0, 0.01),
        ("CUE_SOCIABILITY", "Cue sociability", "with cues, how much more an animal choosing its call values the cue of what it does, times its sociability less a half: the sociable lean to voicing it, the solitary to not", 0.0, 1.0, 0.01),
        ("CUE_COPY", "Cue copying", "with cues, the chance that one who hears and sees another of its kind call as it acts takes that sound as its own cue for that action", 0.0, 1.0, 0.01),
        ("CRY_DRIFT", "Cry drift", "with cues, the chance that a newborn's cry of pain and panic is another sound than its mother's", 0.0, 1.0, 0.01),
        ("NAMING", "Naming", "how much an itch to tell, built by surprise (what it senses from afar against what it foresaw, with its herd near), lifts every sound alike over silence, times its intelligence and a half plus its sociability; which sound, and what it comes to mean, is left to it; 0 for none", 0.0, 5.0, 0.1),
        ("NAMING_KEEP", "Naming itch kept", "with naming, how much of its itch to tell an animal keeps from one turn to the next", 0.0, 0.99, 0.01),
        ("NAMING_RELIEF", "Naming relief", "with naming, how much more a call is taught when one who heard it is then less surprised than it was, the itch passing", 0.0, 5.0, 0.1),
        ("ALERTS", "Alerts", "1: a grazer seeing a hunter, or a hunter a fresh kill, may alert its herd to it; 0: no alert, only the five calls", 0, 1, 1),
        ("COPY_OWN_SENSES", "Copy in own senses", "1: a child or a reader of body language learns a heard call where it was itself; 0: where the caller was", 0, 1, 1),
        ("MEMORY_STEPS", "Memory steps", "choices back a lesson's error reaches the memory gates that wrote what the brain read; 0: memory is only as born",
         0, 4, 1),
        ("MEMORY_LEARNING", "Memory learning", "how fast memory gates learn, times the brain's learning rate", 0.0, 5.0, 0.05),
        ("MAP_PLACES", "Map places", "places an animal's mental map holds, the one visited longest ago forgotten for a new one; 0: no map",
         0, 32, 1),
        ("MAP_LEARNING", "Map learning", "how far each visit moves what an animal remembers of a place in that season toward what it finds",
         0.0, 1.0, 0.05),
        ("ACQUAINTANCES", "Acquaintances", "others an animal remembers by what they did to it, the one it dealt with longest ago forgotten for a new one; 0: none",
         0, 16, 1),
        ("REGARD_LEARNING", "Regard learning", "how far each deed moves an animal's regard for the doer toward the deed's worth", 0.0, 1.0, 0.05),
        ("INHERITED_REGARD", "Inherited regard", "share of how one holds a child's parents, regard and reputation, it holds the child at until it judges it by its own deeds", 0.0, 1.0, 0.05),
        ("SHARED_BOND", "Shared bond", "how much shared moments, a fire enjoyed together or a herd-mate defended, move regard, as a share of a deed's weight; 0 for none", 0.0, 1.0, 0.05),
        ("SKILL_COMFORT", "Skill comfort", "ease near one it holds skilled, more if curious; 0 for none", 0.0, 2.0, 0.05),
        ("TUTOR_JOY", "Tutor joy", "joy of giving one of its kind a lesson in a craft or skill, the more for the empathetic", 0.0, 5.0, 0.05),
        ("TUTOR_PRACTICE", "Tutored practice", "share of a try, or of a shot's lesson, that a lesson from a tutor is worth", 0.0, 1.0, 0.05),
        ("WATCH_LEARNING", "Watch learning", "chance that a craft seen done by one of its kind it holds no foe is learnt from the watching, and share of a shot's lesson a hit seen with a bow teaches; 0 for none", 0.0, 1.0, 0.01),
        ("CHILD_CALL_LEARNING", "Child call learning", "share of its own learning a child that hears its parent call learns to make that call where the parent made it, and, hearing it, what the parent then did; 0 for none", 0.0, 2.0, 0.05),
        ("BODY_LANGUAGE", "Body language", "chance, for each point of its intellect, that one who sees the caller of its kind it hears understands what it meant: it learns to make that call where the caller made it, and, hearing it, what the caller then did; 0 for none", 0.0, 0.2, 0.01),
        ("ELDER_COMFORT", "Elder comfort", "gladness at staying near the elder who taught it, more if trusting; 0 for none", 0.0, 2.0, 0.05),
        ("ELDER_REACH", "Elder reach", "steps from the elder who taught it within which a student is glad of it", 0, 20, 1),
        ("FOLLOW_SAFETY", "Follow safety", "share of its dread spared, and felt as relief, with a friend or one it holds skilled close by; 0 for none", 0.0, 1.0, 0.05),
        ("FOLLOW_REACH", "Follow reach", "steps within which a friend or one it holds skilled eases its dread", 0, 20, 1),
        ("DIARY_ENTRIES", "Diary entries", "most vivid experiences an animal keeps and re-learns from, one each turn, the faintest forgotten for a more vivid one; 0: none",
         0, 32, 1),
        ("DIARY_FADING", "Diary fading", "how much of a diary entry's vividness is left after each turn", 0.9, 1.0, 0.001),
        ("HUNTING_INSTINCT", "Hunting instinct", "how much more a predator values hunting and attacking, times its hunger",
         0.0, 5.0, 0.05),
        ("PRIMAL_FEAR", "Primal fear", "chance a turn of being overcome by an overwhelmingly stronger hunter in sight",
         0.0, 1.0, 0.05),
        ("PRIMAL_FEAR_PREDATORS", "Primal fear, predators", "how susceptible predators are, as a share", 0.0, 1.0, 0.05),
        ("PREDATOR_LOOMS", "Predators loom", "steps larger and stronger a predator counts as, sizing up a fight for primal fear",
         0, 7, 1),
        ("PREDATOR_INTELLIGENCE", "Predator intelligence", "times over a predator's intellect counts, up to the full range",
         1.0, 5.0, 0.1),
        ("BRAIN_COST", "Brain cost", "how much faster the brightest animal's hunger burns than the dullest's", 0.0, 1.0, 0.01),
        ("PREDATOR_BRAIN_COST", "Predator brain cost", "the share of the brain cost a predator pays", 0.0, 1.0, 0.05),
        ("PREDATOR_FEAR", "Predator fear", "share of fear's effects a predator feels: dread, panic, keeping to its herd",
         0.0, 1.0, 0.05),
        ("PRIMAL_FREEZE", "Primal freeze", "chance an overcome animal freezes rather than flees, at the lowest intellect",
         0.0, 1.0, 0.05),
    ],
    "Fighting": [
        ("STRIKE_FORCE", "Strike force", "scales every blow", 0.0, 10.0, 0.1),
        ("PREDATOR_SIGHT", "Predator sight", "cells further a predator sees prey", 0, 3, 1),
        ("HUNT_STEPS", "Hunt steps", "steps further a predator chases prey a turn", 0, 5, 1),
        ("GRAZER_DEFENCE_WEAPONS", "Grazer defence weapons", "share of its weapons a grazer brings to defence", 0.0, 1.0, 0.05),
        ("OMNIVORE_DEFENCE_WEAPONS", "Omnivore defence weapons", "share of its weapons an omnivore brings to defence", 0.0, 1.0, 0.05),
        ("PREDATOR_WEAPONS", "Predator weapons", "a predator's natural weapons count this many times over; an omnivore's 3/4, a grazer's half as many", 0.0, 10.0, 0.1),
        ("GRAZER_DEFENCE_STRENGTH", "Grazer defence strength", "a grazer can strike back only with strength above this",
         0, 7, 1),
        ("GRAZER_FORCE", "Grazer force", "the share of a grazer's base force in a blow it strikes, not in defence, before weapons, gear and other bonuses: it has no natural weapons to attack with", 0.0, 1.0, 0.05),
        ("STRIKE_EFFORT", "Strike effort", "food burned for each point of damage a blow deals", 0.0, 2.0, 0.05),
        ("THORN_FORCE", "Thorn force", "a plant's counterblow at full weapons and size", 0, 100, 1),
        ("THORN_SIZE_SHARE", "Thorn size share", "share of that from the plant's size; the rest its weapons",
         0.0, 1.0, 0.05),
        ("INFANT_THORNS", "Infant thorns", "the share of a thorny plant's counterblow an infant takes, of what a grown animal would", 0.0, 1.0, 0.05),
        ("PACK_BONUS", "Pack bonus", "extra force at prey for each herd-mate also alongside it, up to three", 0.0, 5.0, 0.05),
        ("JOINED_HUNT_JOY", "Joined hunt joy", "pleasure of striking prey that a pack-mate is also alongside, times a half and its herd instinct", 0.0, 5.0, 0.05),
        ("MEAT_PORTION", "Meat portion", "hunger a portion of meat from a pile sates, times how well the eater digests carrion", 0, 2000, 5),
        ("FRUIT_PORTION", "Fruit portion", "hunger a portion of fruit from a pile sates, times the eater's share of the food in plants", 0, 500, 5),
        ("PILE_LASTS", "Pile life", "times as long as a corpse a pile lasts, since food was last dropped on it", 0.0, 20.0, 0.5),
        ("WOOD_LASTS", "Wood life", "times as long as a pile of food a pile of wood or bone lasts, unless sheltered", 0.0, 100.0, 1.0),
        ("STONE_SCATTER_PER", "Stone scatter", "squares to each stone scattered, once in a world's life", 10, 1000, 10),
        ("DISCOVERY_CHANCE", "Discovery chance", "chance each turn, times intelligence, of discovering a craft whose materials it holds", 0.0, 1.0, 0.0001),
        ("CRAFT_TRIES", "Craft attempts", "tries by which a craft is sure to work", 1, 50, 1),
        ("CRAFT_THRILL", "Craft thrill", "the thrill of a failed craft, times curiosity", 0.0, 5.0, 0.1),
        ("DISCOVERY_JOY", "Discovery joy", "the joy of first discovering a craft; more if curious", 0.0, 20.0, 0.5),
        ("CRAFT_JOY", "Craft joy", "the joy of a craft that works, a weapon's in full and armour's in half; more if curious", 0.0, 5.0, 0.1),
        ("TOOL_JOY", "Tool joy", "the joy of the harm a weapon added to a blow, times the share of its victim's health that harm was", 0.0, 10.0, 0.1),
        ("ARMOUR_JOY", "Armour joy", "the joy of the harm its armour took off a blow, times the share of its health that harm was", 0.0, 10.0, 0.1),
        ("GEAR_EASE", "Gear ease", "ease, each turn, of a weapon or bow in hand, and as much again of armour worn", 0.0, 2.0, 0.05),
        ("GEAR_COURAGE", "Gear courage", "share of its fear's effects a weapon or bow in hand takes off, and as much again armour worn", 0.0, 0.5, 0.05),
        ("DON_JOY", "Donning joy", "the joy of gaining a weapon or bow, as this share of tool joy, and armour, as this share of armour joy; dropping one is as much a loss", 0.0, 2.0, 0.05),
        ("GEAR_DESIRE", "Gear desire", "desire for a weapon, armour or a bow stirred on seeing one pay off, friend or foe, up to 1; envy counts that gear one plus its desire times", 0.0, 1.0, 0.01),
        ("WAGON_DRAG", "Wagon drag", "share more tiring each step is, dragging a wagon", 0.0, 3.0, 0.05),
        ("READY_JOY", "Ready joy", "joy, for each part, of first holding all a thing it knows how to make needs: a recipe's materials, a shelter's worth of stone or wood, or a road section's stones, with the tool each wants; mostly more if patient, a little if clever; breaking the set but by using or giving it is as much a loss", 0.0, 2.0, 0.05),
        ("BUILD_THRILL", "Build thrill", "the thrill of each stage of a shelter built, growing with every stage and doubled for finishing it, times curiosity", 0.0, 5.0, 0.1),
        ("SHELTER_UNITS", "Shelter size", "units of stone or wood that finish a shelter", 1, 50, 1),
        ("SHELTER_COMFORT", "Shelter comfort", "comfort of resting inside a finished shelter, half of it standing in one", 0.0, 2.0, 0.05),
        ("SHELTER_LONGING", "Shelter longing", "how much more building thrills one that has rested in a shelter, in full after a day inside, whatever its curiosity", 0.0, 5.0, 0.1),
        ("ROAD_FATIGUE", "Road fatigue", "stamina each turn of laying road costs, beyond the action's own", 0.0, 200.0, 5.0),
        ("ROAD_THRILL", "Road thrill", "the thrill of each turn's work on a road, growing with every turn and doubled for finishing it, times curiosity", 0.0, 5.0, 0.1),
        ("ROAD_LINK_JOY", "Road link joy", "gratification of finishing a road that joins two shelters no road joined before", 0.0, 10.0, 0.1),
        ("ROAD_EASE", "Road ease", "share of a step's stamina a step from a road square costs", 0.0, 1.0, 0.05),
        ("ROAD_JOY", "Road joy", "pleasure of going by road, growing to the full of it after a day of road travel", 0.0, 2.0, 0.05),
        ("ROAD_LONGING", "Road longing", "how much more laying road thrills one that has travelled by road, in full after a day of it, whatever its curiosity", 0.0, 5.0, 0.1),
        ("FERTILIZING_JOY", "Fertilizing joy", "pleasure of feeding crushed bone to worked land, times curiosity", 0.0, 5.0, 0.1),
        ("MARROW_LEAN", "Marrow lean", "how far each marrow meal edges a grazer toward meat; at the full of it, it and its young are omnivores; 0 for never", 0.0, 1.0, 0.01),
        ("COOKED_LEAN", "Cooked food lean", "share of a marrow meal's lean toward omnivore that each cooked meal of another's food gives: cooked meat to a grazer, cooked fruit to a predator; 0 for none", 0.0, 1.0, 0.05),
        ("PREDATOR_COOKED_FRUIT", "Predator cooked fruit", "share of an omnivore's nourishment from fruit that a predator gets from cooked fruit; raw fruit it cannot eat; 0 for none", 0.0, 1.0, 0.05),
        ("SKINNING_JOY", "Skinning joy", "pleasure of wearing a new, finer hide, times curiosity, the more for a finer one", 0.0, 5.0, 0.1),
        ("HIDE_WARMTH", "Hide warmth", "steps of cold a worn hide keeps out", 0.0, 3.0, 0.25),
        ("HIDE_ARMOUR_SHARE", "Hide armour share", "share of the armour of the body a hide came from that it adds to its wearer's", 0.0, 1.0, 0.05),
        ("WARRIOR_GARB", "Warrior garb", "regard an omnivore's hide, a warrior's garb, draws those who see it toward", 0.0, 1.0, 0.05),
        ("PRESTIGE_GARB", "Prestige garb", "regard a predator's hide, a mark of prestige, draws those who see it toward", 0.0, 1.0, 0.05),
        ("GARB_REGARD", "Garb regard", "how much of a deed's weight each turn's sight of a garb has on regard", 0.0, 1.0, 0.01),
        ("LIE_DISREGARD", "Lie disregard", "disregard for one seen fleeing a fight in a garb, as a share of the regard its garb had won, which is lost too", 0.0, 3.0, 0.1),
        ("ENVY_LEVEL", "Envy level", "share of its own kind, ranked by moodiness or dominance, that envies no one: 0.75 leaves the top quarter envious; 1 for none", 0.0, 1.0, 0.05),
        ("ENVY_UNEASE", "Envy unease", "unease of seeing possessions better than its own, by how much better, at the most envious", 0.0, 3.0, 0.05),
        ("ENVY_JOY", "Envy joy", "gladness of the envious at gaining a better possession, by how much better", 0.0, 3.0, 0.05),
        ("FOE_LEVEL", "Foe level", "disregard beyond which one is a foe, times 0.5 plus half the beholder's empathy and sociability together", 0.0, 1.0, 0.05),
        ("VICTORY_REGARD", "Victory regard", "how much a victory seen moves regard for the winner: up for a friend, down for a foe", 0.0, 1.0, 0.05),
        ("CULTURE_REGARD", "Culture regard", "esteem for a deed its kind values, seen done by one of its kind (twice from family, or a predator's pack): victories for predators, crafting and building for omnivores, sowing and fruit picked without uprooting for grazers", 0.0, 1.0, 0.05),
        ("FOE_DEFEAT_REGARD", "Foe defeat regard", "how much more regard a victory over one's foe wins", 0.0, 1.0, 0.05),
        ("COWARD_DISREGARD", "Coward disregard", "how much fleeing a fight costs one's regard with those who hold it a friend or foe", 0.0, 1.0, 0.05),
        ("GIVE_LEVEL", "Give level", "share of its own kind, ranked by empathy or sociability, that never gives a friend a possession: 0.75 leaves the top quarter giving; 1 for none", 0.0, 1.0, 0.05),
        ("GIVE_JOY", "Give joy", "pleasure of giving a friend a possession, times 0.5 plus empathy or sociability", 0.0, 3.0, 0.05),
        ("TAKE_LEVEL", "Take level", "share of its own kind, ranked by aggression or dominance, that never demands or steals a possession: 0.75 leaves the top quarter taking; 1 for none", 0.0, 1.0, 0.05),
        ("TRADE_MISJUDGE", "Trade misjudgement", "how far an animal may misjudge a good's worth in a deal, either way, times one less its intelligence", 0.0, 2.0, 0.05),
        ("TRADE_REGARD", "Trade regard", "how much of a deed's weight an accepted offer has on regard, a kindness both ways", 0.0, 1.0, 0.05),
        ("TRADE_JOY", "Trade joy", "pleasure a side takes in a deal it truly gained by; one that truly lost thinks as much the worse of the other", 0.0, 3.0, 0.05),
        ("TRADE_WARINESS", "Trade wariness", "how much worse an offer seems from one its owner thinks ill of, times that ill regard", 0.0, 2.0, 0.05),
        ("LEADER_FOLLOWERS", "Leader followers", "members of its colour group who must hold one as their standout for it to be recognised a leader, kept until fewer than half as many do; 0 for no leaders", 0, 100, 1),
        ("LEADER_DEFERENCE", "Leader deference", "how much kinder a follower judges its leader's offers; it yields to its leader's demands without a fight", 0.0, 2.0, 0.05),
        ("RANK_BONUS", "Rank bonus", "share more of each leader's bonuses (deference, rally, joy in followers, allure) for each rank above chief", 0.0, 2.0, 0.05),
        ("TITLE_REGARD", "Title regard", "esteem a titled leader wins each turn from each who sees it and holds it no foe, times its rank", 0.0, 0.2, 0.005),
        ("LEADER_JOY", "Leader joy", "pleasure a leader takes in its followers in sight, in full with as many as Leader followers", 0.0, 3.0, 0.05),
        ("LEADER_ALLURE", "Leader allure", "how much more joy breeding with a leader gives its partner; a leader is courted first by any that may breed with it", 0.0, 3.0, 0.05),
        ("THEFT_DISREGARD", "Theft disregard", "how much a theft seen costs the thief's regard, twice with its victim; regard instead with those who hold the victim a foe", 0.0, 1.0, 0.05),
        ("WARRIOR_MARK", "Warrior mark", "regard the mark of the warrior, earned by defeating an omnivore, draws those who see it toward", 0.0, 1.0, 0.05),
        ("PRESTIGE_MARK", "Prestige mark", "regard the mark of prestige, earned by defeating a predator, draws those who see it toward", 0.0, 1.0, 0.05),
        ("MARK_REGARD", "Mark regard", "how much of a deed's weight each turn's sight of a mark has on regard, or, for a foe, on how dangerous it holds the bearer", 0.0, 1.0, 0.01),
        ("MARK_DREAD", "Mark dread", "unease of a foe at the sight of a mark, times its worth and a half plus its fear", 0.0, 3.0, 0.05),
        ("LAND_SIZE", "Land size", "members a colour group needs to claim land; 0 for no lands", 0, 500, 1),
        ("LAND_PATCH", "Land square", "cells on a side of a square of land; a change begins all claims again", 2, 64, 1),
        ("LAND_HOLD", "Land hold", "members of a group standing in a square that claim it", 1, 50, 1),
        ("LAND_MEMORY", "Land memory", "counts a claim lasts with none of its group in the square, or while clearly losing the war for it", 1, 100, 1),
        ("LAND_TURNS", "Land count", "turns between counts of the groups and their lands", 1, 1000, 1),
        ("LAND_HEALING", "Land healing", "how much faster an animal heals in its own land: 1 for twice as fast", 0.0, 5.0, 0.1),
        ("LAND_DEFENCE", "Land defence", "share of a blow from an outsider kept off one in its own land", 0.0, 1.0, 0.05),
        ("LAND_HERD", "Land herd", "how much more its herd counts in its own land, for safety and the pack's blows: 1 for twice", 0.0, 3.0, 0.05),
        ("LAND_COMFORT", "Land comfort", "ease in its own land, and unease as much away from it while its group holds land", 0.0, 2.0, 0.05),
        ("TRESPASS_REGARD", "Trespass regard", "how much of a deed's weight a trespasser's first turn in a land costs it with those of the land who see it", 0.0, 1.0, 0.01),
        ("TRESPASS_GROWTH", "Trespass growth", "how much that grows, of itself, each turn the trespasser stays, to a whole deed", 0.0, 2.0, 0.05),
        ("LAND_AMBITION", "Land ambition", "pull to stay in the land of a group it dislikes, and unease in the land of one it likes unless welcome there, times its regard for that group", 0.0, 1.0, 0.01),
        ("WELCOME_LEVEL", "Welcome level", "how well a group's members must think of a colour group, on average, to welcome it in their land as guests who neither trespass nor contest it; above 1 for none", 0.0, 1.5, 0.05),
        ("CALL_PAYOFF", "Call payoff", "joy a caller takes in each of its kind that heard its call and then got away from a hunter or, hungry, ate; times a half and its empathy or sociability", 0.0, 2.0, 0.05),
        ("COMMUNICATION_JOY", "Communication joy", "share of its own gain each takes when one answers a call it heard, within a day, with a deed good for both: the caller glad to be understood, the listener to understand", 0.0, 2.0, 0.05),
        ("HEEDED_REGARD", "Heeded regard", "how much better one helped so thinks of its caller, of a deed's weight", 0.0, 1.0, 0.01),
        ("LEADER_RALLY", "Leader rally", "share of the chance to panic or break that a follower in sight of its leader overcomes", 0.0, 1.0, 0.05),
        ("LAND_RALLY", "Land rally", "share of the chance to panic, break or be overcome by primal fear that defending its own land overcomes, a born coward's too", 0.0, 1.0, 0.05),
        ("WAR_ENMITY", "War enmity", "how much of a deed's weight each count of a war sets each member against its enemy's colour", 0.0, 1.0, 0.05),
        ("WAR_KILLS", "War kills", "deaths at each other's hands before a war counts as clearly won (two thirds of them or more) or lost (a third or less)", 1, 100, 1),
        ("WAR_MORALE", "War morale", "gladness of a war clearly being won, and unease as much of one clearly being lost, each turn", 0.0, 2.0, 0.05),
        ("WAR_NERVE", "War nerve", "how much less likely to panic a war clearly being won leaves its side, and more the losing side", 0.0, 1.0, 0.05),
        ("WAR_FADE", "War fade", "share of a war's kills forgotten at each count of the lands, so recent deaths decide it and an old defeat is in time forgiven", 0.0, 1.0, 0.01),
        ("STANDING_JOY", "Standing joy", "pleasure of being held a friend by those about it, in full with three or more, times a half and its dominance", 0.0, 2.0, 0.05),
        ("FOLLOW_COMFORT", "Follow comfort", "ease near the member of its own colour group in sight it esteems most, times that esteem and closeness, and a half plus its trust and sociability together", 0.0, 2.0, 0.05),
        ("PROTECTION_JOY", "Protection joy", "relief at the weather kept off it by a shelter, a fire, or a parent close by, by as much unease as it spares", 0.0, 5.0, 0.1),
        ("GEAR_USES", "Gear uses", "uses before a tool or armour of bone or wood breaks", 1, 20, 1),
        ("WEAPON_FORCE", "Weapon force", "times as hard an axe or club strikes", 1.0, 5.0, 0.1),
        ("BOW_USES", "Bow uses", "shots a bow makes before it is spent", 1, 100, 1),
        ("ARCHERY_JOY", "Archery joy", "joy of a hit with a bow, in practice too, the more for the skilled", 0.0, 5.0, 0.05),
        ("ARCHERY_REGARD", "Archery regard", "respect of all who see a hit with a bow, of a deed's weight, in skill and liking", 0.0, 1.0, 0.01),
        ("ARCHERY_LEARNING", "Archery learning", "share of the skill left to learn that each shot with a bow teaches", 0.0, 0.5, 0.01),
        ("STONE_FORCE", "Stone force", "times as hard a wielded stone strikes", 1.0, 5.0, 0.1),
        ("ARMOUR_FORCE", "Armour force", "times its armour that worn armour counts for", 1.0, 5.0, 0.1),
        ("ARMOUR_LEAST", "Least armour", "the armour worn armour counts from, at the least", 0, 7, 1),
        ("FIRE_TURNS", "Fire life", "turns a fire burns, and each wood fed to it adds", 1, 200, 1),
        ("FIRE_PANIC", "Fire panic", "chance an animal panics, seeing a fire not its own or its herd's", 0.0, 1.0, 0.05),
        ("FIRE_JOY", "Fire joy", "the sense of accomplishment in lighting a fire, beside the comfort of it", 0.0, 5.0, 0.1),
        ("FIRE_COMFORT", "Fire comfort", "comfort in sight of a fire that is its own or its herd's", 0.0, 2.0, 0.05),
        ("FIRE_FEED_JOY", "Fire feeding joy", "the satisfaction of feeding a fire, so that it burns on", 0.0, 5.0, 0.1),
        ("SEED_FOOD", "Seed food", "how much of a fruit's portion a seed feeds", 0.0, 1.0, 0.05),
        ("PLANTING_JOY", "Planting joy", "the gladness of planting a seed; more if curious", 0.0, 5.0, 0.1),
        ("HARVEST_JOY", "Harvest joy", "the gladness of picking a fruit from a plant without uprooting it", 0.0, 5.0, 0.05),
        ("REGROWTH_JOY", "Regrowth joy", "the gladness of seeing a plant it last picked bear fruit again", 0.0, 5.0, 0.05),
        ("ARABLE_FRUIT", "Arable fruit", "times as much fruit a plant on worked land bears each morning", 1, 5, 1),
        ("FALLOW_YEARS", "Fallow years", "years worked land lies untended before it goes wild, its bone with it; 0, never", 0.0, 10.0, 0.25),
        ("FERTILIZED_FRUIT", "Fertilized fruit", "times as much again on worked land fed bone; the bone is spent", 1, 5, 1),
        ("DIG_CHANCE", "Dig chance", "chance that digging finds a stone", 0.0, 1.0, 0.05),
        ("COOKED_FOOD", "Cooked food", "times as nourishing as raw", 1.0, 5.0, 0.1),
        ("CARRIED_SPOIL", "Carried spoilage", "the chance each turn a raw portion carried goes off; cooked food keeps", 0.0, 0.5, 0.005),
        ("COOKED_JOY", "Cooked delight", "pleasure of eating a cooked portion, times a half and its taste for cooked food", 0.0, 5.0, 0.1),
        ("RAW_DISTASTE", "Raw distaste", "letdown of eating a raw portion once it knows the taste of cooked, times that taste", 0.0, 5.0, 0.1),
        ("COOK_JOY", "Cooking joy", "pleasure of cooking a portion, times a half and its taste for cooked food", 0.0, 5.0, 0.1),
        ("TASTE_JOY", "Taste joy", "how tasty a cooked portion is, and how untasty a raw one from a pile or carried, times its appetite", 0.0, 5.0, 0.1),
        ("GLUTTONY_UNEASE", "Gluttony unease", "unease at a meal eaten while less than half hungry, the more the fuller, times what it lacks of gluttony (a half, plus half its appetite, less half its kinship); none to the wholly gluttonous", 0.0, 5.0, 0.1),
        ("SMELL_REACH", "Smell reach", "how many times as far as it sees an animal smells cooked food and a fire cooked on", 1.0, 4.0, 0.1),
        ("KILL_MEAL", "Kill meal", "hunger a kill sates", 0, 200, 5),
        ("KILL_MEAL_FOR_SIZE", "Kill meal for size", "more hunger sated, up to this for the largest prey", 0, 200, 5),
        ("FRESH_KILL_TURNS", "Fresh kill turns", "turns after the one it died in that a body is a fresh kill", 0, 30, 1),
        ("FRESH_KILL_MEALS", "Fresh kill meals", "times the kill meal a predator or omnivore gets eating a fresh kill",
         0.0, 10.0, 0.5),
        ("KIND_DIFFERENCE", "Kind difference", "predators whose rings differ no more are one kind, to hunt and breed with",
         0.0, 1.0, 0.01),
        ("FLEE_RANGE", "Flee range", "a hunter is only worth fleeing within this many steps, unless alarmed",
         0, 12, 1),
        ("HERD_VIGILANCE", "Herd vigilance", "how far a close herd shrinks that range", 0.0, 1.0, 0.05),
        ("SWITCH_SHARE", "Prey switching", "a prey is hunted while it is at least this share of the food",
         0.0, 1.0, 0.05),
        ("OMNIVORE_PREDATOR_FOOD", "Omnivores' last resort", "an omnivore hunts predators only while no more grazers, plants and grassy cells than this are near",
         0, 100, 1),
    ],
    "Territory": [
        ("TERRITORY_RAMP", "Territory ramp", "rivals past the limit that bring rivalry to its full strength", 1, 20, 1),
        ("RIVAL_UNEASE", "Rival unease", "discomfort of a fully overcrowded territory", 0.0, 5.0, 0.05),
        ("PACK_RIVALRY", "Pack rivalry", "how much a predator's pack-mates count as rivals crowding its territory", 0.0, 1.0, 0.05),
        ("RIVALRY_WEAR", "Rivalry wear", "how much an overcrowded territory wears down reluctance to kill its own species",
         0.0, 1.0, 0.05),
        ("RIVAL_KILL_JOY", "Rival kill joy", "relief at killing a rival in an overcrowded territory", 0.0, 5.0, 0.05),
        ("HOSTILITY_IN_WANT", "Hostility in want", "1: an overcrowded territory wears down reluctance to kill its own kind, killing a rival is a joy, and lowering its own group's numbers is an aim (dropped once fed), only while its home band holds too little food for it to breed; 0: whatever the food", 0, 1, 1),
        ("TERRITORY.predator.fewest", "Predator rivals, cruel", "rivals the least empathetic predator tolerates",
         0, 50, 1),
        ("TERRITORY.predator.most", "Predator rivals, kind", "rivals the most empathetic predator tolerates",
         0, 50, 1),
        ("TERRITORY.predator.strength", "Predator rivalry", "how strongly overcrowding moves a predator",
         0.0, 2.0, 0.05),
        ("TERRITORY.omnivore.fewest", "Omnivore rivals, cruel", "rivals the least empathetic omnivore tolerates",
         0, 50, 1),
        ("TERRITORY.omnivore.most", "Omnivore rivals, kind", "rivals the most empathetic omnivore tolerates",
         0, 50, 1),
        ("TERRITORY.omnivore.strength", "Omnivore rivalry", "how strongly overcrowding moves an omnivore",
         0.0, 2.0, 0.05),
    ],
    "Herding": [
        ("HERD_SPREAD", "Herd spread", "how far from the herd's middle a fearless animal is content", 0, 12, 1),
        ("HERD_JOY", "Herd joy", "pleasure of being among three or more herd-mates", 0.0, 5.0, 0.05),
        ("HERD_LONELINESS", "Herd loneliness", "unease at having no herd-mate in range", 0.0, 5.0, 0.05),
        ("INFANT_HERDING", "Infant herding", "how much more an infant values keeping to its kind; it herds once a cell from their middle",
         0.0, 5.0, 0.05),
        ("NURSING_REACH", "Nursing reach", "each turn, a grown animal feeds the hungriest infant of its kind this close", 0, 10, 1),
        ("NURSING_LIMIT", "Nursing limit", "a grown animal feeds infants only while its own hunger stays below this", 0, 100, 5),
        ("SOLITARY_OMNIVORES", "Solitary omnivores", "1: omnivores live alone, in neither herd nor pack; 0: they herd with their own kind",
         0, 1, 1),
        ("HERD_SAFETY", "Herd safety", "how much three close herd-mates calm fear", 0.0, 1.0, 0.05),
        ("HERD_DEFEND_JOY", "Herd defence joy", "pleasure of striking back at a herd-mate's attacker", 0.0, 5.0, 0.05),
    ],
    "Food": [
        ("GRASS_MEAL", "Grass meal", "hunger a tuft of grass sates", 0, 200, 1),
        ("PLANT_MEAL", "Plant meal", "hunger a bite of a plant sates; a plant has a bite for every 15 of its health, "
         "more the larger", 0, 200, 1),
        ("PLANT_MEAL_FOR_SIZE", "Plant meal for size", "more hunger a bite sates, up to this for the largest plants",
         0, 200, 1),
        ("PLANT_BITE", "Plant bite", "health that makes a fruit: a plant holds one fruit for each, at least one, and picking one leaves it unhurt", 0, 200, 1),
        ("PLANT_REGROWTH", "Plant regrowth", "share of its health a plant regrows each turn", 0.0, 1.0, 0.005),
        ("PLANT_RECOVERY", "Plant recovery", "turns after it is uprooted from before a plant can be torn into again", 0, 300, 1),
        ("UPROOT_HUNGER", "Uproot hunger", "a grazer this hungry, with food scarce, may uproot a plant", 0, 100, 1),
        ("UPROOT_MEALS", "Uproot meals", "share of a feast on a body of the plant's size an uprooting grazer eats", 0.0, 10.0, 0.05),
        ("UPROOT_BITE", "Uproot bite", "health a plant loses to being uprooted", 0, 200, 1),
        ("UPROOT_SORROW", "Uproot sorrow", "a grazer's sorrow at tearing into a plant, times a half plus its empathy", 0.0, 5.0, 0.05),
        ("CARRION_MEAT", "Carrion meat", "meat a well-fed corpse leaves", 0, 200, 5),
        ("CARRION_MEAT_FOR_SIZE", "Carrion meat for size", "more meat, up to this for the largest", 0, 200, 5),
        ("GRAZING.grazer", "Grazer on grass", "share of the food in grass a grazer gets", 0.0, 3.0, 0.05),
        ("GRAZING.omnivore", "Omnivore on grass", "share of the food in grass an omnivore gets", 0.0, 3.0, 0.05),
        ("BROWSING.grazer", "Grazer on plants", "share of the food in plants a grazer gets", 0.0, 3.0, 0.05),
        ("BROWSING.omnivore", "Omnivore on plants", "share of the food in plants an omnivore gets", 0.0, 3.0, 0.05),
    ],
    "Plants": [
        ("SEED_CHANCE.0", "Seeding, least fertile", "a well-fed plant's chance each turn of seeding", 0.0, 1.0, 0.01),
        ("SEED_CHANCE.1", "Seeding, most fertile", "a well-fed plant's chance each turn of seeding", 0.0, 1.0, 0.01),
        ("SEED_COST", "Seed cost", "hunger seeding costs a plant", 0, 100, 1),
        ("SEED_HUNGER", "Seed hunger", "a plant hungrier than this doesn't seed", 0, 100, 1),
        ("SEEDLING_HUNGER", "Seedling hunger", "hunger a seedling starts with", 0, 100, 1),
        ("SEED_REACH", "Seed reach", "how far, in cells, a seed may land from its plant", 1, 9, 1),
        ("AVIAN_SEED_REACH", "Avian seed reach", "an avian plant's seeds land this many times as far", 1.0, 4.0, 0.25),
        ("LIGHT_RADIUS", "Light radius", "plants compete for light with plants within this many cells", 0, 12, 1),
        ("LIGHT_LIMIT", "Light limit", "this many such plants stop a plant seeding", 1, 50, 1),
        ("PLANT_CROWDING", "Plant crowding", "a seed doesn't take where more plants grow within two cells", 0, 50, 1),
    ],
    "Climate": [
        ("ARCTIC_SHARE", "Arctic", "share of the map's height that is arctic, at the top and again at the bottom",
         0.0, 0.5, 0.05),
        ("EQUATOR_SHARE", "Equator", "share of the map's height across its middle that is the equator", 0.0, 1.0, 0.05),
        ("POLAR_SHARE", "Polar caps", "share of the map's height at the top and again at the bottom that is very cold all year",
         0.0, 0.5, 0.005),
        ("WINTER_EQUATOR_SHARE", "Winter equator", "share of the map's height at the equator's end away from the winter that stays hot "
         "through both winters", 0.0, 1.0, 0.01),
        ("CLIMATE_HUNGER", "Climate hunger", "times the food needed a step or more from home (a plant, only too hot)",
         1.0, 5.0, 0.1),
        ("CLIMATE_DAMAGE", "Climate damage", "share of full health lost a turn for each step from home (plants from two)",
         0.0, 1.0, 0.01),
        ("RESTING_EXPOSURE", "Resting exposure", "times the weather's harm to an animal that chose to rest out of its "
         "temperature (not asleep or spent)", 1.0, 10.0, 0.1),
        ("WEATHER_GRACE", "Weather grace", "days an animal endures out of its temperature, uneasy but unharmed, before the weather harms it (half for the least resilient, half again for the most; twice as long weather it is hardy to)",
         0.0, 30.0, 0.5),
        ("PLANT_CLIMATE_YIELD", "Plant climate yield", "share of its food a plant gives out of its temperature",
         0.0, 1.0, 0.05),
        ("SEASON_TURNING", "Season turning",
         "share of a winter season its wave takes to wash in, and again to ebb out at its end (at most half)", 0.0, 0.5, 0.05),
        ("WAVE_BAND", "Wave band", "rows in each band of a season's wave", 1, 20, 1),
        ("WAVE_FRONT", "Wave front", "bands at a wave's front, part way between the old weather and the new", 1, 30, 1),
        ("GRASS_DIEBACK", "Grass dieback", "where it isn't medium, turns for each tuft of a cell's grass to die", 1, 500, 1),
        ("SEED_VIABILITY", "Seed viability", "years a seed waits for its temperature before it dies", 0.0, 5.0, 0.05),
        ("STARTING_TOLERANCE", "Starting tolerance", "chance a random starting genome carries each tolerance flag",
         0.0, 1.0, 0.05),
        ("HARDINESS", "Hardiness", "share of exposure's extra hunger and damage an animal tolerant of it suffers",
         0.0, 1.0, 0.05),
        ("FRAILTY", "Frailty", "share of exposure's extra hunger and damage an animal tolerant of the other extreme suffers",
         1.0, 4.0, 0.05),
    ],
    "Experiences": [
        (f"EXPERIENCES.{event}.{trait}", f"{event.replace('_', ' ').capitalize()}: {trait}",
         f"how {event.replace('_', ' ')} shifts {trait}, as a share of its range", -0.5, 0.5, 0.0005)
        for event, shifts in world.EXPERIENCES.items() for trait in shifts
    ],
}

# Rules that can't be changed here, and why: the Python version has no GPU to think on.
LOCKED: dict[str, str] = {
    "USE_GPU": "The Python version always thinks on the CPU; the C++ version thinks on a GPU where there is one.",
}

# Rules a world can't change once it is running, and why: changed on the map, they would wait for the next
# new world, so they are greyed out there.
WORLD_LOCKED: dict[str, str] = {
    "USE_GPU": "A running world keeps thinking where it started; this takes effect in a new world.",
}

# Warnings shown with a rule, in full when it is chosen: costs it runs up beyond the simulation itself.
WARNINGS: dict[str, str] = {
}

RULES = tuple(Rule(key, group, label, hint, minimum, maximum, step)
              for group, rows in _TABLE.items() for key, label, hint, minimum, maximum, step in rows)
BY_KEY = {rule.key: rule for rule in RULES}
# Pairs that must stay in order, smaller first, and pairs that must differ (lifespan divides by the gap).
ORDERED = (("LITTER_RANGE.0", "LITTER_RANGE.1"), ("GESTATION.0", "GESTATION.1"), ("LIFESPAN_YEARS.0", "LIFESPAN_YEARS.1"),
           ("LIFESPAN_YEARS.1", "LIFESPAN_YEARS.2"), ("IMPATIENT_DISCOUNT", "PATIENT_DISCOUNT"))
DISTINCT = (("METABOLISM.0", "METABOLISM.1"),)
_PARTS = {"fewest": (0, 0), "most": (0, 1), "strength": (1,)}


def _path(key: str) -> tuple[object, str, list]:
    """The module holding a rule's constant, the constant's name, and the steps into it."""
    name, *steps = key.split(".")
    if name == "TERRITORY":
        steps = [steps[0], *_PARTS[steps[1]]]
    return _MODULES.get(name, world), name, [int(step) if isinstance(step, str) and step.isdigit() else step for step in steps]


def get(key: str) -> float:
    module, name, steps = _path(key)
    value = getattr(module, name)
    for step in steps:
        value = value[step]
    return value


def _replaced(container, steps: list, value):
    """The container with the value at the end of `steps` replaced; tuples are rebuilt, tables
    changed in place."""
    if not steps:
        return value
    head, rest = steps[0], steps[1:]
    if isinstance(container, tuple):
        items = list(container)
        items[head] = _replaced(items[head], rest, value)
        return tuple(items)
    container[head] = _replaced(container[head], rest, value)
    return container


def _set(key: str, value: float) -> None:
    module, name, steps = _path(key)
    setattr(module, name, _replaced(getattr(module, name), steps, value))


DEFAULTS: dict[str, float] = {rule.key: get(rule.key) for rule in RULES}


def current() -> dict[str, float]:
    return {rule.key: get(rule.key) for rule in RULES}


def problems(values: Mapping[str, float]) -> list[str]:
    """Why a set of values can't be applied, if it can't."""
    found = ([f"{BY_KEY[low].label} must not exceed {BY_KEY[high].label.lower()}"
              for low, high in ORDERED if values[low] > values[high]]
             + [f"{BY_KEY[first].label} must differ from {BY_KEY[second].label.lower()}"
                for first, second in DISTINCT if values[first] == values[second]])
    if 2 * values["ARCTIC_SHARE"] + values["EQUATOR_SHARE"] > 1 + 1e-9:
        found.append("The two arctic zones and the equator must fit in the map together")
    return found


def apply(values: Mapping[str, float]) -> None:
    """Set every rule: those given to their (clamped) values, the rest to their defaults."""
    for rule in RULES:
        _set(rule.key, rule.clamp(values.get(rule.key, DEFAULTS[rule.key])))
    world.SEED_OFFSETS = sum(abs(dx) + abs(dy) <= world.SEED_REACH for dx, dy in world.SIGHT_OFFSETS)


def changed(values: Mapping[str, float]) -> dict[str, float]:
    return {key: value for key, value in values.items() if key in DEFAULTS and value != DEFAULTS[key]}  # a rule since retired is dropped


def shown(rule: Rule, value: float) -> str:
    """A value as the panel shows it: up to six significant digits, no trailing zeros."""
    return str(int(value)) if rule.integer else f"{value:.6g}"
