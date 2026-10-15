"""The simulation's tunable rules: every constant that shapes how the world plays out, so the
advanced panel can change them before a world starts.

Each rule names a module constant, or one entry of a tuple or table constant, by a key such as
FORAGE_RADIUS, ANIMAL_LIFESPAN.0, GRAZING.grazer or EXPERIENCES.kill.aggression. The C++ port lists
the same rules with the same keys, limits and steps (`bitzkey_cli --rules`). Rules are module-wide:
`apply` sets them all, and a world plays by whatever was set when it runs. Settings the
configuration panel already covers (world size, population, grass density, the population cap and
mutation rates) are not repeated here, nor are numbers that fix the genome's or the brain's layout,
or how starting groups are sized. How much rewind history is kept is a rule, with a warning about the
memory it takes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tools import Brain as brain
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


_MODULES = {"TEMPERATURE": brain, "MUTATION": brain, "TRACE_DECAY": brain, "SMART_TRACE_DECAY": brain,
            "HEADING_RELUCTANCE": brain}

# (key, label, hint, minimum, maximum, step) by group. A step that is an int makes a whole-number rule.
_TABLE: dict[str, list[tuple]] = {
    "World": [
        ("GRASS_REGROWTH", "Grass regrowth", "share of the starting grass that grows back each turn", 0, 0.1, 0.001),
        ("CORPSE_FERTILITY", "Corpse fertility", "grass grows this many times as fast on and around a corpse", 1, 10, 1),
        ("CORPSE_TURNS", "Corpse turns", "turns a corpse lasts while its meat rots away", 1, 500, 5),
        ("CORPSE_PLANT", "Corpse plant", "chance an animal's body that rots away entirely leaves the seed of a new plant", 0.0, 1.0, 0.05),
        ("CELL_SPACE", "Square space", "spaces in a square; a Bitzkey takes 1 + its size, and none is born, moves or spawns "
         "into a square without room", 1, 80, 1),
        ("SPAWN_SPREAD", "Spawn spread", "a starting group appears within this many cells of its centre", 0, 4, 1),
        ("USE_GPU", "Think on the GPU", "C++ only: 1 thinks and learns on every GPU found, in floats, with no rewinding; 0 on the CPU, exactly", 0, 1, 1),
        ("SPAWN_AT_HOME", "Spawn at home", "1: starting and imported groups appear where their home temperature is usual; 0: anywhere", 0, 1, 1),
        ("HISTORY_TURNS", "History turns", "turns of the world kept whole, to rewind the map to; at the top, all",
         1, world.HISTORY_ALL, 10),
        ("HISTORY_MEMORY", "History memory", "most memory, in GB, the rewind history may take", 0.1, 64.0, 0.1),
    ],
    "Bodies": [
        ("LIFESPAN_YEARS.0", "Lifespan, shortest", "years of life expected for the smallest bodies, plant or animal",
         0.5, 30.0, 0.5),
        ("LIFESPAN_YEARS.1", "Lifespan, middle", "years of life expected for middling bodies", 0.5, 30.0, 0.5),
        ("LIFESPAN_YEARS.2", "Lifespan, longest", "years of life expected for the largest bodies", 0.5, 30.0, 0.5),
        ("STARVING_DAMAGE", "Starving damage", "share of its full health a Bitzkey at full hunger loses each turn",
         0.0, 1.0, 0.01),
        ("MEAT_DIGESTION", "Meat digestion", "the share of its fullness that is meat empties at this share of the usual rate",
         0.05, 1.0, 0.05),
        ("AGEING", "Ageing", "from this share of its lifespan, an animal may die of old age, and is feeble", 0.0, 0.95, 0.05),
        ("FEEBLE_YOUTH", "Feeble youth", "an animal is feeble, an infant, for this share of its lifespan", 0.0, 0.5, 0.05),
        ("PREDATOR_INFANCY", "Predator infancy", "a predator is an infant this share as long as other animals", 0.0, 1.0, 0.05),
        ("FEEBLE_SHARE", "Feeble share", "a feeble animal's body traits and health are at this share of its own",
         0.1, 1.0, 0.05),
        ("METABOLISM.0", "Metabolism, smallest", "metabolic rate of the smallest bodies", 0.1, 5.0, 0.1),
        ("METABOLISM.1", "Metabolism, largest", "metabolic rate of the largest bodies", 0.1, 5.0, 0.1),
        ("STOMACH.0", "Stomach, smallest", "food the smallest body holds, relative to the middle", 0.1, 5.0, 0.1),
        ("STOMACH.1", "Stomach, largest", "food the largest body holds, relative to the middle", 0.1, 5.0, 0.1),
        ("SIZE_IN_MIGHT", "Size in might", "share of strength and toughness that comes from body size", 0.0, 1.0, 0.05),
        ("GRAZER_SPEED", "Grazer speed", "grazers move this many times as far a step, rounded half up", 0.5, 4.0, 0.25),
        ("WALK_REST", "Walk rest", "share of the turn's burn a one-cell walk saves, as resting does", 0.0, 1.0, 0.05),
        ("GO_EFFORT", "Go effort", "share of a turn's burn each cell a go_ moves beyond the first costs", 0.0, 2.0, 0.05),
    ],
    "Breeding": [
        ("BREED_RESERVE", "Breeding reserve", "hunger a parent must still have spare after paying to breed", 0, 100, 5),
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
        ("GESTATION.0", "Shortest gestation", "turns the smallest animals carry their young", 1, 500, 1),
        ("GESTATION.1", "Longest gestation", "turns the largest animals carry their young", 1, 500, 1),
        ("BIRTH_REACH", "Birth reach", "young are born on free cells up to this many cells from their mother", 1, 5, 1),
        ("PLANT_GESTATION", "Plant gestation", "turns a seed lies in the ground before it can sprout", 0, 500, 1),
        ("CROWDING_LIMIT", "Crowding limit", "litters shrink as crowding nears this; at it, no breeding; 0: no limit", 0, 100, 1),
        ("FORAGE_RADIUS", "Forage radius", "food and mouths are weighed within this many cells", 1, 12, 1),
        ("FOOD_NEEDED", "Food needed", "meals each mouth needs nearby to breed; twice this for a full litter; 0: no limit",
         0.0, 20.0, 0.5),
        ("UPROOT_FOOD", "Uproot food", "a starving grazer uproots a plant only with fewer meals a mouth than this about it",
         0.0, 20.0, 0.5),
        ("BREED_COOLDOWN", "Breeding cooldown", "share of the usual rest after breeding (5 to 20 turns by fertility); 0: none",
         0.0, 2.0, 0.05),
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
        ("GRASS_OUTLOOK", "Grass outlook", "grass is judged by what regrows in this many turns", 0, 1000, 10),
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
        ("EMPATHY_PAIN", "Empathy pain", "how much an empathetic animal feels another's wound", 0.0, 5.0, 0.05),
        ("OTHER_KIND", "Other kind", "how much of that pain reaches it for another species", 0.0, 1.0, 0.05),
    ],
    "Society": [
        ("SOCIAL_LEARNING", "Social learning", "how much a herd-mate's success or failure seen teaches, by empathy and intellect",
         0.0, 3.0, 0.05),
        ("SOCIAL_MOOD", "Social mood", "how much a herd-mate's success or failure seen moves the mood", 0.0, 3.0, 0.05),
        ("SOCIAL_MEMORY", "Social memory", "share of a seen success or failure still sensed a turn later", 0.0, 1.0, 0.05),
        ("LOYALTY_PULL", "Loyalty pull", "how much more an animal values keeping to its herd, times its loyalty", 0.0, 5.0, 0.05),
        ("DOMINANCE_EASE", "Dominance ease", "share less a crowded territory troubles the most dominant, and more the most submissive",
         0.0, 1.0, 0.05),
        ("TRUST_HEED", "Trust heed", "share more the most trusting take in what herd-mates do, and less the least", 0.0, 1.0, 0.05),
        ("TRUST_REACH", "Trust reach", "times its trust times its sight away a herd-mate's alert is heeded", 0.0, 5.0, 0.25),
        ("KINSHIP_SHUN", "Kinship shun", "at full kinship and no tolerance, a herd-mate's or partner's social background may "
         "differ by no more than 1 less this", 0.0, 1.0, 0.05),
    ],
    "Mind": [
        ("LEARNING_RATE", "Learning rate", "how fast a brain learns, at the highest intellect", 0.0, 0.5, 0.005),
        ("TRACE_DECAY", "Trace decay", "how far back a choice's outcome reaches earlier choices (lambda)",
         0.0, 1.0, 0.05),
        ("SMART_TRACE_DECAY", "Trace decay, smartest", "trace decay at the highest intellect; the lowest keeps the one above",
         0.0, 1.0, 0.01),
        ("LEARNING_BOOST", "Learning boost", "the highest intellect learns this much faster again; the lowest, no faster",
         0.0, 5.0, 0.1),
        ("REPLAY_PER_INTELLECT", "Replays per intellect", "recent experiences re-learnt each turn per point of intellect",
         0, 4, 1),
        ("BREED_LEARNING", "Breeding learning", "times as much any animal learns from breeding; more unsettles its brain for all else", 0.0, 20.0, 0.1),
        ("COURT_LEARNING", "Courting learning", "times as much any animal learns from courting; more unsettles its brain for all else", 0.0, 20.0, 0.1),
        ("KILL_FOOD_LEARNING", "Kill food learning", "times as much any animal learns from hunting, attacking and feasting",
         0.0, 10.0, 0.1),
        ("PLANT_FOOD_LEARNING", "Plant food learning", "times as much any animal learns from eating, foraging, browsing and uprooting",
         0.0, 10.0, 0.1),
        ("PREDATOR_LEARNING", "Predator learning", "times as fast a predator's brain learns from its own choices; an omnivore gets half the bonus",
         1.0, 5.0, 0.1),
        ("TEMPERATURE", "Choice temperature", "how loosely a brain picks among its options", 0.01, 5.0, 0.05),
        ("MUTATION", "Brain mutation", "how much a newborn's brain varies from its parents'", 0.0, 1.0, 0.01),
        ("HEADING_RELUCTANCE", "Heading reluctance", "how much less a new brain values setting off in a direction",
         0.0, 2.0, 0.05),
        ("TEMPERAMENT_RELAPSE", "Temperament relapse", "share of the gap back to the inborn value closed each turn",
         0.0, 0.1, 0.001),
        ("MEMORY_TURNS", "Memory turns", "turns the best memory recalls where it last fed or was struck; fewer the poorer", 0, 1000, 10),
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
        ("RARE_PREY", "Rare prey", "a hunter doesn't breed while fewer of its prey species, all together, live", 0, 1000, 1),
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
        ("HUNTING_INSTINCT", "Hunting instinct", "how much more a predator values hunting and attacking, times its hunger",
         0.0, 5.0, 0.05),
        ("PRIMAL_FEAR", "Primal fear", "chance a turn of being overcome by an overwhelmingly stronger hunter in sight",
         0.0, 1.0, 0.05),
        ("PRIMAL_FEAR_PREDATORS", "Primal fear, predators", "how susceptible predators are, as a share", 0.0, 1.0, 0.05),
        ("PREDATOR_LOOMS", "Predators loom", "steps larger and stronger a predator counts as, sizing up a fight for primal fear",
         0, 7, 1),
        ("PREDATOR_INTELLIGENCE", "Predator intelligence", "times over a predator's intellect counts, up to the full range",
         1.0, 5.0, 0.1),
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
        ("STRIKE_EFFORT", "Strike effort", "food burned for each point of damage a blow deals", 0.0, 2.0, 0.05),
        ("THORN_FORCE", "Thorn force", "a plant's counterblow at full weapons and size", 0, 100, 1),
        ("THORN_SIZE_SHARE", "Thorn size share", "share of that from the plant's size; the rest its weapons",
         0.0, 1.0, 0.05),
        ("PACK_BONUS", "Pack bonus", "extra force at prey that a herd-mate is also alongside", 0.0, 5.0, 0.05),
        ("KILL_MEAL", "Kill meal", "hunger a kill sates", 0, 200, 5),
        ("KILL_MEAL_FOR_SIZE", "Kill meal for size", "more hunger sated, up to this for the largest prey", 0, 200, 5),
        ("FRESH_KILL_TURNS", "Fresh kill turns", "turns after the one it died in that a body is a fresh kill", 0, 10, 1),
        ("FRESH_KILL_MEALS", "Fresh kill meals", "times the kill meal a predator or omnivore gets eating a fresh kill",
         0.0, 10.0, 0.5),
        ("KIND_DIFFERENCE", "Kind difference", "predators whose rings differ no more are one kind, to hunt and breed with",
         0.0, 1.0, 0.01),
        ("FLEE_RANGE", "Flee range", "a hunter is only worth fleeing within this many steps, unless alarmed",
         0, 12, 1),
        ("HERD_VIGILANCE", "Herd vigilance", "how far a close herd shrinks that range", 0.0, 1.0, 0.05),
        ("SWITCH_SHARE", "Prey switching", "a prey is hunted while it is at least this share of the food",
         0.0, 1.0, 0.05),
    ],
    "Territory": [
        ("TERRITORY_RAMP", "Territory ramp", "rivals past the limit that bring rivalry to its full strength", 1, 20, 1),
        ("RIVAL_UNEASE", "Rival unease", "discomfort of a fully overcrowded territory", 0.0, 5.0, 0.05),
        ("PACK_RIVALRY", "Pack rivalry", "how much a predator's pack-mates count as rivals crowding its territory", 0.0, 1.0, 0.05),
        ("RIVALRY_WEAR", "Rivalry wear", "how much an overcrowded territory wears down reluctance to kill its own species",
         0.0, 1.0, 0.05),
        ("RIVAL_KILL_JOY", "Rival kill joy", "relief at killing a rival in an overcrowded territory", 0.0, 5.0, 0.05),
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
        ("PLANT_MEAL", "Plant meal", "hunger a bite of a plant sates", 0, 200, 1),
        ("PLANT_MEAL_FOR_SIZE", "Plant meal for size", "more hunger sated, up to this for the largest plants",
         0, 200, 1),
        ("PLANT_BITE", "Plant bite", "health a plant loses to each bite", 0, 200, 1),
        ("PLANT_REGROWTH", "Plant regrowth", "share of its health a plant regrows each turn", 0.0, 1.0, 0.005),
        ("PLANT_RECOVERY", "Plant recovery", "turns after a bite before a plant can be bitten again", 0, 100, 1),
        ("UPROOT_HUNGER", "Uproot hunger", "a grazer this hungry, with food scarce, may uproot a plant", 0, 100, 1),
        ("UPROOT_MEALS", "Uproot meals", "share of a feast on a body of the plant's size an uprooting grazer eats", 0.0, 10.0, 0.05),
        ("UPROOT_BITE", "Uproot bite", "health a plant loses to being uprooted", 0, 200, 1),
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
        ("LIGHT_RADIUS", "Light radius", "plants compete for light with plants within this many cells", 0, 12, 1),
        ("LIGHT_LIMIT", "Light limit", "this many such plants stop a plant seeding", 1, 50, 1),
        ("PLANT_CROWDING", "Plant crowding", "a seed doesn't take where more plants grow within two cells", 0, 50, 1),
    ],
    "Climate": [
        ("ARCTIC_SHARE", "Arctic", "share of the map's height that is arctic, at the top and again at the bottom",
         0.0, 0.5, 0.05),
        ("EQUATOR_SHARE", "Equator", "share of the map's height across its middle that is the equator", 0.0, 1.0, 0.05),
        ("CLIMATE_HUNGER", "Climate hunger", "times the food needed a step or more from home (a plant, only too hot)",
         1.0, 5.0, 0.1),
        ("CLIMATE_DAMAGE", "Climate damage", "share of full health lost a turn for each step from home (plants from two)",
         0.0, 1.0, 0.01),
        ("PLANT_CLIMATE_YIELD", "Plant climate yield", "share of its food a plant gives out of its temperature",
         0.0, 1.0, 0.05),
        ("SEASON_TURNING", "Season turning", "share of each season its wave takes to wash across a hemisphere",
         0.0, 1.0, 0.05),
        ("WAVE_BAND", "Wave band", "rows in each band of a season's wave", 1, 20, 1),
        ("WAVE_FRONT", "Wave front", "bands at a wave's front, part way between the old weather and the new", 1, 30, 1),
        ("GRASS_DIEBACK", "Grass dieback", "where it isn't medium, turns for each tuft of a cell's grass to die", 1, 500, 1),
        ("SEED_VIABILITY", "Seed viability", "years a seed waits for its temperature before it dies", 0.0, 5.0, 0.05),
        ("STARTING_TOLERANCE", "Starting tolerance", "chance a random starting genome carries each tolerance flag",
         0.0, 1.0, 0.05),
        ("HARDINESS", "Hardiness", "share of exposure's extra hunger and damage an animal tolerant of it suffers",
         0.0, 1.0, 0.05),
    ],
    "Experiences": [
        (f"EXPERIENCES.{event}.{trait}", f"{event.replace('_', ' ').capitalize()}: {trait}",
         f"how {event.replace('_', ' ')} shifts {trait}, as a share of its range", -0.5, 0.5, 0.0005)
        for event, shifts in world.EXPERIENCES.items() for trait in shifts
    ],
}

# Rules whose top value means "all": history turns there keep every turn, so far as the memory allows.
ALL_AT_TOP = {"HISTORY_TURNS"}
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
    "HISTORY_TURNS": "MEMORY: each turn kept holds the whole world, about 200 KB a brain (some 20 MB a turn with 100 animals), up to the history memory limit",
    "HISTORY_MEMORY": "MEMORY: keep this well below your computer's free memory, or Bitzkey will slow to a crawl or crash",
}

RULES = tuple(Rule(key, group, label, hint, minimum, maximum, step)
              for group, rows in _TABLE.items() for key, label, hint, minimum, maximum, step in rows)
BY_KEY = {rule.key: rule for rule in RULES}
# Pairs that must stay in order, smaller first, and pairs that must differ (lifespan divides by the gap).
ORDERED = (("LITTER_RANGE.0", "LITTER_RANGE.1"), ("GESTATION.0", "GESTATION.1"), ("LIFESPAN_YEARS.0", "LIFESPAN_YEARS.1"),
           ("LIFESPAN_YEARS.1", "LIFESPAN_YEARS.2"))
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
    return {key: value for key, value in values.items() if value != DEFAULTS[key]}


def shown(rule: Rule, value: float) -> str:
    """A value as the panel shows it: up to six significant digits, no trailing zeros; a rule whose top
    means "all" says so there."""
    if rule.key in ALL_AT_TOP and value >= rule.maximum:
        return "ALL"
    return str(int(value)) if rule.integer else f"{value:.6g}"
