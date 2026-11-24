"""Per-animal neural network: what a Bitzkey senses, what it can do, and how it learns."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

import numpy as np

from tools.Genome import (BODY_TRAITS, SPECIES, TEMPERAMENT, TEMPERATURES, TRAIT_MAX, decode_body, decode_climate, decode_temperament,
                          decode_nocturnal, decode_terrain, share, swims, walks)

ACTIONS = ("eat", "forage", "browse", "uproot", "scavenge", "feast", "rest", "court", "breed", "hunt", "attack", "defend",
           "flee", "herd", "go_north", "go_south", "go_east", "go_west", "walk_north", "walk_south", "walk_east",
           "walk_west", "revisit", "feed_young", "teach", "take_meat", "take_fruit", "drop_meat", "drop_fruit", "eat_pile", "eat_carried",
           "take_stone", "take_bone", "take_wood", "drop_stone", "drop_bone", "drop_wood", "dress", "make_bone_axe", "make_bone_armour", "make_club", "make_wood_armour", "make_stone_axe", "make_bow", "make_wagon", "take_gear", "drop_weapon", "drop_armour", "cut", "dig", "light_fire", "feed_fire", "cook", "construct", "eat_seed", "plant_seed", "build_road", "eat_marrow", "fertilize", "skin", "demand", "steal", "give", "offer", "shoot", "practise", "tutor", "hitch", "unhitch", "store_food", "craft_arrows", "address", "learn")
# Alongside its action, every animal that thinks keeps silent or calls: the alert, to its herd, of a hunter (grazers)
# or a fresh kill (hunters) it sees, or one of five calls that mean only what its kind come to make of them, heard by
# all close enough. A brain values them as it values its actions, after them, and learns them the same way.
CALLS = ("silent", "alert", "call_1", "call_2", "call_3", "call_4", "call_5")
SILENT, ALERT_CALL, FIRST_CALL = 0, 1, 2  # the five calls from FIRST_CALL on
CALL_COUNT = 5
# Memory: a brain carries MEMORY_NEURONS values from one choice to the next, each read back the next time as a sense
# ("memory_1" on). Each turn, beside valuing its actions and calls, it gives each a gate, z = sigmoid of its output, and a
# value to write, c = tanh of its output, and the neuron becomes (1 - z) times what it held plus z times c: kept as it
# was (z near 0), overwritten (z near 1), or cleared (writing 0). Its gates read the top hidden layer, which reads what
# the memory held with everything else it senses. They learn back in time (Brain.learn_memory): how a lesson's error
# moves with what the memory held is carried back to the gates that wrote it, as many choices back as Memory steps. As
# the C++ version's.
MEMORY_NEURONS = 16  # 8 until 3.206 / 1.148
MEMORY_GATES = len(ACTIONS) + len(CALLS)  # the first gate's output, then each neuron's in turn,
MEMORY_VALUES = MEMORY_GATES + MEMORY_NEURONS  # then the first value to write, and each neuron's
MAX_MEMORY_STEPS = 4  # the furthest back Memory steps may reach
# Foresight: last of all its outputs, what a brain expects of its next turn, each by how much it will change, about -1 to
# 1. As the C++ version's.
FORESEEN = ("foresee_hunger", "foresee_health", "foresee_stamina", "foresee_food")
FORESIGHT = MEMORY_VALUES + MEMORY_NEURONS  # the first of them
# A call may run on: after a first sound (one of the five calls, not the alert), a second, and after that a third, each
# one of the same five sounds. Each place after the first is valued apart: stopping there, or each sound. As the C++
# version's.
MAX_SOUNDS = 3
SOUND_CHOICES = 1 + CALL_COUNT  # stopping, then the five sounds
MORE_SOUNDS = FORESIGHT + len(FORESEEN)  # the second place's stopping and sounds, then the third's
LATER_SOUNDS = tuple(f"{place}_{choice}" for place in ("second", "third")
                     for choice in ("stop", *(f"call_{call}" for call in range(1, CALL_COUNT + 1))))
# Foresight of the outside: last of all its outputs, what a brain expects to sense next turn of each thing about it it
# senses from afar (the nearest hunter, prey, meat and so on, so anything may come to be named), each as that sense's
# level, 0..1, taught by what it then senses (see OUTSIDE_FORESIGHT). New things go at the end. As the C++ version's.
OUTSIDE_THINGS = ("threat", "prey", "carrion", "kill", "grass", "plant", "mate", "herd", "parent", "known", "elder", "foe",
                  "standout", "meat_pile", "fruit_pile", "stone_pile", "bone_pile", "wood_pile", "gear_pile", "fire", "comfort")
OUTSIDE_FORESEEN = tuple(f"foresee_{thing}" for thing in OUTSIDE_THINGS)
NEED_NAMES = ("hunger", "fatigue", "pain", "fear", "loneliness", "discomfort")  # see GOAL_JOY
OUTSIDE = MORE_SOUNDS + len(LATER_SOUNDS)  # the first of them
# Wishes: last of all its outputs, what a caller expects to gain were one who heard its call then to come nearer, go
# further off, flee, rest, eat, trade or fight; learnt from how its own turn went (see WISH_LEARNING and INTENT). As the
# C++ version's.
WISHES = ("wish_near", "wish_away", "wish_flee", "wish_rest", "wish_eat", "wish_trade", "wish_fight")
WISH_NEAR, WISH_AWAY, WISH_FLEE, WISH_REST, WISH_EAT, WISH_TRADE, WISH_FIGHT = range(len(WISHES))
WISH = OUTSIDE + len(OUTSIDE_FORESEEN)  # the first of them
# Pointing: last of all its outputs, how a caller values pointing, as it calls, at each outside thing it foresees, the
# same 21 in the same order; taught by how much less surprised by that thing its hearers then were (see POINTING). As
# the C++ version's.
POINTS = tuple(f"point_{thing}" for thing in OUTSIDE_THINGS)
POINT = WISH + len(WISHES)  # the first of them
# Attention: last of all its outputs, how it values attending to the loudest caller it hears (or, hearing none, the
# nearest it sees pointing), and not; only one that attended sees that one's finger next turn (see ATTENTION). As the
# C++ version's.
ATTENDS = ("attend", "ignore")
ATTEND_YES, ATTEND_NO = range(len(ATTENDS))
ATTEND = POINT + len(POINTS)  # the first of them
# Pointing at one animal (names): last of all its outputs, how it values pointing at one animal in its sight rather
# than a kind of thing; that point numbered POINT_ONE_THING, after the 21 things. As the C++ version's.
POINT_ONE = ATTEND + len(ATTENDS)
POINT_ONE_THING = len(POINTS)
# And pointing at itself (deed-words): what it is doing as it calls. As the C++ version's.
POINT_SELF = POINT_ONE + 1
POINT_SELF_THING = POINT_ONE_THING + 1
# A second word (see CALL_WORDS): places four to six, each stopping or one of the five sounds. As the C++ version's.
SECOND_WORD = POINT_SELF + 1
SECOND_WORD_OUTPUTS = tuple(f"{place}_{choice}" for place in ("fourth", "fifth", "sixth")
                            for choice in ("stop", "call_1", "call_2", "call_3", "call_4", "call_5"))


def word2_output(place: int, sound: int) -> int:
    """The output valuing, at a second word's `place` (0..2), stopping (sound -1) or a sound 0..4."""
    return SECOND_WORD + place * SOUND_CHOICES + 1 + sound




def point_output(point: int) -> int:
    """The output valuing a point: a thing's, or pointing at one animal."""
    return POINT_ONE if point == POINT_ONE_THING else POINT_SELF if point == POINT_SELF_THING else POINT + point


# A brain's outputs by name: its actions, then silence and the calls, then its memory's gates and values to write, then
# what it foresees, then a call's later sounds, then what it foresees of the outside, then its wishes of its hearers.
OUTPUTS = (ACTIONS + CALLS + tuple(f"memory_gate_{neuron}" for neuron in range(1, MEMORY_NEURONS + 1))
           + tuple(f"memory_value_{neuron}" for neuron in range(1, MEMORY_NEURONS + 1)) + FORESEEN + LATER_SOUNDS
           + OUTSIDE_FORESEEN + WISHES + POINTS + ATTENDS + ("point_one", "point_self") + SECOND_WORD_OUTPUTS)


def sound_output(place: int, sound: int) -> int:
    """The output valuing, at `place` (1 the second sound, 2 the third), stopping (sound -1) or a sound 0..4."""
    return MORE_SOUNDS + (place - 1) * SOUND_CHOICES + 1 + sound


def later_sound(output: int) -> bool:
    """Whether an output values a second or third sound (not stopping): a new brain holds back from them as from calling."""
    return ((MORE_SOUNDS <= output < OUTSIDE and (output - MORE_SOUNDS) % SOUND_CHOICES != 0)
            or (SECOND_WORD <= output < len(OUTPUTS) and (output - SECOND_WORD) % SOUND_CHOICES != 0))  # and a second word's


def call_output(call: int) -> int:
    """The output valuing a call (or silence)."""
    return len(ACTIONS) + call


VOICE_SENSES = ("voice_ring_likeness", "voice_dot_likeness", "voice_social_likeness", "voice_ring_red", "voice_ring_green",
                "voice_ring_blue", "voice_dot_red", "voice_dot_green", "voice_dot_blue", "voice_social_red",
                "voice_social_green", "voice_social_blue", "voice_pitch", "voice_harshness", "voice_urgency")
SENSES = (
    "hunger",          # 0..1
    "health",          # 0..1
    "grass_here",       # 1 when grass lies on the animal's own cell
    "grass_seen",       # 1 when grass is visible elsewhere
    "grass_closeness",  # 1 next to the nearest visible grass, falling to 0 at the edge of sight
    "mate_near",       # 1 when a compatible, ready mate is adjacent
    "mate_closeness",  # 1 when a willing mate is adjacent, falling to 0 beyond sight; 0 when none is seen
    "ready_to_breed",  # 1 when the animal's breeding cooldown has passed
    "prey_near",       # adjacent prey, 0..4 scaled to 0..1
    "prey_frailty",    # share of the frailest adjacent prey's health one blow would take
    "prey_closeness",  # 1 when prey is adjacent, falling to 0 beyond sight; 0 when none is seen
    "prey_pregnant",   # 1 when the nearest prey in sight carries young
    "threat_near",     # adjacent hunters, 0..4 scaled to 0..1
    "threat_danger",   # share of the animal's own health the worst adjacent hunter's blow would take
    "threat_closeness",  # 1 when a hunter is adjacent, falling to 0 beyond sight; 0 when none is seen
    "kin_near",        # same-species animals within two cells, 0..3 scaled to 0..1
    "others_near",     # animals of other species within two cells, 0..3 scaled to 0..1
    "carrion_closeness",  # 1 when meat lies alongside, falling to 0 beyond sight; 0 when none is seen (not grazers)
    "kill_closeness",     # 1 when a fresh kill it eats lies alongside, falling to 0 beyond sight (hunters only)
    "plant_closeness",    # 1 when a plant stands alongside, falling to 0 beyond sight (grazers and omnivores)
    "assailant_near",     # 1 when the animal that just struck it is still alongside
    "counter_frailty",    # share of that assailant's health one counterblow would take
    "rivalry",            # predators and omnivores: how overcrowded its territory is with rivals, 0..1
    "herd_near",          # herd-mates within two cells, 0..3 scaled to 0..1
    "herd_closeness",     # 1 at the middle of its herd in range, falling to 0 at the range's edge; 0 alone
    "alarm",              # 1 when a herd-mate close by was struck, or alerted it to a hunter, since it last acted
    "kill_alert",         # 1 when a pack-mate alerted it to a fresh kill, still there, since it last acted
    "herd_success",       # 1 when it has just seen a herd-mate succeed (a kill), fading turn by turn
    "herd_failure",       # 1 when it has just seen a herd-mate fail (die), fading turn by turn
    "climate_discomfort",  # how far the temperature here is from home, as it feels it: 0.5 a step, to 1 two or more
    "comfort_closeness",  # 1 where it feels at home or next to such a row, falling to 0 beyond sight; 0 when none is seen
    "age",             # 0 at birth, 1 at 400 turns or older
    # The time of year, as two hands of a clock a quarter of a year apart, so the year's turn is smooth:
    "time_of_year_a",  # 1 at the start of the year, falling to 0 halfway through, and rising back to 1
    "time_of_year_b",  # the same a quarter of a year later: 0 a quarter in, 1 three quarters in
    "latitude",        # -1 at the top (north) edge of the map, 0 on the equator, 1 at the bottom (south) edge
    # How much warmer than home it is, in steps, 1 two or more too hot, -1 two or more too cold, part
    # way while the weather turns, as the zones' colours show:
    "warmth",          # here
    "warmth_north",    # at the edge of its sight to the north, or the map's edge if nearer
    "warmth_south",    # and to the south
    "food_remembered",    # 1 just after it fed somewhere, fading as its memory lets the place go; 0 when forgotten
    "danger_remembered",  # 1 at the place it was last struck, while it remembers it, falling to 0 beyond sight
    "home_food",          # how well fed its home band is: plants per grazer, prey per predator, 1 at twice what breeding needs; 0 for omnivores
    # Where the nearest row it would feel at home in lies, however far beyond sight:
    "home_direction",     # -1 to the north, 1 to the south, 0 at home or with no such row
    "home_distance",      # 0 at home, rising to 1 half the map's height away or with no such row
    "young_near",         # its own young, still infants, within its sight, 0..3 scaled to 0..1
    "sea_here",           # 1 when it stands in the sea
    # A shore it can't cross, sea for a land animal or land for a sea animal: 1 when the next square that way is beyond
    # it, falling to 0 at the edge of its sight, and 0 with none in sight (and always, for the avian and amphibious):
    "shore_north", "shore_south", "shore_east", "shore_west",
    "stamina",            # 1 fresh, falling to 0 exhausted, when it can only rest
    "night",              # 1 when it is night where it stands
    # An infant's nearest living parent within its sight: 1 beside it, falling to 0 at the edge of its sight, and 0 with
    # none in sight or grown; and which way it lies, -1 to 1, west to east and north to south:
    "parent_closeness", "parent_east", "parent_south",
    # The calls it heard since it last chose, each of the five as the loudest who made it made it: how loud, 1 from
    # beside it, falling to 0 at the call's reach (the larger the caller, the further it carries); which way it lay, -1
    # to 1, west to east and north to south; and whether it was of its own species, 1 or 0. All 0 for one not heard.
    *(f"heard_call_{call}{part}" for call in range(1, 6) for part in ("", "_east", "_south", "_kin")),
    *(f"memory_{neuron}" for neuron in range(1, MEMORY_NEURONS + 1)),  # what its memory neurons held, -1..1 (see Memory)
    # Its mental map (see Simulation's MapPlace): of the places it remembers, for this season, how good the best for food
    # was, 0..1, and which way it lies, -1 to 1, west to east and north to south, the nearest first among equals; the
    # same of the most dangerous; and of where it was most comfortable in the season to come.
    *(f"map_{kind}{part}" for kind in ("food", "danger", "comfort") for part in ("", "_east", "_south")),
    # Those it knows (see Simulation's Acquaintance): of the nearest it knows within its sight, how close, 1 beside it
    # falling to 0 at the edge of its sight, how well it regards it, -1 to 1, and which way it lies; and whether the
    # loudest caller it heard is one it knows, 1 or 0, and how well it regards it.
    "known_closeness", "known_regard", "known_east", "known_south", "caller_known", "caller_regard", "known_danger", "known_skill",
    "elder_age", "elder_closeness", "elder_east", "elder_south", "student_near",
    "load", "meat_carried", "fruit_carried", "meat_pile_closeness", "fruit_pile_closeness",
    "stone_carried", "bone_carried", "wood_carried", "stone_pile_closeness", "bone_pile_closeness", "wood_pile_closeness", "gear_pile_closeness", "fire_closeness", "fire_friendly", "wielding", "armoured", "foe_sheltered", "shelter_here", "arable_here", "seed_carried", "boredom", "road_here", "fertilized_here", "hide_worn", "regarded", "envy", "fighting", "foe_closeness", "cooking_smell", "food_plenty", "prey_flanked", "own_land", "foreign_land", "own_land_east", "own_land_south", "trespasser", "war_standing", "marked", "mark_seen", "standout_esteem", "standout_closeness", "standout_east", "standout_south", "leader", "standout_leads", "standout_heard", "rank", "standout_rank", "land_owner_regard", "bow_carried", "archery",
) + TEMPERAMENT + (    # then its current temperament, mind and social traits, each 0..7 scaled to 0..1
    *(f"inborn_{trait}" for trait in TEMPERAMENT),        # its dot's and social background's traits, as born, 0..1
    *BODY_TRAITS,                                         # its ring's traits, 0..1 (species 0 plant to 1 predator)
    "home_temperature", "heat_tolerance", "cold_tolerance",  # its climate genes: home 0 hot to 1 very cold, flags 0 or 1
    "swims", "walks", "flies",  # its terrain: whether it may go into the sea and onto the land, 0 or 1, and whether it flies
    "nocturnal",  # its activity gene: 1 nocturnal, 0 diurnal
    # The call of the loudest it heard, sound by sound: for each place, first to third, 1 for the sound made there.
    *(f"heard_{place}_{call}" for place in ("first", "second", "third") for call in range(1, CALL_COUNT + 1)),
    # The tag every call carries, read by all who hear it, of the same loudest caller (all 0 when it heard none): how
    # alike each of the caller's three colours is to its own, each as red, green and blue, and its pitch, harshness and
    # urgency, 0 to 1. As the C++ version's.
    *VOICE_SENSES,
    # The loudest caller's pointing finger (see POINTING): 1 when it pointed as it called, and how close the thing it
    # pointed at is to the hearer, as the hearer senses that kind of thing; where to look, not what it is.
    "pointed", "pointed_closeness",
    # Whether it attends to someone this turn, 1 or 0, how close the nearest it sees pointing is (see ATTENTION),
    # whether one in its sight attends to it (see WATCHED_REPEAT), and whether the call it heard was said to it alone,
    # from beside it (see the action address), 1 or 0.
    "attending", "sees_pointing", "watched", "addressed",
    # Its goal (see GOAL_JOY): 1 while it holds one; its urgency; how close what it seeks is; for an animal, its regard
    # and 1 for family; 1 for a desire; its way, east and south, to a remembered place; how deep in a chain of steps (see OPENS_RATE); the need it serves; then what it seeks: an outside thing, an animal, or a deed. As the C++ version's.
    "goal", "goal_urgency", "goal_closeness", "goal_one_regard", "goal_one_family", "goal_desire", "goal_east", "goal_south", "goal_depth",
    *(f"goal_need_{need}" for need in NEED_NAMES), *(f"goal_{thing}" for thing in OUTSIDE_THINGS), "goal_one",
    *(f"goal_deed_{action}" for action in ACTIONS),
    # A word it holds in its lexicon, heard (see LEXICON_SIZE): 1 when the loudest call it heard is one, and its own
    # closeness to the thing that word means. As the C++ version's.
    "meant", "meant_closeness",
    # A name it holds, heard (see NAMES_SIZE): 1 when heard; how close that one is; how it regards it, -1 to 1; and 1 when
    # the name is its own. As the C++ version's.
    "named", "named_closeness", "named_regard", "named_me",
    # A deed-word it holds, heard (see DEEDS_SIZE): 1 for the deed it means, one for each action. As the C++ version's.
    *(f"meant_{action}" for action in ACTIONS),
    # The loudest call's second word, sound by sound (see CALL_WORDS). As the C++ version's.
    *(f"heard_{place}_{sound}" for place in ("fourth", "fifth", "sixth") for sound in range(1, 6)),
    # Imagining: the action it pictures itself taking, a -1 or 1 for each bit of its number, or all 0 when it pictures
    # nothing. Only its foresight is taught with them set. As the C++ version's.
    "imagined_1", "imagined_2", "imagined_3", "imagined_4", "imagined_5", "imagined_6", "imagined_7",
)
INPUT_SIZE = len(SENSES)
MEMORY_SENSE = SENSES.index("memory_1")  # the first sense of what the memory held
IMAGINED_SENSE = SENSES.index("imagined_1")
UTTERANCE_SENSE = SENSES.index("heard_first_1")
POINTED_SENSE = SENSES.index("pointed")  # then "pointed_closeness"
GOAL_SENSE = SENSES.index("goal")  # then urgency, closeness, regard, family, the needs, the things, one, the deeds
GOAL_NEED_SENSE = SENSES.index("goal_need_hunger")
GOAL_DESIRE_SENSE = SENSES.index("goal_desire")  # 1 when its goal is a desire (see DESIRE), not a need's
GOAL_THING_SENSE = SENSES.index("goal_threat")
GOAL_ONE_SENSE = SENSES.index("goal_one")
GOAL_DEED_SENSE = SENSES.index("goal_deed_eat")
MEANT_SENSE = SENSES.index("meant")  # then "meant_closeness"
NAMED_SENSE = SENSES.index("named")  # then "named_closeness", "named_regard", "named_me"
MEANT_DEED_SENSE = SENSES.index("meant_eat")  # then one for each action, in order
UTTERANCE2_SENSE = SENSES.index("heard_fourth_1")


def utterance2_sense(place: int, sound: int) -> int:
    """The sense of hearing `sound` (0..4) at a second word's `place` (0..2) of the loudest call heard."""
    return UTTERANCE2_SENSE + CALL_COUNT * place + sound


OUTSIDE_SENSES = tuple(SENSES.index(f"{thing}_closeness") for thing in OUTSIDE_THINGS)


def utterance_sense(place: int, sound: int) -> int:
    """The sense of hearing `sound` (0..4) at `place` (0 first) of the loudest call heard."""
    return UTTERANCE_SENSE + CALL_COUNT * place + sound
IMAGINED_BITS = 7  # enough to number every action (65 since archery)
# The senses as a brain reads them: each that runs 0..1 centred, read as -1..1, so that a lesson moves a neuron's weights
# both ways (senses never below 0 push all its weights the same way, lesson after lesson, until the neuron jams at its
# limit, the same whatever it senses, and learns no more); those already running -1..1 read as they are. As the C++
# version's read_sense.
SIGNED_SENSES = frozenset(("goal_east", "goal_south", "latitude", "warmth", "warmth_north", "warmth_south", "home_direction", "parent_east", "parent_south",
                           *(f"heard_call_{call}_{way}" for call in range(1, 6) for way in ("east", "south")),
                           *(f"memory_{neuron}" for neuron in range(1, MEMORY_NEURONS + 1)),
                           *(f"map_{kind}_{way}" for kind in ("food", "danger", "comfort") for way in ("east", "south")),
                           "known_regard", "known_east", "known_south", "caller_regard", "elder_east", "elder_south",
                           "own_land_east", "own_land_south", "war_standing", "standout_east", "standout_south", "land_owner_regard", "named_regard", "goal_one_regard",
                           *(f"imagined_{bit}" for bit in range(1, IMAGINED_BITS + 1))))
# Senses that are mostly nothing, an event or a thing near, carried or worn, read as 0..2: off, they push nothing and
# teach nothing, where read as -1 each was a steady push that, over dozens of them, jammed neurons; on, they count as
# the centred senses do. Those off for 70% of the animals of every species, 135,000 turns into a world (2026-11-10).
SPARSE_SENSES = frozenset((
    "prey_pregnant", "threat_near", "threat_danger", "carrion_closeness", "kill_closeness", "assailant_near",
    "counter_frailty", "alarm", "kill_alert", "young_near", "shore_north", "shore_south", "shore_east", "shore_west",
    "parent_closeness", "heard_call_5", "heard_call_5_kin", "map_comfort", "caller_known", "known_skill",
    "student_near", "meat_carried", "fruit_carried", "meat_pile_closeness", "stone_carried", "wood_carried",
    "gear_pile_closeness", "fire_closeness", "fire_friendly", "wielding", "foe_sheltered", "shelter_here",
    "seed_carried", "road_here", "hide_worn", "envy", "fighting", "cooking_smell", "leader", "standout_leads", "standout_heard", "rank", "standout_rank",
    "bow_carried", "archery",
    *(f"heard_{place}_{call}" for place in ("first", "second", "third") for call in range(1, CALL_COUNT + 1)),
    "pointed", "pointed_closeness", "attending", "sees_pointing", "watched", "addressed",
    "goal", "goal_urgency", "goal_closeness", "goal_one_family", "goal_desire", *(f"goal_need_{need}" for need in NEED_NAMES),
    *(f"goal_{thing}" for thing in OUTSIDE_THINGS), "goal_one", *(f"goal_deed_{action}" for action in ACTIONS), "meant", "meant_closeness", "named", "named_closeness", "named_me", *(f"meant_{action}" for action in ACTIONS),
    *(f"heard_{place}_{sound}" for place in ("fourth", "fifth", "sixth") for sound in range(1, 6))))
_READ_SCALE = np.array([1.0 if name in SIGNED_SENSES else 2.0 for name in SENSES])
_READ_SHIFT = np.array([0.0 if name in SIGNED_SENSES or name in SPARSE_SENSES else -1.0 for name in SENSES])


# A brain holds its values in hundredths: what an action is worth over the turns it looks ahead runs to ten and more,
# and its neurons, each between -1 and 1, would jam at their limits to make that much, the same whatever it senses.
# Held small, the output layer makes them of neurons free to follow the senses. It gives and learns them whole. As the
# C++ version's.
VALUE_SCALE = 0.01


def read_senses(inputs: np.ndarray) -> np.ndarray:
    """The senses (or rows of them) as a brain reads them."""
    return inputs * _READ_SCALE + _READ_SHIFT
MAX_DEPTH = 4      # hidden layers: one for each point of inborn intellect, at least one and up to this many
BASE_WIDTH = 38    # neurons in each hidden layer, up to an intellect of MAX_DEPTH
WIDTH_STEP = 5     # neurons more in each layer for each point of intellect beyond MAX_DEPTH
MAX_WIDTH = BASE_WIDTH + WIDTH_STEP * (TRAIT_MAX - MAX_DEPTH)
FLAT_SPOT = 0.1  # added to each hidden neuron's slope when it learns, so one pinned at either end of tanh still learns a little
LESSON_LIMIT = 2.0  # the most one lesson's error counts for, so a rare but large surprise teaches, but no lesson runs wild
TEMPERATURE = 0.5  # loose enough that an action valued a little below the best is still tried
MUTATION = 0.0
TRACE_DECAY = 0.7  # lambda: how much of each earlier choice's credit carries on to the next, beside the discount
SMART_TRACE_DECAY = 0.95  # lambda at the highest intellect: the clever trace credit further back; the least, TRACE_DECAY
# A new brain values setting off in a direction this much below its other actions, so an untaught
# brain doesn't march straight away from its range, its herd and its food before it has learned where to go.
HEADING_RELUCTANCE = 0.5
CALL_RELUCTANCE = 0.5  # and values each call this much below silence, so an untaught world isn't all calling at once
MEMORY_PERSISTENCE = 1.4  # and its memory gates this far below their middle, a fifth open, so what it holds fades slowly


def fresh_bias(output: int) -> float:
    """What a new brain's bias holds an output at, whole: below stopping, a call's later sounds; a memory gate, a fifth
    open; else nothing. An output a brain from an older version never had starts so. As the C++ version's."""
    if later_sound(output):
        return -CALL_RELUCTANCE
    if MEMORY_GATES <= output < MEMORY_VALUES:
        return -MEMORY_PERSISTENCE
    return 0.0
HEADINGS = ("go_north", "go_south", "go_east", "go_west", "walk_north", "walk_south", "walk_east", "walk_west")
RECORD_DIGITS = 6  # decimal places kept when a brain is exported


def output_spread(width: int) -> float:
    """A new brain's output weights are drawn within this of zero: smaller the wider its layers, so its
    untaught values spread as they did when layers were twelve neurons wide."""
    return 0.3 * math.sqrt(12 / width)


def depth_for(intellect: int) -> int:
    """How many hidden layers a brain has, by its owner's inborn intellect: one a point, from one to MAX_DEPTH."""
    return max(1, min(MAX_DEPTH, intellect))


def width_for(intellect: int) -> int:
    """How many neurons each of its hidden layers has: BASE_WIDTH, and WIDTH_STEP more for each point of
    intellect beyond MAX_DEPTH."""
    return BASE_WIDTH + WIDTH_STEP * max(0, intellect - MAX_DEPTH)


def imagined_call(call: int) -> int:
    """A call (or silence) pictured is numbered after the actions, so foresight may learn what follows a call too."""
    return len(ACTIONS) + call


def imagining(senses: Sequence[float], action: int) -> list[float]:
    """The same senses, picturing itself taking `action`."""
    return [*senses[:IMAGINED_SENSE], *(1.0 if action >> bit & 1 else -1.0 for bit in range(IMAGINED_BITS))]


def make_senses(**values: float) -> list[float]:
    """Build an input vector by sense name; unnamed senses are zero."""
    unknown = set(values) - set(SENSES)
    if unknown:
        raise ValueError(f"Unknown senses: {', '.join(sorted(unknown))}.")
    return [float(values.get(name, 0.0)) for name in SENSES]


def gene_senses(dna: int) -> dict[str, float]:
    """The inputs for a genome's traits, ring, dot and social background, its climate genes and its terrain."""
    inborn, body = decode_temperament(dna), decode_body(dna)
    temperature, heat, cold = decode_climate(dna)
    values = {f"inborn_{trait}": share(inborn[trait]) for trait in TEMPERAMENT}
    values["species"] = SPECIES.index(str(body["species"])) / (len(SPECIES) - 1)
    values.update({trait: share(int(body[trait])) for trait in BODY_TRAITS[1:]})
    values.update(home_temperature=temperature / (len(TEMPERATURES) - 1), heat_tolerance=float(heat),
                  cold_tolerance=float(cold))
    terrain = decode_terrain(dna)
    values.update(swims=float(swims(terrain)), walks=float(walks(terrain)), flies=float(terrain == "avian"),
                  nocturnal=float(decode_nocturnal(dna)))
    return values


def memory_after(values: Sequence[float], held: Sequence[float]) -> list[float]:
    """What memory neurons hold after a choice valued so, holding `held` before it."""
    now = []
    for neuron in range(MEMORY_NEURONS):
        gate = 1 / (1 + math.exp(-values[MEMORY_GATES + neuron]))
        now.append((1 - gate) * held[neuron] + gate * math.tanh(values[MEMORY_VALUES + neuron]))
    return now


def choice_odds(values: Sequence[float], options: Sequence[int], temperature: float | None = None) -> dict[int, float]:
    """Softmax over the available actions' values, at the choice temperature unless another is given."""
    temperature = TEMPERATURE if temperature is None else temperature
    best = max(values[option] for option in options)
    weights = {option: math.exp((values[option] - best) / temperature) for option in options}
    total = sum(weights.values())
    return {option: weight / total for option, weight in weights.items()}


def _tanh(values: np.ndarray) -> np.ndarray:
    """tanh as the C library works it out, which the C++ version uses too; numpy's own differs in the
    last bit now and then."""
    return np.fromiter(map(math.tanh, values), float, len(values))


def _identity_layer(width: int) -> tuple[np.ndarray, np.ndarray]:
    """A hidden layer that passes its input on, near enough: added to deepen a brain without
    changing what it has learned."""
    return np.eye(width), np.zeros(width)


def _fitted(array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """A copy of `array` cut down or padded with zeros to `shape`: neurons, and weights from neurons, a
    layer no longer has are dropped, and those it gains are idle until mutation moves them."""
    fitted = np.zeros(shape)
    common = tuple(slice(0, min(have, want)) for have, want in zip(array.shape, shape))
    fitted[common] = array[common]
    return fitted


_MASK = (1 << 64) - 1


def _ziggurat() -> tuple[list[int], list[float], list[float]]:
    """Marsaglia and Tsang's ziggurat for the normal distribution, 128 layers, as the C++ version builds it."""
    k, w, f = [0] * 128, [0.0] * 128, [0.0] * 128
    m1, vn = 2147483648.0, 9.91256303526217e-3
    dn = tn = 3.442619855899
    q = vn / math.exp(-0.5 * dn * dn)
    k[0] = int((dn / q) * m1)
    k[1] = 0
    w[0], w[127] = q / m1, dn / m1
    f[0], f[127] = 1.0, math.exp(-0.5 * dn * dn)
    for i in range(126, 0, -1):
        dn = math.sqrt(-2.0 * math.log(vn / dn + math.exp(-0.5 * dn * dn)))
        k[i + 1] = int((dn / tn) * m1)
        tn = dn
        f[i] = math.exp(-0.5 * dn * dn)
        w[i] = dn / m1
    return k, w, f


_ZIGGURAT = _ziggurat()


class _Stream:
    """A fast stream of random numbers of a child's own: xoshiro256+, seeded by splitmix64, with normal draws by the
    ziggurat. It draws exactly what the C++ version's does from the same seed."""

    def __init__(self, seed: int) -> None:
        self.state = []
        for _ in range(4):
            seed = (seed + 0x9E3779B97F4A7C15) & _MASK
            z = seed
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
            self.state.append(z ^ (z >> 31))

    def next(self) -> int:
        s = self.state
        result = (s[0] + s[3]) & _MASK
        shifted = (s[1] << 17) & _MASK
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= shifted
        s[3] = ((s[3] << 45) & _MASK) | (s[3] >> 19)
        return result

    def random(self) -> float:
        """In [0, 1), from the top 53 bits."""
        return (self.next() >> 11) * (1.0 / 9007199254740992.0)

    def normal(self) -> float:
        """Of mean 0 and spread 1."""
        k, w, f = _ZIGGURAT
        tail = 3.442620
        while True:
            hz = self.next() >> 32
            if hz >= 1 << 31:
                hz -= 1 << 32
            iz = hz & 127
            x = hz * w[iz]
            if abs(hz) < k[iz]:
                return x
            if iz == 0:  # beyond the base: the tail, by Marsaglia's method
                while True:
                    tx = -math.log(1.0 - self.random()) / tail
                    ty = -math.log(1.0 - self.random())
                    if not ty + ty < tx * tx:
                        break
                return tail + tx if hz > 0 else -tail - tx
            if f[iz] + self.random() * (f[iz - 1] - f[iz]) < math.exp(-0.5 * x * x):
                return x


class Brain:
    """A tanh network of hidden layers estimating the long-run value of each action.

    Its sums are worked out as products added up by numpy's sum, in its fixed pairwise order, and
    its tanh by the C library, so that the C++ version, which follows the same order, matches it to
    the last bit; numpy's matrix product adds up in an order that varies with the machine.

    It learns online from temporal-difference targets and is inherited neuron by neuron, so a
    child keeps whole learned features from one parent or the other. Its weights are NumPy
    arrays: each hidden layer's weights are one row per neuron, the output weights one row per
    action. The first hidden layer reads the senses; each later one reads the layer before.
    Learning from its own choices, it keeps an eligibility trace for every weight (TD(lambda)), so
    what comes of a choice reaches back to the choices that led to it; the traces belong to one
    life and are neither inherited nor exported.
    """

    def __init__(self, layers: Sequence[tuple[Sequence, Sequence]], output_weights: Sequence, output_biases: Sequence):
        self.layers = [(np.array(weights, dtype=float), np.array(biases, dtype=float)) for weights, biases in layers]
        self.output_weights = np.array(output_weights, dtype=float)
        self.output_biases = np.array(output_biases, dtype=float)
        # Each hidden layer's (weights, biases) traces, then the output layer's; None until it first learns.
        self.traces: list[tuple[np.ndarray, np.ndarray]] | None = None

    @property
    def depth(self) -> int:
        return len(self.layers)

    @property
    def width(self) -> int:
        """Neurons in each hidden layer."""
        return len(self.layers[0][1])

    @classmethod
    def random(cls, rng: random.Random, depth: int = 1, width: int = BASE_WIDTH) -> Brain:
        layers = []
        for index in range(depth):
            reads = INPUT_SIZE if index == 0 else width
            bound = math.sqrt(6 / (reads + width))
            layers.append(([[rng.uniform(-bound, bound) for _ in range(reads)] for _ in range(width)], [0.0] * width))
        spread = output_spread(width)
        brain = cls(layers, [[rng.uniform(-spread, spread) for _ in range(width)] for _ in OUTPUTS],
                    [rng.uniform(-0.2, 0.2) - (HEADING_RELUCTANCE if output in HEADINGS else 0.0) for output in OUTPUTS])
        brain.output_biases[call_output(ALERT_CALL):MEMORY_GATES] -= CALL_RELUCTANCE
        for output in range(MORE_SOUNDS, len(OUTPUTS)):
            if later_sound(output):
                brain.output_biases[output] -= CALL_RELUCTANCE  # and as much below stopping, each sound more
        brain.output_biases[MEMORY_GATES:MEMORY_VALUES] -= MEMORY_PERSISTENCE
        brain.scale_values()  # valuing as it would whole
        return brain

    @classmethod
    def inherit(cls, first: Brain, second: Brain, rng: random.Random, depth: int = 1, width: int = BASE_WIDTH,
                mutation: float | None = None) -> Brain:
        """A child's brain, `depth` layers of `width` neurons. Where both parents have a layer, each of
        its neurons, with its incoming weights, comes from one parent or the other, or from the one that
        has it where only one does; where only one has the layer, from that one; deeper still, it starts
        passing its input on. A parent's layers are cut down or padded to the child's width (see
        _fitted). Each action's output comes from one parent or the other. Then it mutates (by MUTATION
        unless given)."""
        mutation = MUTATION if mutation is None else mutation
        # Thousands of draws a brain, so they come from a stream of the child's own, seeded from the world's: one
        # draw of the world's a child, as in the C++ version, which makes the same child from the same seed.
        stream = _Stream(rng.getrandbits(64))
        layers = []
        for index in range(depth):
            shape = (width, INPUT_SIZE if index == 0 else width)
            donors = [parent for parent in (first, second) if index < parent.depth]
            if len(donors) == 2:
                from_first = np.array([stream.random() < 0.5 for _ in range(width)])
                neuron = np.arange(width)
                from_first = (from_first & (neuron < first.width)) | (neuron >= second.width)
                layers.append((np.where(from_first[:, None], _fitted(first.layers[index][0], shape),
                                        _fitted(second.layers[index][0], shape)),
                               np.where(from_first, _fitted(first.layers[index][1], (width,)),
                                        _fitted(second.layers[index][1], (width,)))))
            elif donors:
                layers.append((_fitted(donors[0].layers[index][0], shape), _fitted(donors[0].layers[index][1], (width,))))
            else:
                layers.append(_identity_layer(width))
        from_first = np.array([stream.random() < 0.5 for _ in OUTPUTS])
        outputs = (len(OUTPUTS), width)
        child = cls(layers, np.where(from_first[:, None], _fitted(first.output_weights, outputs),
                                     _fitted(second.output_weights, outputs)),
                    np.where(from_first, first.output_biases, second.output_biases))
        if mutation:
            # The output layer's as much as in whole values, held in hundredths.
            for weights, size in ([(weights, mutation) for weights, _ in child.layers] + [(child.output_weights, mutation * VALUE_SCALE)]
                                  + [(biases, mutation) for _, biases in child.layers] + [(child.output_biases, mutation * VALUE_SCALE)]):
                weights += size * np.array([stream.normal() for _ in range(weights.size)]).reshape(weights.shape)
        return child

    def centre_inputs(self) -> None:
        """Made to read the senses centred (read_senses), from a brain that read them as they were, choosing just as
        before: each weight on a sense x read as 2x - 1 halved, and the half it gave at 0 given by the neuron's bias; one
        read as 2x (a sparse sense), halved alone. What it was learning from lets go."""
        weights, biases = self.layers[0]
        weights[:, _READ_SCALE == 2.0] /= 2
        centred = _READ_SHIFT == -1.0
        for neuron in range(weights.shape[0]):
            for sense in np.flatnonzero(centred):
                biases[neuron] += weights[neuron, sense]
        self.traces = None

    def scale_values(self) -> None:
        """Made to hold its values in hundredths (VALUE_SCALE), from a brain that held them whole, valuing just as
        before. What it was learning from lets go."""
        self.output_weights *= VALUE_SCALE
        self.output_biases *= VALUE_SCALE
        self.traces = None

    def resized(self, depth: int, width: int) -> Brain:
        """A copy `depth` layers of `width` neurons: deeper with layers that pass their input on, or
        shallower without its last layers, and each layer cut down or padded (see _fitted)."""
        layers = [(_fitted(weights, (width, INPUT_SIZE if index == 0 else width)), _fitted(biases, (width,)))
                  for index, (weights, biases) in enumerate(self.layers[:depth])]
        layers += [_identity_layer(width) for _ in range(depth - len(layers))]
        return Brain(layers, _fitted(self.output_weights, (len(OUTPUTS), width)), self.output_biases)

    def _activations(self, inputs: Sequence[float]) -> list[np.ndarray]:
        """The senses, then each hidden layer's output."""
        activation = np.asarray(inputs, dtype=float)
        if activation.shape != (INPUT_SIZE,):
            raise ValueError(f"Expected {INPUT_SIZE} inputs, got {activation.size}.")
        activation = read_senses(activation)
        activations = [activation]
        for weights, biases in self.layers:
            activation = _tanh((weights * activation).sum(axis=1) + biases)
            activations.append(activation)
        return activations

    def values(self, inputs: Sequence[float]) -> list[float]:
        """Return the estimated value of every output, its actions' then its calls', in this situation."""
        return (((self.output_weights * self._activations(inputs)[-1]).sum(axis=1) + self.output_biases) / VALUE_SCALE).tolist()

    def learn(self, inputs: Sequence[float], action: int, target: float, learning_rate: float) -> None:
        """Move one action's value toward a target, with the error clipped for stability, back
        through every layer."""
        if learning_rate <= 0:
            return
        activations = self._activations(inputs)
        top = activations[-1]
        value = float((self.output_weights[action] * top).sum() + self.output_biases[action]) / VALUE_SCALE
        error = max(-LESSON_LIMIT, min(LESSON_LIMIT, value - target))
        gradient = error * self.output_weights[action]
        output_step = learning_rate * error * VALUE_SCALE  # the output layer moves the whole value as it would
        self.output_weights[action] -= output_step * top
        for index in range(self.depth - 1, -1, -1):
            weights, biases = self.layers[index]
            delta = gradient * (1 - activations[index + 1] * activations[index + 1] + FLAT_SPOT)
            if index:
                gradient = (weights * delta[:, None]).sum(axis=0)
            biases -= learning_rate * delta
            weights -= learning_rate * np.outer(delta, activations[index])
        self.output_biases[action] -= output_step

    def learn_traced(self, inputs: Sequence[float], action: int, target: float, learning_rate: float,
                     discount: float, sharpness: float = 0.0, decay: float | None = None) -> None:
        """Learn from its own choice by TD(lambda): every weight's eligibility fades by the discount
        times lambda and gains how this action's value moves with it; then every weight moves by the
        clipped error times its eligibility, so the choices before this one share in it. Lambda runs
        from TRACE_DECAY up to SMART_TRACE_DECAY with `sharpness`, its owner's intellect, 0 to 1; or `decay` gives the
        discount times lambda whole."""
        if learning_rate <= 0:
            return
        if decay is None:  # unless given whole, the discount times lambda
            decay = discount * (TRACE_DECAY + (SMART_TRACE_DECAY - TRACE_DECAY) * sharpness)
        if self.traces is None:
            self.traces = [(np.zeros_like(weights), np.zeros_like(biases)) for weights, biases in self.layers]
            self.traces.append((np.zeros_like(self.output_weights), np.zeros_like(self.output_biases)))
        *hidden, (output_weights, output_biases) = self.traces
        activations = self._activations(inputs)
        top = activations[-1]
        value = float((self.output_weights[action] * top).sum() + self.output_biases[action]) / VALUE_SCALE
        error = max(-LESSON_LIMIT, min(LESSON_LIMIT, value - target))
        output_weights *= decay
        output_weights[action] += VALUE_SCALE * top  # gathering in hundredths, so it moves the whole value as it would
        output_biases *= decay
        output_biases[action] += VALUE_SCALE
        gradient = self.output_weights[action]
        for index in range(self.depth - 1, -1, -1):
            weights, _ = self.layers[index]
            delta = gradient * (1 - activations[index + 1] * activations[index + 1] + FLAT_SPOT)
            if index:
                gradient = (weights * delta[:, None]).sum(axis=0)
            weight_trace, bias_trace = hidden[index]
            weight_trace *= decay
            weight_trace += np.outer(delta, activations[index])
            bias_trace *= decay
            bias_trace += delta
        step = learning_rate * error
        for (weights, biases), (weight_trace, bias_trace) in zip(self.layers, hidden):
            weights -= step * weight_trace
            biases -= step * bias_trace
        self.output_weights -= step * output_weights
        self.output_biases -= step * output_biases

    def learn_memory(self, inputs: Sequence[float], past: Sequence[Sequence[float]], action: int, target: float,
                     learning_rate: float) -> None:
        """Learn back in time, after the lesson on `inputs` (its senses at its choice): how that action's value, with its
        error, moves with what its memory held then is carried back to the gates that wrote it, at `past` (its senses at
        the choices before, most recent first, each the memory was written at), each moving its gates and values to write,
        and the layers beneath them, to hold what would have valued the choice after rightly; and so on back."""
        if learning_rate <= 0 or not past:
            return
        held = slice(MEMORY_SENSE, MEMORY_SENSE + MEMORY_NEURONS)
        made = slice(MEMORY_GATES, MEMORY_VALUES + MEMORY_NEURONS)
        # How the action's value, with its error, moves with what each memory neuron held: the error carried down
        # through every layer as a lesson's, nothing moving, in whole values.
        activations = self._activations(inputs)
        value = float((self.output_weights[action] * activations[-1]).sum() + self.output_biases[action]) / VALUE_SCALE
        gradient = max(-LESSON_LIMIT, min(LESSON_LIMIT, value - target)) * self.output_weights[action]
        carried = np.zeros(MEMORY_NEURONS)
        for index in range(self.depth - 1, -1, -1):
            weights, _ = self.layers[index]
            delta = gradient * (1 - activations[index + 1] * activations[index + 1] + FLAT_SPOT)
            if index:
                gradient = (weights * delta[:, None]).sum(axis=0)
            else:
                carried = (weights[:, held] * delta[:, None]).sum(axis=0) / VALUE_SCALE
        for step, before in enumerate(past):
            # The choice before: each gate's and value's error, as an output's in a lesson, from what the neuron came to
            # hold; and what of the error reaches what it held before, through what it kept of it.
            activations = self._activations(before)
            top = activations[-1]
            valued = ((self.output_weights[made] * top).sum(axis=1) + self.output_biases[made]) / VALUE_SCALE
            gate = np.array([1 / (1 + math.exp(-value)) for value in valued[:MEMORY_NEURONS]])
            write = _tanh(valued[MEMORY_NEURONS:])
            was = np.asarray(before, dtype=float)[held]
            errors = np.clip(np.concatenate((carried * (write - was) * gate * (1 - gate), carried * gate * (1 - write * write))),
                             -LESSON_LIMIT, LESSON_LIMIT)
            kept = carried * (1 - gate)
            gradient = (errors[:, None] * self.output_weights[made]).sum(axis=0)
            self.output_weights[made] -= (learning_rate * errors * VALUE_SCALE)[:, None] * top
            reached = np.zeros(MEMORY_NEURONS)
            for index in range(self.depth - 1, -1, -1):
                weights, biases = self.layers[index]
                delta = gradient * (1 - activations[index + 1] * activations[index + 1] + FLAT_SPOT)
                if index:
                    gradient = (weights * delta[:, None]).sum(axis=0)
                elif step + 1 < len(past):
                    reached = (weights[:, held] * delta[:, None]).sum(axis=0) / VALUE_SCALE
                biases -= learning_rate * delta
                weights -= learning_rate * np.outer(delta, activations[index])
            self.output_biases[made] -= learning_rate * errors * VALUE_SCALE
            carried = kept + reached

    @property
    def nbytes(self) -> int:
        """The memory its weights and traces take."""
        arrays = [array for layer in self.layers for array in layer] + [self.output_weights, self.output_biases]
        if self.traces is not None:
            arrays += [array for layer in self.traces for array in layer]
        return sum(array.nbytes for array in arrays)

    def copy(self) -> Brain:
        brain = Brain(self.layers, self.output_weights, self.output_biases)
        if self.traces is not None:
            brain.traces = [(weights.copy(), biases.copy()) for weights, biases in self.traces]
        return brain

    def to_record(self) -> dict:
        """The brain's weights for export, labelled with the senses and actions they belong to."""
        def rounded(values: np.ndarray) -> list[float]:
            return [round(value, RECORD_DIGITS) for value in values.tolist()]
        return {
            "senses": list(SENSES),
            "centred_senses": True,
            "actions": list(OUTPUTS),  # its outputs: actions, then calls
            "layers": [{"weights": [rounded(row) for row in weights], "biases": rounded(biases)}
                       for weights, biases in self.layers],
            # The output layer exported whole, as it values (it is held in hundredths), lest rounding lose it.
            "output_weights": [rounded(weights / VALUE_SCALE) for weights in self.output_weights],
            "output_biases": rounded(self.output_biases / VALUE_SCALE),
        }

    @classmethod
    def from_record(cls, record: Mapping) -> Brain:
        """Rebuild an exported brain. Weights are matched to senses and actions by name, so a
        brain saved before a sense or action was added still loads; what it never had starts
        neutral (zero) and what this version no longer has is dropped. A brain saved with a single
        hidden layer, before brains grew deeper, loads as one layer deep. It keeps the width it was
        saved with, and the Bitzkey given it resizes it to its own. Raises ValueError."""
        try:
            senses = [str(name) for name in record["senses"]]
            actions = [str(name) for name in record["actions"]]
            if "layers" in record:
                layers = [([[float(weight) for weight in row] for row in layer["weights"]],
                           [float(bias) for bias in layer["biases"]]) for layer in record["layers"]]
            else:
                layers = [([[float(weight) for weight in row] for row in record["hidden_weights"]],
                           [float(bias) for bias in record["hidden_biases"]])]
            output_weights = [[float(weight) for weight in row] for row in record["output_weights"]]
            output_biases = [float(bias) for bias in record["output_biases"]]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"The brain is incomplete or not numeric ({error}).") from None
        if not layers:
            raise ValueError("The brain has no hidden layers.")
        width = len(layers[0][0])
        for index, (weights, biases) in enumerate(layers):
            if len(weights) != width or len(biases) != width or not 0 < width <= MAX_WIDTH:
                raise ValueError(f"The brain has {len(weights)} neurons in a layer; this version uses up to {MAX_WIDTH}, "
                                 "the same in every layer.")
            reads = len(senses) if index == 0 else width
            if any(len(row) != reads for row in weights):
                raise ValueError("The brain's weights don't match its listed senses and actions.")
        if (len(output_weights) != len(actions) or len(output_biases) != len(actions)
                or any(len(row) != width for row in output_weights)):
            raise ValueError("The brain's weights don't match its listed senses and actions.")

        sense_at = {name: index for index, name in enumerate(senses)}
        action_at = {name: index for index, name in enumerate(actions)}
        matched = [([[row[sense_at[name]] if name in sense_at else 0.0 for name in SENSES] for row in weights]
                    if index == 0 else weights, biases) for index, (weights, biases) in enumerate(layers)]
        brain = cls(
            matched,
            [output_weights[action_at[name]] if name in action_at else [0.0] * width for name in OUTPUTS],
            [output_biases[action_at[name]] if name in action_at else fresh_bias(at) for at, name in enumerate(OUTPUTS)],
        )
        if not record.get("centred_senses"):  # exported before brains read their senses centred
            brain.centre_inputs()
        brain.scale_values()  # its output layer exported whole, held in hundredths
        return brain

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Brain) and self.depth == other.depth
                and all(np.array_equal(mine[0], theirs[0]) and np.array_equal(mine[1], theirs[1])
                        for mine, theirs in zip(self.layers, other.layers))
                and np.array_equal(self.output_weights, other.output_weights)
                and np.array_equal(self.output_biases, other.output_biases))


class BrainBatch:
    """Brains of one shape worked as one: their weights stacked, so a turn's valuing and learning are a few large
    array operations rather than many small ones, each slow for Python however little it computes. Each brain's
    lessons still come one after another, in its own order; a batch learns every brain's first lesson at once, then
    every brain's second, and so on. Write the weights back to the brains before anything else reads them."""

    def __init__(self, brains: Sequence[Brain]):
        self.brains = list(brains)
        depth = brains[0].depth
        self.weights = [np.stack([brain.layers[index][0] for brain in brains]) for index in range(depth)]
        self.biases = [np.stack([brain.layers[index][1] for brain in brains]) for index in range(depth)]
        self.output_weights = np.stack([brain.output_weights for brain in brains])
        self.output_biases = np.stack([brain.output_biases for brain in brains])
        self.rows = np.arange(len(brains))
        self.traces: list[np.ndarray] | None = None  # stacked as the weights, then the output layer's, once any learns traced
        self.traced = np.array([brain.traces is not None for brain in brains])

    def _activations(self, inputs: np.ndarray) -> list[np.ndarray]:
        activations = [read_senses(inputs)]
        for weights, biases in zip(self.weights, self.biases):
            activations.append(np.tanh(np.matmul(weights, activations[-1][:, :, None])[:, :, 0] + biases))
        return activations

    def values(self, inputs: np.ndarray) -> np.ndarray:
        """Every brain's value of every action, a row a brain, for its row of senses."""
        top = self._activations(inputs)[-1]
        return (np.matmul(self.output_weights, top[:, :, None])[:, :, 0] + self.output_biases) / VALUE_SCALE

    def _backward(self, activations: list[np.ndarray], gradient: np.ndarray):
        """Each hidden layer's delta, from the top down, for a gradient at the top, worked out from the weights as
        they stand: (layer, delta) pairs."""
        for index in range(len(self.weights) - 1, -1, -1):
            delta = gradient * (1 - activations[index + 1] * activations[index + 1] + FLAT_SPOT)
            if index:
                gradient = np.matmul(delta[:, None, :], self.weights[index])[:, 0, :]
            yield index, delta

    def learn(self, inputs: np.ndarray, actions: np.ndarray, targets: np.ndarray, rates: np.ndarray) -> None:
        """Brain.learn for every brain at once; a brain with no lesson this round has a rate of 0."""
        activations = self._activations(inputs)
        top = activations[-1]
        chosen = self.output_weights[self.rows, actions]
        errors = np.clip((np.einsum("bi,bi->b", chosen, top) + self.output_biases[self.rows, actions]) / VALUE_SCALE - targets,
                         -LESSON_LIMIT, LESSON_LIMIT)
        gradient = errors[:, None] * chosen
        output_steps = rates * errors * VALUE_SCALE  # the output layer moves the whole value as it would
        self.output_weights[self.rows, actions] -= output_steps[:, None] * top
        for index, delta in self._backward(activations, gradient):
            self.biases[index] -= rates[:, None] * delta
            self.weights[index] -= rates[:, None, None] * delta[:, :, None] * activations[index][:, None, :]
        self.output_biases[self.rows, actions] -= output_steps

    def learn_traced(self, inputs: np.ndarray, actions: np.ndarray, targets: np.ndarray, rates: np.ndarray,
                     decays: np.ndarray, taking: np.ndarray) -> None:
        """Brain.learn_traced for the brains `taking` a lesson (by the decay its owner's discount and intellect give);
        the rest, traces and all, are left as they were."""
        if self.traces is None:
            self.traces = [np.stack([brain.traces[index][0] if brain.traces is not None else np.zeros_like(brain.layers[index][0])
                                     for brain in self.brains]) for index in range(len(self.weights))]
            self.traces += [np.stack([brain.traces[index][1] if brain.traces is not None else np.zeros_like(brain.layers[index][1])
                                      for brain in self.brains]) for index in range(len(self.weights))]
            depth = len(self.weights)
            self.traces += [np.stack([brain.traces[depth][0] if brain.traces is not None else np.zeros_like(brain.output_weights)
                                      for brain in self.brains]),
                            np.stack([brain.traces[depth][1] if brain.traces is not None else np.zeros_like(brain.output_biases)
                                      for brain in self.brains])]
        depth = len(self.weights)
        weight_traces, bias_traces = self.traces[:depth], self.traces[depth:2 * depth]
        output_weights, output_biases = self.traces[2 * depth], self.traces[2 * depth + 1]
        decay = np.where(taking, decays, 1.0)
        adding = taking.astype(float)
        activations = self._activations(inputs)
        top = activations[-1]
        chosen = self.output_weights[self.rows, actions]
        errors = np.clip((np.einsum("bi,bi->b", chosen, top) + self.output_biases[self.rows, actions]) / VALUE_SCALE - targets,
                         -LESSON_LIMIT, LESSON_LIMIT)
        output_weights *= decay[:, None, None]
        output_weights[self.rows, actions] += (adding * VALUE_SCALE)[:, None] * top  # gathering in hundredths
        output_biases *= decay[:, None]
        output_biases[self.rows, actions] += adding * VALUE_SCALE
        for index, delta in self._backward(activations, chosen.copy()):
            weight_traces[index] *= decay[:, None, None]
            weight_traces[index] += adding[:, None, None] * delta[:, :, None] * activations[index][:, None, :]
            bias_traces[index] *= decay[:, None]
            bias_traces[index] += adding[:, None] * delta
        steps = np.where(taking, rates * errors, 0.0)
        for index in range(depth):
            self.weights[index] -= steps[:, None, None] * weight_traces[index]
            self.biases[index] -= steps[:, None] * bias_traces[index]
        self.output_weights -= steps[:, None, None] * output_weights
        self.output_biases -= steps[:, None] * output_biases
        self.traced |= taking

    def write_back(self) -> None:
        """Every brain's weights, and traces, as the batch has left them."""
        depth = len(self.weights)
        for row, brain in enumerate(self.brains):
            for index in range(depth):
                brain.layers[index][0][...] = self.weights[index][row]
                brain.layers[index][1][...] = self.biases[index][row]
            brain.output_weights[...] = self.output_weights[row]
            brain.output_biases[...] = self.output_biases[row]
            if self.traces is not None and self.traced[row]:
                brain.traces = ([(self.traces[index][row].copy(), self.traces[depth + index][row].copy()) for index in range(depth)]
                                + [(self.traces[2 * depth][row].copy(), self.traces[2 * depth + 1][row].copy())])
