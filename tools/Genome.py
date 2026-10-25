"""The 96-bit Bitzkey genome.

The low 32 bits are the inner dot: mental and emotional traits.
The next 32 bits are the outer ring: the body and its physical attributes.
The high 32 bits are the social background, behind the species icon: how it treats others.
The dot and ring each hold ten three-bit traits (0 to 7), from their least-significant bits, and
their top two bits hold the climate genes: in the ring, the home temperature (hot, medium, cold
or very cold); in the dot, the heat tolerance flag (bit 30) and the cold tolerance flag (bit 31).
The ring's first trait is the species, in its lower two bits; its third bit sets no species, but a
mutation landing there moves the terrain along its chain. The social background holds seven
three-bit traits, then the four-bit neurodivergence gene (0 to 15), which does nothing yet, then
the two-bit terrain gene; its top five bits are spare. Like the rest, they all shade the colour:
read as four CMYK bytes, each word sets the colour of its part of the Bitzkey.

Version 1 genomes held eight four-bit traits (0 to 15) per word, and version 2 ten three-bit
traits in each of the dot and ring, with sociability, empathy and tolerance in the dot. Exported
records without a version are version 1; older versions are converted when imported. Version 3
genomes had no terrain, its bits spare and random: they are terrestrial when converted.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping


GENOME_VERSION = 4
TRAIT_BITS = 3
TRAIT_MAX = (1 << TRAIT_BITS) - 1
SPECIES = ("plant", "grazer", "omnivore", "predator")
MIND_TRAITS = ("aggression", "libido", "curiosity", "fear", "appetite", "patience", "intellect", "memory",
               "resilience", "moodiness")
SOCIAL_TRAITS = ("sociability", "empathy", "tolerance", "loyalty", "dominance", "trust", "kinship")
TEMPERAMENT = MIND_TRAITS + SOCIAL_TRAITS  # the traits experience reshapes: the dot's and the social background's
BODY_TRAITS = ("species", "size", "strength", "armour", "weapons", "speed", "sight", "fertility", "litter",
               "carrion")
WORD = 0xFFFFFFFF
GENOME = (1 << 96) - 1
SOCIAL_SHIFT = 64  # the social background: the genome's top word
NEURODIVERGENCE_SHIFT = TRAIT_BITS * len(SOCIAL_TRAITS)  # in the social background, after its traits
NEURODIVERGENCE_MAX = 15  # four bits
# Terrain: where a Bitzkey can live, in the two bits after neurodivergence. The terrestrial keep to the land and the
# aquatic to the sea; the avian and the amphibious go freely between them.
TERRAINS = ("terrestrial", "aquatic", "avian", "amphibious")
TERRAIN_SHIFT = SOCIAL_SHIFT + NEURODIVERGENCE_SHIFT + 4
# A terrain mutation moves it a step along this chain, from the sea to the air.
TERRAIN_CHAIN = ("aquatic", "amphibious", "terrestrial", "avian")
# Default chances, per birth, of a mutation; a world can set its own.
MIND_MUTATION_CHANCE = 0.25  # for each of MIND_MUTATION_ROLLS: one temperament trait steps one up or down
MIND_MUTATION_ROLLS = 3     # mind mutations a birth may carry, each with the mind mutation chance
BODY_MUTATION_CHANCE = 0.25  # one bit of the ring flips
SPECIES_REROLL = 0.5  # a body mutation landing on the species is moved to another bit, rolled afresh, this often
SPARE_BITS = 0xC0000000  # the dot's two bits that hold no mind trait: the tolerance flags
SOCIAL_SPARE = WORD & ~((1 << NEURODIVERGENCE_SHIFT) - 1)  # the social background's bits that hold no trait
TEMPERATURES = ("hot", "medium", "cold", "very cold")  # home temperatures, from hottest to coldest
TEMPERATURE_SHIFT = 62  # the home temperature: the ring's top two bits
HEAT_TOLERANCE = 1 << 30  # in the dot
COLD_TOLERANCE = 1 << 31
HEX_WORD = re.compile(r"#?([0-9A-Fa-f]{8})")

# Version 1 held the first eight traits of each word, four bits each. Converted genomes start
# the traits added in version 2 at these values.
V1_MAX = 15
V1_TRAIT_DEFAULTS = {"empathy": 0, "tolerance": 3, "litter": 0, "carrion": 0}
V1_TEMPERATURE = 1  # medium
# Version 2's dot, in order. Converted genomes start the traits added in version 3 at these values, and
# with no neurodivergence.
V2_MIND_TRAITS = ("aggression", "libido", "sociability", "curiosity", "fear", "appetite", "patience", "intellect",
                  "empathy", "tolerance")
V2_TRAIT_DEFAULTS = {"memory": 3, "resilience": 3, "moodiness": 3, "loyalty": 3, "dominance": 3, "trust": 3,
                     "kinship": 3}


def share(value: float) -> float:
    """A trait value as a share of its full range, 0 to 1."""
    return value / TRAIT_MAX


def mind_word(dna: int) -> int:
    return dna & WORD


def body_word(dna: int) -> int:
    return (dna >> 32) & WORD


def social_word(dna: int) -> int:
    return (dna >> SOCIAL_SHIFT) & WORD


def _traits(word: int, count: int, bits: int = TRAIT_BITS) -> list[int]:
    mask = (1 << bits) - 1
    return [(word >> (bits * index)) & mask for index in range(count)]


def decode_mind(dna: int) -> dict[str, int]:
    return dict(zip(MIND_TRAITS, _traits(mind_word(dna), len(MIND_TRAITS)), strict=True))


def decode_social(dna: int) -> dict[str, int]:
    return dict(zip(SOCIAL_TRAITS, _traits(social_word(dna), len(SOCIAL_TRAITS)), strict=True))


def decode_temperament(dna: int) -> dict[str, int]:
    """Its inborn temperament: the dot's traits and the social background's."""
    return {**decode_mind(dna), **decode_social(dna)}


def decode_neurodivergence(dna: int) -> int:
    return (social_word(dna) >> NEURODIVERGENCE_SHIFT) & NEURODIVERGENCE_MAX


def decode_terrain(dna: int) -> str:
    return TERRAINS[(dna >> TERRAIN_SHIFT) & 0x3]


def with_terrain(dna: int, terrain: str) -> int:
    return (dna & ~(0x3 << TERRAIN_SHIFT)) | TERRAINS.index(terrain) << TERRAIN_SHIFT


def swims(terrain: str) -> bool:
    """Whether a Bitzkey of this terrain may be in the sea."""
    return terrain != "terrestrial"


def walks(terrain: str) -> bool:
    """Whether it may be on the land."""
    return terrain != "aquatic"


def fits(terrain: str, sea: bool) -> bool:
    return swims(terrain) if sea else walks(terrain)


def decode_body(dna: int) -> dict[str, int | str]:
    body: dict[str, int | str] = dict(zip(BODY_TRAITS, _traits(body_word(dna), len(BODY_TRAITS)), strict=True))
    body["species"] = SPECIES[int(body["species"]) % len(SPECIES)]
    return body


def decode_climate(dna: int) -> tuple[int, bool, bool]:
    """The climate genes: the home temperature (an index into TEMPERATURES), and whether it
    tolerates heat or cold; both flags together cancel out, tolerating neither."""
    heat, cold = bool(dna & HEAT_TOLERANCE), bool(dna & COLD_TOLERANCE)
    return (dna >> TEMPERATURE_SHIFT) & 0x3, heat and not cold, cold and not heat


def encode_dna(species: str = "plant", temperature: int = V1_TEMPERATURE, heat_tolerance: bool = False,
               cold_tolerance: bool = False, terrain: str = "terrestrial", **traits: int) -> int:
    """Build a genome from named traits, each 0 to 7 (neurodivergence 0 to 15), climate genes and a
    terrain; unnamed traits are zero, the home temperature medium, neither tolerance, and terrestrial."""
    if terrain not in TERRAINS:
        raise ValueError(f"Unknown terrain: {terrain}.")
    unknown = set(traits) - set(TEMPERAMENT) - set(BODY_TRAITS[1:]) - {"neurodivergence"}
    if unknown:
        raise ValueError(f"Unknown traits: {', '.join(sorted(unknown))}.")
    outside = {trait: value for trait, value in traits.items()
               if not 0 <= value <= (NEURODIVERGENCE_MAX if trait == "neurodivergence" else TRAIT_MAX)}
    if outside:
        raise ValueError(f"Traits run from 0 to {TRAIT_MAX}, neurodivergence to {NEURODIVERGENCE_MAX}: {outside}.")
    if not 0 <= temperature < len(TEMPERATURES):
        raise ValueError(f"Temperatures run from 0 to {len(TEMPERATURES) - 1}: {temperature}.")
    dna = SPECIES.index(species) << 32 | temperature << TEMPERATURE_SHIFT
    dna |= (HEAT_TOLERANCE if heat_tolerance else 0) | (COLD_TOLERANCE if cold_tolerance else 0)
    for index, trait in enumerate(MIND_TRAITS):
        dna |= traits.get(trait, 0) << (TRAIT_BITS * index)
    for index, trait in enumerate(BODY_TRAITS[1:], start=1):
        dna |= traits.get(trait, 0) << (32 + TRAIT_BITS * index)
    for index, trait in enumerate(SOCIAL_TRAITS):
        dna |= traits.get(trait, 0) << (SOCIAL_SHIFT + TRAIT_BITS * index)
    dna |= traits.get("neurodivergence", 0) << (SOCIAL_SHIFT + NEURODIVERGENCE_SHIFT)
    return with_terrain(dna, terrain)


def convert_v1(dna: int) -> int:
    """Re-encode a version 1 genome in this layout: each old trait scaled from 0-15 to 0-7, the
    species kept, and the traits version 1 lacked at their defaults: a medium home temperature
    and neither tolerance among them; the traits version 2 lacked at theirs too."""
    old_mind = dict(zip(V2_MIND_TRAITS[:8], _traits(mind_word(dna), 8, 4), strict=True))
    old_body = dict(zip(BODY_TRAITS[:8], _traits(body_word(dna), 8, 4), strict=True))
    traits = {trait: round(value * TRAIT_MAX / V1_MAX) for trait, value in {**old_mind, **old_body}.items()
              if trait != "species"}
    return encode_dna(SPECIES[old_body["species"] % len(SPECIES)],
                      **{**V1_TRAIT_DEFAULTS, **V2_TRAIT_DEFAULTS, **traits})


def convert_v2(dna: int) -> int:
    """Re-encode a version 2 genome in this layout: its ring and climate genes kept, its dot's
    sociability, empathy and tolerance moved to the social background, and the traits version 2
    lacked at their defaults, with no neurodivergence."""
    old_mind = dict(zip(V2_MIND_TRAITS, _traits(mind_word(dna), len(V2_MIND_TRAITS)), strict=True))
    traits = {**V2_TRAIT_DEFAULTS, **old_mind}
    social = sum(traits[trait] << (TRAIT_BITS * index) for index, trait in enumerate(SOCIAL_TRAITS))
    mind = sum(traits[trait] << (TRAIT_BITS * index) for index, trait in enumerate(MIND_TRAITS))
    return social << SOCIAL_SHIFT | body_word(dna) << 32 | (mind_word(dna) & SPARE_BITS) | mind


def convert_v3(dna: int) -> int:
    """Re-encode a version 3 genome in this layout: terrestrial, whatever its spare bits held."""
    return with_terrain(dna, "terrestrial")


def color_from_word(word: int) -> tuple[int, int, int]:
    """Convert one 32-bit word's four bytes (CMYK) to an RGB color."""
    cyan, magenta, yellow, black = ((word >> shift) & 0xFF for shift in (24, 16, 8, 0))
    return (round((255 - cyan) * (255 - black) / 255),
            round((255 - magenta) * (255 - black) / 255),
            round((255 - yellow) * (255 - black) / 255))


def colour_difference(first_word: int, second_word: int) -> float:
    """How far apart two colour words are: the mean difference of their four CMYK bytes, from 0
    (the same colour) to 1."""
    return sum(abs(((first_word >> shift) & 0xFF) - ((second_word >> shift) & 0xFF))
               for shift in (24, 16, 8, 0)) / (4 * 255)


def ring_difference(first: int, second: int) -> float:
    """How far apart two genomes' ring colours are, from 0 to 1."""
    return colour_difference(body_word(first), body_word(second))


def dot_difference(first: int, second: int) -> float:
    """How far apart two genomes' dot colours are, from 0 to 1."""
    return colour_difference(mind_word(first), mind_word(second))


def social_difference(first: int, second: int) -> float:
    """How far apart two genomes' social background colours are, from 0 to 1."""
    return colour_difference(social_word(first), social_word(second))


def make_dna(species: int | None = None, rng: random.Random | None = None) -> int:
    """A random genome; given a species, the species trait is set to pick it (value mod 4)."""
    dna = (rng or random).getrandbits(96)
    if species is not None:
        dna = (dna & ~(0x3 << 32)) | ((species % len(SPECIES)) << 32)
    return dna


def inherit_mind(first: Mapping[str, float], second: Mapping[str, float], rng: random.Random,
                 chance: float | None = None) -> dict[str, float]:
    """A child's starting temperament: the average of its parents' temperaments. Then, up to
    MIND_MUTATION_ROLLS times, each with `chance` (MIND_MUTATION_CHANCE by default), a random
    trait mutates, stepping one up or down, or the other way where that would leave 0 to 7. Its
    social traits are part of its temperament, inherited the same way."""
    chance = MIND_MUTATION_CHANCE if chance is None else chance
    mind = {trait: (first[trait] + second[trait]) / 2 for trait in TEMPERAMENT}
    for _ in range(MIND_MUTATION_ROLLS):
        if rng.random() >= chance:
            continue
        trait = rng.choice(TEMPERAMENT)
        step = rng.choice((-1, 1))
        if not 0 <= mind[trait] + step <= TRAIT_MAX:
            step = -step
        mind[trait] = max(0.0, min(float(TRAIT_MAX), mind[trait] + step))
    return mind


def _write(traits: tuple[str, ...], spare: int, mind: Mapping[str, float], rng: random.Random) -> int:
    """A word for some traits of a temperament, so it shows the mind it holds: each trait's
    fraction rounds up with a chance equal to it, so writing carries no drift. `spare` gives the
    word's other bits."""
    word = spare
    for index, trait in enumerate(traits):
        whole, fraction = divmod(mind[trait], 1)
        value = min(TRAIT_MAX, int(whole) + (1 if fraction and rng.random() < fraction else 0))
        word |= value << (TRAIT_BITS * index)
    return word


def write_mind(spare: int, mind: Mapping[str, float], rng: random.Random) -> int:
    """A dot word for a temperament's mind traits; `spare` gives the two spare bits."""
    return _write(MIND_TRAITS, spare & SPARE_BITS, mind, rng)


def write_social(spare: int, mind: Mapping[str, float], rng: random.Random) -> int:
    """A social background word for a temperament's social traits; `spare` gives its
    neurodivergence and spare bits."""
    return _write(SOCIAL_TRAITS, spare & SOCIAL_SPARE, mind, rng)


def cross(first: int, second: int, rng: random.Random, mind: Mapping[str, float] | None = None,
          mind_chance: float | None = None, body_chance: float | None = None, terrain_mutation: bool = True) -> int:
    """Make a child's genome.

    The ring (body) is a single-point crossover of the parents' body words, with a
    `body_chance` (BODY_MUTATION_CHANCE by default) of one flipped bit. The dot (mind) and the
    social background are written from the child's starting temperament: `mind` if given, which
    a living child has from its parents' current minds, or else the average of its parents'
    inborn traits, with a `mind_chance` of a mutation. The dot's two spare bits come from either
    parent, and so, separately, do the social background's neurodivergence, terrain and spare bits.
    With `terrain_mutation`, a flip of the species' third bit moves the terrain a step along
    TERRAIN_CHAIN, toward the air if the bit was set, toward the sea if cleared (no further than its ends).
    """
    body_chance = BODY_MUTATION_CHANCE if body_chance is None else body_chance
    mask = (1 << rng.randrange(1, 32)) - 1
    body = (body_word(first) & mask) | (body_word(second) & ~mask & WORD)
    flipped = -1
    if rng.random() < body_chance:
        bit = rng.randrange(32)
        if bit < 2 and rng.random() < SPECIES_REROLL:  # the bits that set the species: often rolled afresh
            bit = rng.randrange(32)
        body ^= 1 << bit
        flipped = bit
    if mind is None:
        mind = inherit_mind(decode_temperament(first), decode_temperament(second), rng, mind_chance)
    spare = mind_word(rng.choice((first, second)))
    social_spare = social_word(rng.choice((first, second)))
    dot = write_mind(spare, mind, rng)
    child = write_social(social_spare, mind, rng) << SOCIAL_SHIFT | (body << 32) | dot
    if flipped == 2 and terrain_mutation:  # the species' third bit: the terrain moves a step along its chain
        place = TERRAIN_CHAIN.index(decode_terrain(child)) + (1 if body >> 2 & 1 else -1)
        child = with_terrain(child, TERRAIN_CHAIN[max(0, min(3, place))])
    return child


# A genome is written as its layout version and three colour words in hex:
# {"version": 4, "ring": "7421FF41", "dot": "637E4B01", "social": "0A1B2C3D"}.
# Versions 1 and 2 had no social background.

def genome_record(dna: int) -> dict:
    return {"version": GENOME_VERSION, "ring": f"{body_word(dna):08X}", "dot": f"{mind_word(dna):08X}",
            "social": f"{social_word(dna):08X}"}


def genome_from_record(record: dict) -> int:
    """Read the genome from a record's "ring", "dot" and (from version 3) "social" hex words,
    converting an older genome. Raises ValueError."""
    version = record_version(record)
    words = []
    for part in ("ring", "dot", "social") if version >= 3 else ("ring", "dot"):
        if part not in record:
            raise ValueError(f'A version {version} genome needs "{part}".')
        match = HEX_WORD.fullmatch(str(record[part]).strip())
        if match is None:
            raise ValueError(f'"{part}" must be eight hex digits, not "{record[part]}".')
        words.append(int(match.group(1), 16))
    dna = (words[0] << 32) | words[1]
    if version == 1:
        return convert_v1(dna)
    if version == 2:
        return convert_v2(dna)
    whole = words[2] << SOCIAL_SHIFT | dna
    return convert_v3(whole) if version == 3 else whole


def record_version(record: dict) -> int:
    """The genome layout a record was written in: 1 when it says nothing. Raises ValueError for
    a version this program doesn't know."""
    version = record.get("version", 1)
    if version not in (1, 2, 3, GENOME_VERSION):
        raise ValueError(f"Genome version {version} is not one this Bitzkey knows (1 to {GENOME_VERSION}).")
    return version
