"""The 64-bit Bitzkey genome.

The low 32 bits are the inner dot: mental, emotional and social traits.
The high 32 bits are the outer ring: the body and its physical attributes.
Each word holds ten three-bit traits (0 to 7), from its least-significant bits. Its top two
bits hold the climate genes: in the ring, the home temperature (hot, medium, cold or very
cold); in the dot, the heat tolerance flag (bit 30) and the cold tolerance flag (bit 31).
Like the rest, they shade the colour: read as four CMYK bytes, each word sets the colour of
its part of the Bitzkey.

Version 1 genomes held eight four-bit traits (0 to 15) per word. Exported records without
a version are version 1, and are converted when imported.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping


GENOME_VERSION = 2
TRAIT_BITS = 3
TRAIT_MAX = (1 << TRAIT_BITS) - 1
SPECIES = ("plant", "grazer", "omnivore", "predator")
MIND_TRAITS = ("aggression", "libido", "sociability", "curiosity", "fear", "appetite", "patience", "intellect",
               "empathy", "tolerance")
BODY_TRAITS = ("species", "size", "strength", "armour", "weapons", "speed", "sight", "fertility", "litter",
               "carrion")
WORD = 0xFFFFFFFF
GENOME = (1 << 64) - 1
# Default chances, per birth, of a mutation; a world can set its own.
MIND_MUTATION_CHANCE = 0.5  # for each of MIND_MUTATION_ROLLS: one mind trait steps one up or down
MIND_MUTATION_ROLLS = 3     # mind mutations a birth may carry, each with the mind mutation chance
BODY_MUTATION_CHANCE = 0.5  # one bit of the ring flips
SPARE_BITS = 0xC0000000  # the dot's two bits that hold no mind trait: the tolerance flags
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


def share(value: float) -> float:
    """A trait value as a share of its full range, 0 to 1."""
    return value / TRAIT_MAX


def mind_word(dna: int) -> int:
    return dna & WORD


def body_word(dna: int) -> int:
    return (dna >> 32) & WORD


def _traits(word: int, count: int, bits: int = TRAIT_BITS) -> list[int]:
    mask = (1 << bits) - 1
    return [(word >> (bits * index)) & mask for index in range(count)]


def decode_mind(dna: int) -> dict[str, int]:
    return dict(zip(MIND_TRAITS, _traits(mind_word(dna), len(MIND_TRAITS)), strict=True))


def decode_body(dna: int) -> dict[str, int | str]:
    body: dict[str, int | str] = dict(zip(BODY_TRAITS, _traits(body_word(dna), len(BODY_TRAITS)), strict=True))
    body["species"] = SPECIES[int(body["species"]) % len(SPECIES)]
    return body


def decode_climate(dna: int) -> tuple[int, bool, bool]:
    """The climate genes: the home temperature (an index into TEMPERATURES), and whether it
    tolerates heat and cold."""
    return (dna >> TEMPERATURE_SHIFT) & 0x3, bool(dna & HEAT_TOLERANCE), bool(dna & COLD_TOLERANCE)


def encode_dna(species: str = "plant", temperature: int = V1_TEMPERATURE, heat_tolerance: bool = False,
               cold_tolerance: bool = False, **traits: int) -> int:
    """Build a genome from named traits, each 0 to 7, and climate genes; unnamed traits are
    zero, the home temperature medium, and neither tolerance."""
    unknown = set(traits) - set(MIND_TRAITS) - set(BODY_TRAITS[1:])
    if unknown:
        raise ValueError(f"Unknown traits: {', '.join(sorted(unknown))}.")
    outside = {trait: value for trait, value in traits.items() if not 0 <= value <= TRAIT_MAX}
    if outside:
        raise ValueError(f"Traits run from 0 to {TRAIT_MAX}: {outside}.")
    if not 0 <= temperature < len(TEMPERATURES):
        raise ValueError(f"Temperatures run from 0 to {len(TEMPERATURES) - 1}: {temperature}.")
    dna = SPECIES.index(species) << 32 | temperature << TEMPERATURE_SHIFT
    dna |= (HEAT_TOLERANCE if heat_tolerance else 0) | (COLD_TOLERANCE if cold_tolerance else 0)
    for index, trait in enumerate(MIND_TRAITS):
        dna |= traits.get(trait, 0) << (TRAIT_BITS * index)
    for index, trait in enumerate(BODY_TRAITS[1:], start=1):
        dna |= traits.get(trait, 0) << (32 + TRAIT_BITS * index)
    return dna


def convert_v1(dna: int) -> int:
    """Re-encode a version 1 genome in this layout: each old trait scaled from 0-15 to 0-7, the
    species kept, and the traits version 1 lacked at their defaults: a medium home temperature
    and neither tolerance among them."""
    old_mind = dict(zip(MIND_TRAITS[:8], _traits(mind_word(dna), 8, 4), strict=True))
    old_body = dict(zip(BODY_TRAITS[:8], _traits(body_word(dna), 8, 4), strict=True))
    traits = {trait: round(value * TRAIT_MAX / V1_MAX) for trait, value in {**old_mind, **old_body}.items()
              if trait != "species"}
    return encode_dna(SPECIES[old_body["species"] % len(SPECIES)], **{**V1_TRAIT_DEFAULTS, **traits})


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


def make_dna(species: int | None = None, rng: random.Random | None = None) -> int:
    """A random genome; given a species, the species trait is set to pick it (value mod 4)."""
    dna = (rng or random).getrandbits(64)
    if species is not None:
        dna = (dna & ~(0x3 << 32)) | ((species % len(SPECIES)) << 32)
    return dna


def inherit_mind(first: Mapping[str, float], second: Mapping[str, float], rng: random.Random,
                 chance: float | None = None) -> dict[str, float]:
    """A child's starting temperament: the average of its parents' temperaments. Then, up to
    MIND_MUTATION_ROLLS times, each with `chance` (MIND_MUTATION_CHANCE by default), a random
    trait mutates, stepping one up or down, or the other way where that would leave 0 to 7."""
    chance = MIND_MUTATION_CHANCE if chance is None else chance
    mind = {trait: (first[trait] + second[trait]) / 2 for trait in MIND_TRAITS}
    for _ in range(MIND_MUTATION_ROLLS):
        if rng.random() >= chance:
            continue
        trait = rng.choice(MIND_TRAITS)
        step = rng.choice((-1, 1))
        if not 0 <= mind[trait] + step <= TRAIT_MAX:
            step = -step
        mind[trait] = max(0.0, min(float(TRAIT_MAX), mind[trait] + step))
    return mind


def write_mind(spare: int, mind: Mapping[str, float], rng: random.Random) -> int:
    """A dot word for a temperament, so the dot shows the mind it holds: each trait's fraction
    rounds up with a chance equal to it, so writing carries no drift. `spare` gives the two
    spare bits."""
    word = spare & SPARE_BITS
    for index, trait in enumerate(MIND_TRAITS):
        whole, fraction = divmod(mind[trait], 1)
        value = min(TRAIT_MAX, int(whole) + (1 if fraction and rng.random() < fraction else 0))
        word |= value << (TRAIT_BITS * index)
    return word


def cross(first: int, second: int, rng: random.Random, mind: Mapping[str, float] | None = None,
          mind_chance: float | None = None, body_chance: float | None = None) -> int:
    """Make a child's genome.

    The ring (body) is a single-point crossover of the parents' body words, with a
    `body_chance` (BODY_MUTATION_CHANCE by default) of one flipped bit. The dot (mind) is
    written from the child's starting temperament: `mind` if given, which a living child has
    from its parents' current minds, or else the average of its parents' inborn traits, with a
    `mind_chance` of a mutation. Its two spare bits come from either parent.
    """
    body_chance = BODY_MUTATION_CHANCE if body_chance is None else body_chance
    mask = (1 << rng.randrange(1, 32)) - 1
    body = (body_word(first) & mask) | (body_word(second) & ~mask & WORD)
    if rng.random() < body_chance:
        body ^= 1 << rng.randrange(32)
    if mind is None:
        mind = inherit_mind(decode_mind(first), decode_mind(second), rng, mind_chance)
    spare = mind_word(rng.choice((first, second)))
    return (body << 32) | write_mind(spare, mind, rng)


# A genome is written as its layout version and two colour words in hex:
# {"version": 2, "ring": "7421FF41", "dot": "637E4B01"}.

def genome_record(dna: int) -> dict:
    return {"version": GENOME_VERSION, "ring": f"{body_word(dna):08X}", "dot": f"{mind_word(dna):08X}"}


def genome_from_record(record: dict) -> int:
    """Read the genome from a record's "ring" and "dot" hex words, converting a version 1
    genome. Raises ValueError."""
    words = []
    for part in ("ring", "dot"):
        match = HEX_WORD.fullmatch(str(record[part]).strip())
        if match is None:
            raise ValueError(f'"{part}" must be eight hex digits, not "{record[part]}".')
        words.append(int(match.group(1), 16))
    dna = (words[0] << 32) | words[1]
    version = record_version(record)
    return convert_v1(dna) if version == 1 else dna


def record_version(record: dict) -> int:
    """The genome layout a record was written in: 1 when it says nothing. Raises ValueError for
    a version this program doesn't know."""
    version = record.get("version", 1)
    if version not in (1, GENOME_VERSION):
        raise ValueError(f"Genome version {version} is not one this Bitzkey knows (1 or {GENOME_VERSION}).")
    return version
