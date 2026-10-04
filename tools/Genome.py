"""The 64-bit Bitzkey genome.

The low 32 bits are the inner dot: mental, emotional and social traits.
The high 32 bits are the outer ring: the body and its physical attributes.
Each word holds eight four-bit traits, from its least-significant bits, and,
read as four CMYK bytes, sets the colour of its part of the Bitzkey.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable


SPECIES = ("plant", "grazer", "omnivore", "predator")
MIND_TRAITS = ("aggression", "libido", "sociability", "curiosity", "fear", "appetite", "patience", "intellect")
BODY_TRAITS = ("species", "size", "strength", "armour", "weapons", "speed", "sight", "fertility")
WORD = 0xFFFFFFFF
GENOME = (1 << 64) - 1
MUTATION_CHANCE = 0.2  # per word, per birth: one bit flips
HEX_WORD = re.compile(r"#?([0-9A-Fa-f]{8})")


def mind_word(dna: int) -> int:
    return dna & WORD


def body_word(dna: int) -> int:
    return (dna >> 32) & WORD


def _nibbles(word: int) -> list[int]:
    return [(word >> shift) & 0xF for shift in range(0, 32, 4)]


def decode_mind(dna: int) -> dict[str, int]:
    return dict(zip(MIND_TRAITS, _nibbles(mind_word(dna)), strict=True))


def decode_body(dna: int) -> dict[str, int | str]:
    body: dict[str, int | str] = dict(zip(BODY_TRAITS, _nibbles(body_word(dna)), strict=True))
    body["species"] = SPECIES[int(body["species"]) % len(SPECIES)]
    return body


def encode_dna(species: str = "plant", **traits: int) -> int:
    """Build a genome from named traits; unnamed traits are zero."""
    unknown = set(traits) - set(MIND_TRAITS) - set(BODY_TRAITS[1:])
    if unknown:
        raise ValueError(f"Unknown traits: {', '.join(sorted(unknown))}.")
    dna = SPECIES.index(species) << 32
    for index, trait in enumerate(MIND_TRAITS):
        dna |= (traits.get(trait, 0) & 0xF) << (4 * index)
    for index, trait in enumerate(BODY_TRAITS[1:], start=1):
        dna |= (traits.get(trait, 0) & 0xF) << (32 + 4 * index)
    return dna


def color_from_word(word: int) -> tuple[int, int, int]:
    """Convert one 32-bit word's four bytes (CMYK) to an RGB color."""
    cyan, magenta, yellow, black = ((word >> shift) & 0xFF for shift in (24, 16, 8, 0))
    return (round((255 - cyan) * (255 - black) / 255),
            round((255 - magenta) * (255 - black) / 255),
            round((255 - yellow) * (255 - black) / 255))


def make_dna(species: int | None = None, rng: random.Random | None = None) -> int:
    dna = (rng or random).getrandbits(64)
    if species is not None:
        dna = (dna & ~(0xF << 32)) | ((species % len(SPECIES)) << 32)
    return dna


def cross(first: int, second: int, rng: random.Random) -> int:
    """Make a child's genome, then give each word a chance of one flipped bit.

    The ring (body) is a single-point crossover of the parents' body words. The dot
    (mind) is the average of the parents' dot colours, CMYK byte by byte; an odd
    half rounds up or down at random so blending carries no drift.
    """
    mask = (1 << rng.randrange(1, 32)) - 1
    body = (body_word(first) & mask) | (body_word(second) & ~mask & WORD)
    mind = 0
    for shift in (0, 8, 16, 24):
        byte, odd = divmod(((first >> shift) & 0xFF) + ((second >> shift) & 0xFF), 2)
        if odd and rng.random() < 0.5:
            byte += 1
        mind |= byte << shift
    dna = (body << 32) | mind
    for offset in (0, 32):
        if rng.random() < MUTATION_CHANCE:
            dna ^= 1 << (offset + rng.randrange(32))
    return dna


# Export and import. A genome is written as its two colour words in hex:
# {"ring": "7421FF41", "dot": "637E4B01"}. Several genomes form a JSON array.

def genome_record(dna: int) -> dict[str, str]:
    return {"ring": f"{body_word(dna):08X}", "dot": f"{mind_word(dna):08X}"}


def export_genomes(dnas: Iterable[int]) -> str:
    """One genome as a JSON object; several as a JSON array, one genome per line."""
    records = [json.dumps(genome_record(dna)) for dna in dnas]
    if len(records) == 1:
        return records[0]
    return "[\n" + ",\n".join(f"  {record}" for record in records) + "\n]"


def import_genomes(text: str) -> list[int]:
    """Read every genome in the text: exported objects, arrays of them, or several pasted one
    after another. Raises ValueError, naming the problem, if any part cannot be read."""
    decoder = json.JSONDecoder()
    genomes: list[int] = []
    position = 0
    while True:
        while position < len(text) and (text[position].isspace() or text[position] == ","):
            position += 1
        if position == len(text):
            return genomes
        try:
            value, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as error:
            raise ValueError(f"Not valid JSON at line {error.lineno}, column {error.colno}.") from None
        genomes.extend(_genomes_in(value))


def _genomes_in(value: object) -> list[int]:
    if isinstance(value, list):
        return [dna for item in value for dna in _genomes_in(item)]
    if isinstance(value, dict) and "bitzkeys" in value:
        return _genomes_in(value["bitzkeys"])
    if isinstance(value, dict) and {"ring", "dot"} <= set(value):
        words = []
        for part in ("ring", "dot"):
            match = HEX_WORD.fullmatch(str(value[part]).strip())
            if match is None:
                raise ValueError(f'"{part}" must be eight hex digits, not "{value[part]}".')
            words.append(int(match.group(1), 16))
        return [(words[0] << 32) | words[1]]
    raise ValueError(f"Expected a genome with \"ring\" and \"dot\", found: {json.dumps(value)[:60]}")
