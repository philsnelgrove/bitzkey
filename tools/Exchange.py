"""Exporting and importing Bitzkeys as JSON, so they can move between simulations.

A Bitzkey is written as its two colour words in hex; an animal adds its current temperament
and its brain:
{"ring": "7421FF41", "dot": "637E4B01", "mind": {"aggression": 9.412, ...},
 "brain": {"senses": [...], "actions": [...], ...}}
Several Bitzkeys form a JSON array, one per line.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tools.Brain import Brain
from tools.Genome import MIND_TRAITS, TRAIT_MAX, V1_MAX, genome_from_record, genome_record, record_version

if TYPE_CHECKING:
    from tools.Simulation import Bitzkey


MIND_DIGITS = 3  # decimal places kept for exported temperament


@dataclass(frozen=True, eq=True)
class Imported:
    """A genome read from an export, with the temperament and trained brain saved with it."""
    dna: int
    brain: Brain | None = None
    mind: Mapping[str, float] | None = None  # saved traits only; any missing start at the genes


def bitzkey_record(agent: Bitzkey) -> dict:
    record: dict = genome_record(agent.dna)
    if agent.species != "plant":
        record["mind"] = {trait: round(agent.mind[trait], MIND_DIGITS) for trait in MIND_TRAITS}
    if agent.brain is not None:
        record["brain"] = agent.brain.to_record()
    return record


def export_bitzkeys(agents: Iterable[Bitzkey]) -> str:
    """One Bitzkey as a JSON object; several as a JSON array, one Bitzkey per line."""
    return format_records(bitzkey_record(agent) for agent in agents)


def format_records(records: Iterable[dict]) -> str:
    """One record as a JSON object; several as a JSON array, one record per line."""
    lines = [json.dumps(record) for record in records]
    if len(lines) == 1:
        return lines[0]
    return "[\n" + ",\n".join(f"  {line}" for line in lines) + "\n]"


def import_bitzkeys(text: str) -> list[Imported]:
    """Read every Bitzkey in the text: exported objects, arrays of them, or several pasted one
    after another. Raises ValueError, naming the problem, if any part cannot be read."""
    decoder = json.JSONDecoder()
    found: list[Imported] = []
    position = 0
    while True:
        while position < len(text) and (text[position].isspace() or text[position] == ","):
            position += 1
        if position == len(text):
            return found
        try:
            value, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as error:
            raise ValueError(f"Not valid JSON at line {error.lineno}, column {error.colno}.") from None
        found.extend(_bitzkeys_in(value))


def _bitzkeys_in(value: object) -> list[Imported]:
    if isinstance(value, list):
        return [item for element in value for item in _bitzkeys_in(element)]
    if isinstance(value, dict) and "bitzkeys" in value:
        return _bitzkeys_in(value["bitzkeys"])
    if isinstance(value, dict) and {"ring", "dot"} <= set(value):
        dna = genome_from_record(value)
        try:
            brain = None if value.get("brain") is None else Brain.from_record(value["brain"])
            mind = None if value.get("mind") is None else _mind_from_record(value["mind"], record_version(value))
        except ValueError as error:
            raise ValueError(f"Bitzkey {value['ring']}/{value['dot']}: {error}") from None
        return [Imported(dna, brain, mind)]
    raise ValueError(f"Expected a Bitzkey with \"ring\" and \"dot\", found: {json.dumps(value)[:60]}")


def _mind_from_record(value: object, version: int) -> dict[str, float]:
    """Read saved temperament: known traits only, each kept within 0 to 7. Version 1 saved them
    on a 0 to 15 scale."""
    if not isinstance(value, dict):
        raise ValueError('"mind" must map trait names to numbers.')
    scale = TRAIT_MAX / V1_MAX if version == 1 else 1.0
    mind = {}
    for trait in MIND_TRAITS:
        if trait in value:
            try:
                mind[trait] = max(0.0, min(float(TRAIT_MAX), float(value[trait]) * scale))
            except (TypeError, ValueError):
                raise ValueError(f'"mind" trait "{trait}" must be a number, not {json.dumps(value[trait])}.') from None
    return mind
