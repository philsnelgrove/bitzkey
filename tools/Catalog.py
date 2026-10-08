"""The catalog of favourite Bitzkeys, kept between runs in catalog.json beside exports/.

Each entry is an export record (its genome, its temperament and its trained brain, as
Exchange writes them), so the catalog file can itself be imported, plus a "catalog" note of how
it was when catalogued, for the inspector to show: its identifier, the turn, its hunger,
health, age, generation, children, last action and the chances it last gave its options.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from config.Globals import Config
from tools.Exchange import Imported, bitzkey_record, format_records, import_bitzkeys
from tools.Genome import decode_body
from tools.Simulation import Bitzkey


CATALOG_PATH = Config.APP_DIR / "catalog.json"
STATE_DIGITS = 3  # decimal places kept for hunger, health and chances


def entry_for(agent: Bitzkey, turn: int) -> dict:
    """A catalog entry for an animal as it is now, its brain among it."""
    record = bitzkey_record(agent)
    record["catalog"] = {
        "identifier": agent.identifier,
        "turn": turn,
        "added": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "hunger": round(agent.hunger, STATE_DIGITS),
        "health": round(agent.health, STATE_DIGITS),
        "age": agent.age,
        "generation": agent.generation,
        "children": agent.children,
        "last_action": agent.last_action,
        "odds": {action: round(chance, STATE_DIGITS) for action, chance in agent.odds.items()},
    }
    return record


def load(path: Path | None = None) -> list[dict]:
    """The catalogued entries, oldest first; none if there is no catalog yet. Raises ValueError if
    the file can't be read as a catalog."""
    path = path or CATALOG_PATH
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"The catalog {path.name} can't be read: {error}") from None
    entries = value if isinstance(value, list) else [value]
    if not all(isinstance(entry, dict) and {"ring", "dot"} <= set(entry) for entry in entries):
        raise ValueError(f"The catalog {path.name} holds something that isn't a Bitzkey.")
    return entries


def save(entries: list[dict], path: Path | None = None) -> None:
    (path or CATALOG_PATH).write_text(format_records(entries) if entries else "[]", encoding="utf-8")


def add(new: Iterable[dict], path: Path | None = None) -> int:
    """Add entries to the catalog, leaving out any already in it exactly. Returns how many were added."""
    entries = load(path)
    known = {json.dumps(entry, sort_keys=True) for entry in entries}
    added = 0
    for entry in new:
        text = json.dumps(entry, sort_keys=True)
        if text not in known:
            entries.append(entry)
            known.add(text)
            added += 1
    if added:
        save(entries, path)
    return added


def remove(index: int, path: Path | None = None) -> None:
    entries = load(path)
    del entries[index]
    save(entries, path)


def imported(entry: dict) -> Imported:
    """The Bitzkey an entry holds, with its brain and temperament, ready to import or inspect."""
    return import_bitzkeys(json.dumps(entry))[0]


def inspected(entry: dict) -> Bitzkey:
    """The Bitzkey rebuilt as it was when catalogued, for the inspector to draw."""
    bitzkey = imported(entry)
    state = entry.get("catalog", {})
    agent = Bitzkey(int(state.get("identifier", 0)), bitzkey.dna, 0, 0, brain=bitzkey.brain)
    if bitzkey.mind:
        agent.mind.update(bitzkey.mind)
    agent.health = float(state.get("health", agent.max_health))
    agent.hunger = float(state.get("hunger", 0.0))
    agent.age = int(state.get("age", 0))
    agent.generation = int(state.get("generation", 0))
    agent.children = int(state.get("children", 0))
    agent.last_action = str(state.get("last_action", "catalogued"))
    agent.odds = {str(action): float(chance) for action, chance in state.get("odds", {}).items()}
    return agent


def describe(entry: dict) -> str:
    """A line naming an entry in the catalog's list."""
    state = entry.get("catalog", {})
    species = decode_body(imported(entry).dna)["species"]
    identifier = state.get("identifier")
    name = f"#{identifier:03d} {species}" if isinstance(identifier, int) else species
    when = f"turn {state['turn']}" if "turn" in state else ""
    return "   ".join(part for part in (name, when, state.get("added", "")) if part)
