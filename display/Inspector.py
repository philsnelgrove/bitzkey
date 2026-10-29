"""A Bitzkey's details, as the sidebar shows the selected one and the colour graph's tooltip shows
the one under the mouse: its look, genome, temperament (mind and social), body, condition and latest
choice."""

from __future__ import annotations

from collections.abc import Callable

import pygame as pg

from display.Icons import draw_icon
from display.Look import DOT_RADIUS_RANGE, RING_WIDTH, dot_radius, draw_bitzkey, genome_colour
from display.Language import as_shown, tr
from display.Ui import UI, wrapped_lines
from tools.Genome import (BODY_TRAITS, MIND_TRAITS, NEURODIVERGENCE_MAX, NEURODIVERGENCE_SHIFT, SOCIAL_TRAITS, SPECIES, TEMPERATURES,
                          TERRAINS, TRAIT_BITS, TRAIT_MAX, body_word, decode_neurodivergence, mind_word, social_word)
from tools.Simulation import Bitzkey

# Heights, in design pixels, of the details: in full, and for a Bitzkey known only by its genome.
DETAILS_HEIGHT = 382
GENOME_HEIGHT = 334
# Each temperature's colour, from hot to very cold, for the map's zones and the details.
TEMPERATURE_COLOURS = ((238, 160, 110), (175, 192, 172), (140, 185, 230), (200, 225, 250))
SHORT_TEMPERATURES = ("hot", "medium", "cold", "v. cold")


def hardiness(heat: bool, cold: bool) -> str:
    """What its tolerance flags spare it, in a word or two."""
    return "HEAT-HARDY" if heat else "COLD-HARDY" if cold else ""


def _draw_climate(screen: pg.Surface, agent: Bitzkey, x: int, y: int, width: int, here: float | None) -> None:
    """Its home temperature, the temperature where it is (coloured by how far that is from home, as
    it feels it), and what it tolerates; temperatures are shortened if the line is too narrow."""
    small = UI.font("segoeui", 12)
    home, heat, cold = agent.climate
    hardy = small.render(hardiness(heat, cold), True, (149, 170, 150))
    for names in (TEMPERATURES, SHORT_TEMPERATURES):
        parts = [(f"HOME {names[home].upper()}", TEMPERATURE_COLOURS[home])]
        if here is not None:
            steps = agent.climate_steps(here)  # part steps while the weather turns: shown as the nearer
            colour = (175, 192, 172) if steps < 0.5 else (238, 183, 110) if steps < 1.5 else (235, 100, 90)
            parts.append((f"HERE {names[int(here + 0.5)].upper()}", colour))
        if sum(small.size(text)[0] for text, _ in parts) + UI(14) * len(parts) + hardy.get_width() <= width:
            break
    left = x
    for text, colour in parts:
        _text(screen, text, (left, y), colour, small)
        left += small.size(text)[0] + UI(14)
    screen.blit(hardy, (x + width - hardy.get_width(), y))


def _text(screen: pg.Surface, text: str, position: tuple[int, int], color: tuple[int, int, int],
          font: pg.font.Font) -> None:
    screen.blit(font.render(text, True, color), position)


def _bar(screen: pg.Surface, x: int, y: int, width: int, value: float, color: tuple[int, int, int]) -> None:
    pg.draw.rect(screen, (44, 57, 52), (x, y, width, UI(7)))
    pg.draw.rect(screen, color, (x, y, round(width * max(0, min(1, value))), UI(7)))


# Where the last details drew the mind's, body's and social background's chits, and each trait, for their popups.
LAST_SWATCHES: list[pg.Rect | None] = [None, None, None]
LAST_TRAITS: list[tuple[pg.Rect, tuple[str, str]]] = []

# What each trait governs, for the trait popup, as the C++ version's.
TRAIT_ABOUT = {
    "aggression": "How much it enjoys fighting and attacking, softened by its empathy and tolerance.",
    "libido": "How much it enjoys mating, and so how eagerly it seeks a partner.",
    "curiosity": "How much it enjoys setting off to explore.",
    "fear": "How strongly it shies from hunters and danger, and how much a blow frightens it.",
    "appetite": "How much it enjoys eating.",
    "patience": "How far ahead it looks for the weather turning.",
    "intellect": "How fast it learns and how many past experiences it re-learns from; a cleverer brain burns more energy.",
    "memory": "How long it remembers where it last fed and where it was last struck.",
    "resilience": "How quickly it gets over a herd-mate's failure, how long it bears bad weather, and how little a bad experience frightens it.",
    "moodiness": "How much experience reshapes its temperament over its life.",
    "sociability": "How much it enjoys the company of its own kind.",
    "empathy": "How much it shares in others' joys and sorrows: caring for its young, grieving for them, and holding back from fights.",
    "tolerance": "How readily it accepts others unlike itself, and how much less it enjoys fighting and rivalry.",
    "loyalty": "How much it values keeping to its herd.",
    "dominance": "How sought-after it is as a mate, and how little crowding troubles it.",
    "trust": "How much it learns from what its herd-mates do, and how far off a herd-mate's alert reaches it.",
    "kinship": "How closely a herd-mate or partner must share its social background for it to accept them.",
    "size": "A larger body is mightier and tougher but has fewer young at a time; a small one may slip away from an attacker.",
    "strength": "How hard it hits; it costs energy to keep.",
    "armour": "How well it withstands blows; it costs energy to keep.",
    "weapons": "How much harm it does in a fight (thorns, for a plant); it costs energy to keep.",
    "speed": "How fast it moves; the costliest trait to keep.",
    "sight": "How far it sees: three squares further for each step.",
    "fertility": "How cheaply and how often it can breed (how freely a plant spreads).",
    "litter": "How many young it tends to have at once.",
    "carrion": "How well it digests what it eats, carrion included.",
    "neurodivergence": "A gene that does nothing yet.",
    "burn": "The energy its body and brain use each turn.",
}


def _genes_of(part: int, word: int) -> list[tuple[str, int, int, str]]:
    """A word's genes, from its highest bits to its lowest, as they read left to right in binary: each its name, lowest
    bit, number of bits, and what its value means if more than the number; as the C++ version's."""
    def value(low: int, bits: int) -> int:
        return (word >> low) & ((1 << bits) - 1)
    yes_no = ("no", "yes")
    if part == 0:
        return [("cold tolerance", 31, 1, yes_no[value(31, 1)]), ("heat tolerance", 30, 1, yes_no[value(30, 1)]),
                *((MIND_TRAITS[trait], TRAIT_BITS * trait, TRAIT_BITS, "") for trait in reversed(range(len(MIND_TRAITS))))]
    if part == 1:
        return [("home temperature", 30, 2, TEMPERATURES[value(30, 2)]),
                *((BODY_TRAITS[trait], TRAIT_BITS * trait, TRAIT_BITS, "") for trait in reversed(range(1, len(BODY_TRAITS)))),
                ("nocturnal", 2, 1, yes_no[value(2, 1)]), ("species", 0, 2, SPECIES[value(0, 2)])]
    terrain = NEURODIVERGENCE_SHIFT + 4
    return [("(unused)", terrain + 2, 32 - terrain - 2, ""), ("terrain", terrain, 2, TERRAINS[value(terrain, 2)]),
            ("neurodivergence", NEURODIVERGENCE_SHIFT, 4, ""),
            *((SOCIAL_TRAITS[trait], TRAIT_BITS * trait, TRAIT_BITS, "") for trait in reversed(range(len(SOCIAL_TRAITS))))]


def _binary(word: int, low: int, bits: int) -> str:
    return "".join("1" if (word >> bit) & 1 else "0" for bit in range(low + bits - 1, low - 1, -1))


def _popup_box(screen: pg.Surface, mouse: tuple[int, int], width: int, height: int) -> pg.Rect:
    """A popup's box, to the left of the mouse (the sidebar is at the right), kept on the screen."""
    box = pg.Rect(mouse[0] - UI(18) - width, mouse[1] + UI(12), width, height)
    if box.left < 0:
        box.left = mouse[0] + UI(18)
    box.clamp_ip(screen.get_rect())
    pg.draw.rect(screen, (19, 26, 27), box, border_radius=UI(6))
    pg.draw.rect(screen, (131, 205, 143), box, 1, border_radius=UI(6))
    return box


def draw_word_popup(screen: pg.Surface, agent: Bitzkey, part: int, mouse: tuple[int, int]) -> None:
    """One genome word (0 mind, 1 body, 2 social): its hex, its binary, and its genes labelled."""
    word = (mind_word, body_word, social_word)[part](agent.dna)
    genes = _genes_of(part, word)
    mono, bold = UI.font("consolas", 12), UI.font("segoeui", 13, bold=True)
    whole = " ".join(_binary(word, low, bits) for _, low, bits, _ in genes)
    name_width = max(mono.size(tr(name))[0] for name, *_ in genes)
    line, padding = UI(15), UI(10)
    width = max(mono.size(whole)[0], name_width + mono.size("  bits 00-00  00000  15  very cold")[0]) + 2 * padding
    box = _popup_box(screen, mouse, width, padding * 2 + UI(20) + line * (3 + len(genes)))
    x, y = box.left + padding, box.top + padding
    pg.draw.rect(screen, genome_colour(word), (x, y + UI(2), UI(12), UI(12)))
    pg.draw.rect(screen, (190, 190, 190), (x, y + UI(2), UI(12), UI(12)), 1)
    _text(screen, ("MIND", "BODY", "SOCIAL")[part], (x + UI(18), y), (229, 239, 220), bold)
    y += UI(20)
    _text(screen, as_shown(f"{tr('hex')}  {word:08X}"), (x, y), (220, 229, 211), mono)
    y += line
    _text(screen, "binary", (x, y), (149, 170, 150), mono)
    y += line
    _text(screen, as_shown(whole), (x, y), (220, 229, 211), mono)
    y += line
    for name, low, bits, meaning in genes:  # each gene: its name, where it lies, its bits, and its value
        where = f"{tr('bit')} {low}" if bits == 1 else f"{tr('bits')} {low}-{low + bits - 1}"
        _text(screen, name, (x, y), (175, 192, 172), mono)
        value = (word >> low) & ((1 << bits) - 1)
        _text(screen, as_shown(f"{where:<10} {_binary(word, low, bits):>6} {value:>3}  {tr(meaning) if meaning else ''}"),
              (x + name_width + UI(10), y), (211, 219, 201), mono)
        y += line


def draw_trait_popup(screen: pg.Surface, agent: Bitzkey, spot: tuple[str, str], known: bool, mouse: tuple[int, int]) -> None:
    """One trait, in larger text: its name, its value, and what it governs."""
    kind, trait = spot
    if kind == "temperament":
        name = trait
        value = (f"{tr('inborn')} {agent.inborn[trait]}   {tr('now')} {agent.mind[trait]:.1f}" if known
                 else f"{tr('inborn')} {agent.inborn[trait]}")
    elif kind == "neurodivergence":
        name, value = "neurodivergence", f"{tr('value')} {decode_neurodivergence(agent.dna)} / {NEURODIVERGENCE_MAX}"
    elif kind == "burn":
        name, value = "burn", f"{tr('value')} {agent.burn:.2f}"
    else:
        name, value = trait, f"{tr('value')} {agent.body[trait]} / {TRAIT_MAX}"
    title, text = UI.font("segoeui", 17, bold=True), UI.font("segoeui", 15)
    padding, width, line = UI(12), UI(320), UI(20)
    lines = wrapped_lines(text, TRAIT_ABOUT[name], width - 2 * padding)
    box = _popup_box(screen, mouse, width, padding * 2 + UI(32) + line * (1 + len(lines)))
    x, y = box.left + padding, box.top + padding
    _text(screen, as_shown(tr(name).upper()), (x, y), (229, 239, 220), title)
    y += UI(26)
    _text(screen, as_shown(value), (x, y), (131, 205, 143), text)
    y += line + UI(6)
    for shown in lines:
        _text(screen, as_shown(shown), (x, y), (211, 219, 201), text)
        y += line


def draw_details(screen: pg.Surface, agent: Bitzkey, x: int, at: Callable[[float], int], width: int,
                 known: bool = True, here: float | None = None, note: str | None = None) -> None:
    """Draw the details from x across `width`; `at` turns a design offset from their top into a
    screen y. A Bitzkey not `known` is one no longer (or not yet) in the world, rebuilt from its
    genome, so only what its genes fix is shown, with `note` (how it died, say) beneath. `here` is the
    temperature where it is."""
    font, small = UI.font("segoeui", 15), UI.font("segoeui", 12)
    mono_small = UI.font("consolas", 12)
    # At twice the size it has on the map, beside its name and its colours
    draw_bitzkey(screen, agent.dna, (x + UI(16), at(16)), 2 * dot_radius(int(agent.body["size"]), UI(DOT_RADIUS_RANGE[1])),
                 ring_width=2 * RING_WIDTH, alive=agent.alive or not known)
    draw_icon(screen, agent.species, (x + width - UI(12), at(11)), UI(9), (220, 229, 211))
    _text(screen, as_shown(f"#{agent.identifier:03d}  {tr(agent.species).upper()}  {tr('nocturnal' if agent.nocturnal else 'diurnal').upper()}"),
          (x + UI(40), at(0)), (238, 238, 220), font)
    terrain = as_shown(tr(agent.terrain).upper())  # on the line below, at the right, clear of the species icon
    _text(screen, terrain, (x + width - small.size(terrain)[0], at(25)), (211, 219, 201), small)
    LAST_TRAITS.clear()
    across = x + UI(40)  # its three colours as chits: the mouse over one shows that word, gene by gene
    for part, (name, word) in enumerate((("MIND", mind_word(agent.dna)), ("BODY", body_word(agent.dna)),
                                         ("SOCIAL", social_word(agent.dna)))):
        swatch, start = UI(10), across
        pg.draw.rect(screen, genome_colour(word), (across, at(27), swatch, swatch))
        pg.draw.rect(screen, (190, 190, 190), (across, at(27), swatch, swatch), 1)
        across += swatch + UI(3)
        _text(screen, name, (across, at(26)), (211, 219, 201), mono_small)
        across += mono_small.size(tr(name))[0] + UI(12)
        LAST_SWATCHES[part] = pg.Rect(start, at(24), across - start, UI(16))  # the chit and its name

    # Temperament: the mind's traits on the left, the social background's on the right. Names are in
    # the language shown, and the numbers stand in columns after the longest of them.
    def widest(traits) -> int:
        return max(mono_small.size(name)[0] for name in [*traits, "neurodiv."]) + UI(6)

    digits = mono_small.size("00")[0]
    right = x + max(min(UI(168), width // 2 + UI(18)), widest(MIND_TRAITS) + digits + mono_small.size(" 0.0")[0] + UI(10))
    for left, heading, traits in ((x, "MIND", MIND_TRAITS), (right, "SOCIAL", SOCIAL_TRAITS)):
        numbers = left + widest(traits)
        _text(screen, heading, (left, at(46)), (149, 170, 150), mono_small)
        born_at = max(numbers + digits - mono_small.size("born")[0], left + mono_small.size(heading)[0] + UI(6))
        _text(screen, "born", (born_at, at(46)), (149, 170, 150), mono_small)
        if known:
            _text(screen, "now", (numbers + digits + UI(8), at(46)), (149, 170, 150), mono_small)
        for index, trait in enumerate(traits):
            born, now = agent.inborn[trait], agent.mind[trait]
            color = ((175, 192, 172) if not known or abs(now - born) < 0.25
                     else (131, 205, 143) if now > born else (238, 183, 110))
            # A bar under the row: filled to now, a tick at birth, so the change shows at a glance
            bar_width = numbers + digits + mono_small.size(" 0.0+")[0] - left
            bar_y, thick = at(62 + index * 14) + UI(12), max(1, UI(2))
            pg.draw.rect(screen, (58, 70, 62), (left, bar_y, bar_width, thick))
            shown = now if known else born
            pg.draw.rect(screen, color, (left, bar_y, int(bar_width * min(1.0, max(0.0, shown / TRAIT_MAX))), thick))
            tick = left + int(bar_width * min(1.0, max(0.0, born / TRAIT_MAX)))
            pg.draw.rect(screen, (230, 236, 226), (tick - 1, bar_y - UI(1), 2, max(2, UI(4))))
            _text(screen, trait, (left, at(62 + index * 14)), color, mono_small)
            LAST_TRAITS.append((pg.Rect(left, at(61 + index * 14), bar_width, UI(14)), ("temperament", trait)))
            _text(screen, f"{born:>2}", (numbers, at(62 + index * 14)), color, mono_small)
            change = " " if not known or abs(now - born) < 0.25 else "+" if now > born else "-"  # not by colour alone
            if known:
                _text(screen, f"{now:>5.1f}{change}", (numbers + digits, at(62 + index * 14)), color, mono_small)
        if heading == "SOCIAL":
            _text(screen, "neurodiv.", (left, at(62 + len(traits) * 14)), (175, 192, 172), mono_small)
            LAST_TRAITS.append((pg.Rect(left, at(61 + len(traits) * 14), numbers + digits - left, UI(14)), ("neurodivergence", "")))
            _text(screen, f"{decode_neurodivergence(agent.dna):>2}", (numbers, at(62 + len(traits) * 14)), (175, 192, 172),
                  mono_small)

    # The body, three to a row: each name, and its number at the cell's right.
    _text(screen, "BODY", (x, at(206)), (149, 170, 150), mono_small)
    cells = [(trait, f"{agent.body[trait]:>2}") for trait in BODY_TRAITS[1:]] + [("burn", f"{agent.burn:>5.2f}")]
    for index, (name, number) in enumerate(cells):
        row, column = divmod(index, 3)
        cell = x + column * width // 3
        _text(screen, name, (cell, at(222 + row * 14)), (175, 192, 172), mono_small)
        LAST_TRAITS.append((pg.Rect(cell, at(221 + row * 14), width // 3 - UI(10), UI(14)), ("burn", "") if name == "burn" else ("body", name)))
        _text(screen, number, (cell + width // 3 - UI(10) - mono_small.size(number)[0], at(222 + row * 14)), (175, 192, 172),
              mono_small)
        if name != "burn":  # a bar under it on the traits' shared scale
            bar_width, bar_y, thick = width // 3 - UI(10), at(222 + row * 14) + UI(12), max(1, UI(2))
            pg.draw.rect(screen, (58, 70, 62), (cell, bar_y, bar_width, thick))
            pg.draw.rect(screen, (175, 192, 172), (cell, bar_y, bar_width * agent.body[name] // TRAIT_MAX, thick))
    _draw_climate(screen, agent, x, at(280), width, here if known else None)

    if not known:
        _text(screen, f"LIFESPAN ~{agent.lifespan:.0f}", (x, at(302)), (145, 164, 146), small)
        _text(screen, note or "Not in the world at this turn: only its genome is known.", (x, at(318)),
              (116, 139, 124), small)
        return
    third = (width - UI(24)) // 3  # hunger, stamina and health side by side
    _text(screen, f"HUNGER {min(100.0, agent.hunger):.0f}%", (x, at(302)), (224, 177, 118), mono_small)
    _bar(screen, x, at(320), third, agent.hunger / 100, (224, 150, 94))
    stamina = x + third + UI(12)
    _text(screen, f"STAMINA {agent.stamina:.0f}", (stamina, at(302)), (150, 190, 230), mono_small)
    _bar(screen, stamina, at(320), third, agent.stamina / 100, (110, 160, 220))
    health = x + width - third
    _text(screen, f"HEALTH {max(0.0, agent.health):.0f}/{agent.max_health:.0f}", (health, at(302)), (161, 207, 162),
          mono_small)
    _bar(screen, health, at(320), third, agent.health / agent.max_health, (124, 190, 131))
    _text(screen, f"AGE {agent.age} / ~{agent.lifespan:.0f}{' FEEBLE' if agent.feeble else ''}    GENERATION {agent.generation}    "
          f"CHILDREN {agent.children}", (x, at(334)), (145, 164, 146), small)
    state = f"LAST ACTION: {agent.last_action.upper()}"
    if not agent.alive:
        state += f"     MEAT LEFT {agent.meat:.0f}"
    _text(screen, state, (x, at(350)), (145, 164, 146), small)
    if agent.odds:
        odds = sorted(agent.odds.items(), key=lambda item: -item[1])[:3]
        line = tr("CHANCES") + "  " + "   ".join(f"{tr(action)} {chance:.0%}" for action, chance in odds)  # each in the language shown
        _text(screen, as_shown(line), (x, at(366)), (175, 192, 172), small)


def draw_tooltip(screen: pg.Surface, agent: Bitzkey, known: bool, mouse: tuple[int, int],
                 here: float | None = None, note: str | None = None) -> pg.Rect:
    """The details in a box beside the mouse, kept on the screen. Returns the box."""
    width, padding = UI(300), UI(12)
    box = pg.Rect(0, 0, width + 2 * padding, UI(DETAILS_HEIGHT if known else GENOME_HEIGHT) + 2 * padding)
    box.topleft = (mouse[0] + UI(18), mouse[1] + UI(12))
    if box.right > screen.get_width():
        box.right = mouse[0] - UI(18)
    box.clamp_ip(screen.get_rect())
    pg.draw.rect(screen, (19, 26, 27), box, border_radius=UI(6))
    pg.draw.rect(screen, (131, 205, 143), box, 1, border_radius=UI(6))
    top = box.top + padding
    draw_details(screen, agent, box.left + padding, lambda offset: top + UI(offset), width, known, here, note)
    return box
