"""A Bitzkey's details, as the sidebar shows the selected one and the colour graph's tooltip shows
the one under the mouse: its look, genome, temperament, body, condition and latest choice."""

from __future__ import annotations

from collections.abc import Callable

import pygame as pg

from display.Icons import draw_icon
from display.Look import DOT_RADIUS_RANGE, dot_radius, draw_bitzkey
from display.Ui import UI
from tools.Genome import BODY_TRAITS, MIND_TRAITS, TEMPERATURES, body_word, mind_word
from tools.Simulation import Bitzkey

# Heights, in design pixels, of the details: in full, and for a Bitzkey known only by its genome.
DETAILS_HEIGHT = 385
GENOME_HEIGHT = 308
# Each temperature's colour, from hot to very cold, for the map's zones and the details.
TEMPERATURE_COLOURS = ((238, 160, 110), (175, 192, 172), (140, 185, 230), (200, 225, 250))
SHORT_TEMPERATURES = ("hot", "medium", "cold", "v. cold")


def hardiness(heat: bool, cold: bool) -> str:
    """What its tolerance flags spare it, in a word or two."""
    return "HEAT & COLD-HARDY" if heat and cold else "HEAT-HARDY" if heat else "COLD-HARDY" if cold else ""


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


def draw_details(screen: pg.Surface, agent: Bitzkey, x: int, at: Callable[[float], int], width: int,
                 known: bool = True, here: float | None = None) -> None:
    """Draw the details from x across `width`; `at` turns a design offset from their top into a
    screen y. A Bitzkey not `known` is one no longer (or not yet) in the world, rebuilt from its
    genome, so only what its genes fix is shown. `here` is the temperature where it is."""
    font, small = UI.font("segoeui", 15), UI.font("segoeui", 12)
    mono, mono_small = UI.font("consolas", 14), UI.font("consolas", 12)
    draw_bitzkey(screen, agent.dna, (x + UI(10), at(11)), dot_radius(int(agent.body["size"]), UI(DOT_RADIUS_RANGE[1])),
                 alive=agent.alive or not known)
    draw_icon(screen, agent.species, (x + width - UI(12), at(11)), UI(9), (220, 229, 211))
    _text(screen, f"#{agent.identifier:03d}  {agent.species.upper()}", (x + UI(28), at(0)), (238, 238, 220), font)
    _text(screen, f"DOT / MIND   {mind_word(agent.dna):08X}", (x, at(28)), (211, 219, 201), mono)
    _text(screen, f"RING / BODY  {body_word(agent.dna):08X}", (x, at(47)), (211, 219, 201), mono)

    body_x = x + min(UI(168), width // 2 + UI(18))
    _text(screen, "MIND      born   now" if known else "MIND      born", (x, at(74)), (149, 170, 150), small)
    _text(screen, "BODY", (body_x, at(74)), (149, 170, 150), small)
    for index, trait in enumerate(MIND_TRAITS):
        born, now = agent.inborn[trait], agent.mind[trait]
        if not known:
            _text(screen, f"{trait:<11}{born:>3}", (x, at(92 + index * 16)), (175, 192, 172), mono_small)
            continue
        color = ((175, 192, 172) if abs(now - born) < 0.25
                 else (131, 205, 143) if now > born else (238, 183, 110))
        _text(screen, f"{trait:<11}{born:>3}{now:>6.1f}", (x, at(92 + index * 16)), color, mono_small)
    for index, trait in enumerate(BODY_TRAITS[1:]):
        _text(screen, f"{trait:<9}{agent.body[trait]:>3}", (body_x, at(92 + index * 16)), (175, 192, 172), mono_small)
    _text(screen, f"{'burn':<7}{agent.burn:>5.2f}", (body_x, at(92 + 9 * 16)), (175, 192, 172), mono_small)
    _draw_climate(screen, agent, x, at(254), width, here if known else None)

    if not known:
        _text(screen, f"LIFESPAN ~{agent.lifespan:.0f}", (x, at(276)), (145, 164, 146), small)
        _text(screen, "Not in the world at this turn: only its genome is known.", (x, at(292)), (116, 139, 124),
              small)
        return
    _text(screen, f"HUNGER  {min(100.0, agent.hunger):5.1f}%", (x, at(276)), (224, 177, 118), mono)
    _bar(screen, x, at(295), width, agent.hunger / 100, (224, 150, 94))
    _text(screen, f"HEALTH  {max(0.0, agent.health):5.1f} / {agent.max_health:.0f}", (x, at(305)), (161, 207, 162),
          mono)
    _bar(screen, x, at(324), width, agent.health / agent.max_health, (124, 190, 131))
    _text(screen, f"AGE {agent.age} / ~{agent.lifespan:.0f}    GENERATION {agent.generation}    "
          f"CHILDREN {agent.children}", (x, at(337)), (145, 164, 146), small)
    state = f"LAST ACTION: {agent.last_action.upper()}"
    if not agent.alive:
        state += f"     MEAT LEFT {agent.meat:.0f}"
    _text(screen, state, (x, at(353)), (145, 164, 146), small)
    if agent.odds:
        odds = sorted(agent.odds.items(), key=lambda item: -item[1])[:3]
        _text(screen, "CHANCES  " + "   ".join(f"{action} {chance:.0%}" for action, chance in odds),
              (x, at(369)), (175, 192, 172), small)


def draw_tooltip(screen: pg.Surface, agent: Bitzkey, known: bool, mouse: tuple[int, int],
                 here: float | None = None) -> pg.Rect:
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
    draw_details(screen, agent, box.left + padding, lambda offset: top + UI(offset), width, known, here)
    return box
