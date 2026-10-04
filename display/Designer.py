"""The Bitzkey designer: every trait of a genome, editable, beside a large preview in its true
colours. Opened from the configuration panel; applying a design adds it to the import list."""

from __future__ import annotations

import random

import pygame as pg

from display import Window
from display.Look import RING_WIDTH, dot_radius, draw_bitzkey
from display.Ui import UI
from tools.Genome import BODY_TRAITS, GENOME, MIND_TRAITS, SPECIES, WORD, body_word, decode_body, mind_word
from tools.Simulation import Bitzkey

FIELDS = (tuple(("mind", trait) for trait in MIND_TRAITS) + tuple(("body", trait) for trait in BODY_TRAITS)
          + (("hex", "ring"), ("hex", "dot")))
HEX_DIGITS = "0123456789ABCDEF"
PREVIEW_ZOOM = 16  # the preview is the in-world look, this many times larger


def _shift(field: tuple[str, str]) -> int:
    kind, name = field
    return 4 * MIND_TRAITS.index(name) if kind == "mind" else 32 + 4 * BODY_TRAITS.index(name)


class DesignerForm:
    """The genome being designed, the focused field, and any value or hex code being typed."""

    def __init__(self, dna: int):
        self.dna = dna & GENOME
        self.focus = 0
        self.typed: str | None = None  # partial input for the focused field; None means start afresh

    @property
    def focused(self) -> tuple[str, str]:
        return FIELDS[self.focus]

    def value(self, field: tuple[str, str]) -> int:
        """A trait's four-bit value; for species, the raw value that picks it (value mod 4)."""
        return (self.dna >> _shift(field)) & 0xF

    def set_value(self, field: tuple[str, str], value: int) -> None:
        shift = _shift(field)
        self.dna = (self.dna & ~(0xF << shift) & GENOME) | (max(0, min(15, value)) << shift)

    def word(self, part: str) -> int:
        return body_word(self.dna) if part == "ring" else mind_word(self.dna)

    def code(self, part: str) -> str:
        """The hex code shown for a word: whatever is being typed into it, else its value."""
        if self.focused == ("hex", part) and self.typed is not None:
            return self.typed
        return f"{self.word(part):08X}"

    def select(self, index: int) -> None:
        self.focus = index % len(FIELDS)
        self.typed = None  # an unfinished hex code is dropped

    def move(self, offset: int) -> None:
        self.select(self.focus + offset)

    def adjust(self, steps: int) -> None:
        if self.focused[0] != "hex":
            self.set_value(self.focused, self.value(self.focused) + steps)
            self.typed = None

    def type_text(self, text: str) -> None:
        kind, part = self.focused
        if kind == "hex":
            for character in text.upper():
                if character in HEX_DIGITS and len(self.typed or "") < 8:
                    self.typed = (self.typed or "") + character
            if self.typed is not None and len(self.typed) == 8:
                self._set_word(part, int(self.typed, 16))
                self.typed = None
            return
        for digit in filter(str.isdigit, text):
            self.typed = (self.typed or "") + digit
            self.set_value(self.focused, int(self.typed))
            if len(self.typed) == 2:
                self.typed = None

    def backspace(self) -> None:
        kind, part = self.focused
        if kind == "hex":
            self.typed = (self.code(part))[:-1]
        else:
            value = self.value(self.focused) // 10
            self.set_value(self.focused, value)
            self.typed = str(value) if value else None

    def randomise(self, rng: random.Random) -> None:
        self.dna = rng.getrandbits(64)
        self.typed = None

    def _set_word(self, part: str, word: int) -> None:
        if part == "ring":
            self.dna = (self.dna & WORD) | ((word & WORD) << 32)
        else:
            self.dna = (self.dna & ~WORD & GENOME) | (word & WORD)


class Designer:
    """Runs the designer until it is applied, cancelled or the window closes."""

    def __init__(self, screen: pg.Surface, dna: int | None = None, rng: random.Random | None = None):
        self.screen = screen
        self.rng = rng or random.Random()
        if dna is None:
            dna = self.rng.getrandbits(64) & ~(0x3 << 32) | (self.rng.randrange(1, len(SPECIES)) << 32)
        self.form = DesignerForm(dna)
        self.clock = pg.time.Clock()

    @property
    def font(self) -> pg.font.Font:
        return UI.font("segoeui", 15)

    @property
    def bold(self) -> pg.font.Font:
        return UI.font("segoeui", 22, bold=True)

    @property
    def button_font(self) -> pg.font.Font:
        return UI.font("segoeui", 13, bold=True)

    @property
    def mono(self) -> pg.font.Font:
        return UI.font("consolas", 18)

    @property
    def small(self) -> pg.font.Font:
        return UI.font("segoeui", 12)

    def run(self) -> tuple[str, int | None]:
        """Return ("apply", genome), ("cancel", None) or ("quit", None)."""
        while True:
            self.clock.tick(30)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return "quit", None
                if event.type == pg.VIDEORESIZE:
                    self.screen = Window.resize(event)
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        return "cancel", None
                    if event.key in (pg.K_RETURN, pg.K_KP_ENTER):
                        return "apply", self.form.dna
                    if event.key == pg.K_F11:
                        self.screen = Window.toggle_fullscreen()
                    else:
                        self._key(event)
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    action = self._click(event.pos)
                    if action == "apply":
                        return "apply", self.form.dna
                    if action == "cancel":
                        return "cancel", None
            self._draw()
            pg.display.flip()

    def _key(self, event: pg.event.Event) -> None:
        form = self.form
        if event.key == pg.K_UP or (event.key == pg.K_TAB and event.mod & pg.KMOD_SHIFT):
            form.move(-1)
        elif event.key in (pg.K_DOWN, pg.K_TAB):
            form.move(1)
        elif event.key in (pg.K_RIGHT, pg.K_LEFT):
            form.adjust((5 if event.mod & pg.KMOD_SHIFT else 1) * (1 if event.key == pg.K_RIGHT else -1))
        elif event.key in (pg.K_BACKSPACE, pg.K_DELETE):
            form.backspace()
        elif event.unicode and event.unicode.isprintable():
            form.type_text(event.unicode)

    def _layout(self) -> dict:
        UI.fit(self.screen.get_size())
        width, height = self.screen.get_size()
        card = pg.Rect(0, 0, UI(1040), UI(740))
        card.center = (width // 2, height // 2)
        preview = pg.Rect(card.left + UI(24), card.top + UI(92), UI(320), UI(300))
        codes = {part: pg.Rect(preview.right - UI(160), preview.bottom + UI(16 + 44 * index), UI(160), UI(34))
                 for index, part in enumerate(("ring", "dot"))}
        column_width = (card.right - UI(24) - (preview.right + UI(32)) - UI(24)) // 2
        rows = {}
        for index, field in enumerate(FIELDS[:16]):
            column, row = divmod(index, 8)
            left = preview.right + UI(32) + column * (column_width + UI(24))
            rows[field] = pg.Rect(left, card.top + UI(124 + row * 52), column_width, UI(44))
        buttons = {}
        right = card.right - UI(24)
        for name, button_width in (("apply", 120), ("cancel", 120)):
            buttons[name] = pg.Rect(right - UI(button_width), card.bottom - UI(68), UI(button_width), UI(44))
            right -= UI(button_width + 12)
        buttons["random"] = pg.Rect(card.left + UI(24), card.bottom - UI(68), UI(120), UI(44))
        return {"card": card, "preview": preview, "codes": codes, "rows": rows, **buttons}

    @staticmethod
    def _controls(row: pg.Rect) -> tuple[pg.Rect, pg.Rect, pg.Rect]:
        """The minus button, value box and plus button of a trait row."""
        plus = pg.Rect(row.right - UI(38), row.centery - UI(15), UI(30), UI(30))
        box = pg.Rect(plus.left - UI(112), row.centery - UI(16), UI(104), UI(32))
        minus = pg.Rect(box.left - UI(38), row.centery - UI(15), UI(30), UI(30))
        return minus, box, plus

    def _click(self, position: tuple[int, int]) -> str | None:
        layout = self._layout()
        for action in ("apply", "cancel"):
            if layout[action].collidepoint(position):
                return action
        if layout["random"].collidepoint(position):
            self.form.randomise(self.rng)
        for part, box in layout["codes"].items():
            if box.collidepoint(position):
                self.form.select(FIELDS.index(("hex", part)))
        for field, row in layout["rows"].items():
            if row.collidepoint(position):
                if FIELDS.index(field) != self.form.focus:
                    self.form.select(FIELDS.index(field))
                minus, _, plus = self._controls(row)
                if minus.collidepoint(position):
                    self.form.adjust(-1)
                elif plus.collidepoint(position):
                    self.form.adjust(1)
        return None

    def _draw(self) -> None:
        layout = self._layout()
        card, preview = layout["card"], layout["preview"]
        form = self.form
        self.screen.fill((19, 26, 27))
        pg.draw.rect(self.screen, (27, 37, 37), card, border_radius=UI(8))
        pg.draw.rect(self.screen, (80, 99, 85), card, 1, border_radius=UI(8))
        title = self.bold.render("BITZKEY DESIGNER", True, (229, 239, 220))
        self.screen.blit(title, (card.left + UI(24), card.top + UI(22)))
        self._text("NEW WORLD  /  DESIGN AN IMPORT", (card.left + UI(48) + title.get_width(), card.top + UI(31)),
                   (144, 170, 151), self.small)
        self._text("Every trait is part of the genome, so the dot and ring colours change with them.",
                   (card.left + UI(24), card.top + UI(60)), (157, 177, 159), self.font)

        # The large preview, and the Bitzkey at the size it will have in the world.
        pg.draw.rect(self.screen, (22, 30, 30), preview, border_radius=UI(6))
        pg.draw.rect(self.screen, (54, 68, 62), preview, 1, border_radius=UI(6))
        size = form.value(("body", "size"))
        draw_bitzkey(self.screen, form.dna, (preview.centerx, preview.top + UI(150)),
                     UI(PREVIEW_ZOOM * dot_radius(size)), ring_width=UI(PREVIEW_ZOOM * RING_WIDTH))
        self._text("actual size", (preview.left + UI(12), preview.bottom - UI(26)), (116, 139, 124), self.small)
        draw_bitzkey(self.screen, form.dna, (preview.left + UI(96), preview.bottom - UI(18)), dot_radius(size))

        for part, box in layout["codes"].items():
            label = "RING / BODY" if part == "ring" else "DOT / MIND"
            focused = form.focused == ("hex", part)
            self._text(label, (preview.left, box.top + UI(8)), (149, 170, 150), self.font)
            pg.draw.rect(self.screen, (19, 26, 27), box, border_radius=UI(4))
            pg.draw.rect(self.screen, (131, 205, 143) if focused else (54, 68, 62), box, 1, border_radius=UI(4))
            code = form.code(part) + ("_" if focused and form.typed is not None and pg.time.get_ticks() // 500 % 2 else "")
            text = self.mono.render(code, True, (238, 238, 220))
            self.screen.blit(text, text.get_rect(midleft=(box.left + UI(12), box.centery)))

        species = SPECIES[form.value(("body", "species")) % len(SPECIES)]
        body = decode_body(form.dna)
        sample = Bitzkey(0, form.dna, 0, 0)
        if species == "plant":
            facts = ["A plant: it photosynthesises and never moves,", "thinks, breeds or eats."]
        else:
            fertility = int(body["fertility"])
            facts = [f"A {species}.  Health {sample.max_health:.0f}.  Burns {sample.burn:.2f} hunger a turn.",
                     f"Moves {sample.steps} cell{'s' if sample.steps != 1 else ''} a step.  "
                     f"Sees {2 + int(body['sight']) // 2} cells.",
                     f"Breeding costs each parent {25 - fertility} hunger,",
                     f"then {20 - fertility} turns before it can breed again."]
        for index, fact in enumerate(facts):
            self._text(fact, (preview.left, layout["codes"]["dot"].bottom + UI(20 + index * 22)), (187, 199, 181),
                       self.small)

        # The trait rows: mind on the left, body on the right.
        first_rows = (layout["rows"][FIELDS[0]], layout["rows"][FIELDS[8]])
        for row, heading in zip(first_rows, ("MIND  /  INNER DOT", "BODY  /  OUTER RING")):
            self._text(heading, (row.left, row.top - UI(28)), (220, 229, 211), self.font)
        for field, row in layout["rows"].items():
            self._draw_row(field, row, FIELDS.index(field) == form.focus)

        for name, label, filled in (("random", "RANDOM", False), ("cancel", "CANCEL", False), ("apply", "APPLY", True)):
            button = layout[name]
            pg.draw.rect(self.screen, (131, 205, 143) if filled else (44, 57, 52), button, border_radius=UI(6))
            text = self.bold.render(label, True, (19, 26, 27) if filled else (220, 229, 211))
            self.screen.blit(text, text.get_rect(center=button.center))
        hint = self.small.render("UP/DOWN  choose   LEFT/RIGHT  adjust (SHIFT x5)   type a value or hex code   "
                                 "ENTER  apply   ESC  cancel   F11  window", True, (130, 152, 136))
        self.screen.blit(hint, hint.get_rect(midbottom=(card.centerx, card.bottom - UI(84))))

    def _draw_row(self, field: tuple[str, str], row: pg.Rect, focused: bool) -> None:
        form = self.form
        value = form.value(field)
        pg.draw.rect(self.screen, (36, 49, 47) if focused else (31, 42, 41), row, border_radius=UI(5))
        if focused:
            pg.draw.rect(self.screen, (131, 205, 143), row, 1, border_radius=UI(5))
        minus, box, plus = self._controls(row)
        self._text(field[1], (row.left + UI(12), row.top + UI(6)), (232, 236, 219), self.font)
        meter = pg.Rect(row.left + UI(12), row.bottom - UI(11), minus.left - row.left - UI(28), UI(4))
        pg.draw.rect(self.screen, (44, 57, 52), meter)
        pg.draw.rect(self.screen, (131, 205, 143), (meter.left, meter.top, round(meter.width * value / 15), meter.height))
        for button, sign in ((minus, "-"), (plus, "+")):
            pg.draw.rect(self.screen, (44, 57, 52), button, border_radius=UI(4))
            label = self.bold.render(sign, True, (220, 229, 211))
            self.screen.blit(label, label.get_rect(center=button.center))
        pg.draw.rect(self.screen, (19, 26, 27), box, border_radius=UI(4))
        pg.draw.rect(self.screen, (131, 205, 143) if focused else (54, 68, 62), box, 1, border_radius=UI(4))
        shown = f"{SPECIES[value % len(SPECIES)]} {value}" if field == ("body", "species") else str(value)
        text = self.font.render(shown, True, (238, 238, 220))
        self.screen.blit(text, text.get_rect(midright=(box.right - UI(10), box.centery)))

    def _text(self, text: str, position: tuple[int, int], color: tuple[int, int, int], font: pg.font.Font) -> None:
        self.screen.blit(font.render(text, True, color), position)
