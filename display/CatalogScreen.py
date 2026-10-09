"""The catalog of favourite Bitzkeys: each one listed, and the one picked shown large beside its
inspector panel, as it was when catalogued, trained brain and all. Opened from the configuration
panel, picking one adds it to the import list; opened from a running world to browse, it only shows
them (and can remove them)."""

from __future__ import annotations

import pygame as pg

from display import Window
from display.Dialog import Dialog
from display.Inspector import draw_details
from display.Look import RING_WIDTH, dot_radius, draw_bitzkey
from display.Ui import UI
from tools import Catalog
from tools.Simulation import Bitzkey

ROW_HEIGHT = 40       # each entry's row in the list
PREVIEW_ZOOM = 9      # the large picture is the in-world look, this many times larger


class CatalogScreen:
    """Runs the catalog until Bitzkeys are chosen for import, it is closed, or the window closes. They
    are chosen as in Windows: a click chooses one, CTRL+click adds one or takes it out, SHIFT+click
    chooses all from the last one clicked to this one (CTRL+SHIFT adds them), and CTRL+A all."""

    def __init__(self, screen: pg.Surface, browsing: bool = False):
        self.screen = screen
        self.browsing = browsing  # only to look through: nothing is imported
        self.note = ""
        try:
            self.entries = Catalog.load()
        except ValueError as error:
            self.entries, self.note = [], str(error)
        self.picked = 0       # the entry shown beside the list, the last one clicked or moved to
        self.anchor = 0       # where a SHIFT+click's run of entries starts
        self.chosen: set[int] = {0} if self.entries else set()  # the entries to import or remove
        self.scroll = 0       # the first entry the list shows
        self.dialog: Dialog | None = None
        self._inspected: dict[int, Bitzkey] = {}
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
    def small(self) -> pg.font.Font:
        return UI.font("segoeui", 12)

    def run(self) -> tuple[str, list[dict] | None]:
        """Return ("import", entries) for the Bitzkeys chosen to import, in catalog order, ("close", None)
        or ("quit", None)."""
        while True:
            self.clock.tick(30)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return "quit", None
                if event.type == pg.VIDEORESIZE:
                    self.screen = Window.resize(event)
                elif self.dialog:
                    choice = self.dialog.handle(event, self.screen.get_size())
                    if choice:
                        self.dialog = None
                        if choice == "remove":
                            self._remove()
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        return "close", None
                    if event.key in (pg.K_RETURN, pg.K_KP_ENTER) and self.chosen and not self.browsing:
                        return "import", self._chosen_entries()
                    if event.key == pg.K_F11:
                        self.screen = Window.toggle_fullscreen()
                    elif event.key in (pg.K_UP, pg.K_DOWN) and self.entries:
                        moved = max(0, min(len(self.entries) - 1, self.picked + (1 if event.key == pg.K_DOWN else -1)))
                        self._choose(moved, event.mod & pg.KMOD_SHIFT and pg.KMOD_SHIFT)
                    elif event.key == pg.K_a and event.mod & pg.KMOD_CTRL:
                        self.chosen = set(range(len(self.entries)))
                    elif event.key == pg.K_DELETE and self.chosen:
                        self._ask_to_remove()
                elif event.type == pg.MOUSEWHEEL:
                    self._scroll_to(self.scroll - event.y)
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    action = self._click(event.pos)
                    if action == "import":
                        return "import", self._chosen_entries()
                    if action == "close":
                        return "close", None
            self._draw()
            pg.display.flip()

    def _rows_shown(self) -> int:
        return max(1, self._layout()["list"].height // UI(ROW_HEIGHT))

    def _scroll_to(self, first: int) -> None:
        self.scroll = max(0, min(len(self.entries) - self._rows_shown(), first))

    def _pick(self, index: int) -> None:
        if not self.entries:
            return
        self.picked = max(0, min(len(self.entries) - 1, index))
        if self.picked < self.scroll:  # keep it in view
            self._scroll_to(self.picked)
        elif self.picked >= self.scroll + self._rows_shown():
            self._scroll_to(self.picked - self._rows_shown() + 1)

    def _choose(self, index: int, mods: int = 0) -> None:
        """Choose an entry as in Windows: alone; with CTRL, added or taken out; with SHIFT, the run from
        the last one clicked to it (added, with CTRL as well)."""
        if mods & pg.KMOD_SHIFT:
            run = set(range(min(self.anchor, index), max(self.anchor, index) + 1))
            self.chosen = self.chosen | run if mods & pg.KMOD_CTRL else run
        elif mods & pg.KMOD_CTRL:
            self.chosen ^= {index}
            self.anchor = index
        else:
            self.chosen = {index}
            self.anchor = index
        self._pick(index)

    def _chosen_entries(self) -> list[dict]:
        return [self.entries[index] for index in sorted(self.chosen)]

    def _ask_to_remove(self) -> None:
        count = len(self.chosen)
        what = Catalog.describe(self.entries[min(self.chosen)]) if count == 1 else f"{count} Bitzkeys"
        self.dialog = Dialog("Remove from the catalog?", [what, "They will be gone from the catalog for good." if count > 1
                                                          else "It will be gone from the catalog for good."],
                             [("keep", "KEEP"), ("remove", "REMOVE")], cancel="keep")

    def _remove(self) -> None:
        count = len(self.chosen)
        for index in sorted(self.chosen, reverse=True):  # the last first, so the rest keep their places
            Catalog.remove(index)
        self.entries = Catalog.load()
        self._inspected.clear()
        self._pick(min(self.picked, len(self.entries) - 1))
        self.chosen = {self.picked} if self.entries else set()
        self.anchor = self.picked
        self._scroll_to(self.scroll)
        self.note = f"Removed {count} from the catalog." if count > 1 else "Removed from the catalog."

    def _inspect(self, index: int) -> Bitzkey:
        if index not in self._inspected:
            self._inspected[index] = Catalog.inspected(self.entries[index])
        return self._inspected[index]

    def _layout(self) -> dict:
        UI.fit(self.screen.get_size())
        width, height = self.screen.get_size()
        card = pg.Rect(0, 0, min(UI(940), width - UI(40)), min(UI(760), height - UI(40)))
        card.center = (width // 2, height // 2)
        details = pg.Rect(card.right - UI(24) - UI(330), card.top + UI(96), UI(330), card.height - UI(170))
        listing = pg.Rect(card.left + UI(24), card.top + UI(96), details.left - card.left - UI(48), card.height - UI(170))
        close = pg.Rect(0, 0, UI(120), UI(36))
        close.bottomright = (card.right - UI(24), card.bottom - UI(18))
        add = pg.Rect(0, 0, UI(180), UI(36))
        add.bottomright = (close.left - UI(10), close.bottom)
        remove = pg.Rect(0, 0, UI(110), UI(36))
        remove.bottomleft = (card.left + UI(24), card.bottom - UI(18))
        return {"card": card, "list": listing, "details": details, "close": close, "import": add, "remove": remove}

    def _row(self, index: int) -> pg.Rect:
        listing = self._layout()["list"]
        return pg.Rect(listing.left, listing.top + (index - self.scroll) * UI(ROW_HEIGHT), listing.width,
                       UI(ROW_HEIGHT) - UI(4))

    def _click(self, position: tuple[int, int]) -> str | None:
        layout = self._layout()
        if layout["close"].collidepoint(position):
            return "close"
        if layout["import"].collidepoint(position) and self.chosen and not self.browsing:
            return "import"
        if layout["remove"].collidepoint(position) and self.chosen:
            self._ask_to_remove()
            return None
        if layout["list"].collidepoint(position):
            for index in range(self.scroll, min(len(self.entries), self.scroll + self._rows_shown())):
                if self._row(index).collidepoint(position):
                    self._choose(index, pg.key.get_mods())
        return None

    def _draw(self) -> None:
        layout = self._layout()
        card, listing, details = layout["card"], layout["list"], layout["details"]
        self.screen.fill((19, 26, 27))
        pg.draw.rect(self.screen, (27, 37, 37), card, border_radius=UI(8))
        pg.draw.rect(self.screen, (80, 99, 85), card, 1, border_radius=UI(8))
        title = self.bold.render("BITZKEY CATALOG", True, (229, 239, 220))
        self.screen.blit(title, (card.left + UI(24), card.top + UI(22)))
        self._text("YOUR FAVOURITES" if self.browsing else "NEW WORLD  /  YOUR FAVOURITES",
                   (card.left + UI(48) + title.get_width(), card.top + UI(31)), (144, 170, 151), self.small)
        self._text("Each one kept as it was when catalogued, its trained brain and all." if self.browsing else
                   "Each one kept as it was when catalogued, its trained brain and all. Choose those to import.",
                   (card.left + UI(24), card.top + UI(60)), (157, 177, 159), self.font)

        pg.draw.rect(self.screen, (22, 30, 30), listing, border_radius=UI(6))
        if not self.entries:
            self._text("The catalog is empty.", (listing.left + UI(16), listing.top + UI(16)), (157, 177, 159), self.font)
            self._text("Add Bitzkeys from a running world: ADD TO CATALOG in the sidebar,", (listing.left + UI(16),
                       listing.top + UI(44)), (116, 139, 124), self.small)
            self._text("the export list or the colour graph, or press K.", (listing.left + UI(16), listing.top + UI(62)),
                       (116, 139, 124), self.small)
        for index in range(self.scroll, min(len(self.entries), self.scroll + self._rows_shown())):
            row = self._row(index)
            if index in self.chosen:
                pg.draw.rect(self.screen, (44, 57, 52), row, border_radius=UI(4))
            if index == self.picked:  # the one shown beside the list
                pg.draw.rect(self.screen, (131, 205, 143), row, 1, border_radius=UI(4))
            agent = self._inspect(index)
            draw_bitzkey(self.screen, agent.dna, (row.left + UI(20), row.centery), dot_radius(int(agent.body["size"])))
            self._text(Catalog.describe(self.entries[index]), (row.left + UI(42), row.centery - UI(9)), (220, 229, 211),
                       self.font)
        if len(self.entries) > self._rows_shown():
            self._text(f"{self.scroll + 1}-{min(len(self.entries), self.scroll + self._rows_shown())} of "
                       f"{len(self.entries)}   (wheel or UP/DOWN to scroll)", (listing.left, listing.bottom + UI(6)),
                       (116, 139, 124), self.small)

        pg.draw.rect(self.screen, (22, 30, 30), details, border_radius=UI(6))
        if self.entries:
            agent = self._inspect(self.picked)
            draw_bitzkey(self.screen, agent.dna, (details.centerx, details.top + UI(70)),
                         UI(PREVIEW_ZOOM * dot_radius(int(agent.body["size"]))), ring_width=UI(PREVIEW_ZOOM * RING_WIDTH))
            inside = details.inflate(-UI(24), 0)
            draw_details(self.screen, agent, inside.left, lambda offset: details.top + UI(150 + offset), inside.width)

        adding = "ADD TO IMPORT" if len(self.chosen) <= 1 else f"ADD {len(self.chosen)} TO IMPORT"
        for name, label, lit in (("remove", "REMOVE", False), ("import", adding, True), ("close", "CLOSE", False)):
            if name == "import" and self.browsing:
                continue
            button = layout[name]
            active = lit and bool(self.chosen)
            pg.draw.rect(self.screen, (131, 205, 143) if active else (44, 57, 52), button, border_radius=UI(6))
            text = self.button_font.render(label, True, (19, 26, 27) if active else (220, 229, 211))
            self.screen.blit(text, text.get_rect(center=button.center))
        hint = self.note or ("CLICK, SHIFT or CTRL+CLICK  choose    DELETE  remove    ESC  close" if self.browsing else
                             "CLICK, SHIFT or CTRL+CLICK  choose    ENTER  add to import    DELETE  remove    ESC  close")
        self._text(hint, (layout["remove"].right + UI(16), layout["remove"].centery - UI(8)),
                   (238, 183, 110) if self.note else (116, 139, 124), self.small)
        if self.dialog:
            self.dialog.draw(self.screen)

    def _text(self, text: str, position: tuple[int, int], color: tuple[int, int, int], font: pg.font.Font) -> None:
        self.screen.blit(font.render(text, True, color), position)
