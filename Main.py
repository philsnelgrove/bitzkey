"""PyGame interface for the Bitzkey ecology experiment."""

import argparse
from collections import Counter
from collections.abc import Sequence

import pygame as pg

from config.Globals import Config
from display.Setup import EXPORT_DIR, SetupScreen
from tools.Brain import ACTIONS
from tools.Census import FAMILIES, SWATCHES, bucket
from tools.Exchange import Imported, export_bitzkeys
from tools.Genome import BODY_TRAITS, MIND_TRAITS, body_word, mind_word
from tools.Insights import ANIMALS, brain_report
from tools.Simulation import SPECIES, Bitzkey, Simulation

REPORT_DIR = Config.APP_DIR / "reports"
RING_WIDTH = 2
DOT_RADIUS_RANGE = (1, 5)  # dot radius in pixels at size 0 and size 15; bigger cells raise the top


class BitzkeyApp:
    def __init__(self, width: int, height: int, population: int, grass_density: int, seed: int | None,
                 imported: Sequence[Imported] = ()):
        self.screen = open_window()
        self.font = pg.font.SysFont("segoeui", 15)
        self.bold = pg.font.SysFont("segoeui", 22, bold=True)
        self.mono = pg.font.SysFont("consolas", 14)
        self.mono_small = pg.font.SysFont("consolas", 12)
        self.small = pg.font.SysFont("segoeui", 12)
        self.width = width
        self.height = height
        self.population = population
        self.grass_density = grass_density
        self.seed = seed
        self.imported = list(imported)
        self.simulation = self._new_world()
        self.selected_id: int | None = None
        self.show_insights = False
        self.show_census = False
        self.census_part = "ring"
        self.census_selection: set[str] = set()
        self._census_cache: tuple[tuple | None, list] = (None, [])
        self.message = ""
        self.message_until = 0
        self.playing = True
        self.clock = pg.time.Clock()
        self.elapsed = 0.0

    def run(self) -> None:
        running = True
        while running:
            self.elapsed += self.clock.tick(30) / 1000
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.VIDEORESIZE:
                    self.screen = pg.display.set_mode((max(860, event.w), max(760, event.h)), pg.RESIZABLE)
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        if self.show_census:
                            self.show_census = False
                        else:
                            running = False
                    elif event.key == pg.K_SPACE:
                        self.playing = not self.playing
                    elif event.key == pg.K_RIGHT:
                        self.playing = False
                        self.simulation.advance()
                    elif event.key == pg.K_LEFT:
                        self.playing = False
                        self.simulation.rewind()
                    elif event.key == pg.K_r:
                        self.simulation = self._new_world()
                        self.selected_id = None
                    elif event.key == pg.K_i:
                        self.show_insights = not self.show_insights
                    elif event.key == pg.K_b:
                        self._save_report()
                    elif event.key == pg.K_g:
                        self.show_census = not self.show_census
                    elif event.key == pg.K_TAB and self.show_census:
                        self.census_part = "dot" if self.census_part == "ring" else "ring"
                        self.census_selection.clear()
                    elif event.key == pg.K_c:
                        if self.show_census:
                            self._export_census(to_file=False)
                        else:
                            self._copy_selected()
                    elif event.key == pg.K_s and self.show_census:
                        self._export_census(to_file=True)
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    if self.show_census:
                        self._census_click(event.pos)
                    else:
                        self._select_at(event.pos)

            if self.playing and self.elapsed >= 0.24:
                self.simulation.advance()
                self.elapsed = 0
            self._draw()
            pg.display.flip()
        pg.quit()

    def _select_at(self, position: tuple[int, int]) -> None:
        board, cell_width, cell_height = self._board_geometry()
        nearest = None
        nearest_distance = float("inf")
        for agent in self.simulation.agents:
            point = (board.left + (agent.x + 0.5) * cell_width,
                     board.top + (agent.y + 0.5) * cell_height)
            distance = pg.Vector2(position).distance_to(point)
            if distance < nearest_distance:
                nearest = agent
                nearest_distance = distance
        if nearest and board.collidepoint(position) and nearest_distance < max(9, min(cell_width, cell_height) * 1.6):
            self.selected_id = nearest.identifier

    def _board_geometry(self) -> tuple[pg.Rect, float, float]:
        screen_width, screen_height = self.screen.get_size()
        sidebar_width = min(340, max(290, screen_width // 3))
        board_area = pg.Rect(0, 62, screen_width - sidebar_width, screen_height - 140)
        scale = min((board_area.width - 42) / self.width, (board_area.height - 42) / self.height)
        board_width = scale * self.width
        board_height = scale * self.height
        board = pg.Rect(0, 0, round(board_width), round(board_height))
        board.center = board_area.center
        return board, board_width / self.width, board_height / self.height

    def _draw(self) -> None:
        self.screen.fill((19, 26, 27))
        self._draw_header()
        self._draw_world()
        self._draw_sidebar()
        self._draw_footer()
        if self.show_census:
            self._draw_census()

    def _draw_header(self) -> None:
        pg.draw.rect(self.screen, (27, 37, 37), (0, 0, self.screen.get_width(), 62))
        self._text("BITZKEY", (24, 11), (229, 239, 220), self.bold)
        self._text("ARTIFICIAL ECOLOGY  /  RESEARCH BUILD", (139, 20), (144, 170, 151), self.small)
        status = "RUNNING" if self.playing else "PAUSED"
        status_color = (131, 205, 143) if self.playing else (238, 183, 110)
        label = self.font.render(status, True, status_color)
        self.screen.blit(label, (self.screen.get_width() - label.get_width() - 24, 20))

    def _draw_world(self) -> None:
        board, cell_width, cell_height = self._board_geometry()
        pg.draw.rect(self.screen, (29, 39, 38), board)
        for x, y in self.simulation.grass:
            point = (round(board.left + (x + 0.5) * cell_width),
                     round(board.top + (y + 0.5) * cell_height))
            pg.draw.circle(self.screen, (117, 166, 111), point, max(1, round(min(cell_width, cell_height) * 0.18)))
        largest = max(DOT_RADIUS_RANGE[1], round(min(cell_width, cell_height) * 0.45))
        for agent in self.simulation.agents:
            point = (round(board.left + (agent.x + 0.5) * cell_width),
                     round(board.top + (agent.y + 0.5) * cell_height))
            radius = self._dot_radius(agent, largest)
            self._draw_bitzkey(agent, point, radius)
            if agent.identifier == self.selected_id:
                pg.draw.circle(self.screen, (245, 238, 206), point, radius + RING_WIDTH + 3, 1)
        for x in range(0, self.width + 1, 10):
            px = round(board.left + x * cell_width)
            pg.draw.line(self.screen, (46, 58, 54), (px, board.top), (px, board.bottom), 1)
        for y in range(0, self.height + 1, 10):
            py = round(board.top + y * cell_height)
            pg.draw.line(self.screen, (46, 58, 54), (board.left, py), (board.right, py), 1)
        pg.draw.rect(self.screen, (80, 99, 85), board, 1)

    def _draw_sidebar(self) -> None:
        sidebar_width = min(340, max(290, self.screen.get_width() // 3))
        sidebar_x = self.screen.get_width() - sidebar_width
        pg.draw.line(self.screen, (54, 68, 62), (sidebar_x, 62), (sidebar_x, self.screen.get_height() - 58), 1)
        x = sidebar_x + 20
        width = self.screen.get_width() - x - 20
        counts = Counter(agent.species for agent in self.simulation.living)
        self._text(f"TURN  {self.simulation.turn:05d}", (x, 78), (232, 236, 219), self.bold)
        self._text(f"{len(self.simulation.living)} ALIVE    {self.simulation.deaths} LOST",
                   (x, 108), (157, 177, 159), self.small)
        pg.draw.line(self.screen, (54, 68, 62), (x, 138), (x + width, 138), 1)
        self._text("POPULATION", (x, 153), (220, 229, 211), self.font)
        for index, species in enumerate(SPECIES):
            y = 184 + index * 23
            self._text(species.upper(), (x, y), (149, 170, 150), self.small)
            self._text(str(counts.get(species, 0)), (x + width - 32, y), (231, 236, 222), self.font)

        pg.draw.line(self.screen, (54, 68, 62), (x, 286), (x + width, 286), 1)
        if self.show_insights:
            self._draw_insights(x, width)
            return
        self._text("SELECTED BITZKEY", (x, 301), (220, 229, 211), self.font)
        agent = next((item for item in self.simulation.agents if item.identifier == self.selected_id), None)
        if agent is None:
            self._text("Click a dot to inspect its genome.", (x, 332), (146, 165, 149), self.small)
            self._text("Living: click its current position.", (x, 352), (116, 139, 124), self.small)
            return

        self._draw_bitzkey(agent, (x + 10, 343), self._dot_radius(agent))
        self._text(f"#{agent.identifier:03d}  {agent.species.upper()}", (x + 28, 332), (238, 238, 220), self.font)
        self._text(f"DOT / MIND   {mind_word(agent.dna):08X}", (x, 360), (211, 219, 201), self.mono)
        self._text(f"RING / BODY  {body_word(agent.dna):08X}", (x, 379), (211, 219, 201), self.mono)

        body_x = x + min(168, width // 2 + 18)
        self._text("MIND      born   now", (x, 406), (149, 170, 150), self.small)
        self._text("BODY", (body_x, 406), (149, 170, 150), self.small)
        for index, trait in enumerate(MIND_TRAITS):
            born, now = agent.inborn[trait], agent.mind[trait]
            color = ((175, 192, 172) if abs(now - born) < 0.5
                     else (131, 205, 143) if now > born else (238, 183, 110))
            self._text(f"{trait:<11}{born:>3}{now:>6.1f}", (x, 425 + index * 18), color, self.mono_small)
        for index, trait in enumerate(BODY_TRAITS[1:]):
            self._text(f"{trait:<9}{agent.body[trait]:>3}", (body_x, 425 + index * 18),
                       (175, 192, 172), self.mono_small)
        self._text(f"{'burn':<7}{agent.burn:>5.2f}", (body_x, 425 + 7 * 18), (175, 192, 172), self.mono_small)

        self._text(f"HUNGER  {agent.hunger:5.1f}%", (x, 574), (224, 177, 118), self.mono)
        self._bar(x, 594, width, agent.hunger / 100, (224, 150, 94))
        self._text(f"HEALTH  {max(0.0, agent.health):5.1f} / {agent.max_health:.0f}", (x, 606), (161, 207, 162),
                   self.mono)
        self._bar(x, 626, width, agent.health / agent.max_health, (124, 190, 131))
        self._text(f"AGE {agent.age}    GENERATION {agent.generation}    CHILDREN {agent.children}",
                   (x, 642), (145, 164, 146), self.small)
        self._text(f"LAST ACTION: {agent.last_action.upper()}", (x, 660), (145, 164, 146), self.small)
        if agent.odds:
            odds = sorted(agent.odds.items(), key=lambda item: -item[1])[:3]
            self._text("CHANCES  " + "   ".join(f"{action} {chance:.0%}" for action, chance in odds),
                       (x, 678), (175, 192, 172), self.small)
        self._text("C  copy this genome and brain as JSON", (x, 700), (116, 139, 124), self.small)

    @staticmethod
    def _dot_radius(agent: Bitzkey, largest: int = DOT_RADIUS_RANGE[1]) -> int:
        """The dot's radius shows the body's size: from the smallest radius at size 0 up to
        `largest` at size 15."""
        smallest = DOT_RADIUS_RANGE[0]
        return round(smallest + (largest - smallest) * int(agent.body["size"]) / 15)

    def _draw_bitzkey(self, agent: Bitzkey, point: tuple[int, int], radius: int) -> None:
        """The inner dot shows the mind word's colour; the ring around it shows the body word's."""
        dot, ring = agent.dot_color, agent.ring_color
        if not agent.alive:
            dot, ring = (tuple(channel // 3 for channel in dot), tuple(channel // 3 for channel in ring))
        pg.draw.circle(self.screen, ring, point, radius + RING_WIDTH)
        pg.draw.circle(self.screen, dot, point, radius)

    def _draw_insights(self, x: int, width: int) -> None:
        insights = self.simulation.insights
        self._text("BEHAVIOUR", (x, 301), (220, 229, 211), self.font)
        self._text("% of chances taken, recent turns", (x, 323), (116, 139, 124), self.small)
        columns = [x + 96 + index * (width - 96) // len(ANIMALS) for index in range(len(ANIMALS))]
        for column, species in zip(columns, ANIMALS):
            self._text(species[:4].upper(), (column, 347), (149, 170, 150), self.small)
        for row, action in enumerate(ACTIONS):
            y = 367 + row * 19
            self._text(action.upper(), (x, y), (149, 170, 150), self.small)
            for column, species in zip(columns, ANIMALS):
                share, _ = insights.rate(action, species)
                self._text("  -" if share is None else f"{share:>4.0%}", (column, y), (231, 236, 222), self.mono)

        blocks = (
            ("LEAST KEEN TO BREED", insights.extremes("breed", 3, lowest=True)),
            ("KEENEST TO BREED", insights.extremes("breed", 2, lowest=False)),
            ("LEAST KEEN TO FIGHT", insights.extremes("attack", 2, lowest=True)),
        )
        y = 367 + len(ACTIONS) * 19 + 6
        for title, groups in blocks:
            self._text(title, (x, y), (220, 229, 211), self.small)
            y += 19
            if not groups:
                self._text("not enough chances yet", (x, y), (116, 139, 124), self.small)
                y += 18
            for label, share, _ in groups:
                self._text(label, (x, y), (175, 192, 172), self.small)
                self._text(f"{share:>4.0%}", (x + width - 40, y), (231, 236, 222), self.mono)
                y += 18
            y += 8

    def _save_report(self) -> None:
        report = brain_report(self.simulation.turn, self.simulation.living, self.simulation.insights)
        REPORT_DIR.mkdir(exist_ok=True)
        path = REPORT_DIR / f"brain-report-turn-{self.simulation.turn:05d}.txt"
        path.write_text(report, encoding="utf-8")
        print(report)
        self._say(f"Brain report saved to reports/{path.name}")

    def _new_world(self) -> Simulation:
        return Simulation(self.width, self.height, self.population, self.seed, self.grass_density, self.imported)

    def _say(self, message: str) -> None:
        self.message = message
        self.message_until = pg.time.get_ticks() + 5000

    def _copy(self, text: str, what: str) -> None:
        try:
            pg.scrap.put_text(text)
        except pg.error as error:
            self._say(f"Couldn't copy {what} to the clipboard: {error}")
        else:
            self._say(f"Copied {what} to the clipboard.")

    def _copy_selected(self) -> None:
        agent = next((item for item in self.simulation.agents if item.identifier == self.selected_id), None)
        if agent is None:
            self._say("Click a Bitzkey first, then press C to copy its genome.")
        else:
            what = "genome and brain" if agent.brain else "genome"
            self._copy(export_bitzkeys([agent]), f"#{agent.identifier:03d}'s {what}")

    def _export_census(self, to_file: bool) -> None:
        """Export the living Bitzkeys in the chosen colour families, or all of them."""
        part, chosen = self.census_part, self.census_selection
        agents = [agent for agent in self.simulation.living
                  if not chosen or agent.colour_families[part] in chosen]
        if not agents:
            self._say("No living Bitzkeys have those colours.")
            return
        names = [name for name in FAMILIES if name in chosen]
        what = f"{len(agents)} Bitzkey{'s' if len(agents) != 1 else ''} ({part}: {', '.join(names) or 'all'})"
        text = export_bitzkeys(agents)
        if not to_file:
            self._copy(text, what)
            return
        EXPORT_DIR.mkdir(exist_ok=True)
        path = EXPORT_DIR / f"bitzkeys-turn-{self.simulation.turn:05d}-{part}-{'-'.join(names) or 'all'}.json"
        path.write_text(text + "\n", encoding="utf-8")
        self._say(f"Saved {what} to exports/{path.name}")

    def _draw_footer(self) -> None:
        height = self.screen.get_height()
        pg.draw.rect(self.screen, (27, 37, 37), (0, height - 52, self.screen.get_width(), 52))
        history = f"HISTORY  {self.simulation.history_index + 1}/{len(self.simulation.history)}"
        self._text(history, (22, height - 36), (158, 181, 160), self.small)
        if pg.time.get_ticks() < self.message_until:
            self._text(self.message, (185, height - 36), (238, 183, 110), self.small)
            return
        self._text("SPACE  pause/run   LEFT  rewind   RIGHT  step   I  insights   B  brain report   "
                   "G  colour graph   R  restart   ESC  quit", (185, height - 36), (187, 199, 181), self.small)

    def _census_layout(self) -> dict:
        width, height = self.screen.get_size()
        panel = pg.Rect(0, 0, min(920, width - 60), min(600, height - 100))
        panel.center = (width // 2, height // 2)
        chart = pg.Rect(panel.left + 64, panel.top + 88, panel.width - 92, panel.height - 230)
        save = pg.Rect(0, 0, 84, 30)
        save.topright = (panel.right - 24, panel.top + 18)
        copy = pg.Rect(0, 0, 84, 30)
        copy.topright = (save.left - 8, save.top)
        columns = 5
        legend = {name: pg.Rect(chart.left + (index % columns) * (chart.width // columns) - 4,
                                chart.bottom + 34 + (index // columns) * 22, chart.width // columns - 8, 20)
                  for index, name in enumerate(FAMILIES)}
        return {"panel": panel, "chart": chart, "copy": copy, "save": save, "legend": legend}

    def _census_click(self, position: tuple[int, int]) -> None:
        layout = self._census_layout()
        if layout["copy"].collidepoint(position):
            self._export_census(to_file=False)
        elif layout["save"].collidepoint(position):
            self._export_census(to_file=True)
        for name, rect in layout["legend"].items():
            if rect.collidepoint(position):
                self.census_selection ^= {name}

    def _draw_census(self) -> None:
        """A popup over the world: living Bitzkeys per colour family, as stacked bars over time."""
        width, height = self.screen.get_size()
        shade = pg.Surface((width, height), pg.SRCALPHA)
        shade.fill((8, 12, 12, 175))
        self.screen.blit(shade, (0, 0))
        layout = self._census_layout()
        panel, chart = layout["panel"], layout["chart"]
        pg.draw.rect(self.screen, (30, 40, 40), panel, border_radius=6)
        pg.draw.rect(self.screen, (80, 99, 85), panel, 1, border_radius=6)
        part = "RING / BODY" if self.census_part == "ring" else "DOT / MIND"
        self._text(f"COLOUR CENSUS  /  {part}", (panel.left + 24, panel.top + 18), (229, 239, 220), self.bold)
        self._text("Living Bitzkeys by colour family over time.     TAB  ring / dot     G or ESC  close",
                   (panel.left + 24, panel.top + 52), (144, 170, 151), self.small)
        for name, label in (("copy", "COPY"), ("save", "SAVE")):
            pg.draw.rect(self.screen, (44, 57, 52), layout[name], border_radius=4)
            text = self.font.render(label, True, (220, 229, 211))
            self.screen.blit(text, text.get_rect(center=layout[name].center))
        if pg.time.get_ticks() < self.message_until:
            message = self.small.render(self.message, True, (238, 183, 110))
            self.screen.blit(message, message.get_rect(topright=(panel.right - 24, panel.top + 56)))

        census = self.simulation.census[:self.simulation.turn + 1]
        key = (self.simulation.turn, len(census), self.census_part, chart.width)
        if self._census_cache[0] != key:
            self._census_cache = (key, bucket(census, self.census_part, chart.width // 6))
        bars = self._census_cache[1]
        peak = self._nice_ceiling(max(sum(counts.values()) for _, counts in bars))
        for fraction in (0, 0.5, 1):
            y = round(chart.bottom - fraction * chart.height)
            pg.draw.line(self.screen, (54, 68, 62), (chart.left, y), (chart.right, y), 1)
            label = self.small.render(f"{peak * fraction:g}", True, (149, 170, 150))
            self.screen.blit(label, (chart.left - label.get_width() - 8, y - label.get_height() // 2))

        bar_width = chart.width / len(bars)
        gap = 1 if bar_width > 3 else 0
        for index, (_, counts) in enumerate(bars):
            left, right = round(chart.left + index * bar_width), round(chart.left + (index + 1) * bar_width) - gap
            bottom = float(chart.bottom)
            for name in FAMILIES:
                top = bottom - counts.get(name, 0) / peak * chart.height
                if round(top) < round(bottom):
                    pg.draw.rect(self.screen, SWATCHES[name],
                                 (left, round(top), max(1, right - left), round(bottom) - round(top)))
                bottom = top

        middle = bars[len(bars) // 2][0]
        for turn, x, align in ((0, chart.left, 0), (middle, chart.left + chart.width // 2, 0.5),
                               (self.simulation.turn, chart.right, 1)):
            label = self.small.render(f"TURN {turn}", True, (149, 170, 150))
            self.screen.blit(label, (x - round(label.get_width() * align), chart.bottom + 6))

        latest = census[-1][self.census_part]
        for name, rect in layout["legend"].items():
            chosen = name in self.census_selection
            if chosen:
                pg.draw.rect(self.screen, (131, 205, 143), rect, 1, border_radius=3)
            pg.draw.rect(self.screen, SWATCHES[name], (rect.left + 4, rect.top + 4, 12, 12))
            pg.draw.rect(self.screen, (80, 99, 85), (rect.left + 4, rect.top + 4, 12, 12), 1)
            color = (232, 236, 219) if chosen else (116, 139, 124) if self.census_selection else (187, 199, 181)
            self._text(f"{name:<8}{latest.get(name, 0):>4}", (rect.left + 24, rect.top + 2), color, self.mono_small)

        self._text("Click colours to choose who to export; with none chosen, every living Bitzkey is.     "
                   "C  copy     S  save to exports/", (panel.left + 24, panel.bottom - 28), (144, 170, 151),
                   self.small)

    @staticmethod
    def _nice_ceiling(value: float) -> float:
        """The smallest 1, 2 or 5 times a power of ten at or above the value."""
        scale = 1
        while True:
            for step in (1, 2, 5):
                if step * scale >= value:
                    return step * scale
            scale *= 10

    def _bar(self, x: int, y: int, width: int, value: float, color: tuple[int, int, int]) -> None:
        pg.draw.rect(self.screen, (44, 57, 52), (x, y, width, 7))
        pg.draw.rect(self.screen, color, (x, y, round(width * max(0, min(1, value))), 7))

    def _text(self, text: str, position: tuple[int, int], color: tuple[int, int, int],
              font: pg.font.Font) -> None:
        self.screen.blit(font.render(text, True, color), position)


def open_window() -> pg.Surface:
    """Return the app window, opening it if the configuration panel has not already."""
    pg.init()
    screen = pg.display.get_surface()
    if screen is None:
        screen = pg.display.set_mode((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT), pg.RESIZABLE)
        pg.display.set_caption("Bitzkey | An artificial ecology")
    return screen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Bitzkey artificial ecology.")
    parser.add_argument("--width", type=int, default=Config.GRID_WIDTH, help="world width in cells (default: 100)")
    parser.add_argument("--height", type=int, default=Config.GRID_HEIGHT, help="world height in cells (default: 100)")
    parser.add_argument("--population", type=int, default=Config.INITIAL_POPULATION)
    parser.add_argument("--grass-density", type=int, default=Config.GRASS_DENSITY,
                        help="percent of cells with grass at the start (default: 5)")
    parser.add_argument("--seed", type=int, default=None, help="seed the world for repeatable experiments")
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    settings = SetupScreen(open_window(), vars(options)).run()
    if settings is None:
        pg.quit()
    else:
        BitzkeyApp(**settings).run()