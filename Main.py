"""PyGame interface for the Bitzkey ecology experiment."""

import argparse
from collections import Counter
from pathlib import Path

import pygame as pg

from config.Globals import Config
from tools.Brain import ACTIONS
from tools.Genome import BODY_TRAITS, MIND_TRAITS, body_word, mind_word
from tools.Insights import ANIMALS, brain_report
from tools.Simulation import SPECIES, Bitzkey, Simulation

REPORT_DIR = Path(__file__).resolve().parent / "reports"
RING_WIDTH = 3


class BitzkeyApp:
    def __init__(self, width: int, height: int, population: int, seed: int | None):
        pg.init()
        self.screen = pg.display.set_mode((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT), pg.RESIZABLE)
        pg.display.set_caption("Bitzkey | An artificial ecology")
        self.font = pg.font.SysFont("segoeui", 15)
        self.bold = pg.font.SysFont("segoeui", 22, bold=True)
        self.mono = pg.font.SysFont("consolas", 14)
        self.mono_small = pg.font.SysFont("consolas", 12)
        self.small = pg.font.SysFont("segoeui", 12)
        self.width = width
        self.height = height
        self.population = population
        self.seed = seed
        self.simulation = Simulation(width, height, population, seed)
        self.selected_id: int | None = None
        self.show_insights = False
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
                        self.simulation = Simulation(self.width, self.height, self.population, self.seed)
                        self.selected_id = None
                    elif event.key == pg.K_i:
                        self.show_insights = not self.show_insights
                    elif event.key == pg.K_b:
                        self._save_report()
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
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
        for agent in self.simulation.agents:
            point = (round(board.left + (agent.x + 0.5) * cell_width),
                     round(board.top + (agent.y + 0.5) * cell_height))
            radius = self._dot_radius(agent, max(2, min(5, round(min(cell_width, cell_height) * 0.38))))
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

        self._draw_bitzkey(agent, (x + 10, 343), 4)
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

    @staticmethod
    def _dot_radius(agent: Bitzkey, base: int) -> int:
        """Bigger bodies have bigger dots."""
        return max(2, round(base * (0.8 + 0.8 * int(agent.body["size"]) / 15)))

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
        self.message = f"Brain report saved to reports/{path.name}"
        self.message_until = pg.time.get_ticks() + 5000

    def _draw_footer(self) -> None:
        height = self.screen.get_height()
        pg.draw.rect(self.screen, (27, 37, 37), (0, height - 52, self.screen.get_width(), 52))
        history = f"HISTORY  {self.simulation.history_index + 1}/{len(self.simulation.history)}"
        self._text(history, (22, height - 36), (158, 181, 160), self.small)
        if pg.time.get_ticks() < self.message_until:
            self._text(self.message, (185, height - 36), (238, 183, 110), self.small)
            return
        self._text("SPACE  pause/run    LEFT  rewind    RIGHT  step    I  insights    B  brain report    "
                   "R  restart    ESC  quit", (185, height - 36), (187, 199, 181), self.small)

    def _bar(self, x: int, y: int, width: int, value: float, color: tuple[int, int, int]) -> None:
        pg.draw.rect(self.screen, (44, 57, 52), (x, y, width, 7))
        pg.draw.rect(self.screen, color, (x, y, round(width * max(0, min(1, value))), 7))

    def _text(self, text: str, position: tuple[int, int], color: tuple[int, int, int],
              font: pg.font.Font) -> None:
        self.screen.blit(font.render(text, True, color), position)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Bitzkey artificial ecology.")
    parser.add_argument("--width", type=int, default=Config.GRID_WIDTH, help="world width in cells (default: 100)")
    parser.add_argument("--height", type=int, default=Config.GRID_HEIGHT, help="world height in cells (default: 100)")
    parser.add_argument("--population", type=int, default=Config.INITIAL_POPULATION)
    parser.add_argument("--seed", type=int, default=None, help="seed the world for repeatable experiments")
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    BitzkeyApp(options.width, options.height, options.population, options.seed).run()