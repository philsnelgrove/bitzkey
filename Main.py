"""PyGame interface for the Bitzkey ecology experiment."""

import argparse
from collections import Counter

import pygame as pg

from config.Globals import Config
from tools.Simulation import SPECIES, Simulation


class BitzkeyApp:
    def __init__(self, width: int, height: int, population: int, seed: int | None):
        pg.init()
        self.screen = pg.display.set_mode((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT), pg.RESIZABLE)
        pg.display.set_caption("Bitzkey | An artificial ecology")
        self.font = pg.font.SysFont("segoeui", 15)
        self.bold = pg.font.SysFont("segoeui", 22, bold=True)
        self.mono = pg.font.SysFont("consolas", 14)
        self.small = pg.font.SysFont("segoeui", 12)
        self.width = width
        self.height = height
        self.population = population
        self.seed = seed
        self.simulation = Simulation(width, height, population, seed)
        self.selected_id: int | None = None
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
        for x, y in self.simulation.food:
            point = (round(board.left + (x + 0.5) * cell_width),
                     round(board.top + (y + 0.5) * cell_height))
            pg.draw.circle(self.screen, (117, 166, 111), point, max(1, round(min(cell_width, cell_height) * 0.18)))
        for agent in self.simulation.agents:
            point = (round(board.left + (agent.x + 0.5) * cell_width),
                     round(board.top + (agent.y + 0.5) * cell_height))
            color = agent.color if agent.alive else tuple(channel // 3 for channel in agent.color)
            radius = max(2, min(5, round(min(cell_width, cell_height) * 0.38)))
            pg.draw.circle(self.screen, color, point, radius)
            if agent.identifier == self.selected_id:
                pg.draw.circle(self.screen, (245, 238, 206), point, radius + 3, 1)
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
        self._text(f"{len(self.simulation.living)} ALIVE    {len(self.simulation.agents) - len(self.simulation.living)} LOST",
                   (x, 108), (157, 177, 159), self.small)
        pg.draw.line(self.screen, (54, 68, 62), (x, 138), (x + width, 138), 1)
        self._text("POPULATION", (x, 153), (220, 229, 211), self.font)
        for index, species in enumerate(SPECIES):
            y = 184 + index * 23
            self._text(species.upper(), (x, y), (149, 170, 150), self.small)
            self._text(str(counts.get(species, 0)), (x + width - 32, y), (231, 236, 222), self.font)

        pg.draw.line(self.screen, (54, 68, 62), (x, 286), (x + width, 286), 1)
        self._text("SELECTED BITZKEY", (x, 301), (220, 229, 211), self.font)
        agent = next((item for item in self.simulation.agents if item.identifier == self.selected_id), None)
        if agent is None:
            self._text("Click a dot to inspect its genome.", (x, 332), (146, 165, 149), self.small)
            self._text("Living: click its current position.", (x, 352), (116, 139, 124), self.small)
            return

        pg.draw.circle(self.screen, agent.color, (x + 8, 382), 7)
        self._text(f"#{agent.identifier:03d}  {agent.species.upper()}", (x + 23, 372), (238, 238, 220), self.font)
        self._text(f"DNA / CMYK   {agent.dna:08X}", (x, 405), (211, 219, 201), self.mono)
        binary = f"{agent.dna:032b}"
        self._text(" ".join(binary[i:i + 4] for i in range(0, 32, 4)),
                   (x, 430), (151, 183, 154), self.mono)
        traits = agent.traits
        trait_lines = (
            f"metabolism   {traits['metabolism']:>2}",
            f"aggression   {traits['aggression']:>2} -> {self.simulation.aggression_score(agent):>4.1f}",
            f"constitution {traits['constitution']:>2}",
            f"fertility    {traits['fertility']:>2}",
            f"sight        {traits['sight']:>2}",
            f"social       {traits['social']:>2}",
            f"learning     {traits['learning']:>2}",
        )
        for index, line in enumerate(trait_lines):
            self._text(line, (x, 466 + index * 20), (175, 192, 172), self.mono)
        self._text(f"HUNGER  {agent.hunger:5.1f}%", (x, 620), (224, 177, 118), self.mono)
        self._bar(x, 642, width, agent.hunger / 100, (224, 150, 94))
        self._text(f"HEALTH  {agent.health:5.1f}%", (x, 658), (161, 207, 162), self.mono)
        self._bar(x, 680, width, agent.health / 100, (124, 190, 131))
        self._text(f"AGE {agent.age}    LAST ACTION: {agent.last_action.upper()}",
                   (x, 699), (145, 164, 146), self.small)

    def _draw_footer(self) -> None:
        height = self.screen.get_height()
        pg.draw.rect(self.screen, (27, 37, 37), (0, height - 52, self.screen.get_width(), 52))
        history = f"HISTORY  {self.simulation.history_index + 1}/{len(self.simulation.history)}"
        self._text(history, (22, height - 36), (158, 181, 160), self.small)
        self._text("SPACE  pause/run     LEFT  rewind     RIGHT  step     R  restart     ESC  quit",
                   (185, height - 36), (187, 199, 181), self.small)

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