"""Scaling for text and controls. Screens are laid out for a 1200x820 window; on any other
screen every length is multiplied by one factor, and fonts are rendered at the scaled size so
text stays crisp rather than being stretched.

Fonts translate as they draw: whatever English is rendered or measured with them is shown in the
chosen language (display.Language), in a font with that language's script."""

from __future__ import annotations

import os

import pygame as pg

from display import Sound
from display.Language import AsShown, as_shown, language, tr, visual_order

DESIGN_SIZE = (1200, 820)
MIN_SCALE = 0.8

# The fonts for the scripts Segoe UI and Consolas lack: Microsoft YaHei for Chinese, Yu Gothic for
# Japanese, Nirmala UI for Hindi, and Courier New for monospaced Hebrew.
SCRIPT_FONTS = {"zh": ("msyh.ttc", "msyhbd.ttc"), "ja": ("YuGothM.ttc", "YuGothB.ttc"), "hi": ("Nirmala.ttc", "Nirmala.ttc")}


class Font(pg.font.Font):
    """A font that draws and measures text in the chosen language."""

    key: tuple = ()

    def render(self, text, antialias, color, bgcolor=None, wraplength=0):  # type: ignore[override]
        shown = visual_order(tr(text)) if isinstance(text, str) else text
        return super().render(shown, antialias, color, bgcolor, wraplength)

    def size(self, text):  # type: ignore[override]
        return super().size(tr(text) if isinstance(text, str) else text)


def _font_path(name: str, bold: bool, code: str) -> tuple[str | None, bool]:
    """The font file for a face in a language, and whether it must be made bold."""
    folder = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    if code in SCRIPT_FONTS:
        regular, heavy = SCRIPT_FONTS[code]
        path = os.path.join(folder, heavy if bold else regular)
        if os.path.exists(path):
            return path, bold and heavy == regular
    if code == "he" and name == "consolas":
        name = "couriernew"
    path = pg.font.match_font(name, bold=bold)
    return path, bold and path == pg.font.match_font(name)


class Ui:
    def __init__(self) -> None:
        self.scale = 1.0
        self._fonts: dict[tuple, Font] = {}

    def fit(self, size: tuple[int, int]) -> None:
        """Scale to the screen: as large as fits both its width and height, never below MIN_SCALE."""
        self.scale = max(MIN_SCALE, min(size[0] / DESIGN_SIZE[0], size[1] / DESIGN_SIZE[1]))

    text_size = 1.0  # on top of the scale, for those who need larger text: 1, 1.25, 1.5 or 2 (F9)

    def next_text_size(self) -> None:
        self.text_size = 1.25 if self.text_size < 1.2 else 1.5 if self.text_size < 1.4 else 2.0 if self.text_size < 1.9 else 1.0

    def __call__(self, length: float) -> int:
        """A length from the 1200x820 design, in pixels at the current scale."""
        return round(length * self.scale)

    def font(self, name: str, size: int, bold: bool = False, code: str = "") -> Font:
        """A system font at `size` (design pixels), scaled; fonts are cached by their real size. In
        the chosen language's script, or in `code`'s when one is given."""
        return self._open((name, max(6, round(size * self.scale * self.text_size)), bold, code or language()))

    def fitting(self, font: pg.font.Font, text: str, width: int) -> pg.font.Font:
        """The same font, smaller if it must be for the text (in the language shown) to fit `width`:
        down to two thirds of its size. Longer words in other languages fit the places made for English."""
        key = getattr(font, "key", ())
        if not key:
            return font
        name, pixels, bold, code = key
        shown = tr(text)
        chosen = font
        with AsShown():
            for size in range(pixels, max(6, pixels * 2 // 3) - 1, -1):
                chosen = self._open((name, size, bold, code))
                if chosen.size(shown)[0] <= width:
                    break
        return chosen

    def _open(self, key: tuple) -> Font:
        if key not in self._fonts:
            name, pixels, bold, code = key
            path, embolden = _font_path(name, bold, code)
            font = Font(path, pixels)
            font.key = key
            if embolden:
                font.bold = True
            if code == "hi":
                font.set_script("Deva")  # Devanagari, shaped by HarfBuzz
            self._fonts[key] = font
        return self._fonts[key]

    def close_fonts(self) -> None:
        self._fonts.clear()


def draw_label(surface: pg.Surface, font: pg.font.Font, text: str, color, button: pg.Rect) -> None:
    """A button's label, centred in it, and smaller if it must be to fit."""
    Sound.button_drawn(surface, button)  # a press on it is heard
    fitting = UI.fitting(font, text, button.width - UI(10))
    image = fitting.render(text, True, color)
    surface.blit(image, image.get_rect(center=button.center))


def wrapped_lines(font: pg.font.Font, text: str, width: int) -> list[str]:
    """Text in the language shown, broken into lines no wider than `width`: between words, or
    between any two characters in Chinese and Japanese."""
    shown = tr(text)
    spaced = language() not in ("zh", "ja")
    pieces = shown.split(" ") if spaced else list(shown)
    joiner = " " if spaced else ""
    lines: list[str] = []
    line = ""
    with AsShown():
        for piece in pieces:
            trial = piece if not line else line + joiner + piece
            if line and font.size(trial)[0] > width:
                lines.append(line)
                line = piece
            else:
                line = trial
    if line:
        lines.append(line)
    return [as_shown(each) for each in lines]


def fit_text(font: pg.font.Font, text: str, width: int) -> str:
    """Text in the language shown, cut short with an ellipsis if it is wider than `width`."""
    shown = tr(text)
    with AsShown():
        if font.size(shown)[0] <= width:
            return as_shown(shown)
        while shown and font.size(shown + "...")[0] > width:
            shown = shown[:-1]
    return as_shown(shown.rstrip() + "...")


UI = Ui()
