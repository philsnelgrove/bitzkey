"""The app's window: fullscreen at the desktop's resolution by default; F11 switches to a
resizable window and back."""

from __future__ import annotations

import pygame as pg

from config.Globals import Config
from display import Sound
from display.Ui import UI

MIN_SIZE = (860, 760)
_fullscreen = Config.FULLSCREEN


def open_window(fullscreen: bool | None = None) -> pg.Surface:
    """Return the app window, opening it if it isn't open yet (fullscreen unless told otherwise)."""
    pg.init()
    Sound.start()  # a button pressed clicks
    screen = pg.display.get_surface()
    if screen is None:
        screen = _show(_fullscreen if fullscreen is None else fullscreen)
        pg.display.set_caption("Bitzkey | An artificial ecology")
    return screen


def is_fullscreen() -> bool:
    return _fullscreen


def toggle_fullscreen() -> pg.Surface:
    return _show(not _fullscreen)


def resize(event: pg.event.Event) -> pg.Surface:
    """Follow the user resizing the window, down to a minimum size. Fullscreen keeps the
    desktop's size."""
    if _fullscreen:
        return pg.display.get_surface()
    screen = pg.display.set_mode((max(MIN_SIZE[0], event.w), max(MIN_SIZE[1], event.h)), pg.RESIZABLE)
    UI.fit(screen.get_size())
    return screen


def _show(fullscreen: bool) -> pg.Surface:
    global _fullscreen
    _fullscreen = fullscreen
    if fullscreen:
        screen = pg.display.set_mode((0, 0), pg.FULLSCREEN)
    else:
        screen = pg.display.set_mode((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT), pg.RESIZABLE)
    UI.fit(screen.get_size())
    return screen
