"""The sound of a button pressed: a quarter-second click and beep, made here rather than read from a file, as the C++
version's. Every button's label is drawn through draw_label, which notes where the button lies on the window; a left
press on any button of the frame last shown plays it, whichever screen is open. Screens fetch their events through
events() and show their frames through show_frame() so that it can."""
from __future__ import annotations

import math
import struct

import pygame as pg

RATE = 44100
LENGTH = 0.25    # seconds
PITCH = 1600.0   # the beep's, in hertz

_sound: pg.mixer.Sound | None = None
_drawing: list[pg.Rect] = []  # the buttons of the frame being drawn
_shown: list[pg.Rect] = []    # and of the one last shown


def click_sound(rate: int = RATE) -> list[float]:
    """A click, a few milliseconds of a sharp falling edge with a little noise, under a beep that sounds at once and
    dies away, the whole fading to nothing by its end so it doesn't pop."""
    samples, noise = [], 0x9E3779B9
    for index in range(int(LENGTH * rate)):
        time = index / rate
        noise = (noise * 1664525 + 1013904223) & 0xFFFFFFFF
        hiss = (noise >> 8) / (1 << 24) * 2 - 1
        click = math.exp(-time / 0.0015) * (0.7 + 0.3 * hiss)
        beep = min(1.0, time / 0.002) * math.exp(-time / 0.045) * math.sin(2 * math.pi * PITCH * time)
        fade = min(1.0, (LENGTH - time) / 0.02)
        samples.append(0.3 * (0.6 * click + beep) * fade)
    return samples


def start() -> None:
    """Open the sound; with no sound to be had, buttons are silent."""
    global _sound
    if _sound is not None:
        return
    try:
        if not pg.mixer.get_init():
            pg.mixer.init(frequency=RATE, size=-16, channels=1)
        rate, _, channels = pg.mixer.get_init()
        frames = [int(32767 * sample) for sample in click_sound(rate)]
        _sound = pg.mixer.Sound(buffer=struct.pack(f"<{len(frames) * channels}h", *(frame for frame in frames for _ in range(channels))))
    except (pg.error, NotImplementedError):
        _sound = None


def button_drawn(surface: pg.Surface, button: pg.Rect) -> None:
    """A button drawn: heard, if drawn on the window."""
    if surface is pg.display.get_surface():
        _drawing.append(pg.Rect(button))


def show_frame() -> None:
    """Show the frame drawn, its buttons those a press is heard on until the next."""
    global _shown, _drawing
    pg.display.flip()
    _shown, _drawing = _drawing, []


def events() -> list[pg.event.Event]:
    """The events waiting, as pg.event.get(), a left press on a button shown sounding its click."""
    waiting = pg.event.get()
    for event in waiting:
        if (event.type == pg.MOUSEBUTTONDOWN and event.button == 1 and _sound is not None
                and any(button.collidepoint(event.pos) for button in _shown)):
            _sound.stop()  # a quick second press starts it afresh
            _sound.play()
    return waiting
