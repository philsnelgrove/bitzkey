"""The app's sounds, made here rather than read from files, as the C++ version's.

The sound of a button pressed: a quarter-second click and beep. Every button's label is drawn through draw_label, which
notes where the button lies on the window; a left press on any button of the frame last shown plays it, whichever screen
is open. Screens fetch their events through events() and show their frames through show_frame() so that it can.

The Bitzkeys' calls: each of the five, and the alert, its own short voice, heard from the map when zoomed in close."""
from __future__ import annotations

import math
import struct

import pygame as pg

from tools.Brain import ALERT_CALL, CALLS, FIRST_CALL, SILENT

RATE = 44100
LENGTH = 0.25    # seconds
PITCH = 1600.0   # the beep's, in hertz
VOICES = 12      # sounding at once at most: the click, and the calls

_sound: pg.mixer.Sound | None = None
_calls: dict[int, pg.mixer.Sound] = {}  # each call's sound
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


def _sweep(start: float, end: float, share: float) -> float:
    return start * (end / start) ** share


def call_sound(call: int, rate: int = RATE) -> list[float]:
    """A call's sound, as samples at a rate, at full loudness; none for silence. The alert, a sharp double yelp; then
    the five: a rising chirp, a low hoot, a warbling trill, a croak, and a falling whistle. Each rises in at once and
    fades to nothing so none pops, as the C++ version's."""
    if not SILENT < call < len(CALLS):
        return []
    length = (0.0, 0.26, 0.14, 0.32, 0.26, 0.16, 0.22)[call]
    samples, phase, noise = [], 0.0, 0x2545F491 + call
    for index in range(int(length * rate)):
        time = index / rate
        part = time / length
        envelope, harmonics, rasp = 1.0, 0.0, 0.0
        if call == ALERT_CALL:  # two quick falling yelps
            half = part * 2 if part < 0.5 else part * 2 - 1
            pitch, envelope, harmonics = _sweep(2200.0, 1300.0, half), math.sin(math.pi * half), 0.25
        elif call == FIRST_CALL:  # a chirp, rising
            pitch, envelope = _sweep(1200.0, 2600.0, part), math.sin(math.pi * part)
        elif call == FIRST_CALL + 1:  # a hoot, low and wavering
            pitch = 380.0 * (1 + 0.03 * math.sin(2 * math.pi * 6 * time))
            envelope, harmonics = min(1.0, time / 0.04) * math.exp(-3 * part), 0.15
        elif call == FIRST_CALL + 2:  # a trill, warbling quickly
            pitch = 1800.0 * (1 + 0.06 * math.sin(2 * math.pi * 28 * time))
            envelope = math.sin(math.pi * part) * (0.55 + 0.45 * math.sin(2 * math.pi * 28 * time))
        elif call == FIRST_CALL + 3:  # a croak, rough and low
            pitch = _sweep(260.0, 200.0, part)
            envelope, harmonics, rasp = min(1.0, time / 0.01) * math.exp(-2.5 * part), 0.6, 0.35
        else:  # a whistle, falling
            pitch, envelope = _sweep(2600.0, 1300.0, part), math.sin(math.pi * part)
        phase += 2 * math.pi * pitch / rate
        noise = (noise * 1664525 + 1013904223) & 0xFFFFFFFF
        hiss = (noise >> 8) / (1 << 24) * 2 - 1
        tone = math.sin(phase) + harmonics * (math.sin(2 * phase) / 2 + math.sin(3 * phase) / 3)
        tone = tone / (1 + harmonics) * (1 - rasp) + rasp * hiss * abs(math.sin(phase))
        fade = min(1.0, time / 0.003, (length - time) / 0.01)
        samples.append(0.5 * tone * envelope * max(0.0, fade))
    return samples


def _made(samples: list[float], channels: int) -> pg.mixer.Sound:
    frames = [int(32767 * sample) for sample in samples]
    return pg.mixer.Sound(buffer=struct.pack(f"<{len(frames) * channels}h", *(frame for frame in frames for _ in range(channels))))


def start() -> None:
    """Open the sound; with no sound to be had, buttons and calls are silent."""
    global _sound
    if _sound is not None:
        return
    try:
        if not pg.mixer.get_init():
            pg.mixer.init(frequency=RATE, size=-16, channels=2)
        rate, _, channels = pg.mixer.get_init()
        pg.mixer.set_num_channels(VOICES)
        _sound = _made(click_sound(rate), channels)
        _calls.update({call: _made(call_sound(call, rate), channels) for call in range(SILENT + 1, len(CALLS))})
    except (pg.error, NotImplementedError):
        _sound = None
        _calls.clear()


def calls_heard(calls: list[tuple[int, float, float]]) -> None:
    """Calls heard on the map, each (which, how loud 0 to 1, where across the view -1 left to 1 right), played at once,
    the loudest first, as many as can sound together; panned with equal power, so each sounds as loud wherever it
    comes from."""
    if not _calls:
        return
    for call, loudness, pan in sorted(calls, key=lambda heard: -heard[1]):
        sound = _calls.get(call)
        if sound is None or loudness <= 0:
            continue
        channel = pg.mixer.find_channel()
        if channel is None:
            break
        angle = (max(-1.0, min(1.0, pan)) + 1) * math.pi / 4
        loudness = min(1.0, loudness)
        channel.play(sound)
        channel.set_volume(loudness * math.cos(angle), loudness * math.sin(angle))


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
