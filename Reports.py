"""Bitzkey Reports: a whole world's report, from a save or a running world's probe.

    Reports.py [WORLD.bzworld | --probe PORT | --probe HOST:PORT] [--fullscreen] [--text FILE]

With --text, no window: the report is written to FILE as text, and it ends."""

import argparse
import os
import sys
from pathlib import Path

import pygame as pg

from display import Window
from display.Language import start_language
from tools import WorldReport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report on a whole Bitzkey world, from a save or a probe.")
    parser.add_argument("save", nargs="?", help="a saved world (.bzworld) to report on")
    parser.add_argument("--probe", help="a running world's probe: its port, or host:port")
    parser.add_argument("--fullscreen", action="store_true", help="open fullscreen rather than in a window")
    parser.add_argument("--text", help="write the report to this file as text, with no window")
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    if options.text:
        if bool(options.save) == bool(options.probe):
            sys.exit("--text reports on a save or a --probe: give one")
        report = WorldReport.read_save(options.save) if options.save else WorldReport.Probe(options.probe).read()
        Path(options.text).write_text(WorldReport.as_text(WorldReport.analyse(report)), encoding="utf-8")
        return
    from display.ReportPanel import ReportPanel  # the window's part only when there is a window

    screen = Window.open_window(fullscreen=options.fullscreen)
    pg.display.set_caption("Bitzkey Reports")
    start_language()  # the app's fonts for its language; the reports themselves are in English
    panel = ReportPanel(screen)
    if options.save:
        panel.open_save(Path(options.save))
    elif options.probe:
        panel.connect(options.probe)
    panel.run()
    pg.quit()


if __name__ == "__main__":
    main()
    if sys.stdout:
        sys.stdout.flush()
    os._exit(0)
