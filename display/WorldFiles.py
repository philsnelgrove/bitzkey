"""Choosing where to save a world, and which saved world to load, with the system's file picker."""

from __future__ import annotations

from pathlib import Path

from config.Globals import Config
from tools.WorldFile import WORLD_SUFFIX

WORLD_DIR = Config.APP_DIR / "worlds"
FILE_TYPES = [("Bitzkey worlds", f"*{WORLD_SUFFIX}"), ("All files", "*.*")]


def _picker():
    """The file picker's module and a hidden window for it to belong to; None without one."""
    try:
        import tkinter
        from tkinter import filedialog
    except ImportError:
        return None
    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # in front of the game's window
    return filedialog, root


def choose_save_path(name: str) -> Path | None:
    """Where to save a world, starting in worlds/ with `name` offered; worlds/`name` itself when there is no
    file picker; None if the user cancels."""
    WORLD_DIR.mkdir(parents=True, exist_ok=True)
    picker = _picker()
    if picker is None:
        return WORLD_DIR / name
    filedialog, root = picker
    try:
        path = filedialog.asksaveasfilename(title="Save World", initialdir=str(WORLD_DIR), initialfile=name,
                                            defaultextension=WORLD_SUFFIX, filetypes=FILE_TYPES, parent=root)
    finally:
        root.destroy()
    return Path(path) if path else None


def choose_world_file() -> Path | None:
    """A saved world to load, starting in worlds/; None if the user cancels or there is no file picker."""
    picker = _picker()
    if picker is None:
        return None
    filedialog, root = picker
    try:
        path = filedialog.askopenfilename(title="Load World", initialdir=str(WORLD_DIR) if WORLD_DIR.exists() else None,
                                          filetypes=FILE_TYPES, parent=root)
    finally:
        root.destroy()
    return Path(path) if path else None
