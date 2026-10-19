"""Languages. Every piece of text is written in English and translated where it is drawn: the
catalogs, i18n/<code>.json, map the English to each language. They are the C++ and web versions'
catalogs and work the same way: a key is the English exactly as shown, except that numbers can
stand as {#} and other changing words as {*}; a translation puts them back in order, or in another
order with {#1}, {*2}, ... Keys in capitals are shown in capitals.

BITZKEY_I18N_COLLECT=<file> appends every piece of text with no translation to that file, once."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from config.Globals import Config

LANGUAGES = [("en", "English"), ("fr", "Français"), ("it", "Italiano"), ("es", "Español"), ("de", "Deutsch"),
             ("pt", "Português"), ("pl", "Polski"), ("uk", "Українська"), ("zh", "简体中文"), ("ja", "日本語"),
             ("hi", "हिन्दी"), ("he", "עברית")]
NAMES = {name for _, name in LANGUAGES}

NUMBER = r"[-+]?\d+(?:[.,]\d+)*"
# Not words: genome codes (hex), file names, dates and times.
NOT_WORDS = re.compile(r"^(?:[\s0-9A-Fa-f#:.\-]*\d[\s0-9A-Fa-f#:.\-]*|[\w.\-]+\.(?:txt|json|bkw|bzw|csv))$")
BLANK = re.compile(r"\{([#*])(\d*)\}")
HEBREW = re.compile("[֐-׿יִ-ﭏ]")
CATALOGS = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)) / "i18n"
SETTINGS = Config.APP_DIR / "settings.json"


class _State:
    def __init__(self) -> None:
        self.language = "en"
        self.exact: dict[str, str] = {}
        self.spaced: dict[str, str] = {}   # key with its spaces run together -> key
        self.folded: dict[str, str] = {}   # lowercased plain key -> translation
        self.patterns: list[tuple[re.Pattern, str, str, bool]] = []  # regex, kinds, translation, capitals
        self.cache: dict[str, str] = {}
        self.shown: set[str] = set()       # translations, and text made from them: drawn as they are
        self.as_shown = 0                  # while above 0, nothing is translated
        self.missing: set[str] = set()
        self.collect = os.environ.get("BITZKEY_I18N_COLLECT", "")


_state = _State()


def language() -> str:
    return _state.language


def _capitals(text: str) -> bool:
    return bool(re.search("[A-Z]", text)) and not re.search("[a-z]", text)


def _fill(translation: str, kinds: str, values: list[str]) -> str:
    lists = {"#": [v for k, v in zip(kinds, values) if k == "#"], "*": [v for k, v in zip(kinds, values) if k == "*"]}
    counters = {"#": 0, "*": 0}

    def put(match: re.Match) -> str:
        kind, number = match.group(1), match.group(2)
        if number:
            index = int(number) - 1
        else:
            index = counters[kind]
            counters[kind] += 1
        found = lists[kind]
        return found[index] if index < len(found) else match.group(0)
    return BLANK.sub(put, translation)


def _note_missing(text: str) -> None:
    if not _state.collect or text in _state.missing:
        return
    _state.missing.add(text)
    with open(_state.collect, "a", encoding="utf8") as file:
        file.write(text + "\n")


def _lookup(trimmed: str) -> str | None:
    s = _state
    if trimmed in s.exact:
        return s.exact[trimmed].upper() if _capitals(trimmed) else s.exact[trimmed]
    if trimmed.lower() in s.folded:  # the same words in another case: "grazers" from the key "GRAZERS"
        translation = s.folded[trimmed.lower()]
        if _capitals(trimmed):
            return translation.upper()
        if trimmed[0].islower() and s.language != "de":
            return translation[:1].lower() + translation[1:]
        return translation
    values = re.findall(NUMBER, trimmed)
    generic = re.sub(NUMBER, "{#}", trimmed)
    if values:
        key = generic if generic in s.exact else s.spaced.get(re.sub(r"\s+", " ", generic))
        if key:
            shown = _fill(s.exact[key], "#" * len(values), values)
            return shown.upper() if _capitals(key.replace("{#}", "")) else shown
    for regex, kinds, translation, capitals in s.patterns:
        found = regex.match(trimmed)
        if found:
            parts = [tr(part) if kind == "*" else part for kind, part in zip(kinds, found.groups())]
            shown = _fill(translation, kinds, parts)
            return shown.upper() if capitals else shown
    _note_missing(generic if values else trimmed)
    return None


def _load(code: str) -> None:
    s = _state
    s.exact, s.spaced, s.folded, s.patterns = {}, {}, {}, []
    s.cache.clear()
    s.shown.clear()
    s.language = "en"
    if code == "en":
        return
    try:
        catalog = json.loads((CATALOGS / f"{code}.json").read_text(encoding="utf8"))
    except (OSError, ValueError):
        return  # no catalog: English, rather than broken text
    s.language = code
    patterns = []
    for key, translation in catalog.items():
        if not translation:
            continue
        words, numbers = "{*}" in key, "{#}" in key
        if words:
            kinds = "".join(part[1] for part in re.findall(r"\{[#*]\}", key))
            source = "".join(f"({NUMBER})" if part == "{#}" else "(.+?)" if part == "{*}" else re.escape(part)
                             for part in re.split(r"(\{#\}|\{\*\})", key))
            patterns.append((len(key), re.compile(f"^{source}$", re.S), kinds, translation,
                             _capitals(re.sub(r"\{[#*]\}", "", key))))
        else:
            s.exact[key] = translation
        if not words and not numbers:
            s.folded.setdefault(key.lower(), translation)
        if numbers and not words:
            s.spaced[re.sub(r"\s+", " ", key)] = key
    patterns.sort(key=lambda pattern: -pattern[0])  # longer patterns first, so the most specific wins
    s.patterns = [pattern[1:] for pattern in patterns]


def start_language() -> None:
    """The language saved last time, else the system's."""
    code = ""
    try:
        code = json.loads(SETTINGS.read_text(encoding="utf8")).get("language", "")
    except (OSError, ValueError, AttributeError):
        pass
    if not code:
        try:
            import pygame as pg
            for locale in pg.system.get_pref_locales():
                if not code and any(known == locale["language"] for known, _ in LANGUAGES):
                    code = locale["language"]
        except Exception:  # an older pygame, or none of its locales known
            pass
    set_language(code or "en", remember=False)


def set_language(code: str, remember: bool = True) -> None:
    """Switch now, and remember it."""
    from display.Ui import UI
    _load(code)
    UI.close_fonts()  # other languages draw with other fonts
    if not remember:
        return
    try:
        saved = json.loads(SETTINGS.read_text(encoding="utf8"))
    except (OSError, ValueError):
        saved = {}
    saved["language"] = code
    try:
        SETTINGS.write_text(json.dumps(saved, indent=2), encoding="utf8")
    except OSError:
        pass


def tr(text: str) -> str:
    """The text in the chosen language."""
    s = _state
    if not isinstance(text, str) or s.as_shown or (s.language == "en" and not s.collect) or text in s.shown \
            or text in NAMES:
        return text
    trimmed = text.strip()
    if len(trimmed) < 2 or not re.search(r"[^\W\d_]", trimmed) or NOT_WORDS.match(trimmed):
        return text
    if text in s.cache:
        return s.cache[text]
    shown = text
    if s.language == "en":
        _note_missing(re.sub(NUMBER, "{#}", trimmed))
    else:
        result = _lookup(trimmed)
        if result is not None:
            start = text.index(trimmed)
            shown = text[:start] + result + text[start + len(trimmed):]
    if len(s.cache) > 20000:
        s.cache.clear()
        s.shown.clear()
    s.cache[text] = shown
    if shown != text:
        s.shown.add(shown)
    return shown


def as_shown(text: str) -> str:
    """Text already in the chosen language, drawn as it is: shortened to fit, say."""
    if _state.language != "en":
        _state.shown.add(text)
    return text


class AsShown:
    """While one is open, nothing is translated: for measuring shown text a piece at a time."""

    def __enter__(self) -> AsShown:
        _state.as_shown += 1
        return self

    def __exit__(self, *_) -> None:
        _state.as_shown -= 1


def visual_order(text: str) -> str:
    """Hebrew is written right to left: the text in the order its characters are drawn, left to
    right. A plain form of the Unicode bidirectional algorithm, enough for one line of the app's
    text: runs of Hebrew read right to left, runs of other letters and numbers left to right, and
    the spaces and punctuation between them take the direction of their neighbours."""
    if not HEBREW.search(text):
        return text
    rtl_base = _state.language == "he"
    base = 1 if rtl_base else 0
    directions = [1 if HEBREW.match(c) else -1 if (ord(c) < 0x80 and not c.isalnum() and c not in "#%") else 0 for c in text]
    index = 0
    while index < len(text):
        if directions[index] != -1:
            index += 1
            continue
        end = index
        while end < len(text) and directions[end] == -1:
            end += 1
        before = directions[index - 1] if index > 0 else base
        after = directions[end] if end < len(text) else base
        for at in range(index, end):
            directions[at] = before if before == after else base
        index = end
    runs: list[list] = []
    for char, direction in zip(text, directions):
        if not runs or runs[-1][0] != direction:
            runs.append([direction, ""])
        runs[-1][1] += char
    mirror = str.maketrans("()[]{}<>", ")(][}{><")
    pieces = [run[::-1].translate(mirror) if direction == 1 else run for direction, run in runs]
    return "".join(reversed(pieces) if rtl_base else pieces)
