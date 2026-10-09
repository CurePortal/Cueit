"""Import track lists from DJ software libraries.

Supported sources:
  * Mixxx        — ``mixxxdb.sqlite`` (SQLite)
  * Rekordbox   — ``rekordbox.xml`` (the XML bridge export)
  * Traktor     — ``collection.nml``
  * Serato      — a folder of files (Serato keeps cues in file tags, no central DB)

Each source resolves to a plain list of audio file paths, which the batch
engine consumes exactly like a folder scan.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import urllib.parse
from dataclasses import dataclass
from xml.etree import ElementTree as ET

from .batch import AUDIO_EXTS


@dataclass
class LibrarySource:
    name: str           # human label
    kind: str           # "mixxx" | "rekordbox" | "traktor" | "folder"
    locator: str        # path to db/xml/nml/folder

    def label(self) -> str:
        return f"{self.name} — {self.locator}"


def default_paths() -> dict[str, str]:
    """Best-guess default locations per platform."""
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        mixxx = os.path.join(home, "Library", "Application Support", "Mixxx", "mixxxdb.sqlite")
        rb = os.path.join(home, "Music", "rekordbox", "rekordbox.xml")
        tk = os.path.join(home, "Documents", "Native Instruments", "Traktor 3.x", "collection.nml")
    elif sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA", home)
        mixxx = os.path.join(appdata, "Mixxx", "mixxxdb.sqlite")
        rb = os.path.join(home, "Documents", "rekordbox", "rekordbox.xml")
        tk = os.path.join(home, "Documents", "Native Instruments", "Traktor 3.x", "collection.nml")
    else:
        mixxx = os.path.join(home, ".mixxx", "mixxxdb.sqlite")
        rb = os.path.join(home, "rekordbox.xml")
        tk = os.path.join(home, "Documents", "Native Instruments", "Traktor 3.x", "collection.nml")
    return {"mixxx": mixxx, "rekordbox": rb, "traktor": tk}


def detect_libraries() -> list[LibrarySource]:
    """Return the libraries that exist on this machine."""
    found: list[LibrarySource] = []
    paths = default_paths()
    if os.path.exists(paths["mixxx"]):
        found.append(LibrarySource("Mixxx library", "mixxx", paths["mixxx"]))
    if os.path.exists(paths["rekordbox"]):
        found.append(LibrarySource("Rekordbox XML", "rekordbox", paths["rekordbox"]))
    if os.path.exists(paths["traktor"]):
        found.append(LibrarySource("Traktor collection", "traktor", paths["traktor"]))
    return found


def load_tracks(source: LibrarySource) -> list[str]:
    if source.kind == "mixxx":
        return _load_mixxx(source.locator)
    if source.kind == "rekordbox":
        return _load_rekordbox(source.locator)
    if source.kind == "traktor":
        return _load_traktor(source.locator)
    if source.kind == "folder":
        from .batch import discover
        return discover([source.locator])
    raise ValueError(f"unknown library kind: {source.kind}")


def _load_mixxx(db_path: str) -> list[str]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT DISTINCT tl.location "
            "FROM library l JOIN track_locations tl ON l.location = tl.id "
            "WHERE COALESCE(l.mixxx_deleted, 0) = 0 "
            "ORDER BY tl.location"
        ).fetchall()
    finally:
        conn.close()
    return _existing([r[0] for r in rows if r[0]])


def _load_rekordbox(xml_path: str) -> list[str]:
    root = ET.parse(xml_path).getroot()
    out: list[str] = []
    for track in root.iter("TRACK"):
        loc = track.get("Location")
        if not loc:
            continue
        path = loc
        if path.startswith("file://localhost"):
            path = path[len("file://localhost"):]
        elif path.startswith("file://"):
            path = path[len("file://"):]
        path = urllib.parse.unquote(path)
        out.append(os.path.normpath(path))
    return _existing(out)


def _load_traktor(nml_path: str) -> list[str]:
    root = ET.parse(nml_path).getroot()
    out: list[str] = []
    for entry in root.iter("ENTRY"):
        loc = entry.find("LOCATION")
        if loc is None:
            continue
        directory = loc.get("DIR", "")
        filename = loc.get("FILE", "")
        if filename:
            out.append(os.path.normpath(os.path.join(directory, filename)))
    return _existing(out)


def _existing(paths: list[str]) -> list[str]:
    seen: list[str] = []
    for p in paths:
        if p and os.path.isfile(p) and p.lower().endswith(AUDIO_EXTS) and p not in seen:
            seen.append(p)
    return seen
