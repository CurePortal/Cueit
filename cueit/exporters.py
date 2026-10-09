"""Export analysis + cues to DJ software formats.

Supported targets:
  * Rekordbox  -> rekordbox.xml (imports into Rekordbox 6/7 via the XML bridge)
  * Traktor    -> collection.nml
  * Mixxx      -> writes directly into mixxxdb.sqlite (with a backup)
  * Serato     -> writes ID3 "Serato Markers_" / "Serato Markers2" GEOB frames
  * JSON / CSV -> neutral interchange

All exporters take a :class:`TrackData`.
"""

from __future__ import annotations

import base64
import csv
import json
import os
import shutil
import sqlite3
import struct
import time
from dataclasses import dataclass
from xml.etree import ElementTree as ET

from .analysis import AnalysisResult
from .cue import Cue


@dataclass
class TrackData:
    path: str
    analysis: AnalysisResult
    cues: list[Cue]

    @property
    def title(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _rgb_to_int(color: str) -> int:
    r, g, b = _hex_to_rgb(color)
    return (r << 16) | (g << 8) | b


# --------------------------------------------------------------------------- #
# JSON / CSV
# --------------------------------------------------------------------------- #

def export_json(track: TrackData, out_path: str) -> str:
    data = {
        "file": track.path,
        "title": track.title,
        "bpm": round(track.analysis.bpm, 2),
        "key": f"{track.analysis.key} {track.analysis.scale}".strip(),
        "camelot": track.analysis.camelot,
        "duration": round(track.analysis.duration, 3),
        "cues": [{"name": c.name, "time": c.time, "color": c.color} for c in track.cues],
    }
    with open(out_path, "w") as fh:
        json.dump(data, fh, indent=2)
    return out_path


def export_csv(track: TrackData, out_path: str) -> str:
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "title", "bpm", "key", "camelot", "duration"])
        w.writerow([track.path, track.title, f"{track.analysis.bpm:.2f}",
                    f"{track.analysis.key} {track.analysis.scale}".strip(),
                    track.analysis.camelot, f"{track.analysis.duration:.3f}"])
        w.writerow([])
        w.writerow(["cue", "name", "time_seconds", "color"])
        for i, c in enumerate(track.cues):
            w.writerow([i, c.name, f"{c.time:.3f}", c.color])
    return out_path


# --------------------------------------------------------------------------- #
# Rekordbox XML
# --------------------------------------------------------------------------- #

def export_rekordbox(tracks: list[TrackData], out_path: str) -> str:
    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    ET.SubElement(root, "PRODUCT", Name="CueIt", Version="0.1", Company="CueIt")
    collection = ET.SubElement(root, "COLLECTION", Entries=str(len(tracks)))

    for tid, t in enumerate(tracks, start=1):
        tr = ET.SubElement(
            collection, "TRACK",
            TrackID=str(tid),
            Name=t.title,
            Artist="",
            TotalTime=str(int(round(t.analysis.duration))),
            AverageBpm=f"{t.analysis.bpm:.2f}",
            Tonality=f"{t.analysis.key} {t.analysis.scale}".strip(),
            Location="file://localhost" + t.path,
        )
        ET.SubElement(tr, "TEMPO", Inizio="0.000000", Bpm=f"{t.analysis.bpm:.2f}",
                      Metro="4/4", Battito="1")
        for num, c in enumerate(t.cues):
            r, g, b = _hex_to_rgb(c.color)
            ET.SubElement(
                tr, "POSITION_MARK",
                Name=c.name, Type="0", Start=f"{c.time:.3f}", Num=str(num),
                Red=str(r), Green=str(g), Blue=str(b),
            )

    playlists = ET.SubElement(root, "PLAYLISTS")
    node = ET.SubElement(playlists, "NODE", Type="0", Name="ROOT", Count="1")
    ET.SubElement(node, "NODE", Name="CueIt", Type="1", KeyType="0", Entries=str(len(tracks)))
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="UTF-8", xml_declaration=True)
    return out_path


# --------------------------------------------------------------------------- #
# Traktor NML
# --------------------------------------------------------------------------- #

def export_traktor(tracks: list[TrackData], out_path: str) -> str:
    now = time.localtime()
    date = time.strftime("%Y/%m/%d", now)
    clock = time.strftime("%H:%M:%S", now)

    nml = ET.Element("NML", VERSION="19")
    ET.SubElement(nml, "HEAD", COMPANY="CueIt", PROGRAM="CueIt")
    coll = ET.SubElement(nml, "COLLECTION", ENTRIES=str(len(tracks)))

    for t in tracks:
        entry = ET.SubElement(
            coll, "ENTRY",
            MODIFIED_DATE=date, MODIFIED_TIME=clock, TITLE=t.title, ARTIST="",
        )
        ET.SubElement(entry, "LOCATION", DIR=os.path.dirname(t.path) + os.sep,
                      FILE=os.path.basename(t.path), VOLUME="", VOLUMEID="")
        ET.SubElement(entry, "INFO", BITRATE="0", GENRE="", COMMENT=t.analysis.camelot,
                      IMPORT_DATE=date, RELEASE_DATE="", FLAGS="1",
                      LENGTH=str(int(round(t.analysis.duration))), KEY="",
                      AUTHOR="")
        ET.SubElement(entry, "TEMPO", BPM=f"{t.analysis.bpm:.6f}",
                      BPM_QUALITY="100.000000")
        for i, c in enumerate(t.cues):
            ET.SubElement(
                entry, "CUE_V2",
                NAME=c.name, DISPL_ORDER="0", TYPE="0",
                START=f"{c.time * 1000.0:.6f}", LENGTH="0.000000",
                REPEATS="-1", HOTCUE=str(i),
            )

    ET.indent(nml, space="  ")
    ET.ElementTree(nml).write(out_path, encoding="UTF-8", xml_declaration=True)
    return out_path


# --------------------------------------------------------------------------- #
# Mixxx (direct sqlite write)
# --------------------------------------------------------------------------- #

# Mixxx CueType enum values
_MIXXX_HOTCUE = 1
_MIXXX_MAIN_CUE = 2
_MIXXX_INTRO = 6
_MIXXX_OUTRO = 7


def export_mixxx(track: TrackData, db_path: str = "~/.mixxx/mixxxdb.sqlite") -> str:
    db_path = os.path.expanduser(db_path)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Mixxx database not found: {db_path}")

    backup = db_path + ".cueit.bak"
    if not os.path.exists(backup):
        shutil.copy2(db_path, backup)

    conn = sqlite3.connect(db_path, timeout=5)
    try:
        conn.execute("BEGIN")
        loc = os.path.abspath(track.path)
        row = conn.execute(
            "SELECT l.id FROM library l JOIN track_locations tl ON l.location = tl.id "
            "WHERE tl.location = ?", (loc,),
        ).fetchone()
        if row is None:
            raise KeyError(f"track not in Mixxx library: {loc}")
        track_id = row[0]

        sr = 44100
        srow = conn.execute("SELECT samplerate FROM library WHERE id = ?", (track_id,)).fetchone()
        if srow and srow[0]:
            sr = int(srow[0])

        # Update bpm/key.
        key_str = track.analysis.camelot or f"{track.analysis.key}{track.analysis.scale[:1]}"
        conn.execute("UPDATE library SET bpm = ?, key = ? WHERE id = ?",
                     (round(track.analysis.bpm, 3), key_str, track_id))

        # Replace existing hot/intro/outro cues for this track.
        conn.execute("DELETE FROM cues WHERE track_id = ? AND type IN (?,?,?,?)",
                     (track_id, _MIXXX_HOTCUE, _MIXXX_INTRO, _MIXXX_OUTRO, _MIXXX_MAIN_CUE))

        for i, c in enumerate(track.cues):
            pos = int(round(c.time * sr))
            color = _rgb_to_int(c.color)
            ctype = _MIXXX_HOTCUE
            hot = i
            if c.name.lower().startswith("intro"):
                ctype, hot = _MIXXX_INTRO, -1
            elif c.name.lower().startswith("outro"):
                ctype, hot = _MIXXX_OUTRO, -1
            conn.execute(
                "INSERT INTO cues (track_id, type, position, length, hotcue, label, color) "
                "VALUES (?,?,?,?,?,?,?)",
                (track_id, ctype, pos, 0, hot, c.name, color),
            )
        conn.commit()
    finally:
        conn.close()
    return f"mixxx:{track_id}"


# --------------------------------------------------------------------------- #
# Serato ID3 tags
# --------------------------------------------------------------------------- #

_SERATO_VERSION2 = 0x0101
_SERATO_VERSION_ = 0x0205
_SERATO_MARKERS_PREFIX = b"application/octet-stream\x00\x00Serato Markers_"
_NO_POSITION = 0x7F7F7F7F


def _serato32_from_uint24(value: int) -> int:
    a = (value >> 16) & 0xFF
    b = (value >> 8) & 0xFF
    c = value & 0xFF
    z = c & 0x7F
    y = ((c >> 7) | (b << 1)) & 0x7F
    x = ((b >> 6) | (a << 2)) & 0x7F
    w = a >> 5
    return (w << 24) | (x << 16) | (y << 8) | z


def _serato_b64(data: bytes, chop_padding: bool = False) -> bytes:
    out = bytearray()
    offset = 0
    while offset < len(data):
        if offset > 0:
            out += b"\n"
        block = data[offset:offset + 54]
        enc = base64.b64encode(block).rstrip(b"=")
        if chop_padding and len(block) % 3:
            enc = enc[:-1]
        out += enc
        offset += len(block)
    return bytes(out)


def _serato_markers2_field(cues: list[Cue], track_color: tuple[int, int, int]) -> bytes:
    data = struct.pack(">H", _SERATO_VERSION2)
    for entry in _serato_markers2_entries(cues, track_color):
        name, payload = entry
        data += name.encode("utf-8") + b"\x00" + struct.pack(">I", len(payload)) + payload
    data += b"\x00"
    return data


def _serato_markers2_entries(cues: list[Cue], track_color: tuple[int, int, int]):
    entries = []
    r, g, b = track_color
    entries.append(("COLOR", b"\x00" + bytes([r, g, b])))
    for i, c in enumerate(cues[:8]):
        cr, cg, cb = _hex_to_rgb(c.color)
        payload = (
            b"\x00" + bytes([i]) + struct.pack(">I", int(round(c.time * 1000.0)))
            + b"\x00" + bytes([cr, cg, cb]) + b"\x00\x00"
            + c.name.encode("utf-8") + b"\x00"
        )
        entries.append(("CUE", payload))
    entries.append(("BPMLOCK", b"\x00"))
    return entries


def _serato_markers_field(cues: list[Cue], track_color: tuple[int, int, int] = (0x99, 0x99, 0x99)) -> bytes:
    entries = []
    # 5 cue entries
    for i in range(5):
        if i < len(cues):
            c = cues[i]
            cr, cg, cb = _hex_to_rgb(c.color)
            color32 = _serato32_from_uint24((cr << 16) | (cg << 8) | cb)
            entry = (
                b"\x00" + struct.pack(">I", _serato32_from_uint24(int(round(c.time * 1000.0))))
                + b"\x00" + struct.pack(">I", _serato32_from_uint24(0))
                + b"\x00\x7f\x7f\x7f\x7f\x7f"
                + struct.pack(">I", color32) + bytes([0x00, 0x00])
            )
        else:
            color32 = _serato32_from_uint24(0x7F7F7F)
            entry = (
                b"\x7f" + struct.pack(">I", _NO_POSITION)
                + b"\x7f" + struct.pack(">I", _NO_POSITION)
                + b"\x00\x7f\x7f\x7f\x7f\x7f"
                + struct.pack(">I", color32) + bytes([0x00, 0x00])
            )
        entries.append(entry)
    # 9 loop entries (unset)
    for _ in range(9):
        color32 = _serato32_from_uint24(0x7F7F7F)
        entries.append(
            b"\x7f" + struct.pack(">I", _NO_POSITION)
            + b"\x7f" + struct.pack(">I", _NO_POSITION)
            + b"\x00\x7f\x7f\x7f\x7f\x7f"
            + struct.pack(">I", color32) + bytes([0x00, 0x01])
        )
    tr, tg, tb = track_color
    data = struct.pack(">H", _SERATO_VERSION_) + struct.pack(">I", len(entries))
    data += b"".join(entries)
    data += struct.pack(">I", _serato32_from_uint24((tr << 16) | (tg << 8) | tb))
    return data


def export_serato(track: TrackData, timing_offset_ms: float = 0.0) -> str:
    """Write Serato Markers_ / Markers2 GEOB frames into the file's ID3 tags."""
    from mutagen.id3 import ID3, GEOB, TKEY, TBPM, error as ID3Error

    ext = os.path.splitext(track.path)[1].lower()
    if ext not in (".mp3", ".aiff", ".aif"):
        raise ValueError("Serato tag writing supports MP3/AIFF only (use Rekordbox XML for others).")

    try:
        tags = ID3(track.path)
    except ID3Error:
        tags = ID3()

    def _geo(desc: str, data: bytes) -> GEOB:
        return GEOB(encoding=0, mime="application/octet-stream", desc=desc, data=data)

    # Adjust for decoder timing offset (MP3 only).
    adjusted = [Cue(c.name, max(0.0, c.time - timing_offset_ms / 1000.0), c.color) for c in track.cues]

    markers2 = _serato_markers2_field(adjusted, (0x99, 0x99, 0x99))
    markers_ = _serato_markers_field(adjusted)

    tags.delall("GEOB:Serato Markers2")
    tags.delall("GEOB:Serato Markers_")
    tags.add(_geo("Serato Markers_", markers_))
    tags.add(_geo("Serato Markers2", markers2))

    key = track.analysis.camelot or f"{track.analysis.key} {track.analysis.scale}".strip()
    tags.setall("TKEY", [TKEY(encoding=3, text=key)])
    tags.setall("TBPM", [TBPM(encoding=3, text=f"{track.analysis.bpm:.0f}")])
    tags.save(track.path)
    return track.path
