"""Persist analysis results so long batch runs survive crashes or restarts.

A session is a JSON file mapping absolute track path -> serialized
:class:`~cueit.exporters.TrackData`. On startup the GUI reloads it, repopulates
the queue and skips re-analyzing anything already cached — which is what makes a
2000-track library run resumable instead of a single all-or-nothing hang.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

from .analysis import AnalysisResult, Section
from .cue import Cue
from .exporters import TrackData


def default_session_path() -> str:
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "CueIt", "session.json")


def track_to_dict(t: TrackData) -> dict:
    """Serialize a single track (cheap; call on the GUI thread per completion)."""
    return _track_to_dict(t)


def _track_to_dict(t: TrackData) -> dict:
    a = t.analysis
    return {
        "path": t.path,
        "analysis": {
            "duration": a.duration, "bpm": a.bpm, "bpm_confidence": a.bpm_confidence,
            "key": a.key, "scale": a.scale, "key_strength": a.key_strength, "camelot": a.camelot,
            "beats": a.beats, "downbeats": a.downbeats, "onsets": a.onsets,
            "sections": [{"start": s.start, "end": s.end, "energy": s.energy, "kind": s.kind}
                         for s in a.sections],
            "rms_times": a.rms_times, "rms": a.rms,
            "peak_mins": a.peak_mins, "peak_maxs": a.peak_maxs,
            "integrated_loudness": a.integrated_loudness,
            "beat_energy": a.beat_energy, "beat_low": a.beat_low, "beat_flux": a.beat_flux,
            "downbeat_phase": a.downbeat_phase,
            "downbeat_confidence": a.downbeat_confidence,
            "phrase_length": a.phrase_length,
        },
        "cues": [{"name": c.name, "time": c.time, "color": c.color, "kind": c.kind} for c in t.cues],
    }


def _track_from_dict(d: dict) -> TrackData:
    ad = d["analysis"]
    analysis = AnalysisResult(
        path=d["path"], duration=ad["duration"], bpm=ad["bpm"],
        bpm_confidence=ad.get("bpm_confidence", 0.0), key=ad["key"], scale=ad["scale"],
        key_strength=ad.get("key_strength", 0.0), camelot=ad.get("camelot", ""),
        beats=ad.get("beats", []), downbeats=ad.get("downbeats", []), onsets=ad.get("onsets", []),
        sections=[Section(**s) for s in ad.get("sections", [])],
        rms_times=ad.get("rms_times", []), rms=ad.get("rms", []),
        peak_mins=ad.get("peak_mins", []), peak_maxs=ad.get("peak_maxs", []),
        integrated_loudness=ad.get("integrated_loudness", 0.0),
        beat_energy=ad.get("beat_energy", []), beat_low=ad.get("beat_low", []),
        beat_flux=ad.get("beat_flux", []),
        downbeat_phase=ad.get("downbeat_phase", 0),
        downbeat_confidence=ad.get("downbeat_confidence", 0.0),
        phrase_length=ad.get("phrase_length", 4),
    )
    cues = [Cue(c["name"], c["time"], c["color"], c.get("kind", "hotcue")) for c in d.get("cues", [])]
    return TrackData(path=d["path"], analysis=analysis, cues=cues)


def save(results: dict[str, TrackData], path: str | None = None) -> str:
    path = path or default_session_path()
    payload = {
        "version": 1,
        "saved_at": time.time(),
        "tracks": {p: _track_to_dict(t) for p, t in results.items()},
    }
    _write(payload, path)
    return path


def save_parts(tracks: dict[str, dict], path: str | None = None) -> str:
    """Save from already-serialized track dicts (no re-encoding of TrackData)."""
    path = path or default_session_path()
    payload = {"version": 1, "saved_at": time.time(), "tracks": tracks}
    _write(payload, path)
    return path


def _write(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load(path: str | None = None) -> dict[str, TrackData]:
    path = path or default_session_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            payload = json.load(fh)
        return {p: _track_from_dict(d) for p, d in payload.get("tracks", {}).items()}
    except Exception:  # noqa: BLE001
        return {}


def clear(path: str | None = None) -> None:
    path = path or default_session_path()
    if os.path.exists(path):
        os.unlink(path)
