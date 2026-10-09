"""Audio decoding and waveform peak extraction.

Decoding is delegated to ffmpeg so that every format DJs actually use
(WAV, AIFF, MP3, FLAC, M4A/AAC, OGG) works identically on Linux, macOS
and Windows without per-format code.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

DEFAULT_SR = 44100


class AudioError(RuntimeError):
    pass


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise AudioError(
            "ffmpeg was not found on PATH. Install it (apt/brew/winget) to decode audio."
        )
    return exe


@dataclass
class AudioInfo:
    path: str
    sample_rate: int
    duration: float
    channels: int


def probe(path: str) -> AudioInfo:
    exe = shutil.which("ffprobe")
    if not exe:
        raise AudioError("ffprobe was not found on PATH.")
    out = subprocess.run(
        [
            exe, "-v", "error", "-show_entries",
            "stream=sample_rate,channels,duration:format=duration",
            "-of", "default=noprint_wrappers=1", path,
        ],
        capture_output=True, text=True,
    ).stdout
    fields: dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k] = v
    sr = int(float(fields.get("sample_rate", DEFAULT_SR)))
    duration = float(fields.get("duration", 0.0) or 0.0)
    channels = int(fields.get("channels", 2))
    if duration <= 0:
        duration = float(fields.get("DURATION", 0.0) or 0.0)
    return AudioInfo(path=path, sample_rate=sr, duration=duration, channels=channels)


def load_mono(path: str, sample_rate: int = DEFAULT_SR) -> np.ndarray:
    """Decode *path* to a mono float32 numpy array at *sample_rate*."""
    proc = subprocess.run(
        [
            _ffmpeg(), "-v", "error", "-i", path,
            "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg failed to decode {path}: {proc.stderr.decode()[:400]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def waveform_peaks(audio: np.ndarray, buckets: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Return (mins, maxs) arrays, one entry per bucket, for drawing."""
    n = len(audio)
    if n == 0:
        return np.zeros(buckets), np.zeros(buckets)
    buckets = max(1, min(buckets, n))
    edges = np.linspace(0, n, buckets + 1).astype(int)
    mins = np.empty(buckets)
    maxs = np.empty(buckets)
    for i in range(buckets):
        seg = audio[edges[i]: edges[i + 1]]
        if seg.size == 0:
            mins[i] = maxs[i] = 0.0
        else:
            mins[i] = seg.min()
            maxs[i] = seg.max()
    return mins, maxs


def rms_envelope(audio: np.ndarray, sample_rate: int, hop_s: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, rms) loudness envelope using a hop of *hop_s* seconds."""
    hop = max(1, int(hop_s * sample_rate))
    n = len(audio)
    frames = max(1, n // hop)
    rms = np.empty(frames)
    for i in range(frames):
        seg = audio[i * hop:(i + 1) * hop]
        rms[i] = float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0
    times = np.arange(frames) * hop / sample_rate
    return times, rms
