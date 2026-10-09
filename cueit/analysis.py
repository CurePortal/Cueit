"""Essentia-backed musical analysis: tempo, key, beats, downbeats, energy, structure.

Everything here is pure computation over a mono numpy array so it can run in a
worker thread. The output is a :class:`AnalysisResult` that cue inference and the
GUI consume.
"""

from __future__ import annotations

import os

# Limit every native math/FFT backend to a single thread *before* numpy/essentia
# are imported. Each batch worker is its own process, so without this every
# worker spawns one thread per core (16x oversubscription on a 16-core box),
# which thrashes the CPU and starves the GUI and audio playback.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "OPENBLAS_MAIN_FREE", "GOTO_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import warnings
from dataclasses import dataclass, field, asdict

import numpy as np

import essentia.standard as es
from essentia import log as _elog

_elog.infoActive = False
_elog.warningActive = False

from .audio import load_mono, rms_envelope

warnings.filterwarnings("ignore", message=".*MusicExtractorSVM.*")

# Camelot wheel mapping (8A..12B) for harmonic mixing / Rekordbox colouring.
_CAMELOT = {
    "C major": "8B", "A minor": "8A",
    "G major": "9B", "E minor": "9A",
    "D major": "10B", "B minor": "10A",
    "A major": "11B", "F# minor": "11A",
    "E major": "12B", "C# minor": "12A",
    "B major": "1B", "G# minor": "1A",
    "F# major": "2B", "D# minor": "2A",
    "Db major": "3B", "Bb minor": "3A",
    "Ab major": "4B", "F minor": "4A",
    "Eb major": "5B", "C minor": "5A",
    "Bb major": "6B", "G minor": "6A",
    "F major": "7B", "D minor": "7A",
}


@dataclass
class Section:
    start: float
    end: float
    energy: float
    kind: str = "section"


@dataclass
class AnalysisResult:
    path: str
    duration: float
    bpm: float
    bpm_confidence: float
    key: str
    scale: str
    key_strength: float
    camelot: str
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    onsets: list[float] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    rms_times: list[float] = field(default_factory=list)
    rms: list[float] = field(default_factory=list)
    peak_mins: list[float] = field(default_factory=list)
    peak_maxs: list[float] = field(default_factory=list)
    integrated_loudness: float = 0.0
    # Beat-synchronous features (aligned with `beats`).
    beat_energy: list[float] = field(default_factory=list)
    beat_low: list[float] = field(default_factory=list)
    beat_flux: list[float] = field(default_factory=list)
    # Rhythm structure.
    downbeat_phase: int = 0
    downbeat_confidence: float = 0.0
    phrase_length: int = 4  # bars per phrase (4 or 8)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["sections"] = [asdict(s) for s in self.sections]
        return d


def _beat_sync_features(audio: np.ndarray, sr: int, beats: np.ndarray,
                        frame: int = 2048, hop: int = 512, fmax: float = 180.0
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Kick energy, low-band flux, RMS and snare/mid flux sampled at each beat.

    * kick (<=180 Hz) and low flux line up with bar starts (the "1"),
    * mid-band flux (snare/hat accents) tends to hit on backbeats 2 & 4,
    so a correct downbeat phase has a strong kick on beat 1 *and* snare accents
    on beats 2 and 4. Combining both is far more reliable than loudness alone.
    """
    win = np.hanning(frame).astype("float32")
    kl = max(1, int(fmax / (sr / frame)))
    m0 = max(kl + 1, int(200 / (sr / frame)))
    m1 = min(frame // 2, int(4000 / (sr / frame)))
    n = max(1, (len(audio) - frame) // hop)
    kick = np.zeros(n, dtype="float32")
    low_flux = np.zeros(n, dtype="float32")
    mid_flux = np.zeros(n, dtype="float32")
    prev_low = prev_mid = None
    for i in range(n):
        sp = np.abs(np.fft.rfft(audio[i * hop:i * hop + frame] * win))
        low = sp[:kl]
        mid = sp[m0:m1]
        kick[i] = low.sum()
        if prev_low is not None:
            low_flux[i] = np.maximum(low - prev_low, 0.0).sum()
            mid_flux[i] = np.maximum(mid - prev_mid, 0.0).sum()
        prev_low, prev_mid = low, mid
    dt = hop / sr
    window = max(1, int(0.10 / dt))

    def sample(arr, t):
        i = int(t / dt)
        j = i + window
        if i >= len(arr):
            return 0.0
        return float(arr[i:min(j, len(arr))].max())

    low = np.array([sample(kick, t) for t in beats])
    fl = np.array([sample(low_flux, t) for t in beats])
    snare = np.array([sample(mid_flux, t) for t in beats])
    rms = np.array([
        float(np.sqrt(np.mean(audio[int(t * sr):int(t * sr) + int(0.06 * sr)] ** 2)))
        if int(t * sr) < len(audio) else 0.0
        for t in beats
    ])
    return low, fl, rms, snare


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


def _estimate_downbeats(beats: np.ndarray, low: np.ndarray, flux: np.ndarray,
                        rms: np.ndarray, snare: np.ndarray | None = None
                        ) -> tuple[np.ndarray, int, float]:
    """Return (downbeats, phase, confidence).

    Scores each of the four beat phases by kick + low-flux prominence on the "1"
    plus snare accents on the backbeats (2 & 4), picks the strongest, and reports
    how decisive that choice was.
    """
    if len(beats) < 4:
        return beats.copy(), 0, 0.0

    kick = _z(low) + _z(flux)
    score = np.array([kick[p::4].mean() for p in range(4)])
    if snare is not None and len(snare) == len(beats):
        s = _z(snare)
        backbeat = np.array([
            np.mean([s[(p + off)::4].mean() for off in (1, 3)]) for p in range(4)
        ])
        score = score + backbeat

    order = np.argsort(score)[::-1]
    best = int(order[0])
    margin = float(score[order[0]] - score[order[1]])
    spread = float(score.std()) + 1e-9
    confidence = float(np.clip(margin / (spread + 1e-9), 0.0, 3.0) / 2.0)
    return beats[best::4], best, confidence


def _estimate_phrase_length(beats: np.ndarray, low: np.ndarray, flux: np.ndarray) -> int:
    """Decide whether the track is phrased in 4-bar or 8-bar groups.

    A musical phrase boundary is where bass/kick novelty recurs every N bars.
    We compare the mean novelty on 16-beat (4-bar) versus 32-beat (8-bar) grids;
    whichever is stronger wins. Defaults to 4 (the 32-beat grid then appears
    every 2 phrases anyway, so drops still land correctly).
    """
    if len(beats) < 64:
        return 4
    novelty = _z(low) + _z(flux)
    best_len, best_score = 4, -1e9
    for phrase_beats in (16, 32):
        vals = []
        for p in range(phrase_beats):
            idx = np.arange(p, len(novelty), phrase_beats)
            if len(idx) >= 2:
                vals.append(novelty[idx].mean())
        if vals:
            score = float(np.max(vals) - np.mean(vals))
            if score > best_score:
                best_score, best_len = score, phrase_beats // 4
    return best_len


def _sections_from_features(audio: np.ndarray, sr: int, duration: float) -> list[tuple[float, float]]:
    """Novelty-based structural segmentation from MFCC self-similarity.

    Frames the signal into ~1s windows, builds an MFCC matrix, computes a
    self-similarity matrix and picks peaks in its novelty (checkerboard) curve.
    This is the classic "foote" boundary method and works well on EDM/pop.
    """
    frame = 4096
    hop = 2048
    try:
        mfcc = es.MFCC(numberCoefficients=13, highFrequencyBound=sr // 2)
        spec = es.Spectrum()
        win = es.Windowing(type="hann")
        rows = []
        for f in es.FrameGenerator(audio, frameSize=frame, hopSize=hop, startFromZero=True):
            rows.append(mfcc(spec(win(f)))[1])
        if len(rows) < 16:
            return []
        feat = np.asarray(rows, dtype=np.float64)
        feat = feat - feat.mean(axis=0, keepdims=True)
        std = feat.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        feat /= std
        sim = feat @ feat.T
        sim /= max(1, feat.shape[1])
        n = sim.shape[0]
        kernel = 32
        novelty = np.zeros(n)
        for i in range(n):
            left = sim[max(0, i - kernel):i, max(0, i - kernel):i]
            right = sim[i:i + kernel, i:i + kernel]
            cross = sim[max(0, i - kernel):i, i:i + kernel]
            val = 0.0
            if left.size:
                val += float(left.sum()) / left.size
            if right.size:
                val += float(right.sum()) / right.size
            if cross.size:
                val -= 2.0 * float(cross.sum()) / cross.size
            novelty[i] = val
        times = np.arange(n) * hop / sr
        thr = novelty.mean() + novelty.std()
        boundaries = [0.0]
        for i in range(1, n - 1):
            if novelty[i] > thr and novelty[i] >= novelty[i - 1] and novelty[i] >= novelty[i + 1]:
                t = float(times[i])
                if t - boundaries[-1] > 8.0 and duration - t > 4.0:
                    boundaries.append(t)
        boundaries.append(duration)
        return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    except Exception:
        return []



def analyze(path: str, sample_rate: int = 44100, progress=None) -> AnalysisResult:
    def tick(p: float, msg: str = ""):
        if progress:
            progress(p, msg)

    tick(0.02, "decoding audio")
    audio = load_mono(path, sample_rate)
    sr = sample_rate
    duration = len(audio) / sr
    if duration == 0:
        raise ValueError(f"empty audio: {path}")

    tick(0.20, "tempo & beats")
    rhythm = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, beat_conf, _, _ = rhythm(audio)
    beats = np.asarray(beats, dtype=float)

    tick(0.40, "beat features")
    beat_low, beat_flux, beat_rms, beat_snare = _beat_sync_features(audio, sr, beats)

    tick(0.55, "key detection")
    key, scale, key_strength = es.KeyExtractor()(audio)

    tick(0.68, "onsets & loudness")
    try:
        onsets = np.asarray(es.SuperFluxExtractor()(audio), dtype=float)
    except Exception:
        onsets = np.array([])
    try:
        stereo = np.stack([audio, audio], axis=1)
        _, _, integrated_arr, _ = es.LoudnessEBUR128(sampleRate=sr)(stereo)
        integrated = float(np.asarray(integrated_arr)[-1])
    except Exception:
        integrated = 0.0

    tick(0.80, "structure & downbeats")
    downbeats, phase, db_conf = _estimate_downbeats(beats, beat_low, beat_flux, beat_rms, beat_snare)
    phrase = _estimate_phrase_length(beats, beat_low, beat_flux)
    times, rms = rms_envelope(audio, sr, hop_s=0.5)

    from .audio import waveform_peaks
    peak_mins, peak_maxs = waveform_peaks(audio, buckets=3000)

    raw = _sections_from_features(audio, sr, duration)
    if len(raw) < 2:
        raw = _fallback_sections(rms, times, duration)

    # Snap section boundaries to the downbeat grid so structure lines up with bars.
    raw = _snap_sections(raw, downbeats, beats, duration)

    sections: list[Section] = []
    rms_max = float(rms.max()) or 1.0
    for (a, b) in raw:
        mask = (times >= a) & (times < b)
        e = float(rms[mask].mean()) / rms_max if mask.any() else 0.0
        sections.append(Section(start=a, end=b, energy=e))

    _label_sections(sections)

    tick(1.0, "done")
    camelot = _CAMELOT.get(f"{key} {scale}", "")
    return AnalysisResult(
        path=path, duration=duration, bpm=float(bpm), bpm_confidence=float(beat_conf),
        key=key, scale=scale, key_strength=float(key_strength), camelot=camelot,
        beats=beats.tolist(), downbeats=downbeats.tolist(), onsets=onsets.tolist(),
        sections=sections, rms_times=times.tolist(), rms=rms.tolist(),
        peak_mins=peak_mins.tolist(), peak_maxs=peak_maxs.tolist(),
        integrated_loudness=integrated,
        beat_energy=beat_rms.tolist(), beat_low=beat_low.tolist(), beat_flux=beat_flux.tolist(),
        downbeat_phase=phase, downbeat_confidence=db_conf, phrase_length=phrase,
    )


def _snap_sections(raw: list[tuple[float, float]], downbeats: np.ndarray,
                    beats: np.ndarray, duration: float, tol: float = 1.0
                    ) -> list[tuple[float, float]]:
    """Align section boundaries to the nearest downbeat (fallback: beat)."""
    grid = downbeats if downbeats.size >= 2 else beats
    if grid.size == 0:
        return raw
    snapped: list[float] = []
    for a, b in raw:
        for edge in (a, b):
            if edge <= 0.0:
                snapped.append(0.0)
            elif edge >= duration:
                snapped.append(duration)
            else:
                idx = int(np.argmin(np.abs(grid - edge)))
                snapped.append(float(grid[idx]) if abs(grid[idx] - edge) <= tol else edge)
    out = []
    for i in range(0, len(snapped), 2):
        a, b = snapped[i], snapped[i + 1]
        if b - a > 1.0:
            out.append((max(0.0, a), min(duration, b)))
    return out


def _label_sections(sections: list[Section]) -> None:
    """Assign intro/verse/drop/breakdown/outro from the relative energy curve."""
    if not sections:
        return
    energies = np.array([s.energy for s in sections])
    peak = float(energies.max()) or 1.0
    # A breakdown is a low-energy stretch; a drop is a high-energy jump after it.
    for i, s in enumerate(sections):
        rel = s.energy / peak
        prev = sections[i - 1] if i > 0 else None
        rising = prev is not None and rel >= prev.energy / peak + 0.15
        if i == 0:
            s.kind = "intro"
        elif i == len(sections) - 1:
            s.kind = "outro"
        elif rel <= 0.5:
            s.kind = "breakdown"
        elif rel >= 0.8 or (rising and rel >= 0.65):
            s.kind = "drop"
        else:
            s.kind = "verse"


def _fallback_sections(rms: np.ndarray, times: np.ndarray, duration: float) -> list[tuple[float, float]]:
    if len(rms) < 8:
        return [(0.0, duration)]
    thr = float(np.median(rms)) * 0.6
    loud = rms > thr
    boundaries = [0.0]
    for i in range(1, len(loud)):
        if loud[i] != loud[i - 1]:
            t = float(times[i])
            if t - boundaries[-1] > 8.0:
                boundaries.append(t)
    boundaries.append(duration)
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
