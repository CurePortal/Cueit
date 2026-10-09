"""Cue-point inference: turn an AnalysisResult into labelled hot cues.

The goal is cues a DJ would actually use without editing:

  * **Start** at the first downbeat (the true "1" of the first bar).
  * **Drops** at the *most prominent* energy rises, not every section change.
  * **Breakdowns** at the most prominent energy troughs between drops.
  * **Outro** at the mix-out point near the end.
  * **Phrase awareness**: candidates are preferred on 32-beat phrase starts and
    always snapped to the downbeat grid, so cues land on bar/phase lines.
  * Prominence filtering keeps the result musical: a track with one real drop
    gets one drop cue, not eight.

Cue selection is driven directly by the measured energy contour, so it does not
depend on the (noisier) section labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis import AnalysisResult

# Standard DJ hot-cue colours (index into a palette the GUI/exporters share).
CUE_COLORS = ["#1DB954", "#E91E63", "#2196F3", "#FFC107", "#FF5722", "#9C27B0", "#00BCD4", "#8BC34A"]

MAX_CUES = 8
MAX_DROPS = 3
MAX_BREAKDOWNS = 2
MIN_DROP_GAP = 24.0      # seconds between distinct drops
MIN_BREAKDOWN_GAP = 16.0


@dataclass
class Cue:
    name: str
    time: float
    color: str
    kind: str = "hotcue"


@dataclass
class CueSet:
    cues: list[Cue] = field(default_factory=list)

    def times(self) -> list[float]:
        return [c.time for c in self.cues]

    def add(self, cue: Cue) -> None:
        self.cues.append(cue)


def _snap(t: float, grid: np.ndarray, tol: float = 1.5) -> float:
    if grid.size == 0:
        return max(0.0, t)
    idx = int(np.argmin(np.abs(grid - t)))
    return float(grid[idx]) if abs(grid[idx] - t) <= tol else max(0.0, t)


def _prefer_phrase(t: float, grid: np.ndarray, phrases: np.ndarray, tol: float = 1.0) -> float:
    if phrases.size == 0:
        return _snap(t, grid)
    idx = int(np.argmin(np.abs(phrases - t)))
    return float(phrases[idx]) if abs(phrases[idx] - t) <= tol else _snap(t, grid)


def infer_cues(result: AnalysisResult, use_downbeats: bool = True) -> CueSet:
    grid = np.asarray(
        result.downbeats if use_downbeats and result.downbeats else result.beats, dtype=float
    )
    duration = result.duration
    if grid.size == 0:
        return CueSet([Cue("Intro", 0.0, CUE_COLORS[0])])

    phrase_bars = max(1, int(getattr(result, "phrase_length", 4) or 4))
    phrases = grid[::phrase_bars] if phrase_bars > 1 else grid

    starts = np.array([s.start for s in result.sections], dtype=float)
    energies = np.array([s.energy for s in result.sections], dtype=float)

    candidates: list[tuple[str, float]] = []
    start = float(grid[0])
    candidates.append(("Intro", start))

    if len(energies) >= 3:
        peak = float(energies.max()) or 1.0
        # Drops: biggest *rises* into a high-energy section.
        rises = np.diff(energies)
        drop_order = np.argsort(rises)[::-1]
        drops: list[float] = []
        for j in drop_order:
            if rises[j] <= 0.10:
                break
            idx = j + 1
            t = _prefer_phrase(starts[idx], grid, phrases)
            if t <= start + 2.0 or t >= duration:
                continue
            if energies[idx] < 0.6 * peak:
                continue
            if all(abs(t - d) >= MIN_DROP_GAP for d in drops):
                drops.append(t)
            if len(drops) >= MAX_DROPS:
                break
        candidates.extend(("Drop", t) for t in drops)

        # Breakdowns: biggest *falls* into a low-energy section, ideally between
        # the first drop and the next one.
        falls = np.diff(energies)
        bd_order = np.argsort(falls)[::-1]
        breakdowns: list[float] = []
        for j in bd_order:
            if falls[j] <= 0.10:
                break
            idx = j + 1
            t = _prefer_phrase(starts[idx], grid, phrases)
            if t <= start + 2.0 or t >= duration:
                continue
            if energies[idx] > 0.7 * peak:
                continue
            if all(abs(t - b) >= MIN_BREAKDOWN_GAP for b in breakdowns):
                breakdowns.append(t)
            if len(breakdowns) >= MAX_BREAKDOWNS:
                break
        candidates.extend(("Breakdown", t) for t in breakdowns)
    else:
        # Too few sections: fall back to the single highest-energy point.
        if len(energies):
            idx = int(np.argmax(energies))
            t = _prefer_phrase(starts[idx], grid, phrases)
            if t > start + 2.0:
                candidates.append(("Drop", t))

    # Outro / mix-out at the final section start (or 32 beats before the end).
    outro = _prefer_phrase(starts[-1], grid, phrases) if len(starts) else max(0.0, duration - 32.0)
    candidates.append(("Outro", float(np.clip(outro, 0.0, duration))))

    chosen = _dedupe(candidates, 3.0)
    chosen = _cap(chosen, MAX_CUES)

    cues = [Cue(name, round(t, 3), CUE_COLORS[i % len(CUE_COLORS)])
            for i, (name, t) in enumerate(chosen)]
    return CueSet(cues)


def _dedupe(items: list[tuple[str, float]], min_gap: float = 3.0) -> list[tuple[str, float]]:
    priority = {"Intro": 5, "Drop": 4, "Outro": 3, "Breakdown": 3}
    out: list[tuple[str, float]] = []
    for name, t in sorted(items, key=lambda c: c[1]):
        clash = next((c for c in out if abs(c[1] - t) < min_gap), None)
        if clash is None:
            out.append((name, t))
        elif priority.get(name, 0) > priority.get(clash[0], 0):
            out[out.index(clash)] = (name, t)
    return out


def _cap(items: list[tuple[str, float]], limit: int) -> list[tuple[str, float]]:
    priority = {"Intro": 5, "Drop": 4, "Outro": 3, "Breakdown": 3}
    if len(items) <= limit:
        return sorted(items, key=lambda c: c[1])
    must = [c for c in items if c[0] in ("Intro", "Outro")]
    rest = [c for c in items if c not in must]
    rest.sort(key=lambda c: -priority.get(c[0], 0))
    kept = _dedupe(must + rest[: max(0, limit - len(must))], 3.0)
    return sorted(kept, key=lambda c: c[1])[:limit]
