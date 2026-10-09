# CueIt

**The free, open-source Mixed In Key.** CueIt analyses your tracks and
auto-places hot cues, then exports them to Rekordbox, Mixxx, Serato and Traktor.

```
BPM + key (Camelot)     -> Essentia
beat + downbeat grid    -> Essentia
structure (intro / drop) -> MFCC self-similarity segmentation
cue inference           -> DJ-style heuristics (intro / drop / breakdown / outro)
```

## Features

- Tempo, musical key, Camelot code and danceability-style energy analysis
- Beat grid + downbeat detection
- Structural segmentation: intro / verse / drop / breakdown / outro
- Automatic hot-cue inference, snapped to the downbeat grid
- Waveform GUI with editable cues and audio playback
- **Batch jobs**: add individual files, whole folders, or an entire DJ library,
  queue them, and analyze many tracks in parallel across all CPU cores
- **Load from DJ library**: import the track list from Mixxx, Rekordbox (XML) or
  Traktor (`collection.nml`) and queue every track automatically
- Exporters:
  - **Rekordbox** — `rekordbox.xml` (imports into Rekordbox 6/7)
  - **Traktor** — `collection.nml`
  - **Mixxx** — writes directly into `mixxxdb.sqlite` (automatic backup)
  - **Serato** — writes `Serato Markers_` / `Serato Markers2` ID3 tags (MP3/AIFF)
  - **JSON / CSV** — neutral interchange

## How cue detection works

CueIt does not just drop a cue every 4 beats. It builds a beat grid, finds
the true bar line, detects phrase length, and picks musically meaningful points:

1. **Beat + bar grid** — Essentia `RhythmExtractor2013` gives the beat grid; a
   score combining kick-band energy + low-band flux (bass-note change) *and*
   mid-band snare-flux accents on the backbeats picks the correct downbeat
   phase, with a confidence value shown in the UI.
2. **Phrase length** — compares 4-bar vs 8-bar novelty periodicity per track, so
   cues are preferred on 32-beat phrase starts (where DJs actually mix).
3. **Structure** — MFCC self-similarity segmentation finds section boundaries,
   which are snapped to the downbeat grid.
4. **Cue selection by prominence** — drops are placed at the largest *energy
   rises* and breakdowns at the largest *falls*, de-duplicated and spaced out, so
   a track with one real drop gets one drop cue instead of eight. Everything is
   snapped to bars/phrases, then capped at 8 hot cues (DJ-standard).

Tune the result per track with **Re-detect cues** in the GUI, or by dragging the
cue tabs (each cue line has grab handles at both the top and bottom).

## Parallelism

Batch analysis uses a process pool (`spawn`, so it behaves identically on
Linux, macOS and Windows). It is built for large libraries, unlike tools that
hang on a 2000-track run:

- **Cancel** the batch at any time (GUI button or `Ctrl-C` in the CLI); the pool
  is terminated immediately and unfinished tracks are left queued.
- **Per-track timeout** (`--timeout`, default 600s): a single stuck file is
  marked as an error and skipped instead of wedging the whole run.
- **Resumable session**: results are saved incrementally to
  `~/.local/share/CueIt/session.json` (every 5 tracks and on exit). Restart
  CueIt and previously analyzed tracks reload instantly, so `Run batch`
  continues where you left off. File ▸ Clear saved session resets it.
- **Bounded memory**: workers recycle every 25 tracks.
- **No CPU oversubscription**: each worker is pinned to a single BLAS/FFT thread
  *before* numpy/Essentia import, so N workers use N threads, not N x cores.
  This is what keeps the GUI and audio playback smooth during a big batch.
- **Always-responsive UI**: batch workers run at low OS priority (`nice`), and
  the GUI never blocks — session files are written by a background thread and cue
  drags update a single table cell instead of rebuilding the list. Dragging cue
  selectors while audio plays during a big batch stays smooth.
- Worker count defaults to `min(cpu_count - 1, 8)`; raise it in the spin box or
  with `--workers N` if you have RAM and cores to spare.



## Install

CueIt needs **Python 3.10+** and **ffmpeg** on your PATH.

```bash
git clone https://github.com/CurePortal/Cueit
cd cueit
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

## GUI

```bash
cueit gui                 # or: python -m cueit.app
cueit gui track1.mp3 track2.aiff
```

Add files, a whole folder, or your DJ library (buttons or drag-and-drop).
Libraries load **off the UI thread** and never look frozen. Nothing is analyzed
automatically — the queue fills first, then you choose:
- **Run selected** — analyze only the highlighted rows,
- **Run batch (all)** — analyze every pending track.

Click any analyzed row to inspect its cues. Drag cues on the waveform, edit the
table, click the waveform to seek, and `+ Cue at playhead` adds a cue. Exports act
on every analyzed track in the session.

## CLI

```bash
# Analyse a folder (recursive) and print cues
cueit analyze ~/Music/albums

# Parallel batch -> rekordbox.xml for the whole folder
cueit analyze ~/Music/albums --export rekordbox --out ~/exports --workers 8

# Analyze every track in your Mixxx / Rekordbox / Traktor library
cueit libraries
cueit analyze --library mixxx --export rekordbox --out ~/exports
cueit analyze --library rekordbox --library-file ~/rekordbox.xml --export json --out ~/exports

# Export to Traktor, Mixxx, Serato, JSON, CSV
cueit analyze ~/Music/albums --export traktor   --out ~/exports
cueit analyze ~/Music/albums --export mixxx
cueit analyze ~/Music/albums --export serato
cueit analyze ~/Music/albums --export csv --out ~/exports
```

In the GUI, use **Add files…**, **Add folder…**, or **Load DJ library…**
(auto-detects Mixxx / Rekordbox / Traktor; a Browse button lets you point at any
library file).

### Keyboard shortcuts

| Key | Action |
| --- | --- |
| `Space` | Play / pause |
| `←` / `→` | Jump back / forward 10s |
| `Ctrl+←` / `Ctrl+→` | Previous / next cue |
| `↑` / `↓` | Previous / next track in the queue |
| `Home` / `End` | Go to start / end |
| `C` | Add cue at playhead |
| `Delete` | Delete selected cue |
| Mouse wheel | Zoom around cursor |
| `Ctrl+=` / `Ctrl+-` / `Ctrl+0` | Zoom in / out / fit |
| `Ctrl+D` / `Ctrl+L` | Dark / light mode |
| `Ctrl+R` / `Esc` | Run / cancel batch |
| `Ctrl+Shift+R` | Re-detect cues |
| `Ctrl+S` | Save session |
| `Ctrl+O` / `Ctrl+Shift+O` / `Ctrl+Shift+L` | Add files / folder / DJ library |
| `F1` | Show all shortcuts |

Shortcuts are ignored while editing a cue-table cell, spin box or text field,
so inline editing works normally.

## Building installers (Windows / macOS / Linux)

The GUI is Qt (PySide6), which ships wheels for all three platforms. To ship a
standalone binary, use [PyInstaller](https://pyinstaller.org):

```bash
pip install pyinstaller
pyinstaller --noconfirm --windowed --name CueIt \
  --icon cueit/assets/cueit.ico \
  --add-data "cueit/assets:cueit/assets" \
  --collect-data essentia \
  --hidden-import PySide6.QtMultimedia \
  -m cueit.app
```

Build on each target OS (PyInstaller does not cross-compile). ffmpeg must be
bundled or available on the target machine. On macOS use
`--icon cueit/assets/cueit.icns`.

## Notes / limitations

- Serato tag writing supports MP3 and AIFF only. Use the Rekordbox XML bridge
  for other formats.
- Mixxx writes go straight to the SQLite DB with a one-time `.cueit.bak`
  backup; close Mixxx first.
- Key detection matches Mixed In Key / Mixxx on the test material, but no
  detector is perfect — always sanity-check the odd track.

## Architecture

| module | role |
| --- | --- |
| `cueit/audio.py` | ffmpeg decode, waveform peaks, RMS envelope |
| `cueit/analysis.py` | Essentia tempo/key/beats/downbeats/segmentation |
| `cueit/cue.py` | cue-point inference |
| `cueit/batch.py` | folder discovery + parallel process-pool batch analysis |
| `cueit/libraries.py` | import track lists from Mixxx / Rekordbox / Traktor |
| `cueit/exporters.py` | Rekordbox / Traktor / Mixxx / Serato / JSON / CSV |
| `cueit/app.py` | Qt GUI |
| `cueit/cli.py` | command line |

## License

MIT
