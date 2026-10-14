# CueIt

**The free, open-source Mixed In Key.** CueIt analyses your tracks, auto-places
hot cues, and exports them to Rekordbox, Traktor, Mixxx and Serato.

> ### Alpha 2 — full Rust rewrite
>
> CueIt is now a single **Rust** project. One shared core powers a native desktop
> app (Linux, macOS and **Windows**), a CLI, and a **WebAssembly** build for the
> browser — from the same code. The previous Python version is **deprecated**
> (see [below](#why-the-python-version-is-deprecated)).

**▶ Try the web demo:** **https://cureportal.github.io/Cueit/** — it analyses in
your browser (the pure-Rust DSP engine compiled to WebAssembly); nothing is
uploaded.

```
tempo + key (Camelot)     -> pure-Rust DSP  (optionally Deezer S-KEY model)
beat + downbeat grid      -> pure-Rust DSP
structure (intro / drop)  -> MFCC self-similarity segmentation
cue inference             -> DJ-style heuristics (intro / drop / breakdown / outro)
```

## Why the Python version is deprecated

The first CueIt was Python. Its analysis depended on **Essentia**, and Essentia
publishes Python wheels **only for Linux and macOS** — there are **no Windows
wheels at all**, and no source distribution for current releases. On Windows the
only way to run Essentia was inside **WSL2**, which is a poor prerequisite for a
desktop DJ tool: a full Linux VM, a display server (WSLg), audio plumbing, and
`/mnt/c` file paths.

Rather than ask Windows users to install WSL, CueIt was rewritten in Rust:

- a **pure-Rust DSP engine** replaces Essentia (no native blobs, no WSL),
- the **same code** runs natively on Windows, macOS and Linux, and
- it **compiles to WebAssembly** for the browser build.

The old Python tree (`cueit/`, `pyproject.toml`) is kept only for reference and
is no longer maintained.

## Features

- Tempo, musical key, Camelot code and loudness analysis
- Beat grid + downbeat detection
- Structural segmentation: intro / verse / drop / breakdown / outro
- Automatic hot-cue inference, snapped to the downbeat grid
- Waveform GUI with draggable cue handles, undo/redo, playback and zoom
- **Batch jobs**: files, folders, or an entire DJ library; queue and analyse many
  tracks in parallel across all CPU cores, with a **CPU-limit** control so the
  machine stays responsive
- Import track lists from **Mixxx**, **Rekordbox** and **Traktor**, or a folder
- **Exporters**: Rekordbox (`rekordbox.xml`), Traktor (`collection.nml`),
  **Mixxx** (direct SQLite), **Serato** (ID3 tags), JSON, CSV
- Optional **neural models** (ONNX Runtime): Deezer **S-KEY** for key detection
  (and Beat This! for beats, currently disabled pending tuning)

## Install

Requires **ffmpeg** on your `PATH`.

### Download (desktop)

Prebuilt desktop builds are attached to the
[Releases](https://github.com/CurePortal/Cueit/releases) page
(Linux `.deb`/`.rpm`/AppImage, macOS `.dmg`, Windows `.msi`/`.exe`).

### Build from source

Requires a Rust toolchain (1.75+), Node 18+ and pnpm.

```bash
git clone https://github.com/CurePortal/Cueit
cd Cueit

# CLI only
cargo build --release -p cueit

# Desktop app (Tauri)
cd crates/cueit-tauri && pnpm install
pnpm tauri dev            # develop
pnpm tauri build          # release bundles / installers
```

> **Manual release build note:** Tauri decides dev vs. production from the
> `custom-protocol` feature. `pnpm tauri build` sets it; a plain
> `cargo build --release` must pass it too, or the window will try to load the
> dev-server URL:
> `cargo build --release -p cueit-tauri --features custom-protocol`.

### Web (WebAssembly)

```bash
wasm-pack build crates/cueit-wasm --target web --release --out-dir web/pkg
# then serve web/ (e.g. python3 -m http.server) or deploy to GitHub Pages
```

The web build runs the pure-Rust DSP engine in the browser; the neural models
are desktop-only.

### Neural models (optional, desktop)

Two open, MIT-licensed models via ONNX Runtime:
**Beat This!** (CPJKU, ISMIR 2024) and Deezer **S-KEY** (ICASSP 2025).

Weights download on first use to `~/.local/share/CueIt/models` (S-KEY is ~0.3 MB).
Disable with `cueit analyze --no-models`, or toggle in the app with the
**AI models** button. When unavailable, the pure-Rust DSP engine is used.

## CLI

```bash
cueit analyze ~/Music/albums
cueit analyze ~/Music/albums --export rekordbox --out ~/exports --workers 12 --cpu-limit 75
cueit analyze --library mixxx --export mixxx
cueit analyze track.flac --export serato
cueit libraries
```

## Desktop app

- Queue tracks (files / folder / DJ library), set **Threads** and **CPU** limit,
  run the batch (resumable, cancellable).
- Inspect each track: waveform with section shading and **draggable cue handles**
  (arrows top/bottom), edit or **double-click the waveform to add a cue**,
  undo/redo, playback (native via `rodio`), zoom with the wheel.
- Export the whole session to Rekordbox / Traktor / Mixxx / Serato / JSON / CSV.

Keyboard shortcuts: `Space` play/pause, `←/→` seek, `C` cue at playhead, wheel
zoom, `Ctrl+Z`/`Ctrl+Shift+Z` undo/redo, `Ctrl+R` run, `F1` for all.

## Workspace layout

| crate | role |
| --- | --- |
| `crates/cueit-core` | domain model, pure-Rust DSP, cue inference, exporters, importers, session, batch runner. Compiles native **and** `wasm32`. |
| `crates/cueit-cli` | `cueit` CLI |
| `crates/cueit-tauri` | desktop app (Tauri v2) |
| `crates/cueit-wasm` | `wasm-bindgen` bindings for the web build |
| `frontend/` | React + TypeScript + Vite UI |
| `web/` | static WASM web app (GitHub Pages) |

## Status

Alpha 2. Working: DSP engine, cue inference, all exporters/importers, session
persistence, parallel batch with CPU limiting, CLI, desktop app (waveform, cues,
playback, undo/redo), web build, and the S-KEY key model.

Known limitations: Beat This! beats are disabled pending log-mel front-end
validation (the DSP beat grid is used instead); cue selection heuristics are still
being tuned against labelled tracks.

## License

MIT. Bundled/optional models: Beat This! and S-KEY are MIT.
