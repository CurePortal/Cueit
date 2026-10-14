//! CueIt core: cross-platform automatic cue-point detection for DJ libraries.
//!
//! This crate is pure Rust with no platform-specific native dependencies, so it
//! compiles identically for desktop (Linux/macOS/Windows), servers, and
//! `wasm32` (the web build). It contains:
//!
//! * [`types`] — the domain model shared by every front end and exporter.
//! * [`cue`] — DJ-style hot-cue inference from an [`types::AnalysisResult`].
//! * [`exporters`] — Rekordbox / Traktor / Mixxx / Serato / JSON / CSV output.
//!
//! The DSP engine (tempo, beats, key, structure) lives in [`analysis`].

pub mod analysis;
pub mod batch;
pub mod cue;
pub mod exporters;
pub mod libraries;
#[cfg(feature = "models")]
pub mod models;
pub mod session;
pub mod types;

pub use cue::{infer_cues, CUE_COLORS};
pub use types::{AnalysisResult, Cue, CueSet, Section, TrackData};

/// Audio file extensions CueIt analyzes (lower-case, including the dot).
pub const AUDIO_EXTS: &[&str] = &[
    ".wav", ".aiff", ".aif", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wv",
];
