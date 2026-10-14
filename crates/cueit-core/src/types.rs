//! Domain model shared by the DSP engine, cue inference, exporters and UIs.
//!
//! Field names mirror the on-disk session format (snake_case) so a Rust
//! session file stays compatible with earlier CueIt releases.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// One structural section (intro / verse / drop / breakdown / outro).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Section {
    pub start: f64,
    pub end: f64,
    pub energy: f64,
    #[serde(default = "default_section_kind")]
    pub kind: String,
}

fn default_section_kind() -> String {
    "section".to_string()
}

impl Default for Section {
    fn default() -> Self {
        Self { start: 0.0, end: 0.0, energy: 0.0, kind: default_section_kind() }
    }
}

/// Everything the DSP engine measures about a track.
///
/// Every field is `Default`-able so partially-populated results (e.g. a decode
/// failure midway) are representable; callers should treat zero-length beat /
/// section vectors as "no result".
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub path: String,
    pub duration: f64,
    pub bpm: f64,
    pub bpm_confidence: f64,
    pub key: String,
    pub scale: String,
    pub key_strength: f64,
    pub camelot: String,
    pub beats: Vec<f64>,
    pub downbeats: Vec<f64>,
    pub onsets: Vec<f64>,
    pub sections: Vec<Section>,
    pub rms_times: Vec<f64>,
    pub rms: Vec<f64>,
    pub peak_mins: Vec<f64>,
    pub peak_maxs: Vec<f64>,
    pub integrated_loudness: f64,
    /// Beat-synchronous features, aligned with `beats`.
    pub beat_energy: Vec<f64>,
    pub beat_low: Vec<f64>,
    pub beat_flux: Vec<f64>,
    pub downbeat_phase: i32,
    pub downbeat_confidence: f64,
    /// Bars per phrase (4 or 8).
    pub phrase_length: i32,
}

impl Default for AnalysisResult {
    fn default() -> Self {
        Self {
            path: String::new(),
            duration: 0.0,
            bpm: 0.0,
            bpm_confidence: 0.0,
            key: String::new(),
            scale: String::new(),
            key_strength: 0.0,
            camelot: String::new(),
            beats: Vec::new(),
            downbeats: Vec::new(),
            onsets: Vec::new(),
            sections: Vec::new(),
            rms_times: Vec::new(),
            rms: Vec::new(),
            peak_mins: Vec::new(),
            peak_maxs: Vec::new(),
            integrated_loudness: 0.0,
            beat_energy: Vec::new(),
            beat_low: Vec::new(),
            beat_flux: Vec::new(),
            downbeat_phase: 0,
            downbeat_confidence: 0.0,
            phrase_length: 4,
        }
    }
}

/// A single hot cue.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cue {
    pub name: String,
    pub time: f64,
    pub color: String,
    #[serde(default = "default_cue_kind")]
    pub kind: String,
}

fn default_cue_kind() -> String {
    "hotcue".to_string()
}

impl Cue {
    pub fn new(name: impl Into<String>, time: f64, color: impl Into<String>) -> Self {
        Self { name: name.into(), time, color: color.into(), kind: default_cue_kind() }
    }
}

/// An ordered set of cues for one track.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CueSet {
    pub cues: Vec<Cue>,
}

impl CueSet {
    pub fn new(cues: Vec<Cue>) -> Self {
        Self { cues }
    }

    pub fn times(&self) -> Vec<f64> {
        self.cues.iter().map(|c| c.time).collect()
    }

    pub fn add(&mut self, cue: Cue) {
        self.cues.push(cue);
    }
}

/// A track: its analysis plus the cues inferred (or edited) for it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrackData {
    pub path: String,
    pub analysis: AnalysisResult,
    pub cues: Vec<Cue>,
}

impl TrackData {
    /// File name without its extension (used as the track title in exports).
    pub fn title(&self) -> String {
        Path::new(&self.path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.path.clone())
    }
}
