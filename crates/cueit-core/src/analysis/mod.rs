//! Pure-Rust DSP engine: turn decoded mono PCM into an [`AnalysisResult`].
//!
//! This replaces Essentia entirely. It is `wasm32`-compatible (no FFI, no OS
//! threads required); the caller supplies decoded `f32` samples. Native builds
//! can also decode a file with ffmpeg via [`analyze_file`].
//!
//! Pipeline (mirrors the original `cueit/analysis.py`):
//! 1. spectral flux -> tempo (autocorrelation) -> beat grid (DP tracking)
//! 2. beat-synchronous kick / flux / RMS / snare features
//! 3. chroma -> key/scale (Krumhansl) -> Camelot
//! 4. onsets, integrated loudness, RMS envelope, waveform peaks
//! 5. MFCC novelty segmentation -> snap to grid -> label -> downbeats, phrase

pub mod dsp;
pub mod rhythm;
pub mod structure;

use crate::types::{AnalysisResult, Section};

pub use dsp::{integrated_loudness, rms_envelope, waveform_peaks};

use std::sync::atomic::{AtomicBool, Ordering};

static MODELS_ENABLED: AtomicBool = AtomicBool::new(true);

/// Enable/disable the neural models at runtime (desktop/CLI).
pub fn set_models_enabled(enabled: bool) {
    MODELS_ENABLED.store(enabled, Ordering::SeqCst);
}

/// Whether the neural models may be used.
pub fn models_enabled() -> bool {
    MODELS_ENABLED.load(Ordering::SeqCst)
}

/// Default analysis sample rate (matches the original).
pub const DEFAULT_SR: u32 = 44_100;

const STFT_FRAME: usize = 2048;
const STFT_HOP: usize = 512;
const MFCC_FRAME: usize = 4096;
const MFCC_HOP: usize = 2048;
const MFCC_COEFFS: usize = 13;
const MFCC_MELS: usize = 40;

// --------------------------------------------------------------------------- //
// shared math helpers
// --------------------------------------------------------------------------- //

pub(crate) fn zscore(x: &[f64]) -> Vec<f64> {
    let (m, s) = mean_std(x);
    x.iter().map(|&v| (v - m) / (s + 1e-9)).collect()
}

pub(crate) fn mean_std(x: &[f64]) -> (f64, f64) {
    if x.is_empty() {
        return (0.0, 0.0);
    }
    let m = x.iter().sum::<f64>() / x.len() as f64;
    let v = x.iter().map(|e| (e - m).powi(2)).sum::<f64>() / x.len() as f64;
    (m, v.sqrt())
}

pub(crate) fn argmax(x: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best
}

// --------------------------------------------------------------------------- //
// key detection (Krumhansl-Kessler profiles)
// --------------------------------------------------------------------------- //

const KK_MAJOR: [f64; 12] =
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const KK_MINOR: [f64; 12] =
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

fn correlation(a: &[f64; 12], b: &[f64; 12]) -> f64 {
    let ma = a.iter().sum::<f64>() / 12.0;
    let mb = b.iter().sum::<f64>() / 12.0;
    let mut num = 0.0;
    let (mut sa, mut sb) = (0.0, 0.0);
    for i in 0..12 {
        let (da, db) = (a[i] - ma, b[i] - mb);
        num += da * db;
        sa += da * da;
        sb += db * db;
    }
    num / ((sa.sqrt() * sb.sqrt()) + 1e-9)
}

/// Estimate (key name, scale, strength) from a chroma vector.
pub fn estimate_key(chroma: &[f32; 12]) -> (String, String, f64) {
    let chr: [f64; 12] = std::array::from_fn(|i| chroma[i] as f64);
    let mut best = (0usize, true, f64::NEG_INFINITY);
    for root in 0..12 {
        let rotated: [f64; 12] = std::array::from_fn(|i| chr[(root + i) % 12]);
        for (major, profile) in [(true, &KK_MAJOR), (false, &KK_MINOR)] {
            let c = correlation(&rotated, profile);
            if c > best.2 {
                best = (root, major, c);
            }
        }
    }
    let (root, major, strength) = best;
    let scale = if major { "major" } else { "minor" };
    (dsp::key_name(root).to_string(), scale.to_string(), strength.clamp(0.0, 1.0))
}

/// Camelot wheel code for a pitch class (0 = C) and major/minor mode.
pub fn camelot(pc: usize, major: bool) -> &'static str {
    const MAJ: [&str; 12] = [
        "8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B",
    ];
    const MIN: [&str; 12] = [
        "5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A",
    ];
    if major {
        MAJ[pc % 12]
    } else {
        MIN[pc % 12]
    }
}

// --------------------------------------------------------------------------- //
// orchestration
// --------------------------------------------------------------------------- //

/// Optional progress callback: `(fraction, message)`.
pub type Progress<'a> = dyn Fn(f32, &str) + 'a;

/// Optional beat/key results from the neural models, overriding the DSP.
#[derive(Default, Clone)]
pub struct ModelOverrides {
    pub bpm: Option<f64>,
    pub bpm_confidence: Option<f64>,
    pub beats: Option<Vec<f64>>,
    pub downbeats: Option<Vec<f64>>,
    pub downbeat_confidence: Option<f64>,
    pub key: Option<(String, String, f64)>,
}

/// Analyze decoded mono PCM into a full [`AnalysisResult`].
///
/// `progress` is an optional `(fraction, message)` callback.
pub fn analyze_pcm(
    path: &str,
    audio: &[f32],
    sr: u32,
    progress: Option<&Progress<'_>>,
) -> AnalysisResult {
    analyze_pcm_with(path, audio, sr, &ModelOverrides::default(), progress)
}

/// Analyze decoded mono PCM, optionally using model-provided beats/key.
pub fn analyze_pcm_with(
    path: &str,
    audio: &[f32],
    sr: u32,
    overrides: &ModelOverrides,
    progress: Option<&Progress<'_>>,
) -> AnalysisResult {
    let tick = |p: f32, msg: &str| {
        if let Some(cb) = progress {
            cb(p, msg);
        }
    };
    let sr_f = sr as f64;
    let duration = audio.len() as f64 / sr_f;

    tick(0.20, "tempo & beats");
    let (odf, odf_times) = dsp::spectral_flux(audio, sr, STFT_FRAME, STFT_HOP);
    let hop_t = STFT_HOP as f64 / sr_f;
    let (bpm_dsp, conf_dsp) = rhythm::estimate_tempo(&odf, hop_t);
    let beats = match &overrides.beats {
        Some(b) => b.clone(),
        None => rhythm::track_beats(&odf, hop_t, bpm_dsp, duration),
    };
    let bpm = overrides.bpm.unwrap_or(bpm_dsp);
    let bpm_conf = overrides.bpm_confidence.unwrap_or(conf_dsp);

    tick(0.40, "beat features");
    let (beat_low, beat_flux, beat_rms, beat_snare) =
        rhythm::beat_sync_features(audio, sr, &beats, STFT_FRAME, STFT_HOP, 180.0);

    tick(0.55, "key detection");
    let (key, scale, key_strength) = match &overrides.key {
        Some((k, s, st)) => (k.clone(), s.clone(), *st),
        None => estimate_key(&dsp::chroma(audio, sr)),
    };
    let camelot_code = camelot(
        (0..12).find(|&i| dsp::key_name(i) == key.as_str()).unwrap_or(0),
        scale == "major",
    )
    .to_string();

    tick(0.68, "onsets & loudness");
    let onsets = dsp::pick_onsets(&odf, &odf_times, 0.03);
    let loudness = dsp::integrated_loudness(audio, sr);

    tick(0.80, "structure & downbeats");
    let (downbeats, phase, db_conf) = match &overrides.downbeats {
        Some(d) if !d.is_empty() => (d.clone(), 0, overrides.downbeat_confidence.unwrap_or(0.9)),
        _ => rhythm::estimate_downbeats(&beats, &beat_low, &beat_flux, &beat_rms, Some(&beat_snare)),
    };
    let phrase = rhythm::estimate_phrase_length(&beats, &beat_low, &beat_flux);
    let (times, rms) = dsp::rms_envelope(audio, sr, 0.5);
    let (peak_mins, peak_maxs) = dsp::waveform_peaks(audio, 3000);

    let mfcc = dsp::mfcc_frames(audio, sr, MFCC_FRAME, MFCC_HOP, MFCC_COEFFS, MFCC_MELS);
    let mut raw = structure::sections_from_features(&mfcc, sr, MFCC_HOP, duration);
    if raw.len() < 2 {
        raw = structure::fallback_sections(&rms, &times, duration);
    }
    let raw = structure::snap_sections(&raw, &downbeats, &beats, duration, 1.0);

    let rms_peak = rms.iter().cloned().fold(0.0f64, f64::max).max(1e-9);
    let mut sections: Vec<Section> = raw
        .iter()
        .map(|&(a, b)| {
            let mut sum = 0.0;
            let mut cnt = 0.0;
            for (i, &t) in times.iter().enumerate() {
                if t >= a && t < b {
                    sum += rms[i];
                    cnt += 1.0;
                }
            }
            Section {
                start: a,
                end: b,
                energy: if cnt > 0.0 { sum / cnt / rms_peak } else { 0.0 },
                kind: "section".to_string(),
            }
        })
        .collect();
    structure::label_sections(&mut sections);

    tick(1.0, "done");
    AnalysisResult {
        path: path.to_string(),
        duration,
        bpm,
        bpm_confidence: bpm_conf,
        key,
        scale,
        key_strength,
        camelot: camelot_code,
        beats,
        downbeats,
        onsets,
        sections,
        rms_times: times,
        rms,
        peak_mins,
        peak_maxs,
        integrated_loudness: loudness,
        beat_energy: beat_rms,
        beat_low,
        beat_flux,
        downbeat_phase: phase,
        downbeat_confidence: db_conf,
        phrase_length: phrase,
    }
}

// --------------------------------------------------------------------------- //
// native decode
// --------------------------------------------------------------------------- //

/// Decode an audio file to mono `f32` via ffmpeg (desktop/CLI only).
#[cfg(feature = "native-audio")]
pub fn decode_mono(path: &str, sample_rate: u32) -> Result<Vec<f32>, String> {
    use std::process::Command;
    let out = Command::new("ffmpeg")
        .args([
            "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar",
            &sample_rate.to_string(), "-",
        ])
        .output()
        .map_err(|e| format!("ffmpeg not runnable ({e}); install ffmpeg to decode audio"))?;
    if !out.status.success() {
        return Err(format!(
            "ffmpeg failed to decode {path}: {}",
            String::from_utf8_lossy(&out.stderr).chars().take(400).collect::<String>()
        ));
    }
    Ok(out
        .stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Decode and analyze a file (desktop/CLI only).
#[cfg(feature = "native-audio")]
pub fn analyze_file(
    path: &str,
    sample_rate: u32,
    progress: Option<&Progress<'_>>,
) -> Result<AnalysisResult, String> {
    if let Some(cb) = progress {
        cb(0.02, "decoding audio");
    }
    let audio = decode_mono(path, sample_rate)?;
    if audio.is_empty() {
        return Err(format!("empty audio: {path}"));
    }

    #[cfg(feature = "models")]
    let overrides = model_overrides(path, progress);
    #[cfg(not(feature = "models"))]
    let overrides = ModelOverrides::default();

    Ok(analyze_pcm_with(path, &audio, sample_rate, &overrides, progress))
}

/// Run the neural models (if available) on a 22050 Hz mean-downmix copy.
#[cfg(all(feature = "native-audio", feature = "models"))]
fn model_overrides(path: &str, progress: Option<&Progress<'_>>) -> ModelOverrides {
    if !models_enabled() {
        return ModelOverrides::default();
    }
    let Ok(audio22) = decode_mean_mono(path, 22050) else {
        return ModelOverrides::default();
    };
    if audio22.is_empty() {
        return ModelOverrides::default();
    }
    let cb = |p: f32, m: &str| {
        if let Some(f) = progress {
            f(p, m);
        }
    };
    match crate::models::run(&audio22, &cb) {
        Some(o) => ModelOverrides {
            key: (!o.key.is_empty()).then(|| (o.key.clone(), o.scale.clone(), o.key_strength)),
            ..ModelOverrides::default()
        },
        None => ModelOverrides::default(),
    }
}

/// Decode to mono at `sr`, downmixing stereo by the arithmetic mean (the
/// contract Beat This! expects; ffmpeg's `-ac 1` rematrix is energy-preserving
/// and would shift the log-mel features).
#[cfg(all(feature = "native-audio", feature = "models"))]
fn decode_mean_mono(path: &str, sr: u32) -> Result<Vec<f32>, String> {
    use std::process::Command;
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", &sr.to_string(), "-"])
        .output()
        .map_err(|e| format!("ffmpeg not runnable ({e})"))?;
    if !out.status.success() {
        return Err(format!("ffmpeg failed to decode {path}"));
    }
    let samples: Vec<f32> = out
        .stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(samples.chunks_exact(2).map(|c| 0.5 * (c[0] + c[1])).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f64, sr: u32, dur: f64) -> Vec<f32> {
        (0..(sr as f64 * dur) as usize)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin() as f32)
            .collect()
    }

    #[test]
    fn sine_wave_reports_its_pitch_class() {
        // A5 = 880 Hz is pitch class A (pc 9).
        let sr = 44_100;
        let audio = sine(880.0, sr, 2.0);
        let chr = dsp::chroma(&audio, sr);
        let (key, scale, strength) = estimate_key(&chr);
        assert_eq!(key, "A");
        assert!(scale == "major" || scale == "minor");
        assert!(strength > 0.5, "strength {strength}");
    }

    #[test]
    fn camelot_wheel_is_consistent() {
        assert_eq!(camelot(0, true), "8B"); // C major
        assert_eq!(camelot(9, false), "8A"); // A minor
        assert_eq!(camelot(1, true), "3B"); // Db major
        assert_eq!(camelot(11, true), "1B"); // B major
    }

    #[test]
    fn produces_a_consistent_result_on_a_click_track() {
        let sr = 44_100u32;
        let bpm = 126.0f64;
        let dur = 12.0f64;
        let mut audio = vec![0f32; (sr as f64 * dur) as usize];
        let period = 60.0 / bpm;
        let mut t = 0.0;
        while t < dur {
            let start = (t * sr as f64) as usize;
            for i in 0..300.min(audio.len().saturating_sub(start)) {
                let env = (-(i as f32) / 50.0).exp();
                audio[start + i] += 0.8 * env;
            }
            t += period;
        }
        let r = analyze_pcm("click.wav", &audio, sr, None);
        assert!((r.bpm - bpm).abs() < 3.0, "bpm {}", r.bpm);
        assert!(!r.beats.is_empty());
        assert!(!r.downbeats.is_empty());
        assert_eq!(r.peak_mins.len(), 3000);
        assert!(r.duration > 11.9 && r.duration < 12.1);
    }
}
