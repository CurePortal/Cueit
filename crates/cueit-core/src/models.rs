//! Neural beat/downbeat and key estimation via ONNX Runtime (native only).
//!
//! Two open models, both MIT, packaged as self-describing ONNX graphs:
//!
//! * **Beat This!** (CPJKU, ISMIR 2024) — beat + downbeat probabilities.
//! * **S-KEY / ChromaNet** (Deezer, ICASSP 2025) — 24-way key posterior.
//!
//! Weights are large (beat model ≈190 MB), so they are downloaded to the data
//! dir on first use and cached. Everything runs on 22050 Hz mono audio:
//! Beat This! needs a log-mel front-end (n_fft 1024, hop 441, 128 mels — the
//! shipped filterbank), S-KEY takes raw samples.
//!
//! This module is behind the `models` feature; the pure-Rust DSP stays the
//! fallback when the feature is off, when the models can't be fetched, or on
//! wasm.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ndarray::{Array2, Array3};
use ort::session::Session;
use ort::value::TensorRef;

use crate::analysis::dsp;

/// Beat This! provides beat/downbeat activations, but our log-mel front-end
/// needs validating against torchaudio before its raw beats are trustworthy, so
/// it is currently disabled and only S-KEY (self-contained) runs.
const USE_MODEL_BEATS: bool = false;

/// Shared ONNX Runtime environment (created once).
fn env() -> &'static ort::environment::Environment {
    static ENV: OnceLock<ort::environment::Environment> = OnceLock::new();
    ENV.get_or_init(|| ort::init().build().expect("failed to initialise ONNX Runtime"))
}

const SKY_URL: &str = "https://huggingface.co/musetric/skey-onnx/resolve/main/skey.onnx";
const BEAT_URL: &str = "https://huggingface.co/musetric/beat-this-onnx/resolve/main/beat_this.onnx";
const MEL_URL: &str = "https://huggingface.co/musetric/beat-this-onnx/resolve/main/mel-filterbank.bin";

// Beat This! front-end contract (from the model's config.json).
const N_FFT: usize = 1024;
const HOP: usize = 441;
const MEL_BINS: usize = 128;
const CHUNK: usize = 1500;
const FPS: f64 = 50.0;
const PEAK_KERNEL: usize = 7;

/// Key map from the S-KEY config, index -> key.
const SKEY_MAP: [(&str, bool); 24] = [
    ("A", true), ("A#", true), ("B", true), ("C", true), ("C#", true), ("D", true),
    ("D#", true), ("E", true), ("F", true), ("F#", true), ("G", true), ("G#", true),
    ("B", false), ("C", false), ("C#", false), ("D", false), ("D#", false), ("E", false),
    ("F", false), ("F#", false), ("G", false), ("G#", false), ("A", false), ("A#", false),
];

#[derive(Clone, Debug, Default)]
pub struct ModelOutputs {
    pub bpm: f64,
    pub bpm_confidence: f64,
    pub beats: Vec<f64>,
    pub downbeats: Vec<f64>,
    pub key: String,
    pub scale: String,
    pub key_strength: f64,
}

pub fn model_dir() -> PathBuf {
    let base = std::env::var("XDG_DATA_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            format!("{home}/.local/share")
        });
    PathBuf::from(base).join("CueIt").join("models")
}

/// Are the required models already present locally?
pub fn models_present() -> bool {
    let d = model_dir();
    let skey = d.join("skey.onnx").exists();
    if USE_MODEL_BEATS {
        skey && d.join("beat_this.onnx").exists() && d.join("mel.bin").exists()
    } else {
        skey
    }
}

fn ensure(url: &str, name: &str, on_progress: &dyn Fn(f32, &str)) -> Result<PathBuf, String> {
    let dir = model_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let dest = dir.join(name);
    if dest.exists() {
        return Ok(dest);
    }
    on_progress(0.0, &format!("downloading model {name}…"));
    let resp = ureq::get(url)
        .call()
        .map_err(|e| format!("download {name} failed: {e}"))?;
    let mut reader = resp.into_body().into_reader();
    let tmp = dest.with_extension("part");
    let mut file = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
    std::io::copy(&mut reader, &mut file).map_err(|e| format!("write {name} failed: {e}"))?;
    drop(file);
    std::fs::rename(&tmp, &dest).map_err(|e| e.to_string())?;
    Ok(dest)
}

/// Ensure all required model files are present, downloading what's missing.
pub fn ensure_models(on_progress: &dyn Fn(f32, &str)) -> Result<(), String> {
    ensure(SKY_URL, "skey.onnx", on_progress)?;
    if USE_MODEL_BEATS {
        ensure(BEAT_URL, "beat_this.onnx", on_progress)?;
        ensure(MEL_URL, "mel.bin", on_progress)?;
    }
    Ok(())
}

/// Run the models on 22050 Hz mono audio. `None` if they aren't present and
/// couldn't be fetched.
pub fn run(audio: &[f32], on_progress: &dyn Fn(f32, &str)) -> Option<ModelOutputs> {
    if !models_present() {
        if let Err(e) = ensure_models(on_progress) {
            on_progress(0.0, &format!("models unavailable: {e}"));
            return None;
        }
    }
    let dir = model_dir();

    on_progress(0.0, "S-KEY: key detection");
    let mut out = ModelOutputs::default();
    if let Some((name, scale, strength)) = skey(&dir.join("skey.onnx"), audio) {
        out.key = name;
        out.scale = scale;
        out.key_strength = strength;
    }

    if USE_MODEL_BEATS {
        on_progress(0.0, "Beat This!: beat tracking");
        if let Some((bpm, beats, downbeats)) = beat_this(&dir.join("beat_this.onnx"), &dir.join("mel.bin"), audio) {
            out.bpm = bpm;
            out.beats = beats;
            out.downbeats = downbeats;
            out.bpm_confidence = 0.9;
        }
    }

    if out.beats.is_empty() && out.key.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// S-KEY runs once per process — the ONNX session is cached, not reloaded per
/// track (that reload dominated batch time).
static SKEY_SESSION: Mutex<Option<Session>> = Mutex::new(None);

fn skey(model: &Path, audio: &[f32]) -> Option<(String, String, f64)> {
    let peak = audio.iter().fold(0f32, |a, &b| a.max(b.abs()));
    if peak <= 0.0 {
        return None;
    }
    let norm: Vec<f32> = audio.iter().map(|&x| x / peak).collect();
    let input = Array2::from_shape_vec((1, norm.len()), norm).ok()?;

    let mut guard = SKEY_SESSION.lock().ok()?;
    if guard.is_none() {
        *guard = Some(Session::builder(env()).ok()?.commit_from_file(model).ok()?);
    }
    let session = guard.as_mut()?;
    let outputs = session
        .run(ort::inputs!["audio" => TensorRef::from_array_view(&input).ok()?])
        .ok()?;
    let probs = outputs["probs"].try_extract_array::<f32>().ok()?;
    let flat: Vec<f32> = probs.iter().copied().collect();

    let mut best = 0usize;
    for (i, &p) in flat.iter().enumerate().take(24) {
        if p > flat[best] {
            best = i;
        }
    }
    let (note, major) = SKEY_MAP[best];
    let pc = note_to_pc(note);
    Some((
        dsp::key_name(pc).to_string(),
        if major { "major" } else { "minor" }.to_string(),
        flat[best] as f64,
    ))
}

fn note_to_pc(note: &str) -> usize {
    match note {
        "C" => 0, "C#" | "Db" => 1, "D" => 2, "D#" | "Eb" => 3, "E" => 4, "F" => 5,
        "F#" | "Gb" => 6, "G" => 7, "G#" | "Ab" => 8, "A" => 9, "A#" | "Bb" => 10, "B" => 11,
        _ => 0,
    }
}

// --------------------------------------------------------------------------- //
// Beat This!
// --------------------------------------------------------------------------- //

fn beat_this(model: &Path, mel_file: &Path, audio: &[f32]) -> Option<(f64, Vec<f64>, Vec<f64>)> {
    let fb = load_filterbank(mel_file)?; // [513 x 128] row-major
    let mel = log_mel(audio, &fb); // T x 128
    let t = mel.len();
    if t == 0 {
        return None;
    }

    let mut session = Session::builder(env()).ok()?.commit_from_file(model).ok()?;
    let mut beat_prob = vec![-1000f32; t];
    let mut down_prob = vec![-1000f32; t];

    // Reference split/aggregate (beat_this.inference): chunks of CHUNK frames,
    // overlapping by 2*BORDER, with BORDER frames discarded from each chunk's
    // edges (the model wasn't trained on input edges). The first chunk is
    // zero-padded on the left; the last is shifted to end at the piece's end.
    const BORDER: i64 = 6;
    let step = (CHUNK - 2 * BORDER as usize) as i64;
    let mut starts: Vec<i64> = Vec::new();
    let mut s = -BORDER;
    while s < t as i64 - BORDER {
        starts.push(s);
        s += step;
    }
    if t > step as usize {
        if let Some(last) = starts.last_mut() {
            *last = t as i64 - (CHUNK - BORDER as usize) as i64;
        }
    }

    // keep_first: process in reverse so earlier excerpts overwrite later ones.
    for &start in starts.iter().rev() {
        let mut buf = vec![0f32; CHUNK * MEL_BINS];
        for j in 0..CHUNK {
            let idx = start + j as i64;
            if idx >= 0 && (idx as usize) < t {
                let src = &mel[idx as usize];
                buf[j * MEL_BINS..(j + 1) * MEL_BINS].copy_from_slice(src);
            }
        }
        let input = Array3::from_shape_vec((1, CHUNK, MEL_BINS), buf).ok()?;
        let outputs = session
            .run(ort::inputs!["spect" => TensorRef::from_array_view(&input).ok()?])
            .ok()?;
        let b = outputs["beat"].try_extract_array::<f32>().ok()?;
        let d = outputs["downbeat"].try_extract_array::<f32>().ok()?;
        let bv: Vec<f32> = b.iter().copied().collect();
        let dv: Vec<f32> = d.iter().copied().collect();
        for j in 0..(CHUNK - 2 * BORDER as usize) {
            let idx = start + BORDER + j as i64;
            if idx >= 0 && (idx as usize) < t {
                beat_prob[idx as usize] = bv[BORDER as usize + j];
                down_prob[idx as usize] = dv[BORDER as usize + j];
            }
        }
    }

    let beats: Vec<f64> = peak_frames(&beat_prob).into_iter().map(|f| f / FPS).collect();
    // Reference post-processing: move each downbeat to the nearest beat, dedupe.
    let mut downbeats: Vec<f64> = peak_frames(&down_prob)
        .into_iter()
        .filter_map(|f| {
            let t = f / FPS;
            beats
                .iter()
                .cloned()
                .min_by(|a, b| (a - t).abs().partial_cmp(&(b - t).abs()).unwrap())
        })
        .collect();
    downbeats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    downbeats.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    if beats.len() < 2 {
        return None;
    }
    let mut bpm = bpm_from_beats(&beats);

    // The model fires extra beats/downbeats in busy passages, corrupting a median
    // tempo. The true bar length is the *dominant* downbeat gap (mode), so take
    // that for tempo; leave the model's own grid untouched.
    if downbeats.len() >= 4 {
        let mut rounded: Vec<i64> = downbeats
            .windows(2)
            .map(|w| ((w[1] - w[0]) * 50.0).round() as i64)
            .filter(|g| *g > 5)
            .collect();
        if !rounded.is_empty() {
            rounded.sort_unstable();
            let mut best = (0i64, 0usize);
            for &g in &rounded {
                let cnt = rounded.iter().filter(|x| (**x - g).abs() <= 1).count();
                if cnt > best.1 {
                    best = (g, cnt);
                }
            }
            let bar = best.0 as f64 / 50.0;
            if bar > 0.3 {
                bpm = 240.0 / bar;
            }
        }
    }
    Some((bpm, beats, downbeats))
}

fn bpm_from_beats(beats: &[f64]) -> f64 {
    let mut gaps: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).filter(|g| *g > 0.05).collect();
    if gaps.is_empty() {
        return 0.0;
    }
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = gaps[gaps.len() / 2];
    if median <= 0.0 {
        0.0
    } else {
        60.0 / median
    }
}

/// Peak picking exactly as Beat This! does it: a frame is a peak if it equals
/// the ±3-frame max-pool (kernel 7) and its logit is > 0 (probability > 0.5),
/// then groups of adjacent peaks (≤1 frame apart) are averaged.
fn peak_frames(logits: &[f32]) -> Vec<f64> {
    let k = PEAK_KERNEL / 2;
    let mut peaks: Vec<usize> = Vec::new();
    for i in 0..logits.len() {
        let lo = i.saturating_sub(k);
        let hi = (i + k + 1).min(logits.len());
        let m = logits[lo..hi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if logits[i] == m && logits[i] > 0.0 {
            peaks.push(i);
        }
    }
    deduplicate_peaks(&peaks, 1)
}

/// Replace groups of peaks ≤ `width` frames apart with their average frame.
fn deduplicate_peaks(peaks: &[usize], width: usize) -> Vec<f64> {
    let mut result = Vec::new();
    let mut it = peaks.iter().copied();
    let mut p = match it.next() {
        Some(x) => x as f64,
        None => return result,
    };
    let mut count = 1.0f64;
    for p2 in it {
        if (p2 as f64 - p) <= width as f64 {
            count += 1.0;
            p += (p2 as f64 - p) / count;
        } else {
            result.push(p);
            p = p2 as f64;
            count = 1.0;
        }
    }
    result.push(p);
    result
}

// --------------------------------------------------------------------------- //
// log-mel front-end
// --------------------------------------------------------------------------- //

fn load_filterbank(path: &Path) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 4 {
        return None;
    }
    Some(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Log-mel for Beat This!: periodic Hann, center reflect pad, magnitude /
/// sqrt(n_fft), mel projection, then `log1p(1000 * x)`.
fn log_mel(audio: &[f32], fb: &[f32]) -> Vec<[f32; MEL_BINS]> {
    use rustfft::num_complex::Complex32;
    use rustfft::FftPlanner;

    let n = audio.len();
    if n == 0 {
        return Vec::new();
    }
    let pad = N_FFT / 2;
    let window = dsp::hann(N_FFT);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut buf = vec![Complex32::new(0.0, 0.0); N_FFT];

    let frames = 1 + n / HOP;
    let bins = N_FFT / 2 + 1;
    let norm = 1.0 / (N_FFT as f32).sqrt();
    let mut out = Vec::with_capacity(frames);

    for f in 0..frames {
        let start = f * HOP;
        for i in 0..N_FFT {
            let idx = start + i;
            let s = if idx >= pad && idx - pad < n {
                audio[idx - pad]
            } else {
                // reflect (mirror) padding, clamped for very short clips
                let m = (idx as isize - pad as isize).abs().min(n as isize * 2 - 1);
                audio[(m as usize).min(n - 1)]
            };
            buf[i] = Complex32::new(s * window[i], 0.0);
        }
        fft.process(&mut buf);
        let mut mel = [0f32; MEL_BINS];
        for m in 0..MEL_BINS {
            let mut acc = 0.0f32;
            for k in 0..bins {
                acc += buf[k].norm() * norm * fb[k * MEL_BINS + m];
            }
            mel[m] = (1000.0 * acc).ln_1p();
        }
        out.push(mel);
    }
    out
}