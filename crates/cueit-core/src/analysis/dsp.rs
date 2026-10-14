//! Low-level signal processing primitives (pure Rust, `wasm32`-safe).
//!
//! Deliberately allocation-light and streaming: we never materialise a full
//! spectrogram (that would be ~100 MB for a 5-minute track), only the derived
//! ODF, MFCC frames, chroma and loudness.

use std::f32::consts::PI;
use std::sync::Arc;

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

/// Periodic Hann window.
pub fn hann(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())
        .collect()
}

/// Reusable real-input STFT via a complex FFT, exposing the magnitude spectrum.
pub struct Stft {
    fft: Arc<dyn Fft<f32>>,
    frame: usize,
    window: Vec<f32>,
    buf: Vec<Complex32>,
    bins: usize,
}

impl Stft {
    pub fn new(frame: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        Self {
            fft: planner.plan_fft_forward(frame),
            frame,
            window: hann(frame),
            buf: vec![Complex32::new(0.0, 0.0); frame],
            bins: frame / 2 + 1,
        }
    }

    pub fn bins(&self) -> usize {
        self.bins
    }

    pub fn frame(&self) -> usize {
        self.frame
    }

    /// Magnitude spectrum of the windowed frame starting at `start` (zero-padded
    /// past the end of `audio`). Writes exactly `bins()` values into `out`.
    pub fn magnitude(&mut self, audio: &[f32], start: usize, out: &mut Vec<f32>) {
        for i in 0..self.frame {
            let s = audio.get(start + i).copied().unwrap_or(0.0);
            self.buf[i] = Complex32::new(s * self.window[i], 0.0);
        }
        self.fft.process(&mut self.buf);
        out.clear();
        out.extend(self.buf[..self.bins].iter().map(|c| c.norm()));
    }
}

/// Number of whole frames that fit in `len` samples.
pub fn frame_count(len: usize, frame: usize, hop: usize) -> usize {
    if len < frame {
        0
    } else {
        (len - frame) / hop + 1
    }
}

/// Positive spectral flux (onset detection function) and its frame times.
pub fn spectral_flux(
    audio: &[f32],
    sr: u32,
    frame: usize,
    hop: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut stft = Stft::new(frame);
    let n = frame_count(audio.len(), frame, hop);
    let mut odf = Vec::with_capacity(n);
    let mut times = Vec::with_capacity(n);
    let mut cur = Vec::new();
    let mut prev = vec![0f32; stft.bins()];
    for i in 0..n {
        stft.magnitude(audio, i * hop, &mut cur);
        let mut flux = 0.0f64;
        if i > 0 {
            for k in 0..cur.len() {
                let d = cur[k] - prev[k];
                if d > 0.0 {
                    flux += d as f64;
                }
            }
        }
        odf.push(flux);
        times.push((i * hop) as f64 / sr as f64);
        prev.copy_from_slice(&cur);
    }
    (odf, times)
}

/// Peak-pick the ODF: local maxima above `mean + k*std`, at least `min_gap`
/// seconds apart. Used to expose "onsets" to the UI.
pub fn pick_onsets(odf: &[f64], times: &[f64], min_gap: f64) -> Vec<f64> {
    if odf.is_empty() {
        return Vec::new();
    }
    let mean = odf.iter().sum::<f64>() / odf.len() as f64;
    let var = odf.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / odf.len() as f64;
    let thr = mean + var.sqrt();
    let mut out: Vec<f64> = Vec::new();
    for i in 1..odf.len().saturating_sub(1) {
        if odf[i] > thr && odf[i] >= odf[i - 1] && odf[i] >= odf[i + 1] {
            let t = times[i];
            if out.last().map(|&p| t - p >= min_gap).unwrap_or(true) {
                out.push(t);
            }
        }
    }
    out
}

// --------------------------------------------------------------------------- //
// mel / MFCC
// --------------------------------------------------------------------------- //

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10f32.powf(m / 2595.0) - 1.0)
}

/// Sparse triangular mel filterbank: one `Vec<(bin, weight)>` per filter.
fn mel_filterbank(n_mels: usize, frame: usize, sr: u32) -> Vec<Vec<(usize, f32)>> {
    let n_bins = frame / 2 + 1;
    let fmax = sr as f32 / 2.0;
    let (mel_min, mel_max) = (hz_to_mel(0.0), hz_to_mel(fmax));
    let mut points = vec![0f32; n_mels + 2];
    for (i, p) in points.iter_mut().enumerate() {
        *p = mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32);
    }
    let mut filters = Vec::with_capacity(n_mels);
    for m in 1..=n_mels {
        let (lo, ce, hi) = (points[m - 1], points[m], points[m + 1]);
        let mut f = Vec::new();
        for k in 0..n_bins {
            let hz = k as f32 * sr as f32 / frame as f32;
            let w = if hz >= lo && hz <= ce {
                (hz - lo) / (ce - lo + 1e-9)
            } else if hz > ce && hz <= hi {
                (hi - hz) / (hi - ce + 1e-9)
            } else {
                0.0
            };
            if w > 0.0 {
                f.push((k, w));
            }
        }
        filters.push(f);
    }
    filters
}

/// Per-frame MFCCs (log-mel + DCT-II), no liftering.
pub fn mfcc_frames(
    audio: &[f32],
    sr: u32,
    frame: usize,
    hop: usize,
    n_mfcc: usize,
    n_mels: usize,
) -> Vec<Vec<f32>> {
    let mut stft = Stft::new(frame);
    let n = frame_count(audio.len(), frame, hop);
    let filters = mel_filterbank(n_mels, frame, sr);
    let mut out = Vec::with_capacity(n);
    let mut mags = Vec::new();
    for i in 0..n {
        stft.magnitude(audio, i * hop, &mut mags);
        let mut mel = vec![0f32; n_mels];
        for (m, filt) in filters.iter().enumerate() {
            let mut e = 0.0f32;
            for &(k, w) in filt {
                e += mags[k] * mags[k] * w;
            }
            mel[m] = e.max(1e-10).ln();
        }
        let mut coeffs = vec![0f32; n_mfcc];
        for (c, coeff) in coeffs.iter_mut().enumerate() {
            let mut s = 0.0f32;
            for (m, &energy) in mel.iter().enumerate() {
                s += energy * (PI * c as f32 / n_mels as f32 * (m as f32 + 0.5)).cos();
            }
            *coeff = s;
        }
        out.push(coeffs);
    }
    out
}

// --------------------------------------------------------------------------- //
// chroma / loudness
// --------------------------------------------------------------------------- //

const KEY_NAMES: [&str; 12] =
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

pub fn key_name(pc: usize) -> &'static str {
    KEY_NAMES[pc % 12]
}

/// Pitch-class energy (chroma) accumulated over the whole track.
pub fn chroma(audio: &[f32], sr: u32) -> [f32; 12] {
    let frame = 4096;
    let hop = 2048;
    let mut stft = Stft::new(frame);
    let n = frame_count(audio.len(), frame, hop);
    let mut chr = [0f32; 12];
    let mut mags = Vec::new();
    for i in 0..n {
        stft.magnitude(audio, i * hop, &mut mags);
        for (k, &mag) in mags.iter().enumerate().skip(1) {
            if mag <= 1.0 {
                continue;
            }
            let freq = k as f32 * sr as f32 / frame as f32;
            if !(55.0..=2000.0).contains(&freq) {
                continue;
            }
            let midi = 69.0 + 12.0 * (freq / 440.0).log2();
            let pc = ((midi.round() as i32) % 12 + 12) % 12;
            chr[pc as usize] += mag;
        }
    }
    let norm = chr.iter().cloned().fold(0.0f32, f32::max) + 1e-9;
    for c in chr.iter_mut() {
        *c /= norm;
    }
    chr
}

/// Approximate integrated loudness (dBFS-like). Good enough for display and
/// relative comparison; not a certified EBU R128 meter.
pub fn integrated_loudness(audio: &[f32], sr: u32) -> f64 {
    if audio.is_empty() {
        return 0.0;
    }
    let block = (sr as usize / 4).max(1); // 250 ms
    let mut loud = Vec::new();
    let mut i = 0;
    while i < audio.len() {
        let seg = &audio[i..(i + block).min(audio.len())];
        let ms = seg.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / seg.len() as f64;
        if ms > 1e-10 {
            loud.push(10.0 * ms.log10());
        }
        i += block;
    }
    if loud.is_empty() {
        return -70.0;
    }
    let max = loud.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let gated: Vec<f64> = loud.iter().cloned().filter(|&l| l > max - 20.0).collect();
    let n = gated.len().max(1) as f64;
    gated.iter().sum::<f64>() / n
}

/// (mins, maxs) envelope, one entry per bucket, for drawing waveforms.
pub fn waveform_peaks(audio: &[f32], buckets: usize) -> (Vec<f64>, Vec<f64>) {
    let n = audio.len();
    if n == 0 {
        return (vec![0.0; buckets], vec![0.0; buckets]);
    }
    let buckets = buckets.clamp(1, n);
    let mut mins = vec![0f64; buckets];
    let mut maxs = vec![0f64; buckets];
    for b in 0..buckets {
        let start = b * n / buckets;
        let end = ((b + 1) * n / buckets).max(start + 1).min(n);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &s in &audio[start..end] {
            lo = lo.min(s);
            hi = hi.max(s);
        }
        mins[b] = lo as f64;
        maxs[b] = hi as f64;
    }
    (mins, maxs)
}

/// RMS loudness envelope with a `hop_s`-second hop.
pub fn rms_envelope(audio: &[f32], sr: u32, hop_s: f64) -> (Vec<f64>, Vec<f64>) {
    let hop = ((hop_s * sr as f64) as usize).max(1);
    let frames = (audio.len() / hop).max(1);
    let mut rms = Vec::with_capacity(frames);
    let mut times = Vec::with_capacity(frames);
    for i in 0..frames {
        let seg = &audio[i * hop..((i + 1) * hop).min(audio.len())];
        let v = if seg.is_empty() {
            0.0
        } else {
            (seg.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / seg.len() as f64).sqrt()
        };
        rms.push(v);
        times.push((i * hop) as f64 / sr as f64);
    }
    (times, rms)
}
