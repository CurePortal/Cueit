//! Rhythm analysis: tempo, beat tracking, beat-synchronous features, downbeat
//! phase and phrase length.
//!
//! The tempo/beat stage is original (autocorrelation tempo + Ellis-style
//! dynamic-programming beat tracking). The downbeat / phrase / beat-feature
//! stages are direct ports of the heuristics in the original `analysis.py`, so
//! the musical decisions stay identical once a beat grid exists.

use super::dsp::{frame_count, Stft};
use super::{argmax, mean_std, zscore};

/// Global tempo (BPM) and confidence from an onset-detection function.
///
/// Confidence is `(acf_peak / acf_mean - 1)` clipped to `[0, 2]`, halved, so
/// `0.0..=1.0` like the value the original read from Essentia.
#[allow(clippy::needless_range_loop)] // index math is clearer than iterators here
pub fn estimate_tempo(odf: &[f64], hop_t: f64) -> (f64, f64) {
    let n = odf.len();
    if n < 8 {
        return (0.0, 0.0);
    }
    let min_lag = (60.0 / (200.0 * hop_t)).floor().max(1.0) as usize;
    let max_lag = ((60.0 / (60.0 * hop_t)).ceil() as usize).min(n - 1);
    if min_lag >= max_lag {
        return (0.0, 0.0);
    }
    let mut acf = vec![0f64; max_lag + 1];
    for (lag, a) in acf.iter_mut().enumerate().skip(min_lag) {
        let mut s = 0.0;
        for i in 0..n - lag {
            s += odf[i] * odf[i + lag];
        }
        *a = s / (n - lag) as f64;
    }
    let mut best_lag = min_lag;
    let mut best = f64::NEG_INFINITY;
    for lag in min_lag..=max_lag {
        let bpm = 60.0 / (lag as f64 * hop_t);
        let prior = (-0.5 * ((bpm / 120.0).abs().ln() / 0.9).powi(2)).exp();
        let v = acf[lag] * prior;
        if v > best {
            best = v;
            best_lag = lag;
        }
    }
    let refined = if best_lag > min_lag && best_lag < max_lag {
        let (y0, y1, y2) = (acf[best_lag - 1], acf[best_lag], acf[best_lag + 1]);
        let denom = y0 - 2.0 * y1 + y2;
        let delta = if denom.abs() > 1e-12 { 0.5 * (y0 - y2) / denom } else { 0.0 };
        best_lag as f64 + delta.clamp(-1.0, 1.0)
    } else {
        best_lag as f64
    };
    let bpm = 60.0 / (refined * hop_t);
    let mean = acf[min_lag..=max_lag].iter().sum::<f64>() / (max_lag - min_lag + 1) as f64;
    let conf = if mean.abs() > 1e-12 {
        ((acf[best_lag] / mean - 1.0).clamp(0.0, 2.0)) / 2.0
    } else {
        0.0
    };
    (bpm, conf)
}

/// Ellis-style dynamic-programming beat tracker over an onset function.
#[allow(clippy::needless_range_loop)] // DP indexing over the cumulative score
pub fn track_beats(odf: &[f64], hop_t: f64, bpm: f64, duration: f64) -> Vec<f64> {
    if bpm <= 0.0 {
        return Vec::new();
    }
    let period = 60.0 / (bpm * hop_t);
    if period < 1.0 {
        return Vec::new();
    }
    let n = odf.len();
    let norm = zscore(odf);
    let pos: Vec<f64> = norm.iter().map(|&v| v.max(0.0)).collect();
    const TIGHTNESS: f64 = 100.0;

    let mut cum = vec![f64::NEG_INFINITY; n];
    let mut back = vec![-1i64; n];
    for i in 0..n {
        let lo = (i as f64 - 2.0 * period).max(0.0) as usize;
        let hi = ((i as f64 - 0.5 * period).max(0.0) as usize).min(i);
        let mut best = f64::NEG_INFINITY;
        let mut bj = -1i64;
        for j in lo..hi {
            let ratio = (i - j) as f64 / period;
            let sc = cum[j] - TIGHTNESS * ratio.ln().powi(2);
            if sc > best {
                best = sc;
                bj = j as i64;
            }
        }
        if bj >= 0 {
            cum[i] = pos[i] + best;
            back[i] = bj;
        } else {
            cum[i] = pos[i];
        }
    }

    let mut best_i = 0usize;
    let mut best = f64::NEG_INFINITY;
    for (i, &c) in cum.iter().enumerate() {
        if c > best {
            best = c;
            best_i = i;
        }
    }
    let mut frames = Vec::new();
    let mut i = best_i as i64;
    while i >= 0 {
        frames.push(i as usize);
        i = back[i as usize];
    }
    frames.reverse();

    let mut times: Vec<f64> =
        frames.iter().map(|&f| f as f64 * hop_t).filter(|&t| t <= duration + 1e-6).collect();
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    times
}

/// Kick energy, low-band flux, RMS and mid/snare flux sampled at each beat.
///
/// Kick (<=180 Hz) and low flux line up with bar starts; mid-band flux (snare
/// accents) hits backbeats 2 & 4, so a correct downbeat has a strong kick on
/// beat 1 *and* snare accents on 2 and 4.
pub fn beat_sync_features(
    audio: &[f32],
    sr: u32,
    beats: &[f64],
    frame: usize,
    hop: usize,
    fmax: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut stft = Stft::new(frame);
    let bin = sr as f64 / frame as f64;
    let kl = ((fmax / bin) as usize).max(1);
    let m0 = (kl + 1).max((200.0 / bin) as usize);
    let m1 = (frame / 2).min((4000.0 / bin) as usize);
    let n = frame_count(audio.len(), frame, hop);

    let mut kick = vec![0f32; n];
    let mut low_flux = vec![0f32; n];
    let mut mid_flux = vec![0f32; n];
    let mut mags = Vec::new();
    let (mut prev_low, mut prev_mid): (Option<Vec<f32>>, Option<Vec<f32>>) = (None, None);
    for i in 0..n {
        stft.magnitude(audio, i * hop, &mut mags);
        let low: Vec<f32> = mags[..kl.min(mags.len())].to_vec();
        let mid: Vec<f32> = mags[m0.min(mags.len())..m1.min(mags.len())].to_vec();
        kick[i] = low.iter().sum();
        if let (Some(pl), Some(pm)) = (&prev_low, &prev_mid) {
            low_flux[i] = low.iter().zip(pl).map(|(a, b)| (a - b).max(0.0)).sum();
            mid_flux[i] = mid.iter().zip(pm).map(|(a, b)| (a - b).max(0.0)).sum();
        }
        prev_low = Some(low);
        prev_mid = Some(mid);
    }

    let dt = hop as f64 / sr as f64;
    let window = ((0.10 / dt) as usize).max(1);
    let sample = |arr: &[f32], t: f64| -> f64 {
        let i = (t / dt) as usize;
        if i >= arr.len() {
            return 0.0;
        }
        let j = (i + window).min(arr.len());
        arr[i..j].iter().cloned().fold(0.0f32, f32::max) as f64
    };

    let low = beats.iter().map(|&t| sample(&kick, t)).collect();
    let fl = beats.iter().map(|&t| sample(&low_flux, t)).collect();
    let snare = beats.iter().map(|&t| sample(&mid_flux, t)).collect();
    let rms = beats
        .iter()
        .map(|&t| {
            let a = (t * sr as f64) as usize;
            let b = (a + (0.06 * sr as f64) as usize).min(audio.len());
            if a >= b {
                0.0
            } else {
                (audio[a..b].iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()
                    / (b - a) as f64)
                    .sqrt()
            }
        })
        .collect();
    (low, fl, rms, snare)
}

/// Return (downbeats, phase, confidence) from the four-beat beat grid.
#[allow(clippy::needless_range_loop)] // the four beat phases are naturally indexed
pub fn estimate_downbeats(
    beats: &[f64],
    low: &[f64],
    flux: &[f64],
    _rms: &[f64],
    snare: Option<&[f64]>,
) -> (Vec<f64>, i32, f64) {
    if beats.len() < 4 {
        return (beats.to_vec(), 0, 0.0);
    }
    let zl = zscore(low);
    let zf = zscore(flux);
    let kick: Vec<f64> = zl.iter().zip(&zf).map(|(a, b)| a + b).collect();

    let mut score = [0f64; 4];
    for (p, s) in score.iter_mut().enumerate() {
        let vals: Vec<f64> = kick.iter().skip(p).step_by(4).copied().collect();
        *s = if vals.is_empty() { 0.0 } else { vals.iter().sum::<f64>() / vals.len() as f64 };
    }
    if let Some(sn) = snare {
        if sn.len() == beats.len() {
            let s = zscore(sn);
            for p in 0..4 {
                let mut acc = 0.0;
                let mut cnt = 0.0;
                for off in [1usize, 3] {
                    let vals: Vec<f64> =
                        s.iter().skip(p + off).step_by(4).copied().collect();
                    if !vals.is_empty() {
                        acc += vals.iter().sum::<f64>() / vals.len() as f64;
                        cnt += 1.0;
                    }
                }
                if cnt > 0.0 {
                    score[p] += acc / cnt;
                }
            }
        }
    }

    let best = argmax(&score);
    let mut order: Vec<usize> = (0..4).collect();
    order.sort_by(|&a, &b| score[b].partial_cmp(&score[a]).unwrap());
    let margin = score[order[0]] - score[order[1]];
    let spread = if score.is_empty() { 1e-9 } else { mean_std(&score).1 + 1e-9 };
    let confidence = (margin / (spread + 1e-9)).clamp(0.0, 3.0) / 2.0;

    let downbeats: Vec<f64> = beats.iter().skip(best).step_by(4).copied().collect();
    (downbeats, best as i32, confidence)
}

/// 4-bar vs 8-bar phrase length (returns bars per phrase: 4 or 8).
pub fn estimate_phrase_length(beats: &[f64], low: &[f64], flux: &[f64]) -> i32 {
    if beats.len() < 64 {
        return 4;
    }
    let zl = zscore(low);
    let zf = zscore(flux);
    let novelty: Vec<f64> = zl.iter().zip(&zf).map(|(a, b)| a + b).collect();

    let mut best_len = 4;
    let mut best_score = f64::NEG_INFINITY;
    for phrase_beats in [16usize, 32] {
        let mut vals = Vec::new();
        for p in 0..phrase_beats {
            let idx: Vec<usize> = (p..novelty.len()).step_by(phrase_beats).collect();
            if idx.len() >= 2 {
                vals.push(idx.iter().map(|&i| novelty[i]).sum::<f64>() / idx.len() as f64);
            }
        }
        if !vals.is_empty() {
            let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let score = max - vals.iter().sum::<f64>() / vals.len() as f64;
            if score > best_score {
                best_score = score;
                best_len = (phrase_beats / 4) as i32;
            }
        }
    }
    best_len
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A click train at `bpm` should be tracked back at roughly that tempo.
    #[test]
    fn detects_click_track_tempo() {
        let sr = 44100u32;
        let bpm = 128.0f64;
        let dur = 20.0f64;
        let mut audio = vec![0f32; (sr as f64 * dur) as usize];
        let period = 60.0 / bpm;
        let mut t = 0.0;
        while t < dur {
            let start = (t * sr as f64) as usize;
            for i in 0..200.min(audio.len().saturating_sub(start)) {
                let env = (-(i as f32) / 40.0).exp();
                audio[start + i] += env * (2.0 * std::f32::consts::PI * 1500.0 * i as f32
                    / sr as f32)
                    .sin();
            }
            t += period;
        }
        let (odf, _) = super::super::dsp::spectral_flux(&audio, sr, 2048, 512);
        let hop_t = 512.0 / sr as f64;
        let (bpm_est, conf) = estimate_tempo(&odf, hop_t);
        assert!((bpm_est - bpm).abs() < 3.0, "bpm_est={bpm_est} conf={conf}");
        assert!(conf > 0.0);
        let beats = track_beats(&odf, hop_t, bpm_est, dur);
        assert!(beats.len() > 20);
        let gaps: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
        let avg = gaps.iter().sum::<f64>() / gaps.len() as f64;
        assert!((avg - period).abs() < 0.05, "avg gap {avg} vs period {period}");
    }

    #[test]
    fn phrase_length_defaults_to_four() {
        let beats: Vec<f64> = (0..40).map(|i| i as f64 * 0.5).collect();
        let low: Vec<f64> = (0..40).map(|i| (i as f64).sin()).collect();
        assert_eq!(estimate_phrase_length(&beats, &low, &low), 4);
    }
}
