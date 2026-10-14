//! Structural segmentation and section labelling.
//!
//! Direct ports of `_sections_from_features`, `_snap_sections`,
//! `_label_sections` and `_fallback_sections` from the original `analysis.py`.

use crate::types::Section;

/// Novelty-based segmentation from an MFCC self-similarity matrix (Foote).
///
/// `mfcc` is the per-frame MFCC matrix (frame size 4096, hop 2048 in the
/// caller). Returns section `(start, end)` pairs in seconds.
#[allow(clippy::needless_range_loop)] // i indexes both `feat` and `novelty`
pub fn sections_from_features(
    mfcc: &[Vec<f32>],
    sr: u32,
    hop: usize,
    duration: f64,
) -> Vec<(f64, f64)> {
    let n = mfcc.len();
    if n < 16 {
        return Vec::new();
    }
    let dim = mfcc[0].len();
    let mut feat = vec![0f64; n * dim];
    for (i, row) in mfcc.iter().enumerate() {
        for j in 0..dim {
            feat[i * dim + j] = row[j] as f64;
        }
    }
    // Standardise each coefficient column.
    for j in 0..dim {
        let col: Vec<f64> = (0..n).map(|i| feat[i * dim + j]).collect();
        let mean = col.iter().sum::<f64>() / n as f64;
        let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std = var.sqrt();
        let inv = if std == 0.0 { 1.0 } else { 1.0 / std };
        for i in 0..n {
            feat[i * dim + j] = (feat[i * dim + j] - mean) * inv;
        }
    }
    // Self-similarity is only ever read inside a +/- kernel band around the
    // diagonal, so we compute it on the fly (dot products) instead of
    // materialising an n x n matrix — that would be hundreds of MB for a
    // 5-minute track.
    let dot = |a: usize, b: usize| -> f64 {
        let mut s = 0.0;
        for d in 0..dim {
            s += feat[a * dim + d] * feat[b * dim + d];
        }
        s / dim as f64
    };

    // Checkerboard novelty.
    let kernel = 32usize;
    let mut novelty = vec![0f64; n];
    for i in 0..n {
        let l0 = i.saturating_sub(kernel);
        let r1 = (i + kernel).min(n);
        let (mut left_sum, mut left_n) = (0.0, 0.0);
        let (mut right_sum, mut right_n) = (0.0, 0.0);
        let (mut cross_sum, mut cross_n) = (0.0, 0.0);
        for a in l0..i {
            for b in l0..i {
                left_sum += dot(a, b);
                left_n += 1.0;
            }
            for b in i..r1 {
                cross_sum += dot(a, b);
                cross_n += 1.0;
            }
        }
        for a in i..r1 {
            for b in i..r1 {
                right_sum += dot(a, b);
                right_n += 1.0;
            }
        }
        let mut val = 0.0;
        if left_n > 0.0 {
            val += left_sum / left_n;
        }
        if right_n > 0.0 {
            val += right_sum / right_n;
        }
        if cross_n > 0.0 {
            val -= 2.0 * cross_sum / cross_n;
        }
        novelty[i] = val;
    }

    let mean = novelty.iter().sum::<f64>() / n as f64;
    let var = novelty.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let thr = mean + var.sqrt();

    let mut boundaries = vec![0.0f64];
    for i in 1..n.saturating_sub(1) {
        if novelty[i] > thr && novelty[i] >= novelty[i - 1] && novelty[i] >= novelty[i + 1] {
            let t = i as f64 * hop as f64 / sr as f64;
            if t - boundaries.last().unwrap() > 8.0 && duration - t > 4.0 {
                boundaries.push(t);
            }
        }
    }
    boundaries.push(duration);
    boundaries.windows(2).map(|w| (w[0], w[1])).collect()
}

/// Align section boundaries to the nearest downbeat (fallback: beat).
pub fn snap_sections(
    raw: &[(f64, f64)],
    downbeats: &[f64],
    beats: &[f64],
    duration: f64,
    tol: f64,
) -> Vec<(f64, f64)> {
    let grid: &[f64] = if downbeats.len() >= 2 { downbeats } else { beats };
    if grid.is_empty() {
        return raw.to_vec();
    }
    let mut snapped = Vec::with_capacity(raw.len() * 2);
    for (a, b) in raw {
        for edge in [*a, *b] {
            if edge <= 0.0 {
                snapped.push(0.0);
            } else if edge >= duration {
                snapped.push(duration);
            } else {
                let mut best = grid[0];
                let mut best_d = (grid[0] - edge).abs();
                for &g in grid {
                    let d = (g - edge).abs();
                    if d < best_d {
                        best_d = d;
                        best = g;
                    }
                }
                snapped.push(if best_d <= tol { best } else { edge });
            }
        }
    }
    let mut out = Vec::new();
    for pair in snapped.chunks(2) {
        if pair.len() == 2 {
            let (a, b) = (pair[0], pair[1]);
            if b - a > 1.0 {
                out.push((a.max(0.0), b.min(duration)));
            }
        }
    }
    out
}

/// Assign intro/verse/drop/breakdown/outro from the relative energy curve.
pub fn label_sections(sections: &mut [Section]) {
    if sections.is_empty() {
        return;
    }
    let peak = sections
        .iter()
        .map(|s| s.energy)
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1e-9);
    let energies: Vec<f64> = sections.iter().map(|s| s.energy).collect();
    for i in 0..sections.len() {
        let rel = sections[i].energy / peak;
        let prev_rel = if i > 0 { energies[i - 1] / peak } else { 0.0 };
        let rising = i > 0 && rel >= prev_rel + 0.15;
        sections[i].kind = if i == 0 {
            "intro"
        } else if i == sections.len() - 1 {
            "outro"
        } else if rel <= 0.5 {
            "breakdown"
        } else if rel >= 0.8 || (rising && rel >= 0.65) {
            "drop"
        } else {
            "verse"
        }
        .to_string();
    }
}

/// Fallback segmentation from the RMS envelope when MFCC segmentation fails.
pub fn fallback_sections(rms: &[f64], times: &[f64], duration: f64) -> Vec<(f64, f64)> {
    if rms.len() < 8 {
        return vec![(0.0, duration)];
    }
    let mut sorted = rms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let thr = median * 0.6;
    let loud: Vec<bool> = rms.iter().map(|&r| r > thr).collect();
    let mut boundaries = vec![0.0f64];
    for i in 1..loud.len() {
        if loud[i] != loud[i - 1] {
            let t = times[i];
            if t - boundaries.last().unwrap() > 8.0 {
                boundaries.push(t);
            }
        }
    }
    boundaries.push(duration);
    boundaries.windows(2).map(|w| (w[0], w[1])).collect()
}
