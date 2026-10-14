//! Cue-point inference: turn an [`AnalysisResult`] into labelled hot cues.
//!
//! The goal is cues a DJ would actually use without editing:
//!
//! * **Start** at the first downbeat (the true "1" of the first bar).
//! * **Drops** at the *most prominent* energy rises, not every section change.
//! * **Breakdowns** at the most prominent energy troughs between drops.
//! * **Outro** at the mix-out point near the end.
//! * **Phrase awareness**: candidates are preferred on phrase starts and always
//!   snapped to the downbeat grid, so cues land on bar/phrase lines.
//! * Prominence filtering keeps the result musical: a track with one real drop
//!   gets one drop cue, not eight.
//!
//! This is a direct port of the original `cueit/cue.py`; the selection is driven
//! by the measured energy contour, so it does not depend on section labels.

use crate::types::{AnalysisResult, Cue, CueSet};

/// Standard DJ hot-cue palette (index into colours shared by GUI and exporters).
pub const CUE_COLORS: [&str; 8] = [
    "#1DB954", "#E91E63", "#2196F3", "#FFC107", "#FF5722", "#9C27B0", "#00BCD4", "#8BC34A",
];

pub const MAX_CUES: usize = 8;
pub const MAX_DROPS: usize = 3;
pub const MAX_BREAKDOWNS: usize = 2;
/// Seconds between distinct drops.
pub const MIN_DROP_GAP: f64 = 12.0;
pub const MIN_BREAKDOWN_GAP: f64 = 8.0;

/// Index of the first maximum (matches numpy `argmax` tie-breaking).
fn argmax(xs: &[f64]) -> usize {
    let mut best = 0usize;
    for (i, &x) in xs.iter().enumerate() {
        if x > xs[best] {
            best = i;
        }
    }
    best
}

/// Nearest element of `grid` to `t` if within `tol`, else `max(0, t)`.
fn snap(t: f64, grid: &[f64], tol: f64) -> f64 {
    if grid.is_empty() {
        return t.max(0.0);
    }
    let mut best = grid[0];
    let mut best_d = (grid[0] - t).abs();
    for &g in grid {
        let d = (g - t).abs();
        if d < best_d {
            best_d = d;
            best = g;
        }
    }
    if best_d <= tol {
        best
    } else {
        t.max(0.0)
    }
}

/// Prefer the nearest phrase start if within `tol`, else snap to the grid.
fn prefer_phrase(t: f64, grid: &[f64], phrases: &[f64], tol: f64) -> f64 {
    if phrases.is_empty() {
        return snap(t, grid, 1.5);
    }
    let mut best = phrases[0];
    let mut best_d = (phrases[0] - t).abs();
    for &p in phrases {
        let d = (p - t).abs();
        if d < best_d {
            best_d = d;
            best = p;
        }
    }
    if best_d <= tol {
        best
    } else {
        snap(t, grid, 1.5)
    }
}

/// Indices of `xs` sorted by value descending, ties broken by ascending index.
/// (A deterministic approximation of `numpy.argsort(x)[::-1]`.)
fn argsort_desc(xs: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..xs.len()).collect();
    order.sort_by(|&a, &b| xs[b].partial_cmp(&xs[a]).unwrap().then(a.cmp(&b)));
    order
}

fn priority(name: &str) -> i32 {
    match name {
        "Intro" => 5,
        "Drop" => 4,
        "Outro" => 3,
        "Breakdown" => 3,
        _ => 0,
    }
}

/// Merge cues that are closer than `min_gap`, keeping the higher-priority one.
fn dedupe(items: &[(String, f64)], min_gap: f64) -> Vec<(String, f64)> {
    let mut sorted = items.to_vec();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut out: Vec<(String, f64)> = Vec::with_capacity(sorted.len());
    for (name, t) in sorted {
        if let Some(pos) = out.iter().position(|c| (c.1 - t).abs() < min_gap) {
            if priority(&name) > priority(&out[pos].0) {
                out[pos] = (name, t);
            }
        } else {
            out.push((name, t));
        }
    }
    out
}

/// Cap to `limit` items, always keeping Intro/Outro and the most prominent rest.
fn cap(items: &[(String, f64)], limit: usize) -> Vec<(String, f64)> {
    if items.len() <= limit {
        let mut v = items.to_vec();
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        return v;
    }
    let must: Vec<(String, f64)> =
        items.iter().filter(|c| c.0 == "Intro" || c.0 == "Outro").cloned().collect();
    let mut rest: Vec<(String, f64)> =
        items.iter().filter(|c| !must.contains(c)).cloned().collect();
    rest.sort_by_key(|c| std::cmp::Reverse(priority(&c.0)));
    let take = limit.saturating_sub(must.len());
    let mut combined = must;
    combined.extend(rest.into_iter().take(take));
    let mut kept = dedupe(&combined, 3.0);
    kept.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    kept.truncate(limit);
    kept
}

fn round3(t: f64) -> f64 {
    (t * 1000.0).round() / 1000.0
}

/// Infer DJ hot cues for a track.
pub fn infer_cues(result: &AnalysisResult, use_downbeats: bool) -> CueSet {
    let grid: &[f64] = if use_downbeats && !result.downbeats.is_empty() {
        &result.downbeats
    } else {
        &result.beats
    };
    let duration = result.duration;

    if grid.is_empty() {
        return CueSet::new(vec![Cue::new("Intro", 0.0, CUE_COLORS[0])]);
    }

    let phrase_bars = result.phrase_length.max(1) as usize;
    let phrases: Vec<f64> = if phrase_bars > 1 {
        grid.iter().step_by(phrase_bars).copied().collect()
    } else {
        grid.to_vec()
    };

    let starts: Vec<f64> = result.sections.iter().map(|s| s.start).collect();
    let energies: Vec<f64> = result.sections.iter().map(|s| s.energy).collect();

    let mut candidates: Vec<(String, f64)> = Vec::new();
    let start = grid[0];
    candidates.push(("Intro".to_string(), start));

    if energies.len() >= 3 {
        let peak_raw = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let peak = if peak_raw == 0.0 { 1.0 } else { peak_raw };

        // Drops: biggest rises into a high-energy section.
        let rises: Vec<f64> = energies.windows(2).map(|w| w[1] - w[0]).collect();
        let mut drops: Vec<f64> = Vec::new();
        for &j in &argsort_desc(&rises) {
            if rises[j] <= 0.10 {
                break;
            }
            let idx = j + 1;
            let t = prefer_phrase(starts[idx], grid, &phrases, 1.0);
            if t <= start + 2.0 || t >= duration {
                continue;
            }
            if energies[idx] < 0.6 * peak {
                continue;
            }
            if drops.iter().all(|&d| (t - d).abs() >= MIN_DROP_GAP) {
                drops.push(t);
            }
            if drops.len() >= MAX_DROPS {
                break;
            }
        }
        candidates.extend(drops.into_iter().map(|t| ("Drop".to_string(), t)));

        // Breakdowns: biggest falls into a low-energy section.
        let mut breakdowns: Vec<f64> = Vec::new();
        for &j in &argsort_desc(&rises) {
            if rises[j] <= 0.10 {
                break;
            }
            let idx = j + 1;
            let t = prefer_phrase(starts[idx], grid, &phrases, 1.0);
            if t <= start + 2.0 || t >= duration {
                continue;
            }
            if energies[idx] > 0.7 * peak {
                continue;
            }
            if breakdowns.iter().all(|&b| (t - b).abs() >= MIN_BREAKDOWN_GAP) {
                breakdowns.push(t);
            }
            if breakdowns.len() >= MAX_BREAKDOWNS {
                break;
            }
        }
        candidates.extend(breakdowns.into_iter().map(|t| ("Breakdown".to_string(), t)));
    } else if !energies.is_empty() {
        // Too few sections: fall back to the single highest-energy point.
        let idx = argmax(&energies);
        let t = prefer_phrase(starts[idx], grid, &phrases, 1.0);
        if t > start + 2.0 {
            candidates.push(("Drop".to_string(), t));
        }
    }

    // Outro / mix-out at the final section start (or 32s before the end).
    let outro = if !starts.is_empty() {
        prefer_phrase(*starts.last().unwrap(), grid, &phrases, 1.0)
    } else {
        (duration - 32.0).max(0.0)
    };
    candidates.push(("Outro".to_string(), outro.clamp(0.0, duration)));

    let chosen = cap(&dedupe(&candidates, 3.0), MAX_CUES);

    let cues = chosen
        .into_iter()
        .enumerate()
        .map(|(i, (name, t))| Cue::new(name, round3(t), CUE_COLORS[i % CUE_COLORS.len()]))
        .collect();
    CueSet::new(cues)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Section;

    fn downbeats_every_4s(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 * 4.0).collect()
    }

    fn section(start: f64, energy: f64) -> Section {
        Section { start, end: start + 32.0, energy, kind: "section".to_string() }
    }

    #[test]
    fn picks_intro_drops_and_outro() {
        // 5 sections; two strong rises (0.2->0.9 and 0.3->0.95) => two drops.
        let mut r = AnalysisResult {
            duration: 200.0,
            downbeats: downbeats_every_4s(50), // 0,4,...,196
            phrase_length: 4,
            sections: vec![
                section(0.0, 0.2),
                section(32.0, 0.9),
                section(64.0, 0.3),
                section(96.0, 0.95),
                section(128.0, 0.4),
            ],
            ..Default::default()
        };
        r.beats = r.downbeats.clone();
        let cues = infer_cues(&r, true);
        let names: Vec<&str> = cues.cues.iter().map(|c| c.name.as_str()).collect();
        let times: Vec<f64> = cues.times();
        assert_eq!(names, vec!["Intro", "Drop", "Drop", "Outro"]);
        assert_eq!(times, vec![0.0, 32.0, 96.0, 128.0]);
        // palette is applied in order
        assert_eq!(cues.cues[0].color, CUE_COLORS[0]);
        assert_eq!(cues.cues[1].color, CUE_COLORS[1]);
    }

    #[test]
    fn falls_back_to_highest_energy_with_few_sections() {
        let mut r = AnalysisResult {
            duration: 120.0,
            downbeats: downbeats_every_4s(30),
            sections: vec![section(0.0, 0.1), section(40.0, 0.7)],
            ..Default::default()
        };
        r.beats = r.downbeats.clone();
        let cues = infer_cues(&r, true);
        let names: Vec<&str> = cues.cues.iter().map(|c| c.name.as_str()).collect();
        let times: Vec<f64> = cues.times();
        // Outro lands on the same downbeat as the Drop; dedupe keeps the
        // higher-priority Drop, so only Intro + Drop survive.
        assert_eq!(names, vec!["Intro", "Drop"]);
        assert_eq!(times, vec![0.0, 40.0]);
    }

    #[test]
    fn empty_grid_yields_just_intro() {
        let r = AnalysisResult { duration: 60.0, ..Default::default() };
        let cues = infer_cues(&r, true);
        assert_eq!(cues.cues.len(), 1);
        assert_eq!(cues.cues[0].name, "Intro");
        assert_eq!(cues.cues[0].time, 0.0);
    }

    #[test]
    fn never_exceeds_max_cues() {
        let mut r = AnalysisResult {
            duration: 400.0,
            downbeats: downbeats_every_4s(100),
            phrase_length: 4,
            sections: (0..12)
                .map(|i| section(i as f64 * 32.0, if i % 2 == 0 { 0.1 } else { 0.9 }))
                .collect(),
            ..Default::default()
        };
        r.beats = r.downbeats.clone();
        let cues = infer_cues(&r, true);
        assert!(cues.cues.len() <= MAX_CUES);
        // always sorted by time
        let times = cues.times();
        assert!(times.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn uses_beats_when_downbeats_absent() {
        let r = AnalysisResult {
            duration: 100.0,
            beats: downbeats_every_4s(25),
            downbeats: vec![],
            sections: vec![section(0.0, 0.2), section(32.0, 0.9), section(64.0, 0.3)],
            ..Default::default()
        };
        let cues = infer_cues(&r, true);
        assert_eq!(cues.cues[0].name, "Intro");
        assert_eq!(cues.cues[0].time, 0.0);
    }
}
