//! Batch discovery and parallel analysis.
//!
//! * [`discover`] expands files/folders into a de-duplicated audio path list.
//! * [`run_batch`] analyzes many tracks, in parallel on native targets and
//!   sequentially on `wasm32` (which has no threads).
//!
//! Cancellation is cooperative via a `should_stop` callback, so a huge library
//! run can be stopped between tracks instead of hanging (the Mixed In Key
//! failure mode the original set out to avoid).

#[cfg(feature = "native-audio")]
use crate::cue::infer_cues;
use crate::types::TrackData;
use crate::AUDIO_EXTS;

#[cfg(feature = "native-audio")]
use crate::analysis::{analyze_file, DEFAULT_SR};

/// Expand files and folders into a sorted-ish, de-duplicated list of audio paths.
pub fn discover(paths: &[String], recursive: bool) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for p in paths {
        let expanded = expand_tilde(p);
        let path = std::path::Path::new(&expanded);
        if path.is_dir() {
            if recursive {
                walk(path, &mut found);
            } else if let Ok(rd) = std::fs::read_dir(path) {
                for entry in rd.flatten() {
                    let full = entry.path();
                    if full.is_file() && is_audio(&full) {
                        found.push(full.to_string_lossy().into_owned());
                    }
                }
            }
        } else if path.is_file() && is_audio(path) {
            found.push(path.to_string_lossy().into_owned());
        }
    }
    let mut seen = std::collections::HashSet::new();
    found.retain(|p| seen.insert(p.clone()));
    found
}

fn walk(dir: &std::path::Path, out: &mut Vec<String>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let full = entry.path();
        if full.is_dir() {
            walk(&full, out);
        } else if full.is_file() && is_audio(&full) {
            out.push(full.to_string_lossy().into_owned());
        }
    }
}

fn is_audio(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            let dotted = format!(".{}", e.to_lowercase());
            AUDIO_EXTS.contains(&dotted.as_str())
        })
        .unwrap_or(false)
}

fn expand_tilde(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    p.to_string()
}

/// Default worker count: `min(cpu_count - 1, 16)`.
pub fn default_workers() -> usize {
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
    cpus.saturating_sub(1).clamp(1, 16)
}

/// Target average CPU usage (percent, 5..=100). Below 100, each worker pauses
/// between tracks proportional to its busy time so the whole machine stays
/// responsive and cool at the cost of throughput.
static CPU_LIMIT_PCT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(100);

/// Set the CPU limit as a percentage (clamped to 5..=100).
pub fn set_cpu_limit(pct: u32) {
    CPU_LIMIT_PCT.store(pct.clamp(5, 100), std::sync::atomic::Ordering::SeqCst);
}

/// Current CPU limit percentage.
pub fn cpu_limit() -> u32 {
    CPU_LIMIT_PCT.load(std::sync::atomic::Ordering::SeqCst)
}

/// Sleep to enforce the configured duty cycle after `busy` of work.
fn throttle(busy: std::time::Duration) {
    let limit = cpu_limit();
    if limit < 100 {
        let extra = busy.as_nanos() as u64 * (100 - limit) as u64 / limit as u64;
        std::thread::sleep(std::time::Duration::from_nanos(extra));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobStatus {
    Queued,
    Analyzing,
    Done,
    Error,
    Cancelled,
}

/// One track in a batch.
#[derive(Clone, Debug)]
pub struct Job {
    pub path: String,
    pub status: JobStatus,
    pub error: String,
    pub track: Option<TrackData>,
}

impl Job {
    pub fn new(path: String) -> Self {
        Self { path, status: JobStatus::Queued, error: String::new(), track: None }
    }

    pub fn name(&self) -> String {
        std::path::Path::new(&self.path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.path.clone())
    }
}

pub struct BatchResult {
    pub jobs: Vec<Job>,
    pub cancelled: bool,
}

impl BatchResult {
    pub fn tracks(&self) -> Vec<TrackData> {
        self.jobs.iter().filter_map(|j| j.track.clone()).collect()
    }

    pub fn done(&self) -> usize {
        self.jobs
            .iter()
            .filter(|j| matches!(j.status, JobStatus::Done | JobStatus::Error))
            .count()
    }
}

/// Analyze one track: DSP + cue inference. File-based, so desktop/CLI only.
///
/// The analysis is wrapped in `catch_unwind` so a panic on one pathological
/// file becomes a per-track error instead of taking down the whole batch (or
/// the GUI process).
#[cfg(feature = "native-audio")]
pub fn analyze_one(path: &str, use_downbeats: bool) -> Result<TrackData, String> {
    let owned = path.to_string();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let analysis = analyze_file(&owned, DEFAULT_SR, None)?;
        let cues = infer_cues(&analysis, use_downbeats);
        Ok::<TrackData, String>(TrackData { path: owned.clone(), analysis, cues: cues.cues })
    }));
    match result {
        Ok(r) => r,
        Err(_) => Err("analysis panicked (unsupported or corrupt audio?)".to_string()),
    }
}

#[cfg(feature = "native-audio")]
type OnDone<'a> = Option<&'a (dyn Fn(&Job) + Sync)>;
#[cfg(feature = "native-audio")]
type ShouldStop<'a> = Option<&'a (dyn Fn() -> bool + Sync)>;
/// Worker outcome: `(status, error, track)`.
#[cfg(feature = "native-audio")]
type Outcome = (JobStatus, String, Option<TrackData>);

/// Analyze `paths` (up to `workers` at a time). Callbacks are invoked from
/// worker threads so they must be `Sync`. Desktop/CLI only — the web build gets
/// decoded PCM from the browser and calls [`crate::analysis::analyze_pcm`].
#[cfg(feature = "native-audio")]
pub fn run_batch(
    paths: &[String],
    workers: usize,
    use_downbeats: bool,
    on_done: OnDone<'_>,
    should_stop: ShouldStop<'_>,
) -> BatchResult {
    let workers = workers.max(1);
    if paths.is_empty() {
        return BatchResult { jobs: Vec::new(), cancelled: false };
    }

    let stopped = || should_stop.map(|f| f()).unwrap_or(false);
    let mut jobs: Vec<Job> = paths.iter().cloned().map(Job::new).collect();

    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Mutex;

        let next = AtomicUsize::new(0);
        let results: Mutex<Vec<Option<Outcome>>> = Mutex::new(vec![None; paths.len()]);
        std::thread::scope(|scope| {
            for _ in 0..workers.min(paths.len()) {
                scope.spawn(|| loop {
                    if stopped() {
                        break;
                    }
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= paths.len() {
                        break;
                    }
                    let path = &paths[i];
                    let started = std::time::Instant::now();
                    let outcome = analyze_one(path, use_downbeats);
                    throttle(started.elapsed());
                    let slot = match outcome {
                        Ok(t) => (JobStatus::Done, String::new(), Some(t)),
                        Err(e) => (JobStatus::Error, e, None),
                    };
                    let mut job = Job::new(path.clone());
                    job.status = slot.0;
                    job.error = slot.1.clone();
                    job.track = slot.2.clone();
                    if let Some(cb) = on_done {
                        cb(&job);
                    }
                    results.lock().unwrap()[i] = Some(slot);
                });
            }
        });

        let mut done = results.into_inner().unwrap();
        for (job, slot) in jobs.iter_mut().zip(done.iter_mut()) {
            if let Some((status, error, track)) = slot.take() {
                job.status = status;
                job.error = error;
                job.track = track;
            } else {
                job.status = JobStatus::Cancelled;
            }
        }
    }

    let cancelled = stopped();
    if cancelled {
        for job in jobs.iter_mut() {
            if job.status == JobStatus::Queued {
                job.status = JobStatus::Cancelled;
            }
        }
    }
    BatchResult { jobs, cancelled }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discover_filters_non_audio_and_dedupes() {
        let dir = std::env::temp_dir().join("cueit_discover_test");
        let _ = std::fs::create_dir_all(&dir);
        let a = dir.join("a.mp3");
        let b = dir.join("notes.txt");
        std::fs::write(&a, b"x").unwrap();
        std::fs::write(&b, b"x").unwrap();
        let paths = vec![a.to_string_lossy().into_owned(), a.to_string_lossy().into_owned()];
        let got = discover(&paths, false);
        assert_eq!(got.len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }
}
