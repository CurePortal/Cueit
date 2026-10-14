//! CueIt desktop shell (Tauri v2). Thin command layer over `cueit-core`.

mod player;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use serde::Serialize;
use tauri::{AppHandle, Emitter};

use cueit_core::batch::{self, Job, JobStatus};
use cueit_core::libraries::{self, LibrarySource};
use cueit_core::types::TrackData;
use cueit_core::{exporters, session};

/// Set from the UI to cancel a running batch (checked between tracks).
static CANCEL: AtomicBool = AtomicBool::new(false);

#[derive(Serialize)]
struct JobDto {
    path: String,
    status: String,
    error: String,
    track: Option<TrackData>,
}

/// Streamed to the UI while a batch runs (event `analyze-progress`).
#[derive(Clone, Serialize)]
struct Progress {
    done: usize,
    total: usize,
    path: String,
    status: String,
    error: String,
    track: Option<TrackData>,
}

fn status_str(s: JobStatus) -> &'static str {
    match s {
        JobStatus::Queued => "queued",
        JobStatus::Analyzing => "analyzing",
        JobStatus::Done => "done",
        JobStatus::Error => "error",
        JobStatus::Cancelled => "cancelled",
    }
}

#[derive(Serialize)]
struct LibDto {
    name: String,
    kind: String,
    locator: String,
    tracks: Option<usize>,
}

#[derive(Clone, Serialize)]
struct ModelProgress {
    p: f32,
    msg: String,
}

/// Expand files/folders into audio paths.
#[tauri::command]
fn discover_paths(paths: Vec<String>, recursive: bool) -> Vec<String> {
    batch::discover(&paths, recursive)
}

/// Request cancellation of the running batch.
#[tauri::command]
fn set_cancel(value: bool) {
    CANCEL.store(value, Ordering::SeqCst);
}

/// Analyze many tracks in parallel off the main thread, streaming progress.
///
/// Runs on a blocking worker via `spawn_blocking` so the webview stays
/// responsive during a long batch (a synchronous command would block the main
/// thread and make the window appear hung). Cancellable via [`set_cancel`].
#[tauri::command]
async fn analyze(
    app: AppHandle,
    paths: Vec<String>,
    workers: Option<usize>,
    use_downbeats: bool,
) -> Result<Vec<JobDto>, String> {
    let w = workers.unwrap_or_else(batch::default_workers);
    CANCEL.store(false, Ordering::SeqCst);
    tauri::async_runtime::spawn_blocking(move || {
        let total = paths.len();
        let done = AtomicUsize::new(0);
        let on_done = |job: &Job| {
            let n = done.fetch_add(1, Ordering::SeqCst) + 1;
            let _ = app.emit(
                "analyze-progress",
                Progress {
                    done: n,
                    total,
                    path: job.path.clone(),
                    status: status_str(job.status).to_string(),
                    error: job.error.clone(),
                    track: job.track.clone(),
                },
            );
        };
        let should_stop = || CANCEL.load(Ordering::SeqCst);
        let result = batch::run_batch(&paths, w, use_downbeats, Some(&on_done), Some(&should_stop));
        result
            .jobs
            .into_iter()
            .map(|j| JobDto {
                path: j.path,
                status: status_str(j.status).to_string(),
                error: j.error,
                track: j.track,
            })
            .collect::<Vec<_>>()
    })
    .await
    .map_err(|e| e.to_string())
}

/// Analyze a single track (used by re-detect).
#[tauri::command]
fn analyze_one(path: String, use_downbeats: bool) -> Result<TrackData, String> {
    batch::analyze_one(&path, use_downbeats)
}

/// Re-run cue inference on an existing analysis (no re-DSP).
#[tauri::command]
fn redetect_cues(mut track: TrackData, use_downbeats: bool) -> TrackData {
    track.cues = cueit_core::infer_cues(&track.analysis, use_downbeats).cues;
    track
}

/// Export tracks. Returns the list of written paths.
#[tauri::command]
fn export_tracks(
    format: String,
    tracks: Vec<TrackData>,
    out: String,
    out_file: Option<String>,
    mixxx_db: Option<String>,
) -> Result<Vec<String>, String> {
    if tracks.is_empty() {
        return Err("nothing to export".into());
    }
    let out_dir = PathBuf::from(&out);
    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    let mut written = Vec::new();
    match format.as_str() {
        "rekordbox" => {
            let path = out_file.map(PathBuf::from).unwrap_or_else(|| out_dir.join("rekordbox.xml"));
            written.push(exporters::export_rekordbox(&tracks, &path).map_err(|e| e.to_string())?);
        }
        "traktor" => {
            let path = out_file.map(PathBuf::from).unwrap_or_else(|| out_dir.join("collection.nml"));
            written.push(exporters::export_traktor(&tracks, &path).map_err(|e| e.to_string())?);
        }
        "json" | "csv" => {
            for t in &tracks {
                let base = Path::new(&t.path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "track".into());
                let path = if format == "json" {
                    out_dir.join(format!("{base}.cueit.json"))
                } else {
                    out_dir.join(format!("{base}.cueit.csv"))
                };
                let p = if format == "json" {
                    exporters::export_json(t, &path)
                } else {
                    exporters::export_csv(t, &path)
                };
                written.push(p.map_err(|e| e.to_string())?);
            }
        }
        "serato" => {
            for t in &tracks {
                written.push(exporters::export_serato(t)?);
            }
        }
        "mixxx" => {
            let db = mixxx_db.unwrap_or_else(|| "~/.mixxx/mixxxdb.sqlite".to_string());
            for t in &tracks {
                written.push(exporters::export_mixxx(t, &db)?);
            }
        }
        other => return Err(format!("unknown export format: {other}")),
    }
    Ok(written)
}

/// Detected DJ libraries on this machine (with track counts).
#[tauri::command]
fn detect_libraries() -> Vec<LibDto> {
    libraries::detect_libraries()
        .into_iter()
        .map(|lib| {
            let tracks = libraries::load_tracks(&lib).ok().map(|t| t.len());
            LibDto { name: lib.name, kind: lib.kind, locator: lib.locator, tracks }
        })
        .collect()
}

/// Load a library by kind + locator.
#[tauri::command]
fn load_library(kind: String, locator: String) -> Result<Vec<String>, String> {
    libraries::load_tracks(&LibrarySource::new(kind.clone(), kind, locator))
}

/// Load the saved session.
#[tauri::command]
fn session_load() -> Vec<TrackData> {
    session::load(None).into_values().collect()
}

/// Save the session.
#[tauri::command]
fn session_save(tracks: Vec<TrackData>) -> Result<String, String> {
    let map: BTreeMap<String, TrackData> =
        tracks.into_iter().map(|t| (t.path.clone(), t)).collect();
    session::save(&map, None).map(|p| p.to_string_lossy().into_owned()).map_err(|e| e.to_string())
}

/// Clear the saved session.
#[tauri::command]
fn session_clear() {
    session::clear(None);
}

/// Default worker count for the UI spin box.
#[tauri::command]
fn default_workers() -> usize {
    batch::default_workers()
}

/// Cap the batch's average CPU usage (percent, 5-100).
#[tauri::command]
fn set_cpu_limit(percent: u32) {
    batch::set_cpu_limit(percent);
}

/// Decode a track and load it for native playback. Returns its duration (s).
#[tauri::command]
async fn audio_load(path: String) -> Result<f64, String> {
    tauri::async_runtime::spawn_blocking(move || player::load(&path))
        .await
        .map_err(|e| e.to_string())?
}

/// Start or resume playback.
#[tauri::command]
fn audio_play() -> Result<(), String> {
    player::play()
}

/// Pause playback.
#[tauri::command]
fn audio_pause() {
    player::pause();
}

/// Seek to `secs`.
#[tauri::command]
fn audio_seek(secs: f64) {
    player::seek(secs);
}

/// Current playback position in seconds.
#[tauri::command]
fn audio_position() -> f64 {
    player::position()
}

/// Whether audio is currently playing.
#[tauri::command]
fn audio_is_playing() -> bool {
    player::is_playing()
}

/// Stop playback and release the current track.
#[tauri::command]
fn audio_stop() {
    player::stop();
}

/// Whether the neural models are present locally.
#[tauri::command]
fn models_ready() -> bool {
    cueit_core::models::models_present()
}

/// Enable/disable neural analysis at runtime.
#[tauri::command]
fn set_models_enabled(value: bool) {
    cueit_core::analysis::set_models_enabled(value);
}

/// Download the neural models to the data dir (streams `models-progress`).
#[tauri::command]
async fn download_models(app: AppHandle) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        let cb = |p: f32, msg: &str| {
            let _ = app.emit("models-progress", ModelProgress { p, msg: msg.to_string() });
        };
        cueit_core::models::ensure_models(&cb)
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Entry point used by `main.rs`.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            discover_paths,
            analyze,
            set_cancel,
            analyze_one,
            audio_load,
            audio_play,
            audio_pause,
            audio_seek,
            audio_position,
            audio_is_playing,
            audio_stop,
            models_ready,
            download_models,
            set_models_enabled,
            redetect_cues,
            export_tracks,
            detect_libraries,
            load_library,
            session_load,
            session_save,
            session_clear,
            default_workers,
            set_cpu_limit,
        ])
        .run(tauri::generate_context!())
        .expect("error while running CueIt");
}
