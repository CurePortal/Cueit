//! Persist analysis results so long batch runs survive crashes or restarts.
//!
//! A session is a JSON file mapping absolute track path -> [`TrackData`]. The
//! on-disk shape is identical to the original Python session (a `tracks` map of
//! `{path, analysis, cues}`), so sessions stay interchangeable.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::types::TrackData;

#[derive(Serialize, Deserialize)]
pub struct Session {
    pub version: u32,
    pub saved_at: f64,
    pub tracks: BTreeMap<String, TrackData>,
}

/// `$XDG_DATA_HOME/CueIt/session.json`, else `~/.local/share/CueIt/session.json`.
pub fn default_session_path() -> PathBuf {
    let base = std::env::var("XDG_DATA_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            format!("{home}/.local/share")
        });
    PathBuf::from(base).join("CueIt").join("session.json")
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Atomically write the session (`path` defaults to [`default_session_path`]).
pub fn save(
    tracks: &BTreeMap<String, TrackData>,
    path: Option<&Path>,
) -> std::io::Result<PathBuf> {
    let path = path.map(Path::to_path_buf).unwrap_or_else(default_session_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let payload = Session { version: 1, saved_at: now(), tracks: tracks.clone() };
    let json = serde_json::to_string(&payload).map_err(std::io::Error::other)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

/// Load a session, returning an empty map if it is missing or unreadable.
pub fn load(path: Option<&Path>) -> BTreeMap<String, TrackData> {
    let path = path.map(Path::to_path_buf).unwrap_or_else(default_session_path);
    let Ok(text) = std::fs::read_to_string(&path) else { return BTreeMap::new() };
    match serde_json::from_str::<Session>(&text) {
        Ok(s) => s.tracks,
        Err(_) => BTreeMap::new(),
    }
}

/// Delete the session file if it exists.
pub fn clear(path: Option<&Path>) {
    let path = path.map(Path::to_path_buf).unwrap_or_else(default_session_path);
    let _ = std::fs::remove_file(path);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_round_trips() {
        let dir = std::env::temp_dir().join("cueit_session_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("session.json");

        let mut tracks = BTreeMap::new();
        tracks.insert(
            "/music/a.mp3".to_string(),
            TrackData {
                path: "/music/a.mp3".into(),
                analysis: crate::types::AnalysisResult { bpm: 128.0, ..Default::default() },
                cues: vec![crate::types::Cue::new("Intro", 0.0, "#1DB954")],
            },
        );
        save(&tracks, Some(&path)).unwrap();
        let loaded = load(Some(&path));
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded["/music/a.mp3"].analysis.bpm, 128.0);
        assert_eq!(loaded["/music/a.mp3"].cues[0].name, "Intro");
        std::fs::remove_dir_all(&dir).ok();
    }
}
