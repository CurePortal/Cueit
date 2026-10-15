//! Export analysis + cues to DJ software formats.
//!
//! Supported targets:
//! * Rekordbox  -> `rekordbox.xml` (imports into Rekordbox 6/7 via the XML bridge)
//! * Traktor    -> `collection.nml`
//! * Mixxx      -> writes directly into `mixxxdb.sqlite` (with a backup)
//! * Serato     -> writes ID3 `Serato Markers_` / `Serato Markers2` GEOB frames
//! * JSON / CSV -> neutral interchange
//!
//! All exporters take a [`TrackData`]. This file currently implements the pure
//! formats (JSON, CSV, Rekordbox, Traktor); Mixxx/Serato land with their
//! database / ID3 dependencies in a later phase.

use std::fmt::Write as _;
use std::io;
use std::path::Path;

use serde::Serialize;

use crate::types::TrackData;

// --------------------------------------------------------------------------- //
// helpers
// --------------------------------------------------------------------------- //

fn round_to(x: f64, dp: i32) -> f64 {
    let f = 10f64.powi(dp);
    (x * f).round() / f
}

fn expand_tilde(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    p.to_string()
}

/// `#rrggbb` -> packed 0xRRGGBB integer (Mixxx cue colour).
fn rgb_to_int(color: &str) -> i64 {
    let (r, g, b) = hex_to_rgb(color);
    ((r as i64) << 16) | ((g as i64) << 8) | (b as i64)
}

fn hex_to_rgb(color: &str) -> (u8, u8, u8) {
    let c = color.trim_start_matches('#');
    let p = |i: usize| u8::from_str_radix(&c[i..i + 2], 16).unwrap_or(0);
    (p(0), p(2), p(4))
}

fn key_display(t: &TrackData) -> String {
    format!("{} {}", t.analysis.key, t.analysis.scale).trim().to_string()
}

/// Escape a string for use inside a double-quoted XML attribute.
fn xml_attr(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\n' => out.push_str("&#10;"),
            '\r' => out.push_str("&#13;"),
            '\t' => out.push_str("&#09;"),
            _ => out.push(ch),
        }
    }
    out
}

/// One CSV field with RFC-4180 minimal quoting (Python `csv` dialect).
fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

// --------------------------------------------------------------------------- //
// JSON / CSV
// --------------------------------------------------------------------------- //

#[derive(Serialize)]
struct JsonCue<'a> {
    name: &'a str,
    time: f64,
    color: &'a str,
}

#[derive(Serialize)]
struct JsonTrack<'a> {
    file: &'a str,
    title: String,
    bpm: f64,
    key: String,
    camelot: &'a str,
    duration: f64,
    cues: Vec<JsonCue<'a>>,
}

pub fn json_string(track: &TrackData) -> String {
    let data = JsonTrack {
        file: &track.path,
        title: track.title(),
        bpm: round_to(track.analysis.bpm, 2),
        key: key_display(track),
        camelot: &track.analysis.camelot,
        duration: round_to(track.analysis.duration, 3),
        cues: track
            .cues
            .iter()
            .map(|c| JsonCue { name: &c.name, time: c.time, color: &c.color })
            .collect(),
    };
    serde_json::to_string_pretty(&data).unwrap_or_default()
}

pub fn export_json(track: &TrackData, out_path: impl AsRef<Path>) -> io::Result<String> {
    let path = out_path.as_ref();
    std::fs::write(path, json_string(track))?;
    Ok(path.to_string_lossy().into_owned())
}

pub fn csv_string(track: &TrackData) -> String {
    let mut s = String::new();
    let row = |s: &mut String, fields: &[String]| {
        let line: Vec<String> = fields.iter().map(|f| csv_field(f)).collect();
        s.push_str(&line.join(","));
        s.push_str("\r\n");
    };
    row(
        &mut s,
        &[
            "file".into(),
            "title".into(),
            "bpm".into(),
            "key".into(),
            "camelot".into(),
            "duration".into(),
        ],
    );
    row(
        &mut s,
        &[
            track.path.clone(),
            track.title(),
            format!("{:.2}", track.analysis.bpm),
            key_display(track),
            track.analysis.camelot.clone(),
            format!("{:.3}", track.analysis.duration),
        ],
    );
    row(&mut s, &[]);
    row(
        &mut s,
        &["cue".into(), "name".into(), "time_seconds".into(), "color".into()],
    );
    for (i, c) in track.cues.iter().enumerate() {
        row(
            &mut s,
            &[i.to_string(), c.name.clone(), format!("{:.3}", c.time), c.color.clone()],
        );
    }
    s
}

pub fn export_csv(track: &TrackData, out_path: impl AsRef<Path>) -> io::Result<String> {
    let path = out_path.as_ref();
    std::fs::write(path, csv_string(track))?;
    Ok(path.to_string_lossy().into_owned())
}

// --------------------------------------------------------------------------- //
// Rekordbox XML
// --------------------------------------------------------------------------- //

pub fn rekordbox_string(tracks: &[TrackData]) -> String {
    let mut x = String::new();
    x.push_str("<?xml version='1.0' encoding='UTF-8'?>\n");
    x.push_str("<DJ_PLAYLISTS Version=\"1.0.0\">\n");
    x.push_str("  <PRODUCT Name=\"CueIt\" Version=\"0.1\" Company=\"CueIt\" />\n");
    let _ = writeln!(x, "  <COLLECTION Entries=\"{}\">", tracks.len());
    for (tid, t) in tracks.iter().enumerate() {
        let _ = writeln!(
            x,
            "    <TRACK TrackID=\"{}\" Name=\"{}\" Artist=\"\" TotalTime=\"{}\" \
AverageBpm=\"{:.2}\" Tonality=\"{}\" Location=\"file://localhost{}\">",
            tid + 1,
            xml_attr(&t.title()),
            t.analysis.duration.round() as i64,
            t.analysis.bpm,
            xml_attr(&key_display(t)),
            xml_attr(&t.path),
        );
        let _ = writeln!(
            x,
            "      <TEMPO Inizio=\"0.000000\" Bpm=\"{:.2}\" Metro=\"4/4\" Battito=\"1\" />",
            t.analysis.bpm
        );
        for (num, c) in t.cues.iter().enumerate() {
            let (r, g, b) = hex_to_rgb(&c.color);
            let _ = writeln!(
                x,
                "      <POSITION_MARK Name=\"{}\" Type=\"0\" Start=\"{:.3}\" Num=\"{}\" \
Red=\"{}\" Green=\"{}\" Blue=\"{}\" />",
                xml_attr(&c.name),
                c.time,
                num,
                r,
                g,
                b
            );
        }
        x.push_str("    </TRACK>\n");
    }
    x.push_str("  </COLLECTION>\n");
    x.push_str("  <PLAYLISTS>\n");
    x.push_str("    <NODE Type=\"0\" Name=\"ROOT\" Count=\"1\">\n");
    let _ = writeln!(
        x,
        "      <NODE Name=\"CueIt\" Type=\"1\" KeyType=\"0\" Entries=\"{}\" />",
        tracks.len()
    );
    x.push_str("    </NODE>\n");
    x.push_str("  </PLAYLISTS>\n");
    x.push_str("</DJ_PLAYLISTS>\n");
    x
}

pub fn export_rekordbox(tracks: &[TrackData], out_path: impl AsRef<Path>) -> io::Result<String> {
    let path = out_path.as_ref();
    std::fs::write(path, rekordbox_string(tracks))?;
    Ok(path.to_string_lossy().into_owned())
}

// --------------------------------------------------------------------------- //
// Traktor NML
// --------------------------------------------------------------------------- //

pub fn traktor_string(tracks: &[TrackData]) -> String {
    let now = chrono::Local::now();
    let date = now.format("%Y/%m/%d").to_string();
    let clock = now.format("%H:%M:%S").to_string();

    let mut x = String::new();
    x.push_str("<?xml version='1.0' encoding='UTF-8'?>\n");
    x.push_str("<NML VERSION=\"19\">\n");
    x.push_str("  <HEAD COMPANY=\"CueIt\" PROGRAM=\"CueIt\" />\n");
    let _ = writeln!(x, "  <COLLECTION ENTRIES=\"{}\">", tracks.len());
    for t in tracks {
        let _ = writeln!(
            x,
            "    <ENTRY MODIFIED_DATE=\"{}\" MODIFIED_TIME=\"{}\" TITLE=\"{}\" ARTIST=\"\">",
            date,
            clock,
            xml_attr(&t.title())
        );
        let dir = Path::new(&t.path)
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let file = Path::new(&t.path)
            .file_name()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let _ = writeln!(
            x,
            "      <LOCATION DIR=\"{}{}\" FILE=\"{}\" VOLUME=\"\" VOLUMEID=\"\" />",
            xml_attr(&dir),
            std::path::MAIN_SEPARATOR,
            xml_attr(&file)
        );
        let _ = writeln!(
            x,
            "      <INFO BITRATE=\"0\" GENRE=\"\" COMMENT=\"{}\" IMPORT_DATE=\"{}\" \
RELEASE_DATE=\"\" FLAGS=\"1\" LENGTH=\"{}\" KEY=\"\" AUTHOR=\"\" />",
            xml_attr(&t.analysis.camelot),
            date,
            t.analysis.duration.round() as i64
        );
        let _ = writeln!(
            x,
            "      <TEMPO BPM=\"{:.6}\" BPM_QUALITY=\"100.000000\" />",
            t.analysis.bpm
        );
        for (i, c) in t.cues.iter().enumerate() {
            let _ = writeln!(
                x,
                "      <CUE_V2 NAME=\"{}\" DISPL_ORDER=\"0\" TYPE=\"0\" START=\"{:.6}\" \
LENGTH=\"0.000000\" REPEATS=\"-1\" HOTCUE=\"{}\" />",
                xml_attr(&c.name),
                c.time * 1000.0,
                i
            );
        }
        x.push_str("    </ENTRY>\n");
    }
    x.push_str("  </COLLECTION>\n");
    x.push_str("</NML>\n");
    x
}

pub fn export_traktor(tracks: &[TrackData], out_path: impl AsRef<Path>) -> io::Result<String> {
    let path = out_path.as_ref();
    std::fs::write(path, traktor_string(tracks))?;
    Ok(path.to_string_lossy().into_owned())
}

// --------------------------------------------------------------------------- //
// Mixxx (direct sqlite write)
// --------------------------------------------------------------------------- //

#[cfg(feature = "database")]
mod mixxx {
    use super::*;
    use rusqlite::{Connection, OptionalExtension};

    // Mixxx CueType enum values.
    const HOTCUE: i64 = 1;
    const MAIN_CUE: i64 = 2;
    const INTRO: i64 = 6;
    const OUTRO: i64 = 7;

    pub fn export(track: &TrackData, db_path: &str) -> Result<String, String> {
        let db_path = expand_tilde(db_path);
        if !Path::new(&db_path).exists() {
            return Err(format!("Mixxx database not found: {db_path}"));
        }
        let backup = format!("{db_path}.cueit.bak");
        if !Path::new(&backup).exists() {
            std::fs::copy(&db_path, &backup).map_err(|e| format!("backup failed: {e}"))?;
        }
        let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
        conn.execute_batch("BEGIN").map_err(|e| e.to_string())?;

        let loc = std::path::absolute(&track.path)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| track.path.clone());
        let track_id: Option<i64> = conn
            .query_row(
                "SELECT l.id FROM library l JOIN track_locations tl ON l.location = tl.id \
                 WHERE tl.location = ?1",
                [&loc],
                |r| r.get(0),
            )
            .optional()
            .map_err(|e| e.to_string())?;
        let Some(track_id) = track_id else {
            return Err(format!("track not in Mixxx library: {loc}"));
        };

        let sr: i64 = conn
            .query_row("SELECT samplerate FROM library WHERE id = ?1", [track_id], |r| r.get(0))
            .optional()
            .map_err(|e| e.to_string())?
            .unwrap_or(0);
        let sr = if sr > 0 { sr } else { 44100 };

        let key_str = if !track.analysis.camelot.is_empty() {
            track.analysis.camelot.clone()
        } else {
            let first = track.analysis.scale.chars().next().unwrap_or(' ');
            format!("{}{}", track.analysis.key, first)
        };
        conn.execute(
            "UPDATE library SET bpm = ?1, key = ?2 WHERE id = ?3",
            rusqlite::params![track.analysis.bpm, key_str, track_id],
        )
        .map_err(|e| e.to_string())?;

        conn.execute(
            "DELETE FROM cues WHERE track_id = ?1 AND type IN (?2,?3,?4,?5)",
            rusqlite::params![track_id, HOTCUE, INTRO, OUTRO, MAIN_CUE],
        )
        .map_err(|e| e.to_string())?;

        for (i, c) in track.cues.iter().enumerate() {
            let pos = (c.time * sr as f64).round() as i64;
            let color = rgb_to_int(&c.color);
            let lower = c.name.to_lowercase();
            let (ctype, hot) = if lower.starts_with("intro") {
                (INTRO, -1)
            } else if lower.starts_with("outro") {
                (OUTRO, -1)
            } else {
                (HOTCUE, i as i64)
            };
            conn.execute(
                "INSERT INTO cues (track_id, type, position, length, hotcue, label, color) \
                 VALUES (?1,?2,?3,?4,?5,?6,?7)",
                rusqlite::params![track_id, ctype, pos, 0i64, hot, c.name, color],
            )
            .map_err(|e| e.to_string())?;
        }
        conn.execute_batch("COMMIT").map_err(|e| e.to_string())?;
        Ok(format!("mixxx:{track_id}"))
    }
}

#[cfg(feature = "database")]
pub use mixxx::export as export_mixxx;

// --------------------------------------------------------------------------- //
// Serato ID3 tags
// --------------------------------------------------------------------------- //

#[cfg(feature = "tags")]
mod serato {
    use super::*;
    use crate::types::Cue;
    use id3::frame::{Content, EncapsulatedObject, Frame};
    use id3::{Tag, TagLike, Version};

    const VERSION2: u16 = 0x0101;
    const VERSION_: u16 = 0x0205;
    const NO_POSITION: u32 = 0x7F7F7F7F;
    const TRACK_COLOR: (u8, u8, u8) = (0x99, 0x99, 0x99);
    const MIME: &str = "application/octet-stream";

    fn serato32(value: u32) -> u32 {
        let a = (value >> 16) & 0xFF;
        let b = (value >> 8) & 0xFF;
        let c = value & 0xFF;
        let z = c & 0x7F;
        let y = ((c >> 7) | (b << 1)) & 0x7F;
        let x = ((b >> 6) | (a << 2)) & 0x7F;
        let w = a >> 5;
        (w << 24) | (x << 16) | (y << 8) | z
    }

    fn push_entry(d: &mut Vec<u8>, name: &str, payload: &[u8]) {
        d.extend_from_slice(name.as_bytes());
        d.push(0x00);
        d.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        d.extend_from_slice(payload);
    }

    fn markers2_field(cues: &[Cue], track_color: (u8, u8, u8)) -> Vec<u8> {
        let mut d = Vec::new();
        d.extend_from_slice(&VERSION2.to_be_bytes());
        let (r, g, b) = track_color;
        push_entry(&mut d, "COLOR", &[0x00, r, g, b]);
        for (i, c) in cues.iter().take(8).enumerate() {
            let (cr, cg, cb) = hex_to_rgb(&c.color);
            let mut p = Vec::new();
            p.push(0x00);
            p.push(i as u8);
            p.extend_from_slice(&((c.time * 1000.0).round() as u32).to_be_bytes());
            p.push(0x00);
            p.extend_from_slice(&[cr, cg, cb, 0x00, 0x00]);
            p.extend_from_slice(c.name.as_bytes());
            p.push(0x00);
            push_entry(&mut d, "CUE", &p);
        }
        push_entry(&mut d, "BPMLOCK", &[0x00]);
        d.push(0x00);
        d
    }

    fn markers_field(cues: &[Cue]) -> Vec<u8> {
        let mut entries: Vec<Vec<u8>> = Vec::with_capacity(14);
        for i in 0..5 {
            let mut e = Vec::new();
            if i < cues.len() {
                let (cr, cg, cb) = hex_to_rgb(&cues[i].color);
                let color32 = serato32(((cr as u32) << 16) | ((cg as u32) << 8) | (cb as u32));
                e.push(0x00);
                e.extend_from_slice(&serato32((cues[i].time * 1000.0).round() as u32).to_be_bytes());
                e.push(0x00);
                e.extend_from_slice(&serato32(0).to_be_bytes());
                e.extend_from_slice(&[0x00, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f]);
                e.extend_from_slice(&color32.to_be_bytes());
                e.extend_from_slice(&[0x00, 0x00]);
            } else {
                let color32 = serato32(0x7F7F7F);
                e.push(0x7f);
                e.extend_from_slice(&NO_POSITION.to_be_bytes());
                e.push(0x7f);
                e.extend_from_slice(&NO_POSITION.to_be_bytes());
                e.extend_from_slice(&[0x00, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f]);
                e.extend_from_slice(&color32.to_be_bytes());
                e.extend_from_slice(&[0x00, 0x00]);
            }
            entries.push(e);
        }
        for _ in 0..9 {
            let color32 = serato32(0x7F7F7F);
            let mut e = Vec::new();
            e.push(0x7f);
            e.extend_from_slice(&NO_POSITION.to_be_bytes());
            e.push(0x7f);
            e.extend_from_slice(&NO_POSITION.to_be_bytes());
            e.extend_from_slice(&[0x00, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f]);
            e.extend_from_slice(&color32.to_be_bytes());
            e.extend_from_slice(&[0x00, 0x01]);
            entries.push(e);
        }
        let (tr, tg, tb) = TRACK_COLOR;
        let mut d = Vec::new();
        d.extend_from_slice(&VERSION_.to_be_bytes());
        d.extend_from_slice(&(entries.len() as u32).to_be_bytes());
        for e in &entries {
            d.extend_from_slice(e);
        }
        d.extend_from_slice(
            &serato32(((tr as u32) << 16) | ((tg as u32) << 8) | (tb as u32)).to_be_bytes(),
        );
        d
    }

    pub fn export(track: &TrackData, timing_offset_ms: f64) -> Result<String, String> {
        let ext = Path::new(&track.path)
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();
        if ext != "mp3" && ext != "aiff" && ext != "aif" {
            return Err("Serato tag writing supports MP3/AIFF only (use Rekordbox XML for others).".into());
        }
        if !Path::new(&track.path).exists() {
            return Err(format!("file not found: {}", track.path));
        }

        let mut tag = Tag::read_from_path(&track.path).unwrap_or_else(|_| Tag::new());
        let adjusted: Vec<Cue> = track
            .cues
            .iter()
            .map(|c| Cue {
                name: c.name.clone(),
                time: (c.time - timing_offset_ms / 1000.0).max(0.0),
                color: c.color.clone(),
                kind: c.kind.clone(),
            })
            .collect();

        let markers2 = markers2_field(&adjusted, TRACK_COLOR);
        let markers_ = markers_field(&adjusted);

        tag.remove("GEOB");
        tag.add_frame(Frame::with_content(
            "GEOB",
            Content::EncapsulatedObject(EncapsulatedObject {
                mime_type: MIME.to_string(),
                filename: String::new(),
                description: "Serato Markers_".to_string(),
                data: markers_,
            }),
        ));
        tag.add_frame(Frame::with_content(
            "GEOB",
            Content::EncapsulatedObject(EncapsulatedObject {
                mime_type: MIME.to_string(),
                filename: String::new(),
                description: "Serato Markers2".to_string(),
                data: markers2,
            }),
        ));

        let key = if !track.analysis.camelot.is_empty() {
            track.analysis.camelot.clone()
        } else {
            format!("{} {}", track.analysis.key, track.analysis.scale).trim().to_string()
        };
        tag.set_text("TKEY", key);
        tag.set_text("TBPM", format!("{:.0}", track.analysis.bpm));
        tag.write_to_path(&track.path, Version::Id3v24).map_err(|e| e.to_string())?;
        Ok(track.path.clone())
    }
}

/// Write Serato Markers_ / Markers2 ID3 frames into an MP3/AIFF.
#[cfg(feature = "tags")]
pub fn export_serato(track: &TrackData) -> Result<String, String> {
    serato::export(track, 0.0)
}

/// Serato export with a decoder timing offset (ms) subtracted from cue times.
#[cfg(feature = "tags")]
pub fn export_serato_offset(track: &TrackData, timing_offset_ms: f64) -> Result<String, String> {
    serato::export(track, timing_offset_ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AnalysisResult, Cue};

    fn track() -> TrackData {
        TrackData {
            path: "/music/My Track.aiff".into(),
            analysis: AnalysisResult {
                bpm: 128.0,
                key: "A".into(),
                scale: "minor".into(),
                camelot: "8A".into(),
                duration: 245.6,
                ..Default::default()
            },
            cues: vec![
                Cue::new("Intro", 0.0, "#1DB954"),
                Cue::new("Drop", 64.0, "#E91E63"),
            ],
        }
    }

    #[test]
    fn json_round_trips_fields() {
        let dir = std::env::temp_dir();
        let out = dir.join("cueit_test.json");
        export_json(&track(), &out).unwrap();
        let s = std::fs::read_to_string(&out).unwrap();
        assert!(s.contains("\"title\": \"My Track\""));
        assert!(s.contains("\"key\": \"A minor\""));
        assert!(s.contains("\"bpm\": 128.0"));
        assert!(s.contains("\"name\": \"Drop\""));
        std::fs::remove_file(&out).ok();
    }

    #[test]
    fn csv_quotes_and_orders_rows() {
        let dir = std::env::temp_dir();
        let out = dir.join("cueit_test.csv");
        export_csv(&track(), &out).unwrap();
        let s = std::fs::read_to_string(&out).unwrap();
        assert!(s.starts_with("file,title,bpm,key,camelot,duration\r\n"));
        assert!(s.contains("cue,name,time_seconds,color\r\n0,Intro,0.000,#1DB954\r\n"));
        std::fs::remove_file(&out).ok();
    }

    #[test]
    fn rekordbox_has_collection_and_marks() {
        let dir = std::env::temp_dir();
        let out = dir.join("cueit_test_rb.xml");
        export_rekordbox(std::slice::from_ref(&track()), &out).unwrap();
        let s = std::fs::read_to_string(&out).unwrap();
        assert!(s.starts_with("<?xml version='1.0' encoding='UTF-8'?>\n"));
        assert!(s.contains("<COLLECTION Entries=\"1\">"));
        assert!(s.contains("Location=\"file://localhost/music/My Track.aiff\""));
        assert!(s.contains("<POSITION_MARK Name=\"Drop\""));
        std::fs::remove_file(&out).ok();
    }
}

#[cfg(all(test, feature = "database"))]
mod mixxx_tests {
    use super::*;
    use crate::types::{AnalysisResult, Cue};
    use rusqlite::Connection;

    #[test]
    fn mixxx_writes_bpm_key_and_cues() {
        let dir = std::env::temp_dir().join("cueit_mixxx_test");
        std::fs::create_dir_all(&dir).unwrap();
        let db = dir.join("mixxxdb.sqlite");
        let _ = std::fs::remove_file(&db);

        let track_path = dir.join("song.mp3");
        std::fs::write(&track_path, b"x").unwrap();
        let loc = std::path::absolute(&track_path).unwrap().to_string_lossy().into_owned();

        {
            let conn = Connection::open(&db).unwrap();
            conn.execute_batch(
                "CREATE TABLE track_locations (id INTEGER PRIMARY KEY, location TEXT);
                 CREATE TABLE library (id INTEGER PRIMARY KEY, location INTEGER, samplerate INTEGER,
                                       bpm REAL, key TEXT, mixxx_deleted INTEGER DEFAULT 0);
                 CREATE TABLE cues (id INTEGER PRIMARY KEY, track_id INTEGER, type INTEGER,
                                    position INTEGER, length INTEGER, hotcue INTEGER, label TEXT, color INTEGER);
                 INSERT INTO track_locations (id, location) VALUES (1, 'placeholder');",
            )
            .unwrap();
            conn.execute("UPDATE track_locations SET location = ?1 WHERE id = 1", [&loc]).unwrap();
            conn.execute(
                "INSERT INTO library (id, location, samplerate, bpm, key) VALUES (10, 1, 44100, 0, '')",
                [],
            )
            .unwrap();
        }

        let track = TrackData {
            path: loc.clone(),
            analysis: AnalysisResult {
                bpm: 128.0,
                key: "A".into(),
                scale: "minor".into(),
                camelot: "8A".into(),
                ..Default::default()
            },
            cues: vec![
                Cue::new("Intro", 0.0, "#1DB954"),
                Cue::new("Drop", 32.0, "#E91E63"),
            ],
        };
        let res = export_mixxx(&track, db.to_str().unwrap()).unwrap();
        assert_eq!(res, "mixxx:10");

        let conn = Connection::open(&db).unwrap();
        let (bpm, key): (f64, String) =
            conn.query_row("SELECT bpm, key FROM library WHERE id = 10", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            }).unwrap();
        assert!((bpm - 128.0).abs() < 1e-6);
        assert_eq!(key, "8A");

        let n: i64 = conn.query_row("SELECT COUNT(*) FROM cues WHERE track_id = 10", [], |r| r.get(0)).unwrap();
        assert_eq!(n, 2);
        let (ctype, hot, pos): (i64, i64, i64) = conn
            .query_row("SELECT type, hotcue, position FROM cues WHERE track_id = 10 ORDER BY position LIMIT 1", [], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .unwrap();
        assert_eq!(ctype, 6); // intro
        assert_eq!(hot, -1);
        assert_eq!(pos, 0);

        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(all(test, feature = "tags"))]
mod serato_tests {
    use super::*;
    use crate::types::{AnalysisResult, Cue};
    use id3::TagLike;

    #[test]
    fn serato_writes_markers_and_text_frames() {
        let dir = std::env::temp_dir().join("cueit_serato_test");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("song.mp3");
        // A minimal file with a valid-ish MPEG frame header is enough for id3 to
        // attach a tag; content is irrelevant to the tag writer.
        std::fs::write(&file, [0xFFu8, 0xFB, 0x90, 0x00, 0, 0, 0, 0]).unwrap();

        let track = TrackData {
            path: file.to_string_lossy().into_owned(),
            analysis: AnalysisResult {
                bpm: 128.0,
                key: "A".into(),
                scale: "minor".into(),
                camelot: "8A".into(),
                ..Default::default()
            },
            cues: vec![Cue::new("Intro", 0.0, "#1DB954"), Cue::new("Drop", 32.0, "#E91E63")],
        };
        export_serato(&track).unwrap();

        let tag = id3::Tag::read_from_path(&file).unwrap();
        let geobs: Vec<&id3::frame::Frame> = tag.frames().filter(|f| f.id() == "GEOB").collect();
        assert_eq!(geobs.len(), 2, "expected Serato Markers_ and Markers2");
        assert_eq!(tag.get("TKEY").and_then(|f| f.content().text()).map(|s| s.to_string()).as_deref(), Some("8A"));
        assert_eq!(tag.get("TBPM").and_then(|f| f.content().text()).map(|s| s.to_string()).as_deref(), Some("128"));

        std::fs::remove_dir_all(&dir).ok();
    }
}
