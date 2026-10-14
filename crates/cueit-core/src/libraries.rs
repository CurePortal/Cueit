//! Import track lists from DJ software libraries.
//!
//! * Rekordbox — `rekordbox.xml` (the XML bridge export)
//! * Traktor   — `collection.nml`
//! * Folder    — a directory of files (also covers Serato, which keeps cues in
//!   file tags and has no central database)
//!
//! Mixxx (`mixxxdb.sqlite`) needs a SQLite engine and is a native-only feature;
//! it is intentionally absent from the default/web build. The web app has no
//! library import at all (the browser cannot read local DJ databases).

use crate::batch::discover;
use crate::AUDIO_EXTS;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LibrarySource {
    /// Human label.
    pub name: String,
    /// `"mixxx" | "rekordbox" | "traktor" | "folder"`.
    pub kind: String,
    /// Path to a db/xml/nml/folder.
    pub locator: String,
}

impl LibrarySource {
    pub fn new(name: impl Into<String>, kind: impl Into<String>, locator: impl Into<String>) -> Self {
        Self { name: name.into(), kind: kind.into(), locator: locator.into() }
    }

    pub fn label(&self) -> String {
        format!("{} — {}", self.name, self.locator)
    }
}

/// Best-guess default locations per platform.
pub fn default_paths() -> Vec<(String, String)> {
    let home = std::env::var("HOME").unwrap_or_default();
    let (mixxx, rb, tk) = match std::env::consts::OS {
        "macos" => (
            format!("{home}/Library/Application Support/Mixxx/mixxxdb.sqlite"),
            format!("{home}/Music/rekordbox/rekordbox.xml"),
            format!("{home}/Documents/Native Instruments/Traktor 3.x/collection.nml"),
        ),
        "windows" => {
            let appdata = std::env::var("APPDATA").unwrap_or_else(|_| home.clone());
            (
                format!("{appdata}/Mixxx/mixxxdb.sqlite"),
                format!("{home}/Documents/rekordbox/rekordbox.xml"),
                format!("{home}/Documents/Native Instruments/Traktor 3.x/collection.nml"),
            )
        }
        _ => (
            format!("{home}/.mixxx/mixxxdb.sqlite"),
            format!("{home}/rekordbox.xml"),
            format!("{home}/Documents/Native Instruments/Traktor 3.x/collection.nml"),
        ),
    };
    vec![
        ("mixxx".to_string(), mixxx),
        ("rekordbox".to_string(), rb),
        ("traktor".to_string(), tk),
    ]
}

/// The libraries that exist on this machine.
pub fn detect_libraries() -> Vec<LibrarySource> {
    let mut found = Vec::new();
    for (kind, locator) in default_paths() {
        if std::path::Path::new(&locator).exists() {
            let name = match kind.as_str() {
                "mixxx" => "Mixxx library",
                "rekordbox" => "Rekordbox XML",
                "traktor" => "Traktor collection",
                _ => "Library",
            };
            found.push(LibrarySource::new(name, kind, locator));
        }
    }
    found
}

/// Resolve a library source to a list of existing audio file paths.
pub fn load_tracks(source: &LibrarySource) -> Result<Vec<String>, String> {
    match source.kind.as_str() {
        "mixxx" => load_mixxx(&source.locator),
        "folder" => Ok(discover(std::slice::from_ref(&source.locator), true)),
        "rekordbox" => Ok(existing(load_rekordbox(&source.locator))),
        "traktor" => Ok(existing(load_traktor(&source.locator))),
        other => Err(format!("unknown library kind: {other}")),
    }
}

fn expand_tilde(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    p.to_string()
}

/// Read the track list from `mixxxdb.sqlite` (read-only; native builds only).
#[cfg(feature = "database")]
fn load_mixxx(db_path: &str) -> Result<Vec<String>, String> {
    use rusqlite::{Connection, OpenFlags};
    let path = expand_tilde(db_path);
    if !std::path::Path::new(&path).exists() {
        return Err(format!("Mixxx database not found: {path}"));
    }
    let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare(
            "SELECT DISTINCT tl.location FROM library l JOIN track_locations tl ON l.location = tl.id \
             WHERE COALESCE(l.mixxx_deleted, 0) = 0 ORDER BY tl.location",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt.query_map([], |r| r.get::<_, String>(0)).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for loc in rows.flatten() {
        if !loc.is_empty() {
            out.push(loc);
        }
    }
    Ok(existing(out))
}

/// Mixxx import needs SQLite, which the web/no-default build omits.
#[cfg(not(feature = "database"))]
fn load_mixxx(_db_path: &str) -> Result<Vec<String>, String> {
    Err("Mixxx import requires the native build (SQLite is not in the web build)".to_string())
}

fn load_rekordbox(xml_path: &str) -> Vec<String> {
    let Ok(xml) = std::fs::read_to_string(xml_path) else { return Vec::new() };
    tags(&xml, "TRACK")
        .iter()
        .filter_map(|tag| attr(tag, "Location"))
        .map(|loc| {
            let p = loc
                .strip_prefix("file://localhost")
                .or_else(|| loc.strip_prefix("file://"))
                .unwrap_or(&loc);
            percent_decode(p)
        })
        .collect()
}

fn load_traktor(nml_path: &str) -> Vec<String> {
    let Ok(xml) = std::fs::read_to_string(nml_path) else { return Vec::new() };
    tags(&xml, "LOCATION")
        .iter()
        .filter_map(|tag| {
            let dir = attr(tag, "DIR").unwrap_or_default();
            let file = attr(tag, "FILE")?;
            Some(join_path(&dir, &file))
        })
        .collect()
}

fn join_path(dir: &str, file: &str) -> String {
    if dir.is_empty() {
        return file.to_string();
    }
    let sep = std::path::MAIN_SEPARATOR;
    if dir.ends_with(sep) || dir.ends_with('/') || dir.ends_with('\\') {
        format!("{dir}{file}")
    } else {
        format!("{dir}{sep}{file}")
    }
}

fn existing(paths: Vec<String>) -> Vec<String> {
    let mut seen = Vec::new();
    for p in paths {
        let path = std::path::Path::new(&p);
        let audio = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| AUDIO_EXTS.contains(&format!(".{}", e.to_lowercase()).as_str()))
            .unwrap_or(false);
        if path.is_file() && audio && !seen.contains(&p) {
            seen.push(p);
        }
    }
    seen
}

// --------------------------------------------------------------------------- //
// minimal XML scanning (machine-generated files, so this is more than enough)
// --------------------------------------------------------------------------- //

/// Inner text of every `<name ...>` element.
fn tags(xml: &str, name: &str) -> Vec<String> {
    let needle = format!("<{name}");
    let bytes = xml.as_bytes();
    let mut out = Vec::new();
    let mut search = 0;
    while let Some(rel) = xml[search..].find(&needle) {
        let start = search + rel;
        let after = start + needle.len();
        // must be a real start tag: next char is whitespace, '/' or '>'
        match bytes.get(after).copied() {
            Some(b) if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' || b == b'/' || b == b'>' => {}
            _ => {
                search = after;
                continue;
            }
        }
        // find the closing '>' of the start tag, ignoring quoted attribute values
        let mut i = after;
        let mut quote = 0u8;
        while i < bytes.len() {
            let b = bytes[i];
            if quote != 0 {
                if b == quote {
                    quote = 0;
                }
            } else if b == b'"' || b == b'\'' {
                quote = b;
            } else if b == b'>' {
                break;
            }
            i += 1;
        }
        out.push(xml[start..i.min(xml.len())].to_string());
        search = i + 1;
    }
    out
}

/// Extracts `name="value"` (either quote style) from a start tag, unescaping
/// the basic XML entities.
fn attr(tag: &str, name: &str) -> Option<String> {
    for quote in ['"', '\''] {
        let needle = format!("{name}={quote}");
        if let Some(pos) = tag.find(&needle) {
            let start = pos + needle.len();
            if let Some(end) = tag[start..].find(quote) {
                return Some(unescape(&tag[start..start + end]));
            }
        }
    }
    None
}

fn unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#10;", "\n")
        .replace("&#39;", "'")
        .replace("&amp;", "&")
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
            if let Ok(v) = u8::from_str_radix(hex, 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rekordbox_locations() {
        let xml = r#"<?xml version="1.0"?>
<DJ_PLAYLISTS><COLLECTION Entries="2">
<TRACK TrackID="1" Name="A" Location="file://localhost/music/My%20Song.mp3"/>
<TRACK TrackID="2" Name="B" Location="file://localhost/music/Other.aiff"/>
</COLLECTION></DJ_PLAYLISTS>"#;
        let got: Vec<String> = tags(xml, "TRACK")
            .iter()
            .filter_map(|t| attr(t, "Location"))
            .map(|l| percent_decode(l.strip_prefix("file://localhost").unwrap_or(&l)))
            .collect();
        assert_eq!(got, vec!["/music/My Song.mp3", "/music/Other.aiff"]);
    }

    #[test]
    fn parses_traktor_location() {
        let xml = r#"<NML><COLLECTION><ENTRY TITLE="x"><LOCATION DIR="/music/" FILE="a.aiff" VOLUME=""/></ENTRY></COLLECTION></NML>"#;
        let locations = tags(xml, "LOCATION");
        assert_eq!(locations.len(), 1);
        assert_eq!(attr(&locations[0], "DIR").as_deref(), Some("/music/"));
        assert_eq!(attr(&locations[0], "FILE").as_deref(), Some("a.aiff"));
    }
}

#[cfg(all(test, feature = "database"))]
mod mixxx_import_tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn imports_existing_tracks_from_mixxx() {
        let dir = std::env::temp_dir().join("cueit_mixxx_import");
        std::fs::create_dir_all(&dir).unwrap();
        let db = dir.join("mixxxdb.sqlite");
        let _ = std::fs::remove_file(&db);

        let good = dir.join("good.mp3");
        let missing = dir.join("gone.mp3");
        std::fs::write(&good, b"x").unwrap();

        {
            let conn = Connection::open(&db).unwrap();
            conn.execute_batch(
                "CREATE TABLE track_locations (id INTEGER PRIMARY KEY, location TEXT);
                 CREATE TABLE library (id INTEGER PRIMARY KEY, location INTEGER, mixxx_deleted INTEGER DEFAULT 0);",
            )
            .unwrap();
            conn.execute("INSERT INTO track_locations (id, location) VALUES (1, ?1)", [good.to_string_lossy().as_ref()]).unwrap();
            conn.execute("INSERT INTO track_locations (id, location) VALUES (2, ?1)", [missing.to_string_lossy().as_ref()]).unwrap();
            conn.execute("INSERT INTO library (id, location, mixxx_deleted) VALUES (10, 1, 0)", []).unwrap();
            conn.execute("INSERT INTO library (id, location, mixxx_deleted) VALUES (11, 2, 0)", []).unwrap();
        }

        let src = LibrarySource::new("Mixxx library", "mixxx", db.to_string_lossy().into_owned());
        let found = load_tracks(&src).unwrap();
        assert_eq!(found.len(), 1, "only the existing file should be returned");
        assert_eq!(found[0], good.to_string_lossy());

        std::fs::remove_dir_all(&dir).ok();
    }
}
