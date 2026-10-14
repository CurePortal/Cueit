//! Native audio playback via `rodio` (cpal) — no webview media stack involved.
//!
//! The webview (`<audio>`) could not reliably play local files: WebKitGTK's
//! media backend rejects custom URI schemes, Blobs were unreliable for long
//! tracks, and a loopback HTTP server still hung. So decoding + playback live
//! in Rust: ffmpeg decodes the track to interleaved stereo f32, `rodio` plays
//! it, and the front end polls [`position`] to move the playhead.

use std::sync::mpsc;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use rodio::buffer::SamplesBuffer;
use rodio::{OutputStream, OutputStreamHandle, Sink};

const CHANNELS: u16 = 2;
const SAMPLE_RATE: u32 = 44_100;

/// Create (once) the cpal output stream on a dedicated thread that keeps it
/// alive, and hand back a cloneable handle.
fn handle() -> OutputStreamHandle {
    static H: OnceLock<OutputStreamHandle> = OnceLock::new();
    H.get_or_init(|| {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || match OutputStream::try_default() {
            Ok((_stream, h)) => {
                let _ = tx.send(Ok(h));
                loop {
                    std::thread::park();
                }
            }
            Err(e) => {
                let _ = tx.send(Err(e.to_string()));
            }
        });
        rx.recv()
            .ok()
            .and_then(std::result::Result::ok)
            .expect("no audio output device")
    })
    .clone()
}

struct Player {
    samples: Vec<f32>,
    sink: Option<Sink>,
    /// Playback offset where the current sink started.
    base: Duration,
    playing: bool,
}

static PLAYER: Mutex<Option<Player>> = Mutex::new(None);

fn duration_secs(samples_len: usize) -> f64 {
    samples_len as f64 / (CHANNELS as f64 * SAMPLE_RATE as f64)
}

fn start_from(p: &mut Player, base: Duration) -> Result<(), String> {
    let sink = Sink::try_new(&handle()).map_err(|e| e.to_string())?;
    let frame = (base.as_secs_f64() * SAMPLE_RATE as f64) as usize;
    let off = frame.saturating_mul(CHANNELS as usize).min(p.samples.len());
    let src = SamplesBuffer::new(CHANNELS, SAMPLE_RATE, p.samples[off..].to_vec());
    sink.append(src);
    sink.play();
    p.base = base;
    p.sink = Some(sink);
    p.playing = true;
    Ok(())
}

/// Decode `path` and make it the current track. Returns its duration in seconds.
pub fn load(path: &str) -> Result<f64, String> {
    let samples = decode_stereo(path)?;
    if samples.is_empty() {
        return Err(format!("empty audio: {path}"));
    }
    let dur = duration_secs(samples.len());
    let mut guard = PLAYER.lock().unwrap();
    if let Some(old) = guard.as_ref() {
        if let Some(s) = &old.sink {
            s.stop();
        }
    }
    *guard = Some(Player { samples, sink: None, base: Duration::ZERO, playing: false });
    Ok(dur)
}

pub fn play() -> Result<(), String> {
    let mut guard = PLAYER.lock().unwrap();
    let p = guard.as_mut().ok_or("no track loaded")?;
    if let Some(sink) = &p.sink {
        if !sink.empty() {
            sink.play();
            p.playing = true;
            return Ok(());
        }
    }
    let dur = duration_secs(p.samples.len());
    let base = if p.base.as_secs_f64() >= dur - 0.05 { Duration::ZERO } else { p.base };
    start_from(p, base)
}

pub fn pause() {
    let mut guard = PLAYER.lock().unwrap();
    if let Some(p) = guard.as_mut() {
        if let Some(s) = &p.sink {
            s.pause();
        }
        p.playing = false;
    }
}

pub fn seek(secs: f64) {
    let mut guard = PLAYER.lock().unwrap();
    if let Some(p) = guard.as_mut() {
        let dur = duration_secs(p.samples.len());
        let t = secs.clamp(0.0, dur);
        if let Some(s) = &p.sink {
            s.stop();
        }
        p.sink = None;
        p.base = Duration::from_secs_f64(t);
        if p.playing {
            let _ = start_from(p, p.base);
        }
    }
}

pub fn position() -> f64 {
    let guard = PLAYER.lock().unwrap();
    match guard.as_ref() {
        Some(p) => {
            let dur = duration_secs(p.samples.len());
            let pos = p.base.as_secs_f64()
                + p.sink.as_ref().map(|s| s.get_pos().as_secs_f64()).unwrap_or(0.0);
            pos.min(dur)
        }
        None => 0.0,
    }
}

pub fn is_playing() -> bool {
    let guard = PLAYER.lock().unwrap();
    guard
        .as_ref()
        .map(|p| p.playing && p.sink.as_ref().map(|s| !s.empty()).unwrap_or(false))
        .unwrap_or(false)
}

pub fn stop() {
    let mut guard = PLAYER.lock().unwrap();
    if let Some(p) = guard.as_mut() {
        if let Some(s) = &p.sink {
            s.stop();
        }
        p.sink = None;
        p.playing = false;
        p.base = Duration::ZERO;
    }
}

fn decode_stereo(path: &str) -> Result<Vec<f32>, String> {
    use std::process::Command;
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", "44100", "-"])
        .output()
        .map_err(|e| format!("ffmpeg not runnable ({e}); install ffmpeg for playback"))?;
    if !out.status.success() {
        return Err(format!("ffmpeg failed to decode {path}"));
    }
    Ok(out
        .stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
