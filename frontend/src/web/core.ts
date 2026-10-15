// Stand-in for `@tauri-apps/api/core` used by the web build. Implements the same
// `invoke(command, args)` surface as the Tauri backend, backed by the WASM core
// (analysis, cue inference, exporters), the Web Audio API (playback), and
// localStorage (session).

import init, {
  analyze as wasmAnalyze,
  inferCues,
  exportJson,
  exportCsv,
  exportRekordbox,
  exportTraktor,
} from "../wasm/cueit_wasm.js";
import { emit } from "./event";
import { files, folders, isAudio } from "./store";

type Wasm = {
  wasmAnalyze: typeof wasmAnalyze;
  inferCues: typeof inferCues;
  exportJson: typeof exportJson;
  exportCsv: typeof exportCsv;
  exportRekordbox: typeof exportRekordbox;
  exportTraktor: typeof exportTraktor;
};

let ready: Promise<Wasm> | null = null;
function wasm(): Promise<Wasm> {
  if (!ready) {
    ready = init().then(() => ({
      wasmAnalyze,
      inferCues,
      exportJson,
      exportCsv,
      exportRekordbox,
      exportTraktor,
    }));
  }
  return ready;
}

// --------------------------------------------------------------------------- //
// Web Audio playback
// --------------------------------------------------------------------------- //

let ctx: AudioContext | null = null;
let buffer: AudioBuffer | null = null;
let source: AudioBufferSourceNode | null = null;
let playing = false;
let base = 0;
let startedAt = 0;
const decoded = new Map<string, AudioBuffer>();

function actx(): AudioContext {
  if (!ctx) ctx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
  return ctx;
}
function dur(): number {
  return buffer?.duration ?? 0;
}
function position(): number {
  return playing ? Math.min(dur(), base + (actx().currentTime - startedAt)) : base;
}
function startAt(offset: number) {
  if (!buffer) return;
  source = actx().createBufferSource();
  source.buffer = buffer;
  source.connect(actx().destination);
  const s = source;
  source.onended = () => {
    if (source === s) {
      playing = false;
      base = dur();
    }
  };
  source.start(0, Math.min(offset, dur()));
  base = offset;
  startedAt = actx().currentTime;
  playing = true;
}
async function getBuffer(key: string): Promise<AudioBuffer> {
  const cached = decoded.get(key);
  if (cached) return cached;
  const file = files.get(key);
  if (!file) throw new Error(`file not available: ${key}`);
  const b = await actx().decodeAudioData(await file.arrayBuffer());
  decoded.set(key, b);
  return b;
}
async function toMono(b: AudioBuffer): Promise<Float32Array> {
  const ch = b.numberOfChannels;
  const len = b.length;
  const out = new Float32Array(len);
  for (let c = 0; c < ch; c++) {
    const d = b.getChannelData(c);
    for (let i = 0; i < len; i++) out[i] += d[i];
  }
  for (let i = 0; i < len; i++) out[i] /= ch;
  return out;
}

// --------------------------------------------------------------------------- //
// session (localStorage)
// --------------------------------------------------------------------------- //

function loadSession(): unknown[] {
  try {
    return JSON.parse(localStorage.getItem("cueit.session") || "[]");
  } catch {
    return [];
  }
}

// --------------------------------------------------------------------------- //
// commands
// --------------------------------------------------------------------------- //

let cancelFlag = false;

export async function invoke(command: string, args: Record<string, unknown> = {}): Promise<unknown> {
  switch (command) {
    case "default_workers":
      return Math.max(1, Math.min((navigator.hardwareConcurrency || 4) - 1, 8));
    case "set_cpu_limit":
    case "set_models_enabled":
    case "set_cancel":
      if (command === "set_cancel") cancelFlag = true;
      return null;
    case "models_ready":
      return false;
    case "download_models":
      throw new Error("Neural models are desktop-only; the web build uses the DSP engine");
    case "session_load":
      return loadSession();
    case "session_save":
      localStorage.setItem("cueit.session", JSON.stringify(args.tracks ?? []));
      return "localStorage";
    case "session_clear":
      localStorage.removeItem("cueit.session");
      return null;
    case "discover_paths":
      return discover(args.paths as string[], Boolean(args.recursive));
    case "detect_libraries":
      return [];
    case "load_library":
      throw new Error("DJ library import is desktop-only");
    case "analyze":
      return analyze(args.paths as string[], Boolean(args.useDownbeats));
    case "redetect_cues":
      return redetect(args.track as Record<string, unknown>, Boolean(args.useDownbeats));
    case "export_tracks":
      return exportTracks(args.format as string, args.tracks as Record<string, unknown>[]);
    case "audio_load":
      return audioLoad(args.path as string);
    case "audio_play":
      return audioPlay();
    case "audio_pause":
      return audioPause();
    case "audio_seek":
      return audioSeek(args.secs as number);
    case "audio_position":
      return position();
    case "audio_stop":
      return audioStop();
    default:
      throw new Error(`unknown command: ${command}`);
  }
}

function discover(paths: string[], _recursive: boolean): string[] {
  const out: string[] = [];
  for (const p of paths) {
    if (folders.has(p)) out.push(...folders.get(p)!);
    else if (files.has(p)) out.push(p);
  }
  return out.filter(isAudio).filter((v, i, a) => a.indexOf(v) === i);
}

async function analyze(paths: string[], useDownbeats: boolean) {
  const w = await wasm();
  cancelFlag = false;
  const jobs: {
    path: string;
    status: string;
    error: string;
    track: unknown;
  }[] = [];
  let done = 0;
  for (const key of paths) {
    if (cancelFlag) {
      jobs.push({ path: key, status: "cancelled", error: "", track: null });
    } else {
      let job = { path: key, status: "error", error: "", track: null as unknown };
      try {
        const buf = await getBuffer(key);
        const mono = await toMono(buf);
        const analysis = w.wasmAnalyze(mono, buf.sampleRate, key);
        const cues = w.inferCues(analysis, useDownbeats);
        job = { path: key, status: "done", error: "", track: { path: key, analysis, cues } };
      } catch (e) {
        job = { path: key, status: "error", error: String(e), track: null };
      }
      jobs.push(job);
    }
    done++;
    const last = jobs[jobs.length - 1];
    emit("analyze-progress", {
      done,
      total: paths.length,
      path: key,
      status: last.status,
      error: last.error,
      track: last.track,
    });
    // Yield so the UI stays responsive between tracks.
    await new Promise((r) => setTimeout(r, 0));
  }
  return jobs;
}

async function redetect(track: Record<string, unknown>, useDownbeats: boolean) {
  const w = await wasm();
  const cues = w.inferCues(track.analysis, useDownbeats);
  return { ...track, cues };
}

function download(name: string, text: string, mime: string) {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

async function exportTracks(format: string, tracks: Record<string, unknown>[]) {
  const w = await wasm();
  const base = (p: string) => p.replace(/\.[^.]+$/, "");
  const written: string[] = [];
  if (format === "json" || format === "csv") {
    for (const t of tracks) {
      const path = t.path as string;
      const text = format === "json" ? w.exportJson(t) : w.exportCsv(t);
      const name = base(path) + (format === "json" ? ".cueit.json" : ".cueit.csv");
      download(name, text, format === "json" ? "application/json" : "text/csv");
      written.push(name);
    }
  } else if (format === "rekordbox") {
    download("rekordbox.xml", w.exportRekordbox(tracks), "application/xml");
    written.push("rekordbox.xml");
  } else if (format === "traktor") {
    download("collection.nml", w.exportTraktor(tracks), "application/xml");
    written.push("collection.nml");
  } else {
    throw new Error(`${format} export is desktop-only`);
  }
  return written;
}

async function audioLoad(path: string): Promise<number> {
  const b = await getBuffer(path);
  buffer = b;
  source = null;
  playing = false;
  base = 0;
  return b.duration;
}
async function audioPlay(): Promise<void> {
  if (!buffer) throw new Error("no track loaded");
  await actx().resume();
  if (!playing) startAt(base);
}
async function audioPause(): Promise<void> {
  if (playing && source) {
    try {
      source.onended = null;
      source.stop();
    } catch {
      /* already stopped */
    }
    source = null;
    base = position();
    playing = false;
  }
}
async function audioSeek(secs: number): Promise<void> {
  const t = Math.max(0, Math.min(dur(), secs));
  if (playing) {
    try {
      source!.onended = null;
      source!.stop();
    } catch {
      /* already stopped */
    }
    source = null;
    playing = false;
    startAt(t);
  } else {
    base = t;
  }
}
function audioStop(): void {
  if (source) {
    try {
      source.onended = null;
      source.stop();
    } catch {
      /* already stopped */
    }
    source = null;
  }
  playing = false;
  base = 0;
  buffer = null;
}
