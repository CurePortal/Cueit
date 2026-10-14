import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type * as React from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open } from "@tauri-apps/plugin-dialog";
import type { Cue, JobDto, LibDto, Progress, TrackData } from "./types";

const AUDIO_EXTS = ["wav", "aiff", "aif", "mp3", "flac", "m4a", "aac", "ogg", "opus", "wv"];

const SECTION_FILL: Record<string, string> = {
  drop: "rgba(245,158,11,0.22)",
  breakdown: "rgba(168,139,250,0.22)",
  intro: "rgba(99,102,241,0.22)",
  outro: "rgba(20,184,166,0.22)",
  verse: "rgba(148,163,184,0.12)",
  section: "rgba(148,163,184,0.10)",
};

// Cue grab handles: arrows drawn at the top and bottom of each cue line.
const CUE_HANDLE_HW = 6; // half width of the triangle base
const CUE_HANDLE_H = 12; // height of the triangle

const SHORTCUTS: [string, string][] = [
  ["Space", "Play / pause"],
  ["← / →", "Jump back / forward 10 s"],
  ["Ctrl+← / Ctrl+→", "Previous / next cue"],
  ["↑ / ↓", "Previous / next track"],
  ["Home / End", "Go to start / end"],
  ["C", "Add cue at playhead"],
  ["Delete", "Delete selected cue"],
  ["Ctrl+Z / Ctrl+Shift+Z", "Undo / redo cue edits"],
  ["Mouse wheel", "Zoom around cursor"],
  ["Ctrl+= / Ctrl+- / Ctrl+0", "Zoom in / out / fit"],
  ["Ctrl+D / Ctrl+L", "Dark / light mode"],
  ["Ctrl+R / Esc", "Run / cancel batch"],
  ["Ctrl+Shift+R", "Re-detect cues"],
  ["Ctrl+S", "Save session"],
  ["Ctrl+O / Ctrl+Shift+O / Ctrl+Shift+L", "Add files / folder / library"],
  ["F1", "Show all shortcuts"],
];

type Status = { status: string; error: string };
type View = { start: number; span: number };

function baseName(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

function fmt(t: number): string {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

function loadSetting<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw == null ? fallback : (JSON.parse(raw) as T);
  } catch {
    return fallback;
  }
}

const QUEUE_ROW_H = 44;

/**
 * Virtualised, memoised queue list.
 *
 * Libraries can hold 10k+ tracks; rendering every row (and re-rendering the
 * list on every playback frame) made the UI lag. Only the visible slice is
 * rendered, and `memo` keeps playback ticks from touching it at all.
 */
const QueueList = memo(function QueueList({
  queued,
  tracks,
  status,
  selected,
  onSelect,
}: {
  queued: string[];
  tracks: Record<string, TrackData>;
  status: Record<string, Status>;
  selected: string | null;
  onSelect: (p: string) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [height, setHeight] = useState(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setHeight(el.clientHeight);
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
    return () => ro.disconnect();
  }, []);

  const total = queued.length;
  const start = Math.max(0, Math.floor(scrollTop / QUEUE_ROW_H) - 6);
  const visible = Math.ceil((height || 600) / QUEUE_ROW_H) + 12;
  const end = Math.min(total, start + visible);

  return (
    <div className="queue" ref={ref} onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}>
      <div className="queue-inner" style={{ height: total * QUEUE_ROW_H }}>
        <div style={{ transform: `translateY(${start * QUEUE_ROW_H}px)` }}>
          {queued.slice(start, end).map((p) => {
            const t = tracks[p];
            const st = status[p]?.status;
            const meta = t
              ? `${t.analysis.bpm.toFixed(1)} · ${t.analysis.key} ${t.analysis.scale} · ${t.analysis.camelot}`
              : st === "error"
                ? "error"
                : st === "done"
                  ? "done"
                  : st === "cancelled"
                    ? "cancelled"
                    : "pending";
            return (
              <div
                key={p}
                className={"qrow" + (selected === p ? " sel" : "")}
                style={{ height: QUEUE_ROW_H }}
                onClick={() => onSelect(p)}
                title={p}
              >
                <span className="qname">{baseName(p)}</span>
                <span className="qmeta">{meta}</span>
              </div>
            );
          })}
        </div>
      </div>
      {total === 0 && <div className="empty">No tracks queued</div>}
    </div>
  );
});

export default function App() {
  const [queued, setQueued] = useState<string[]>([]);
  const tracksRef = useRef<Record<string, TrackData>>({});
  const [tracks, setTracksState] = useState<Record<string, TrackData>>({});
  const setTracks = useCallback(
    (updater: React.SetStateAction<Record<string, TrackData>>) => {
      const next =
        typeof updater === "function"
          ? (updater as (p: Record<string, TrackData>) => Record<string, TrackData>)(tracksRef.current)
          : updater;
      tracksRef.current = next;
      setTracksState(next);
    },
    [],
  );

  const [status, setStatus] = useState<Record<string, Status>>({});
  const [selected, setSelected] = useState<string | null>(null);

  const mergeQueued = useCallback((paths: string[]) => {
    setQueued((prev) => {
      const seen = new Set(prev);
      const next = [...prev];
      for (const p of paths) if (!seen.has(p)) next.push(p);
      return next;
    });
  }, []);

  const [workers, setWorkers] = useState(() => loadSetting("cueit.workers", 4));
  const [cpuLimit, setCpuLimit] = useState(() => loadSetting("cueit.cpuLimit", 100));
  const [useDownbeats, setUseDownbeats] = useState(() => loadSetting("cueit.useDownbeats", true));
  const [exportFormat, setExportFormat] = useState(() => loadSetting("cueit.exportFormat", "rekordbox"));
  const [exportDir, setExportDir] = useState(() => loadSetting("cueit.exportDir", "."));
  const [libraries, setLibraries] = useState<LibDto[]>([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [theme, setTheme] = useState<"dark" | "light">(() => loadSetting("cueit.theme", "dark"));
  const [showHelp, setShowHelp] = useState(false);
  const [modelsReady, setModelsReady] = useState(true);
  const [modelsEnabled, setModelsEnabled] = useState(true);
  const [modelMsg, setModelMsg] = useState("");

  // playback (native, via rodio in the backend)
  const [playing, setPlaying] = useState(false);
  const [position, setPosition] = useState(0);

  // waveform view window
  const [view, setView] = useState<View>({ start: 0, span: 60 });
  const [selectedCue, setSelectedCue] = useState<number | null>(null);

  const trackList = useMemo(() => Object.values(tracks), [tracks]);
  const selectedTrack = selected ? tracks[selected] : undefined;
  const selectedExists = selectedTrack != null;
  const duration = selectedTrack?.analysis.duration ?? 0;

  // ---- persistence of settings ----
  useEffect(() => localStorage.setItem("cueit.workers", JSON.stringify(workers)), [workers]);
  useEffect(() => {
    localStorage.setItem("cueit.cpuLimit", JSON.stringify(cpuLimit));
    void invoke("set_cpu_limit", { percent: cpuLimit }).catch(() => {});
  }, [cpuLimit]);
  useEffect(() => localStorage.setItem("cueit.useDownbeats", JSON.stringify(useDownbeats)), [useDownbeats]);
  useEffect(() => localStorage.setItem("cueit.exportFormat", JSON.stringify(exportFormat)), [exportFormat]);
  useEffect(() => localStorage.setItem("cueit.exportDir", JSON.stringify(exportDir)), [exportDir]);
  useEffect(() => {
    localStorage.setItem("cueit.theme", JSON.stringify(theme));
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  // ---- stop native playback on unmount ----
  useEffect(() => () => { void invoke("audio_stop").catch(() => {}); }, []);

  // poll the backend playhead while playing
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      invoke<number>("audio_position")
        .then((p) => setPosition(p))
        .catch(() => {});
    }, 100);
    return () => clearInterval(id);
  }, [playing]);

  // ---- session + libraries + progress ----
  useEffect(() => {
    invoke<TrackData[]>("session_load").then((loaded) => {
      if (loaded.length) {
        const map: Record<string, TrackData> = {};
        for (const t of loaded) map[t.path] = t;
        setTracks(map);
        setMessage(`Restored ${loaded.length} analyzed track(s) from session`);
      }
    }).catch(() => {});
    if (localStorage.getItem("cueit.workers") == null) {
      invoke<number>("default_workers").then(setWorkers).catch(() => {});
    }
    invoke<boolean>("models_ready").then(setModelsReady).catch(() => {});
  }, [setTracks]);

  useEffect(() => {
    const un = listen<{ p: number; msg: string }>("models-progress", (e) => {
      setModelMsg(`${e.payload.msg} ${Math.round(e.payload.p * 100)}%`);
    });
    return () => { void un.then((f) => f()); };
  }, []);

  const toggleModels = useCallback(async () => {
    if (!modelsReady) {
      setModelMsg("starting…");
      try {
        await invoke("download_models");
        setModelsReady(true);
        setModelsEnabled(true);
        setModelMsg("");
        setMessage("AI models ready — model-assisted analysis enabled");
      } catch (e) {
        setModelMsg(String(e));
      }
      return;
    }
    const next = !modelsEnabled;
    setModelsEnabled(next);
    await invoke("set_models_enabled", { value: next }).catch(() => {});
    setMessage(next ? "AI models enabled" : "AI models disabled — using the DSP engine");
  }, [modelsReady, modelsEnabled]);

  useEffect(() => {
    const un = listen<Progress>("analyze-progress", (e) => {
      const p = e.payload;
      setProgress({ done: p.done, total: p.total });
      if (p.track) {
        const t = p.track;
        setTracks((prev) => ({ ...prev, [t.path]: t }));
      }
      setStatus((prev) => ({ ...prev, [p.path]: { status: p.status, error: p.error } }));
    });
    return () => { void un.then((f) => f()); };
  }, [setTracks]);

  // drag & drop files
  useEffect(() => {
    const un = getCurrentWebview().onDragDropEvent((event) => {
      if (event.payload.type === "drop") {
        const paths = event.payload.paths;
        invoke<string[]>("discover_paths", { paths, recursive: true })
          .then((found) => mergeQueued(found))
          .catch(() => {});
      }
    });
    return () => { void un.then((f) => f()); };
  }, [mergeQueued]);

  // ---- load audio when the selected track changes ----
  useEffect(() => {
    setPlaying(false);
    setPosition(0);
    setSelectedCue(null);
    const t = selected ? tracksRef.current[selected] : undefined;
    setView({ start: 0, span: Math.max(4, t?.analysis.duration ?? 60) });
    let cancelled = false;
    if (!t) {
      void invoke("audio_stop").catch(() => {});
      return;
    }
    setMessage("Preparing playback…");
    invoke<number>("audio_load", { path: t.path })
      .then(() => {
        if (!cancelled) setMessage("");
      })
      .catch((e) => {
        if (!cancelled) setMessage(String(e));
      });
    return () => {
      cancelled = true;
    };
    // `selectedExists` matters: if a pending track is selected then analysed,
    // `selected` doesn't change but the audio must (re)load.
  }, [selected, selectedExists]);

  // ---- session persistence helpers ----
  const saveSession = useCallback((map: Record<string, TrackData>) => {
    void invoke("session_save", { tracks: Object.values(map) }).catch(() => {});
  }, []);

  const persistSelected = useCallback(() => saveSession(tracksRef.current), [saveSession]);

  // ---- undo / redo (cue edits only, so analyses are never lost) ----
  const undoRef = useRef<Array<{ path: string; cues: Cue[] }>>([]);
  const redoRef = useRef<Array<{ path: string; cues: Cue[] }>>([]);
  const lastCueAddRef = useRef(0);
  const [hist, setHist] = useState({ undo: 0, redo: 0 });
  const syncHist = useCallback(
    () => setHist({ undo: undoRef.current.length, redo: redoRef.current.length }),
    [],
  );
  const pushUndo = useCallback(
    (path?: string | null) => {
      const p = path ?? selected;
      if (!p) return;
      const t = tracksRef.current[p];
      if (!t) return;
      undoRef.current.push({ path: p, cues: t.cues.map((c) => ({ ...c })) });
      if (undoRef.current.length > 200) undoRef.current.shift();
      redoRef.current.length = 0;
      syncHist();
    },
    [selected, syncHist],
  );
  const applyCues = useCallback(
    (path: string, cues: Cue[]) => {
      setTracks((prev) => {
        const cur = prev[path];
        if (!cur) return prev;
        const next = { ...prev, [path]: { ...cur, cues } };
        saveSession(next);
        return next;
      });
    },
    [setTracks, saveSession],
  );
  const undo = useCallback(() => {
    const entry = undoRef.current.pop();
    if (!entry) return;
    const cur = tracksRef.current[entry.path];
    if (cur) redoRef.current.push({ path: entry.path, cues: cur.cues.map((c) => ({ ...c })) });
    applyCues(entry.path, entry.cues);
    setSelected(entry.path);
    setMessage("Undo");
    syncHist();
  }, [applyCues, syncHist]);
  const redo = useCallback(() => {
    const entry = redoRef.current.pop();
    if (!entry) return;
    const cur = tracksRef.current[entry.path];
    if (cur) undoRef.current.push({ path: entry.path, cues: cur.cues.map((c) => ({ ...c })) });
    applyCues(entry.path, entry.cues);
    setSelected(entry.path);
    setMessage("Redo");
    syncHist();
  }, [applyCues, syncHist]);

  const beginCueDrag = useCallback(() => pushUndo(), [pushUndo]);

  // ---- file/library loading ----
  const addFiles = useCallback(async () => {
    const picked = await open({ multiple: true, filters: [{ name: "Audio", extensions: AUDIO_EXTS }] });
    if (!picked) return;
    const list = Array.isArray(picked) ? picked : [picked];
    const found = await invoke<string[]>("discover_paths", { paths: list, recursive: false });
    mergeQueued(found);
  }, [mergeQueued]);

  const addFolder = useCallback(async () => {
    const picked = await open({ directory: true });
    if (!picked || Array.isArray(picked)) return;
    const found = await invoke<string[]>("discover_paths", { paths: [picked], recursive: true });
    mergeQueued(found);
  }, [mergeQueued]);

  const loadLibraries = useCallback(async () => {
    const libs = await invoke<LibDto[]>("detect_libraries");
    setLibraries(libs);
    if (!libs.length) setMessage("No DJ libraries auto-detected");
  }, []);

  const loadLibrary = useCallback(
    async (lib: LibDto) => {
      setBusy(true);
      setMessage(`Loading ${lib.name}…`);
      try {
        const found = await invoke<string[]>("load_library", { kind: lib.kind, locator: lib.locator });
        mergeQueued(found);
        setMessage(`${lib.name}: ${found.length} tracks queued`);
      } catch (e) {
        setMessage(String(e));
      } finally {
        setBusy(false);
      }
    },
    [mergeQueued],
  );

  // ---- batch ----
  const runPaths = useCallback(
    async (paths: string[]) => {
      if (!paths.length) return;
      setBusy(true);
      setProgress({ done: 0, total: paths.length });
      setMessage(`Analyzing ${paths.length} track(s) with ${workers} threads…`);
      try {
        const jobs = await invoke<JobDto[]>("analyze", { paths, workers, useDownbeats });
        setTracks((prev) => {
          const next = { ...prev };
          for (const j of jobs) if (j.track) next[j.path] = j.track;
          return next;
        });
        setStatus((prev) => {
          const next = { ...prev };
          for (const j of jobs) next[j.path] = { status: j.status, error: j.error };
          return next;
        });
        setSelected((prev) =>
          prev && jobs.some((j) => j.path === prev)
            ? prev
            : jobs.find((j) => j.track)?.path ?? prev ?? null,
        );
        saveSession(tracksRef.current);
        const failed = jobs.filter((j) => j.status === "error").length;
        const cancelled = jobs.some((j) => j.status === "cancelled");
        setMessage(
          cancelled
            ? `Cancelled: ${jobs.length - failed} analyzed`
            : `Done: ${jobs.length - failed} analyzed, ${failed} failed`,
        );
      } catch (e) {
        setMessage(String(e));
      } finally {
        setBusy(false);
      }
    },
    [workers, useDownbeats, saveSession, setTracks],
  );

  const analyzeAll = useCallback(() => {
    const pending = queued.filter((p) => !tracks[p]);
    if (!pending.length) {
      setMessage("Queue is empty — add files, a folder, or a library");
      return;
    }
    void runPaths(pending);
  }, [queued, tracks, runPaths]);

  const runSelected = useCallback(() => {
    if (!selected) {
      setMessage("No track selected");
      return;
    }
    void runPaths([selected]);
  }, [selected, runPaths]);

  const cancelBatch = useCallback(() => {
    void invoke("set_cancel", { value: true });
    setMessage("Cancelling…");
  }, []);

  // ---- cue editing ----
  const patchSelectedCues = useCallback(
    (cues: Cue[], persist: boolean) => {
      if (!selected) return;
      setTracks((prev) => {
        const cur = prev[selected];
        if (!cur) return prev;
        const next = { ...prev, [selected]: { ...cur, cues } };
        if (persist) saveSession(next);
        return next;
      });
    },
    [selected, setTracks, saveSession],
  );

  const setCue = useCallback(
    (index: number, patch: Partial<Cue>, persist = true) => {
      if (!selectedTrack) return;
      const cues = selectedTrack.cues.map((c, i) => (i === index ? { ...c, ...patch } : c));
      patchSelectedCues(cues, persist);
    },
    [selectedTrack, patchSelectedCues],
  );

  const deleteCue = useCallback(
    (index: number) => {
      if (!selectedTrack) return;
      pushUndo();
      setSelectedCue(null);
      patchSelectedCues(selectedTrack.cues.filter((_, i) => i !== index), true);
    },
    [selectedTrack, patchSelectedCues, pushUndo],
  );

  const addCueAt = useCallback(
    (t: number) => {
      if (!selectedTrack) return;
      // Guard against runaway additions (e.g. a focused button being re-clicked
      // by the Space shortcut) — ignore repeats within 250ms.
      const now = performance.now();
      if (now - lastCueAddRef.current < 250) return;
      lastCueAddRef.current = now;
      pushUndo();
      const time = clamp(t, 0, duration);
      const cue: Cue = { name: "Cue", time, color: "#00BCD4", kind: "hotcue" };
      patchSelectedCues([...selectedTrack.cues, cue].sort((a, b) => a.time - b.time), true);
      setSelectedCue(selectedTrack.cues.length);
    },
    [selectedTrack, duration, patchSelectedCues, pushUndo],
  );

  const redetect = useCallback(async () => {
    if (!selectedTrack) return;
    pushUndo();
    const updated = await invoke<TrackData>("redetect_cues", { track: selectedTrack, useDownbeats });
    setTracks((prev) => {
      const next = { ...prev, [updated.path]: updated };
      saveSession(next);
      return next;
    });
    setMessage("Cues re-detected");
  }, [selectedTrack, useDownbeats, setTracks, saveSession, pushUndo]);

  // ---- playback controls ----
  const togglePlay = useCallback(() => {
    if (!selectedTrack) return;
    if (playing) {
      void invoke("audio_pause").catch(() => {});
      setPlaying(false);
    } else {
      invoke("audio_play")
        .then(() => setPlaying(true))
        .catch((e) => setMessage(String(e)));
    }
  }, [selectedTrack, playing]);

  const seek = useCallback(
    (t: number) => {
      const tt = clamp(t, 0, duration || t);
      setPosition(tt);
      void invoke("audio_seek", { secs: tt }).catch(() => {});
    },
    [duration],
  );

  const nudge = useCallback((dt: number) => seek(position + dt), [seek, position]);

  // ---- zoom ----
  const fitZoom = useCallback(() => setView({ start: 0, span: Math.max(4, duration) }), [duration]);
  const zoomAt = useCallback(
    (t: number, factor: number) => {
      setView((v) => {
        const span = clamp(v.span * factor, 2, Math.max(4, duration));
        const start = clamp(t - (t - v.start) * (span / v.span), 0, Math.max(0, duration - span));
        return { start, span };
      });
    },
    [duration],
  );
  const zoomBy = useCallback((factor: number) => zoomAt(position, factor), [zoomAt, position]);

  // ---- track / cue navigation ----
  const selectDelta = useCallback(
    (delta: number) => {
      const list = queued.length ? queued : trackList.map((t) => t.path);
      if (!list.length) return;
      const i = selected ? list.indexOf(selected) : -1;
      const next = clamp((i < 0 ? 0 : i + delta), 0, list.length - 1);
      setSelected(list[next]);
    },
    [queued, trackList, selected],
  );

  const gotoDelta = useCallback(
    (delta: number) => {
      if (!selectedTrack || !selectedTrack.cues.length) return;
      const times = selectedTrack.cues.map((c) => c.time);
      const cur = position;
      let idx = delta > 0 ? times.findIndex((t) => t > cur + 0.01) : -1;
      if (delta < 0) {
        for (let i = times.length - 1; i >= 0; i--) if (times[i] < cur - 0.01) { idx = i; break; }
      }
      if (idx >= 0) {
        seek(times[idx]);
        setSelectedCue(idx);
      }
    },
    [selectedTrack, position, seek],
  );

  // ---- export ----
  const doExport = useCallback(async () => {
    const all = Object.values(tracks);
    if (!all.length) {
      setMessage("Nothing analyzed to export");
      return;
    }
    setBusy(true);
    try {
      const written = await invoke<string[]>("export_tracks", {
        format: exportFormat,
        tracks: all,
        out: exportDir,
        outFile: null,
      });
      setMessage(`Exported ${written.length} file(s) to ${exportDir}`);
    } catch (e) {
      setMessage(String(e));
    } finally {
      setBusy(false);
    }
  }, [tracks, exportFormat, exportDir]);

  const pickExportDir = useCallback(async () => {
    const picked = await open({ directory: true });
    if (picked && !Array.isArray(picked)) setExportDir(picked);
  }, []);

  const saveNow = useCallback(() => {
    saveSession(tracksRef.current);
    setMessage("Session saved");
  }, [saveSession]);

  const clearSession = useCallback(async () => {
    await invoke("session_clear").catch(() => {});
    setMessage("Saved session cleared");
  }, []);

  // ---- keyboard shortcuts ----
  // Clicking a button must not leave it focused: otherwise pressing Space to
  // play/pause re-activates the focused button (e.g. "+ Cue") and adds cues.
  useEffect(() => {
    const onMouseDown = (e: MouseEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && t.closest("button")) e.preventDefault();
    };
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable)) {
        return;
      }
      const mod = e.ctrlKey || e.metaKey;
      const key = e.key;
      const lower = key.toLowerCase();
      const stop = () => { e.preventDefault(); e.stopPropagation(); };
      if (key === " ") {
        stop();
        const ae = document.activeElement as HTMLElement | null;
        if (ae && ae.tagName === "BUTTON") ae.blur();
        togglePlay();
        return;
      }
      if (key === "F1") { stop(); setShowHelp((s) => !s); return; }
      if (mod && lower === "r" && e.shiftKey) { stop(); void redetect(); return; }
      if (mod && lower === "r") { stop(); analyzeAll(); return; }
      if (key === "Escape") { stop(); if (busy) cancelBatch(); else setShowHelp(false); return; }
      if (mod && lower === "s") { stop(); saveNow(); return; }
      if (mod && lower === "z" && e.shiftKey) { stop(); redo(); return; }
      if (mod && lower === "z") { stop(); undo(); return; }
      if (mod && lower === "y") { stop(); redo(); return; }
      if (mod && lower === "o" && e.shiftKey) { stop(); void addFolder(); return; }
      if (mod && lower === "o") { stop(); void addFiles(); return; }
      if (mod && lower === "l" && e.shiftKey) { stop(); void loadLibraries(); return; }
      if (mod && (lower === "d" || lower === "l")) { stop(); setTheme((t) => (t === "dark" ? "light" : "dark")); return; }
      if (mod && (key === "=" || key === "+")) { stop(); zoomBy(0.87); return; }
      if (mod && key === "-") { stop(); zoomBy(1.15); return; }
      if (mod && key === "0") { stop(); fitZoom(); return; }
      if (mod && key === "ArrowLeft") { stop(); gotoDelta(-1); return; }
      if (mod && key === "ArrowRight") { stop(); gotoDelta(1); return; }
      if (key === "ArrowLeft") { stop(); nudge(-10); return; }
      if (key === "ArrowRight") { stop(); nudge(10); return; }
      if (key === "ArrowUp") { stop(); selectDelta(-1); return; }
      if (key === "ArrowDown") { stop(); selectDelta(1); return; }
      if (key === "Home") { stop(); seek(0); return; }
      if (key === "End") { stop(); seek(duration); return; }
      if (lower === "c") { stop(); addCueAt(position); return; }
      if (key === "Delete" || key === "Backspace") { stop(); deleteCue(selectedCue ?? selectedTrack?.cues.length ?? -1); return; }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    togglePlay, redetect, analyzeAll, busy, cancelBatch, saveNow, addFolder, addFiles,
    loadLibraries, zoomBy, fitZoom, gotoDelta, nudge, selectDelta, seek, duration,
    addCueAt, position, deleteCue, selectedCue, selectedTrack, undo, redo,
  ]);

  return (
    <div className="app">
      <header className="toolbar">
        <strong className="brand">CueIt</strong>
        <button
          className={"badge " + (modelsReady && modelsEnabled ? "on" : "off")}
          onClick={toggleModels}
          title={
            !modelsReady
              ? "Download the neural models (Beat This! + Deezer S-KEY, ~190 MB)"
              : modelsEnabled
                ? "Neural models active — click to disable"
                : "DSP engine only — click to enable neural models"
          }
        >
          {!modelsReady ? "Download AI models" : modelsEnabled ? "AI models ✓" : "AI models off"}
        </button>
        <button onClick={addFiles} disabled={busy}>Add files…</button>
        <button onClick={addFolder} disabled={busy}>Add folder…</button>
        <button onClick={loadLibraries} disabled={busy}>Detect libraries</button>
        {libraries.length > 0 && (
          <select
            defaultValue=""
            onChange={(e) => {
              const lib = libraries.find((l) => l.locator === e.target.value);
              if (lib) void loadLibrary(lib);
            }}
          >
            <option value="" disabled>Load library…</option>
            {libraries.map((l) => (
              <option key={l.locator} value={l.locator}>
                {l.name} ({l.tracks ?? "?"} tracks)
              </option>
            ))}
          </select>
        )}
        <span className="spacer" />
        <label>
          Threads
          <input
            type="number"
            min={1}
            max={32}
            value={workers}
            onChange={(e) => setWorkers(Math.max(1, Number(e.target.value) || 1))}
          />
        </label>
        <label>
          CPU
          <select
            value={cpuLimit}
            onChange={(e) => setCpuLimit(Number(e.target.value))}
            title="Cap average CPU usage — lower runs cooler but slower"
          >
            <option value={100}>100%</option>
            <option value={75}>75%</option>
            <option value={50}>50%</option>
            <option value={25}>25%</option>
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={useDownbeats} onChange={(e) => setUseDownbeats(e.target.checked)} />
          Downbeats
        </label>
        {busy ? (
          <button className="danger" onClick={cancelBatch}>Cancel</button>
        ) : (
          <>
            <button onClick={runSelected}>Run selected</button>
            <button className="primary" onClick={analyzeAll}>Run batch</button>
          </>
        )}
      </header>

      <div className="body">
        <aside className="sidebar">
          <div className="sidebar-head">
            Queue ({queued.length}) · Analyzed ({trackList.length})
          </div>
          <QueueList queued={queued} tracks={tracks} status={status} selected={selected} onSelect={setSelected} />
        </aside>

        <main className="main">
          {!selectedTrack && <div className="placeholder">Select an analyzed track to inspect its cues.</div>}
          {selectedTrack && (
            <>
              <div className="readout">
                <span className="title">{baseName(selectedTrack.path)}</span>
                <span className="stat"><b>{selectedTrack.analysis.bpm.toFixed(1)}</b> BPM</span>
                <span className="stat">
                  <b>{selectedTrack.analysis.key} {selectedTrack.analysis.scale}</b> ({selectedTrack.analysis.camelot})
                </span>
                <span className="stat">{selectedTrack.analysis.duration.toFixed(1)}s</span>
                <span className="stat">{selectedTrack.analysis.integrated_loudness.toFixed(1)} dB</span>
                <span className="stat">downbeat {Math.round(selectedTrack.analysis.downbeat_confidence * 100)}%</span>
                <span className="stat">phrase {selectedTrack.analysis.phrase_length}</span>
                <span className="spacer" />
                <button onClick={undo} disabled={hist.undo === 0} title="Undo (Ctrl+Z)">Undo</button>
                <button onClick={redo} disabled={hist.redo === 0} title="Redo (Ctrl+Shift+Z)">Redo</button>
                <button onClick={redetect}>Re-detect cues</button>
                <button onClick={() => addCueAt(position)}>+ Cue</button>
              </div>

              <Waveform
                track={selectedTrack}
                position={position}
                view={view}
                theme={theme}
                selectedCue={selectedCue}
                onSeek={seek}
                onZoomAt={zoomAt}
                onSelectCue={setSelectedCue}
                onCueDrag={(i, t) => setCue(i, { time: t }, false)}
                onCueDragStart={beginCueDrag}
                onCueCommit={persistSelected}
                onAddCue={addCueAt}
              />

              <div className="transport">
                <button onClick={() => seek(0)} title="Start">⏮</button>
                <button onClick={() => nudge(-10)} title="-10s">⏪</button>
                <button className="play" onClick={togglePlay} title="Play / pause (Space)">
                  {playing ? "⏸" : "▶"}
                </button>
                <button onClick={() => nudge(10)} title="+10s">⏩</button>
                <button onClick={() => seek(duration)} title="End">⏭</button>
                <span className="time">{fmt(position)} / {fmt(duration)}</span>
                <span className="spacer" />
                <button onClick={() => zoomBy(1.15)} title="Zoom out (Ctrl+-)">－</button>
                <button onClick={fitZoom} title="Fit (Ctrl+0)">fit</button>
                <button onClick={() => zoomBy(0.87)} title="Zoom in (Ctrl+=)">＋</button>
              </div>

              <table className="cues">
                <thead>
                  <tr>
                    <th>#</th><th>Name</th><th>Time (s)</th><th>Color</th><th></th>
                  </tr>
                </thead>
                <tbody>
                  {selectedTrack.cues.map((c, i) => (
                    <tr
                      key={i}
                      className={selectedCue === i ? "sel" : ""}
                      onClick={() => { setSelectedCue(i); seek(c.time); }}
                    >
                      <td>{i + 1}</td>
                      <td>
                        <input value={c.name} onFocus={() => pushUndo()} onChange={(e) => setCue(i, { name: e.target.value }, false)} onBlur={persistSelected} />
                      </td>
                      <td>
                        <input
                          type="number"
                          step="0.001"
                          value={c.time}
                          onFocus={() => pushUndo()}
                          onChange={(e) => setCue(i, { time: Number(e.target.value) || 0 }, false)}
                          onBlur={persistSelected}
                        />
                      </td>
                      <td><span className="swatch" style={{ background: c.color }} /></td>
                      <td><button onClick={(ev) => { ev.stopPropagation(); deleteCue(i); }}>Delete</button></td>
                    </tr>
                  ))}
                  {selectedTrack.cues.length === 0 && (
                    <tr><td colSpan={5} className="empty">No cues — press C to add one at the playhead</td></tr>
                  )}
                </tbody>
              </table>
            </>
          )}
        </main>
      </div>

      <footer className="status">
        <span>{message}</span>
        {modelMsg && <span className="prog">{modelMsg}</span>}
        {progress && <span className="prog">{progress.done}/{progress.total}</span>}
        <span className="spacer" />
        <label>
          Export
          <select value={exportFormat} onChange={(e) => setExportFormat(e.target.value)}>
            <option value="rekordbox">Rekordbox XML</option>
            <option value="traktor">Traktor NML</option>
            <option value="mixxx">Mixxx database</option>
            <option value="serato">Serato tags</option>
            <option value="json">JSON</option>
            <option value="csv">CSV</option>
          </select>
        </label>
        <button onClick={pickExportDir} title={exportDir}>
          {exportDir === "." ? "Choose output…" : exportDir}
        </button>
        <button className="primary" onClick={doExport} disabled={busy}>Export all</button>
        <button onClick={saveNow} title="Save session (Ctrl+S)">Save</button>
        <button onClick={clearSession}>Clear session</button>
        <button onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))} title="Theme (Ctrl+D)">
          {theme === "dark" ? "☀" : "☾"}
        </button>
        <button onClick={() => setShowHelp(true)} title="Shortcuts (F1)">?</button>
      </footer>

      {showHelp && <ShortcutsDialog onClose={() => setShowHelp(false)} />}
    </div>
  );
}

// --------------------------------------------------------------------------- //
// Waveform
// --------------------------------------------------------------------------- //

function Waveform({
  track,
  position,
  view,
  theme,
  selectedCue,
  onSeek,
  onZoomAt,
  onSelectCue,
  onCueDrag,
  onCueDragStart,
  onCueCommit,
  onAddCue,
}: {
  track: TrackData;
  position: number;
  view: View;
  theme: string;
  selectedCue: number | null;
  onSeek: (t: number) => void;
  onZoomAt: (t: number, factor: number) => void;
  onSelectCue: (i: number | null) => void;
  onCueDrag: (i: number, t: number) => void;
  onCueDragStart: () => void;
  onCueCommit: () => void;
  onAddCue: (t: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<number | null>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });

  const a = track.analysis;
  const dur = a.duration || 1;

  // Read theme colours once per theme change, not on every draw — getComputedStyle
  // is expensive and was running on every playback frame.
  const colors = useMemo(() => {
    const s = getComputedStyle(document.documentElement);
    const get = (k: string, fb: string) => s.getPropertyValue(k).trim() || fb;
    return {
      plot: get("--plot-bg", "#121319"),
      wave: get("--wave", "#1aa3c4"),
      playhead: get("--playhead", "#ffffff"),
      grid: get("--grid", "rgba(255,255,255,0.08)"),
    };
  }, [theme]);

  const timeAt = useCallback(
    (clientX: number) => {
      const c = canvasRef.current;
      if (!c) return view.start;
      const r = c.getBoundingClientRect();
      return view.start + ((clientX - r.left) / Math.max(1, r.width)) * view.span;
    },
    [view],
  );

  const cueX = useCallback(
    (t: number, width: number) => ((t - view.start) / view.span) * width,
    [view],
  );

  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const measure = () => setSize({ w: wrap.clientWidth, h: wrap.clientHeight });
    const ro = new ResizeObserver(measure);
    ro.observe(wrap);
    measure();
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 1.15 : 0.87;
      onZoomAt(timeAt(e.clientX), factor);
    };
    wrap.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      ro.disconnect();
      wrap.removeEventListener("wheel", onWheel);
    };
  }, [timeAt, onZoomAt]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.w === 0 || size.h === 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(size.w * dpr);
    canvas.height = Math.floor(size.h * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const { w: width, h: height } = size;
    const mid = height / 2;
    ctx.fillStyle = colors.plot;
    ctx.fillRect(0, 0, width, height);

    const xOf = (t: number) => ((t - view.start) / view.span) * width;

    // section shading
    for (const s of a.sections) {
      const x0 = xOf(s.start);
      const x1 = xOf(s.end);
      if (x1 < 0 || x0 > width) continue;
      ctx.fillStyle = SECTION_FILL[s.kind] ?? SECTION_FILL.section;
      ctx.fillRect(x0, 0, Math.max(1, x1 - x0), height);
    }

    // waveform: aggregate buckets into screen columns, then draw a filled,
    // lightly-smoothed amplitude envelope. One path → fast, and far less spiky
    // than one min/max stroke per bucket.
    const n = a.peak_mins.length;
    if (n > 0 && width > 0) {
      const cols = Math.max(1, Math.floor(width));
      const top = new Float32Array(cols);
      const bot = new Float32Array(cols);
      const i0 = clamp(Math.floor((view.start / dur) * n), 0, n - 1);
      const i1 = clamp(Math.ceil(((view.start + view.span) / dur) * n), i0 + 1, n);
      for (let i = i0; i < i1; i++) {
        const x = xOf(((i + 0.5) / n) * dur);
        const p = Math.min(cols - 1, Math.max(0, Math.floor((x / width) * cols)));
        if (a.peak_maxs[i] > top[p]) top[p] = a.peak_maxs[i];
        if (a.peak_mins[i] < bot[p]) bot[p] = a.peak_mins[i];
      }
      // Smooth more when zoomed out (many buckets share a column). Trailing box
      // filter; radius grows with density.
      const radius = clamp(Math.round(cols / 400), 1, 32);
      const smooth = (arr: Float32Array): Float32Array => {
        const out = new Float32Array(arr.length);
        let acc = 0;
        for (let p = 0; p < arr.length; p++) {
          acc += arr[p];
          if (p > radius) acc -= arr[p - radius - 1];
          out[p] = acc / Math.min(p + 1, radius + 1);
        }
        return out;
      };
      const st = smooth(top);
      const sb = smooth(bot);
      const scale = mid * 0.92;
      ctx.beginPath();
      for (let p = 0; p < cols; p++) ctx.lineTo(((p + 0.5) / cols) * width, mid - st[p] * scale);
      for (let p = cols - 1; p >= 0; p--) ctx.lineTo(((p + 0.5) / cols) * width, mid - sb[p] * scale);
      ctx.closePath();
      ctx.fillStyle = colors.wave;
      ctx.fill();
    }

    // downbeats
    ctx.strokeStyle = colors.grid;
    ctx.beginPath();
    for (const t of a.downbeats) {
      const x = xOf(t);
      if (x < 0 || x > width) continue;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
    }
    ctx.stroke();

    // cues (line + top/bottom grab-arrow handles)
    ctx.font = "11px system-ui, sans-serif";
    track.cues.forEach((c, i) => {
      const x = xOf(c.time);
      if (x < -20 || x > width + 20) return;
      const isSel = selectedCue === i;
      ctx.strokeStyle = c.color;
      ctx.lineWidth = isSel ? 3 : 2;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();

      ctx.fillStyle = c.color;
      ctx.strokeStyle = "rgba(0,0,0,0.55)";
      ctx.lineWidth = 1;
      // top arrow (points down)
      ctx.beginPath();
      ctx.moveTo(x - CUE_HANDLE_HW, 0);
      ctx.lineTo(x + CUE_HANDLE_HW, 0);
      ctx.lineTo(x, CUE_HANDLE_H);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      // bottom arrow (points up)
      ctx.beginPath();
      ctx.moveTo(x - CUE_HANDLE_HW, height);
      ctx.lineTo(x + CUE_HANDLE_HW, height);
      ctx.lineTo(x, height - CUE_HANDLE_H);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = c.color;
      ctx.fillText(c.name, x + 4, CUE_HANDLE_H + 12);
    });

    // playhead
    const phx = xOf(position);
    if (phx >= 0 && phx <= width) {
      ctx.strokeStyle = colors.playhead;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(phx, 0);
      ctx.lineTo(phx, height);
      ctx.stroke();
    }
  }, [track, position, view, selectedCue, size, a, dur]);

  // Which cue (if any) is under the pointer? Handles at top/bottom, or the line.
  const hitCue = (clientX: number, clientY: number): number | null => {
    const c = canvasRef.current;
    if (!c) return null;
    const r = c.getBoundingClientRect();
    const y = clientY - r.top;
    const nearTop = y <= CUE_HANDLE_H + 5;
    const nearBottom = y >= r.height - CUE_HANDLE_H - 5;
    for (let i = 0; i < track.cues.length; i++) {
      const x = r.left + cueX(track.cues[i].time, r.width);
      const dx = Math.abs(clientX - x);
      if ((nearTop || nearBottom) && dx <= CUE_HANDLE_HW + 5) return i;
      if (dx <= 5) return i;
    }
    return null;
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const c = canvasRef.current;
    if (!c) return;
    const hit = hitCue(e.clientX, e.clientY);
    if (hit != null) {
      dragRef.current = hit;
      onSelectCue(hit);
      onCueDragStart();
      c.setPointerCapture(e.pointerId);
      c.style.cursor = "grabbing";
      return;
    }
    onSelectCue(null);
    onSeek(clamp(timeAt(e.clientX), 0, dur));
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const c = canvasRef.current;
    if (!c) return;
    if (dragRef.current != null) {
      onCueDrag(dragRef.current, clamp(timeAt(e.clientX), 0, dur));
      return;
    }
    c.style.cursor = hitCue(e.clientX, e.clientY) != null ? "ew-resize" : "crosshair";
  };

  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const c = canvasRef.current;
    if (dragRef.current == null) return;
    dragRef.current = null;
    try {
      c?.releasePointerCapture(e.pointerId);
    } catch {
      /* not captured */
    }
    if (c) c.style.cursor = "crosshair";
    onCueCommit();
  };

  return (
    <div className="waveform" ref={wrapRef}>
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", touchAction: "none" }}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerCancel={onPointerUp}
                onDoubleClick={(e) => {
                  const t = clamp(timeAt(e.clientX), 0, dur);
                  onSelectCue(null);
                  onSeek(t);
                  onAddCue(t);
                }}
      />
    </div>
  );
}

// --------------------------------------------------------------------------- //
// Shortcuts dialog
// --------------------------------------------------------------------------- //

function ShortcutsDialog({ onClose }: { onClose: () => void }) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <strong>Keyboard shortcuts</strong>
          <button onClick={onClose}>Close</button>
        </div>
        <table className="shortcuts">
          <tbody>
            {SHORTCUTS.map(([k, v]) => (
              <tr key={k}><td className="key">{k}</td><td>{v}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
