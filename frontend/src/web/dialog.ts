// Stand-in for `@tauri-apps/plugin-dialog` used by the web build. Opens the
// browser's file/folder picker and registers the selected files.

import { register, registerFolder } from "./store";

type OpenOptions = {
  multiple?: boolean;
  directory?: boolean;
  filters?: { name?: string; extensions?: string[] }[];
};

// Dev/testing hook: `?demo` feeds a synthetic, obviously non-periodic WAV
// through the normal flow so the render can be checked headlessly.
function demoEnabled(): boolean {
  return location.search.includes("demo");
}

function synthWav(): File {
  const sr = 44100;
  const dur = 40;
  const n = sr * dur;
  const ab = new ArrayBuffer(44 + n * 2);
  const dv = new DataView(ab);
  const str = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i));
  };
  str(0, "RIFF");
  dv.setUint32(4, 36 + n * 2, true);
  str(8, "WAVE");
  str(12, "fmt ");
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true);
  dv.setUint16(22, 1, true);
  dv.setUint32(24, sr, true);
  dv.setUint32(28, sr * 2, true);
  dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true);
  str(36, "data");
  dv.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    const env = t < 6 ? 0.12 : t < 12 ? 0.35 : t < 18 ? 0.05 : t < 26 ? 0.9 : t < 34 ? 0.45 : 0.08;
    const s = Math.sin(2 * Math.PI * 220 * t) * env + Math.sin(2 * Math.PI * 330 * t) * env * 0.4;
    dv.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 32767, true);
  }
  return new File([ab], "demo.wav", { type: "audio/wav" });
}

export function open(opts: OpenOptions = {}): Promise<string | string[] | null> {
  if (demoEnabled() && !opts.directory) {
    const key = register(synthWav());
    return Promise.resolve(opts.multiple ? [key] : key);
  }
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    if (opts.directory) {
      input.setAttribute("webkitdirectory", "");
      input.multiple = true;
    } else if (opts.multiple) {
      input.multiple = true;
      const exts = opts.filters?.flatMap((f) => f.extensions ?? []) ?? [];
      if (exts.length) input.accept = exts.map((e) => "." + e).join(",");
    }
    input.onchange = () => {
      const list = input.files;
      if (!list || list.length === 0) {
        resolve(null);
        return;
      }
      if (opts.directory) {
        resolve(registerFolder(list));
        return;
      }
      const keys = Array.from(list).map(register);
      resolve(opts.multiple ? keys : keys[0]);
    };
    input.click();
  });
}
