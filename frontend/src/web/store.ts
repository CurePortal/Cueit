// In-memory registry of files the user picked / dropped, keyed by a synthetic
// "path" (the file name). The web backend uses this everywhere the desktop
// backend would use a filesystem path.

export const files = new Map<string, File>();
export const folders = new Map<string, string[]>();

let counter = 0;

export function register(file: File): string {
  let key = file.name;
  while (files.has(key)) key = `${file.name}#${++counter}`;
  files.set(key, file);
  return key;
}

export function registerFolder(list: FileList | File[]): string {
  const keys = Array.from(list).map(register);
  const key = `folder#${++counter}`;
  folders.set(key, keys);
  return key;
}

const AUDIO = [".wav", ".aiff", ".aif", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wv"];

export function isAudio(name: string): boolean {
  const lower = name.toLowerCase();
  return AUDIO.some((e) => lower.endsWith(e));
}
