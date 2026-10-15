// Minimal stand-in for `@tauri-apps/api/event` used by the web build.

type Handler = (event: { payload: unknown }) => void;

const listeners = new Map<string, Set<Handler>>();

export function listen(event: string, handler: Handler): Promise<() => void> {
  let set = listeners.get(event);
  if (!set) {
    set = new Set();
    listeners.set(event, set);
  }
  set.add(handler);
  return Promise.resolve(() => {
    set?.delete(handler);
  });
}

export function emit(event: string, payload: unknown): void {
  listeners.get(event)?.forEach((h) => h({ payload }));
}
