// Stand-in for `@tauri-apps/api/webview` used by the web build: HTML5 drag/drop
// instead of the Tauri drag-drop event.

import { register } from "./store";

export function getCurrentWebview() {
  return {
    onDragDropEvent(cb: (event: { payload: { type: string; paths: string[]; position: { x: number; y: number } } }) => void) {
      const onOver = (e: DragEvent) => e.preventDefault();
      const onDrop = (e: DragEvent) => {
        e.preventDefault();
        const list = e.dataTransfer?.files;
        if (!list || list.length === 0) return;
        const paths = Array.from(list).map(register);
        cb({ payload: { type: "drop", paths, position: { x: e.clientX, y: e.clientY } } });
      };
      window.addEventListener("dragover", onOver);
      window.addEventListener("drop", onDrop);
      return Promise.resolve(() => {
        window.removeEventListener("dragover", onOver);
        window.removeEventListener("drop", onDrop);
      });
    },
  };
}
