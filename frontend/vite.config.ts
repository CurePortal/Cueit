import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

const r = (p: string) => fileURLToPath(new URL(p, import.meta.url));

// The same app is built two ways:
//   - desktop (default): talks to the Tauri backend via `@tauri-apps/*`.
//   - web (`--mode web`): those packages are aliased to browser shims that back
//     the same `invoke`/`listen`/`open` surface with WASM + Web Audio, so the
//     identical UI runs in the browser.
export default defineConfig(({ mode }) => {
  const web = mode === "web";
  return {
    plugins: [react()],
    clearScreen: false,
    css: { postcss: { plugins: [] } },
    base: web ? "/Cueit/" : "/",
    resolve: web
      ? {
          alias: {
            "@tauri-apps/api/core": r("./src/web/core.ts"),
            "@tauri-apps/api/event": r("./src/web/event.ts"),
            "@tauri-apps/api/webview": r("./src/web/webview.ts"),
            "@tauri-apps/plugin-dialog": r("./src/web/dialog.ts"),
          },
        }
      : undefined,
    server: {
      port: 1420,
      strictPort: true,
    },
    build: {
      target: "es2021",
      outDir: web ? r("../web") : "dist",
      emptyOutDir: true,
    },
  };
});
