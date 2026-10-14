import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Tauri expects a fixed dev port and does not want Vite to clear the screen.
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  // Use an inline PostCSS config so Vite does not walk up the filesystem and
  // pick up an unrelated config from a parent directory.
  css: { postcss: { plugins: [] } },
  server: {
    port: 1420,
    strictPort: true,
  },
  build: {
    target: "es2021",
    outDir: "dist",
  },
});
