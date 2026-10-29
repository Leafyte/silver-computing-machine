import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// The ArQ backend has no CORS middleware, so the browser must reach it
// same-origin. In development, Vite proxies /api to VITE_PROXY_TARGET.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_PROXY_TARGET || "http://127.0.0.1:8000";
  return {
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: 5173,
      proxy: { "/api": { target, changeOrigin: false } },
    },
    preview: {
      host: "127.0.0.1",
      port: 4173,
      proxy: { "/api": { target, changeOrigin: false } },
    },
    build: {
      chunkSizeWarningLimit: 900,
    },
  };
});

