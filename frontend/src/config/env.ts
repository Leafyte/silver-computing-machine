/**
 * Runtime configuration. Env values are build-time defaults; the operator may
 * override API base URL and timeout from Settings (stored in localStorage).
 * No credentials are ever read or stored here.
 */
import { settingsStore } from "../lib/storage";

function parseView(raw: string | undefined): { lat: number; lng: number; zoom: number } {
  const parts = (raw ?? "").split(",").map((p) => Number(p.trim()));
  if (parts.length === 3 && parts.every((n) => Number.isFinite(n))) {
    return { lat: parts[0], lng: parts[1], zoom: parts[2] };
  }
  return { lat: 20, lng: 0, zoom: 2 };
}

export const env = {
  apiBaseUrl: (import.meta.env.VITE_API_BASE_URL ?? "").trim().replace(/\/$/, ""),
  apiTimeoutMs: Number(import.meta.env.VITE_API_TIMEOUT_MS) > 0 ? Number(import.meta.env.VITE_API_TIMEOUT_MS) : 30000,
  tileUrl: (import.meta.env.VITE_TILE_URL ?? "").trim(),
  tileAttribution: (import.meta.env.VITE_TILE_ATTRIBUTION ?? "").trim(),
  routingUrl: (import.meta.env.VITE_ROUTING_URL ?? "").trim().replace(/\/$/, ""),
  defaultView: parseView(import.meta.env.VITE_MAP_DEFAULT_VIEW),
};

export function apiBaseUrl(): string {
  const override = settingsStore.get().apiBaseUrl?.trim();
  return (override ? override : env.apiBaseUrl).replace(/\/$/, "");
}

export function apiTimeoutMs(): number {
  const override = settingsStore.get().timeoutMs;
  return override && override > 0 ? override : env.apiTimeoutMs;
}

export const mapsConfigured = () => env.tileUrl.length > 0;
export const routingConfigured = () => env.routingUrl.length > 0;

