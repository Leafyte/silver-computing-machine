/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_API_TIMEOUT_MS?: string;
  readonly VITE_TILE_URL?: string;
  readonly VITE_TILE_ATTRIBUTION?: string;
  readonly VITE_ROUTING_URL?: string;
  readonly VITE_MAP_DEFAULT_VIEW?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

