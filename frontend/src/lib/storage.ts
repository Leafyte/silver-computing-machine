/**
 * Local persistence. Storage policy (enforced here, nowhere else writes
 * localStorage):
 *   - scenario drafts, run records (job IDs / local run IDs + returned
 *     results), recovery audit IDs, settings (API base URL, timeout),
 *     demo toggle, local approval records, last QpiAI-local verification.
 *   - NEVER tokens, secrets or credentials. sanitize() strips any key that
 *     looks like a credential before writing, as a defensive backstop.
 */
import { useSyncExternalStore } from "react";
import type { Scenario } from "../api/types";

const PREFIX = "arq.v1.";
const FORBIDDEN = /(token|secret|password|authorization|api[_-]?key|bearer|credential)/i;

function sanitize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sanitize);
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      if (FORBIDDEN.test(k)) continue;
      out[k] = sanitize(v);
    }
    return out;
  }
  return value;
}

function createStore<T>(key: string, initial: T) {
  const storageKey = PREFIX + key;
  let cache: T = initial;
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (raw !== null) cache = JSON.parse(raw) as T;
  } catch {
    cache = initial;
  }
  const listeners = new Set<() => void>();
  const notify = () => listeners.forEach((l) => l());
  window.addEventListener("storage", (e) => {
    if (e.key !== storageKey) return;
    try {
      cache = e.newValue === null ? initial : (JSON.parse(e.newValue) as T);
    } catch {
      cache = initial;
    }
    notify();
  });
  return {
    get: () => cache,
    set(next: T | ((prev: T) => T)) {
      cache = typeof next === "function" ? (next as (p: T) => T)(cache) : next;
      try {
        window.localStorage.setItem(storageKey, JSON.stringify(sanitize(cache)));
      } catch {
        /* quota exceeded: keep in memory only */
      }
      notify();
    },
    reset() {
      cache = initial;
      window.localStorage.removeItem(storageKey);
      notify();
    },
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

export type Store<T> = ReturnType<typeof createStore<T>>;

export function useStore<T>(store: Store<T>): T {
  return useSyncExternalStore(store.subscribe, store.get, store.get);
}

// ---------------- Models ----------------
export interface DraftOrder {
  id: string;
  lat: number | null;
  lng: number | null;
  demand: number;
  windowOpen: number;
  windowClose: number;
  service: number;
  penalty: number;
  hardWindow: boolean;
}

export interface DraftVehicle {
  id: string;
  capacity: number;
}

export interface DraftScenario {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  depot: { id: string; lat: number | null; lng: number | null };
  vehicles: DraftVehicle[];
  orders: DraftOrder[];
  /** derived = straight-line km / speed; manual = operator-entered matrix. */
  travelMode: "derived" | "manual";
  speedKmh: number;
  manualTravel: Record<string, Record<string, number>>;
  /** Provenance of the draft itself. */
  origin: "local draft" | "demo";
  lastValidation?: { at: string; valid: boolean; hash: string };
}

export type MethodId = "exact" | "exact_job" | "financial" | "vqc" | "qaoa" | "qaoa_job" | "risk" | "recovery";

export interface RunRecord {
  id: string;
  source: "job" | "sync";
  method: MethodId;
  createdAt: string;
  draftId?: string;
  draftName?: string;
  scenario: Scenario;
  scenarioHash: string;
  settings?: Record<string, unknown>;
  status: string;
  roundTripMs?: number;
  result?: unknown;
  error?: { code?: string; message: string; status?: number };
  lastFetchedAt?: string;
  auditId?: string;
  /** For risk/recovery: the run whose plan was used as input. */
  parentRunId?: string;
  origin: "backend" | "demo";
}

export interface Settings {
  apiBaseUrl?: string;
  timeoutMs?: number;
}

export interface ApprovalRecord {
  runId: string;
  approvedAt: string;
  note: string;
}

export interface QpiaiVerification {
  at: string;
  outcome: "verified_local" | "unsupported" | "failed";
  versions?: Record<string, string>;
  runId?: string;
}

// ---------------- Stores ----------------
export const settingsStore = createStore<Settings>("settings", {});
export const demoStore = createStore<boolean>("demo", false);
export const draftsStore = createStore<DraftScenario[]>("drafts", []);
export const activeDraftStore = createStore<string | null>("activeDraft", null);
export const runsStore = createStore<RunRecord[]>("runs", []);
export const selectedRunStore = createStore<string | null>("selectedRun", null);
export const approvalsStore = createStore<Record<string, ApprovalRecord>>("approvals", {});
export const qpiaiStore = createStore<QpiaiVerification | null>("qpiaiVerification", null);

const MAX_RUNS = 60;

export function upsertRun(run: RunRecord) {
  runsStore.set((prev) => {
    const rest = prev.filter((r) => r.id !== run.id);
    return [run, ...rest].slice(0, MAX_RUNS);
  });
}

export function patchRun(id: string, patch: Partial<RunRecord>) {
  runsStore.set((prev) => prev.map((r) => (r.id === id ? { ...r, ...patch } : r)));
}

export function removeRun(id: string) {
  runsStore.set((prev) => prev.filter((r) => r.id !== id));
  approvalsStore.set((prev) => {
    const next = { ...prev };
    delete next[id];
    return next;
  });
}

export function upsertDraft(draft: DraftScenario) {
  draftsStore.set((prev) => {
    const exists = prev.some((d) => d.id === draft.id);
    return exists ? prev.map((d) => (d.id === draft.id ? draft : d)) : [draft, ...prev];
  });
}

export function removeDraft(id: string) {
  draftsStore.set((prev) => prev.filter((d) => d.id !== id));
  if (activeDraftStore.get() === id) activeDraftStore.set(null);
}

export function clearAllLocalData() {
  [settingsStore, demoStore, draftsStore, activeDraftStore, runsStore, selectedRunStore, approvalsStore, qpiaiStore].forEach((s) =>
    (s as Store<unknown>).reset(),
  );
}

export function newId(prefix: string): string {
  const rand = typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : Math.random().toString(36).slice(2);
  return `${prefix}-${rand}`;
}

