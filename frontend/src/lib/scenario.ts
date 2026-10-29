/**
 * Draft <-> backend Scenario conversion. Coordinates are frontend-only draft
 * metadata: the backend scenario schema has no location fields and rejects
 * unknown keys, so they are stripped here. Travel durations are either
 * derived locally from straight-line distance at an operator-set speed, or
 * entered manually. Neither is road-network travel time.
 */
import type { Plan, Scenario, TravelMatrix } from "../api/types";
import type { DraftScenario } from "./storage";
import { newId } from "./storage";
import { env } from "../config/env";

export function haversineKm(a: { lat: number; lng: number }, b: { lat: number; lng: number }): number {
  const R = 6371;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const dLat = toRad(b.lat - a.lat);
  const dLng = toRad(b.lng - a.lng);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(a.lat)) * Math.cos(toRad(b.lat)) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

export function locationIds(draft: DraftScenario): string[] {
  return [draft.depot.id, ...draft.orders.map((o) => o.id)];
}

type Point = { id: string; lat: number | null; lng: number | null };

function points(draft: DraftScenario): Point[] {
  return [draft.depot, ...draft.orders];
}

/** Returns null when any location lacks coordinates (derived mode cannot compute). */
export function derivedTravel(draft: DraftScenario): TravelMatrix | null {
  const pts = points(draft);
  if (pts.some((p) => p.lat === null || p.lng === null) || !(draft.speedKmh > 0)) return null;
  const matrix: TravelMatrix = {};
  for (const a of pts) {
    matrix[a.id] = {};
    for (const b of pts) {
      const km = a.id === b.id ? 0 : haversineKm({ lat: a.lat!, lng: a.lng! }, { lat: b.lat!, lng: b.lng! });
      matrix[a.id][b.id] = Math.round((km / draft.speedKmh) * 60 * 10) / 10;
    }
  }
  return matrix;
}

export function effectiveTravel(draft: DraftScenario): TravelMatrix | null {
  if (draft.travelMode === "derived") return derivedTravel(draft);
  const ids = locationIds(draft);
  const m: TravelMatrix = {};
  for (const a of ids) {
    m[a] = {};
    for (const b of ids) {
      const v = draft.manualTravel?.[a]?.[b];
      if (a === b) m[a][b] = typeof v === "number" ? v : 0;
      else if (typeof v === "number" && Number.isFinite(v)) m[a][b] = v;
      else return null;
    }
  }
  return m;
}

export interface DraftIssue {
  path: string;
  message: string;
}

/** Local pre-checks (the backend validation route remains authoritative). */
export function localIssues(draft: DraftScenario): DraftIssue[] {
  const issues: DraftIssue[] = [];
  if (!draft.depot.id.trim()) issues.push({ path: "depot", message: "Depot ID is required" });
  if (draft.vehicles.length === 0) issues.push({ path: "vehicles", message: "Add at least one vehicle" });
  const ids = locationIds(draft);
  const dup = ids.filter((id, i) => ids.indexOf(id) !== i);
  if (dup.length) issues.push({ path: "orders", message: `Duplicate location IDs: ${[...new Set(dup)].join(", ")}` });
  const vids = draft.vehicles.map((v) => v.id);
  if (vids.some((id, i) => vids.indexOf(id) !== i)) issues.push({ path: "vehicles", message: "Vehicle IDs must be unique" });
  draft.orders.forEach((o, i) => {
    if (!o.id.trim()) issues.push({ path: `orders[${i}].id`, message: "Order ID is required" });
    if (o.windowOpen > o.windowClose) issues.push({ path: `orders[${i}].window`, message: "Window opens after it closes" });
  });
  if (effectiveTravel(draft) === null) {
    issues.push({
      path: "travel",
      message: draft.travelMode === "derived" ? "Place every location on the map (or enter coordinates) to derive travel times" : "Fill every cell of the travel matrix",
    });
  }
  return issues;
}

export function toScenario(draft: DraftScenario): Scenario | null {
  const travel = effectiveTravel(draft);
  if (!travel) return null;
  return {
    depot: draft.depot.id,
    vehicles: draft.vehicles.map((v) => ({ id: v.id, capacity: v.capacity })),
    orders: draft.orders.map((o) => ({
      id: o.id,
      demand: o.demand,
      window: [o.windowOpen, o.windowClose] as [number, number],
      service: o.service,
      penalty: o.penalty,
      ...(o.hardWindow ? { hard_window: true } : {}),
    })),
    travel,
  };
}

export function blankDraft(name = "Untitled scenario"): DraftScenario {
  const now = new Date().toISOString();
  return {
    id: newId("draft"),
    name,
    createdAt: now,
    updatedAt: now,
    depot: { id: "DEPOT", lat: null, lng: null },
    vehicles: [{ id: "V1", capacity: 3 }],
    orders: [],
    travelMode: "derived",
    speedKmh: 30,
    manualTravel: {},
    origin: "local draft",
  };
}

export function nextOrderId(draft: DraftScenario): string {
  const taken = new Set(locationIds(draft));
  for (let i = 1; i < 1000; i += 1) {
    const id = `O${i}`;
    if (!taken.has(id)) return id;
  }
  return `O${Date.now()}`;
}

export function nextVehicleId(draft: DraftScenario): string {
  const taken = new Set(draft.vehicles.map((v) => v.id));
  for (let i = 1; i < 1000; i += 1) if (!taken.has(`V${i}`)) return `V${i}`;
  return `V${Date.now()}`;
}

/** Rename a location everywhere (manual travel keys included). */
export function renameLocation(draft: DraftScenario, from: string, to: string): DraftScenario["manualTravel"] {
  const m: DraftScenario["manualTravel"] = {};
  for (const [a, row] of Object.entries(draft.manualTravel ?? {})) {
    const ka = a === from ? to : a;
    m[ka] = {};
    for (const [b, v] of Object.entries(row)) m[ka][b === from ? to : b] = v;
  }
  return m;
}

export function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    const keys = Object.keys(value as Record<string, unknown>).sort();
    return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalJson((value as Record<string, unknown>)[k])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

export async function sha256(text: string): Promise<string> {
  if (!crypto?.subtle) return "unavailable";
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

export const scenarioHash = (s: Scenario) => sha256(canonicalJson(s));

export function defaultCenter(draft?: DraftScenario | null): [number, number, number] {
  const pts = draft ? points(draft).filter((p) => p.lat !== null && p.lng !== null) : [];
  if (pts.length) {
    const lat = pts.reduce((s, p) => s + p.lat!, 0) / pts.length;
    const lng = pts.reduce((s, p) => s + p.lng!, 0) / pts.length;
    return [lat, lng, 13];
  }
  return [env.defaultView.lat, env.defaultView.lng, env.defaultView.zoom];
}

/** Demand loaded per vehicle in a plan, using the scenario the plan was solved on. */
export function vehicleLoads(scenario: Scenario, plan: Plan): Array<{ vehicle: string; load: number; capacity: number }> {
  const demand = new Map(scenario.orders.map((o) => [o.id, o.demand]));
  return scenario.vehicles.map((v) => {
    const route = plan.routes.find((r) => r.vehicle_id === v.id);
    const load = route ? route.order_ids.reduce((s, id) => s + (demand.get(id) ?? 0), 0) : 0;
    return { vehicle: v.id, load, capacity: v.capacity };
  });
}

/** Map a backend field path such as "orders[2].window" to a draft path. */
export function issuesByPath(paths: string[]): Map<string, string[]> {
  const map = new Map<string, string[]>();
  for (const p of paths) {
    const key = p.match(/^orders\[\d+\]/)?.[0] ?? p.match(/^vehicles\[\d+\]/)?.[0] ?? p.split(".")[0];
    map.set(key, [...(map.get(key) ?? []), p]);
  }
  return map;
}

