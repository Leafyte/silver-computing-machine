import * as SecureStore from 'expo-secure-store';

export type DriverIdentity = { driver_id: string; vehicle_id: string; expires_at: string };
export type BackendDriverState = {
  driver_id: string;
  vehicle_id: string;
  manifest: {
    plan_id: string;
    plan_version: number;
    manifest_version: number;
    ordered_stops: Array<Record<string, any>>;
  };
  stop_status: Record<string, 'pending' | 'arrived' | 'completed'>;
  reports: Array<Record<string, any>>;
};

const API_BASE_URL = (process.env.EXPO_PUBLIC_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
let sessionToken: string | null = null;
let identity: DriverIdentity | null = null;

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (sessionToken) headers.Authorization = `Bearer ${sessionToken}`;
  Object.assign(headers, init.headers ?? {});
  const response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers, signal: init.signal ?? AbortSignal.timeout(10_000) });
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(body?.error?.message ?? `Backend request failed (${response.status})`);
  return body as T;
}

export const driverClient = {
  async getHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/health`, { signal: AbortSignal.timeout(5_000) });
      return response.ok;
    } catch { return false; }
  },
  async pair(pairingCode: string): Promise<DriverIdentity> {
    const response = await call<{ access_token: string; driver_id: string; vehicle_id: string; expires_at: string }>(
      '/api/v1/driver/sessions', { method: 'POST', body: JSON.stringify({ pairing_code: pairingCode }) },
    );
    sessionToken = response.access_token;
    identity = { driver_id: response.driver_id, vehicle_id: response.vehicle_id, expires_at: response.expires_at };
    await SecureStore.setItemAsync('driver_session_token', sessionToken);
    return identity;
  },
  async restoreSession(): Promise<DriverIdentity | null> {
    const savedToken = await SecureStore.getItemAsync('driver_session_token');
    if (!savedToken) return null;
    sessionToken = savedToken;
    try {
      identity = await call<DriverIdentity>('/api/v1/driver/me');
      return identity;
    } catch {
      sessionToken = null;
      identity = null;
      await SecureStore.deleteItemAsync('driver_session_token');
      return null;
    }
  },
  getIdentity(): DriverIdentity | null { return identity; },
  isPaired(): boolean { return Boolean(sessionToken); },
  async getState(): Promise<BackendDriverState> {
    return call('/api/v1/driver/state');
  },
  async report(payload: { event_id: string; manifest_version: number; kind: 'stop_arrived' | 'stop_completed' | 'delay_report'; stop_id?: string; severity?: 'light' | 'moderate' | 'heavy'; note?: string }): Promise<void> {
    await call('/api/v1/driver/reports', { method: 'POST', body: JSON.stringify(payload) });
  },
  async logout(): Promise<void> {
    try { await call('/api/v1/driver/sessions/current', { method: 'DELETE' }); }
    finally {
      sessionToken = null;
      identity = null;
      await SecureStore.deleteItemAsync('driver_session_token');
    }
  },
};

