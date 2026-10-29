# Client applications

The client apps are kept separate from the Python backend:

- `web-frontend/` — React, TypeScript, and Vite dispatch dashboard.
- `mobile-app/` — React Native with Expo driver app.

The web app preserves the NMIT dashboard's visual structure and connects demo dispatcher login, scenario validation, dispatch jobs, cancellation, health, risk replay, recovery, synthetic manifest activation, and the live driver activity feed to the local API. The driver app adapts the NMIT workflow and connects demo driver login, assigned manifest reads, stop-arrival/completion reports, delay reports, and dispatch messages. The only maintained frontend source projects are `web-frontend/` and `mobile-app/`.

## Stack and integration boundary

- Web: React 19, TypeScript, Vite, Tailwind CSS, React Query, MapLibre GL, and Lucide icons.
- Mobile: Expo SDK 58 (`expo ~58.0.21`), React Native 0.88.1, and React 19.3.0.
- Backend: the existing Python API at `http://127.0.0.1:8000` by default. Set `VITE_API_URL` for the web app or `EXPO_PUBLIC_API_URL` for mobile when using another address.

Web runs and jobs come from the backend when available. The bundled tiny scenario is submitted through the backend. Preview plans and intelligence evidence remain clearly marked fixtures. The mobile app pairs with a server-configured one-time code and stores its opaque session token in Expo SecureStore. Live manifest state and driver reports use backend routes. GPS/navigation, consignee contacts, proof of delivery, shift controls, cargo details, dock checklist, and gate clearance remain unavailable because the backend has no corresponding services; sample cargo is hidden while paired.

## Run locally

```sh
cd apps/web-frontend
npm ci
npm run dev
```

```sh
cd apps/mobile-app
npm ci
npm start
```

If using a physical device, set `EXPO_PUBLIC_API_URL` to a backend address reachable from that device. The default loopback address works only when the API is reachable as `127.0.0.1` from the app runtime.

Configure one-time driver pairing entries on the backend through `DEBUGLEAF_DRIVER_PAIRING_CODES` (JSON array of `{ "code", "driver_id", "vehicle_id" }`). Keep those values server-side and use a sufficiently long random code. For web browser access, allow the exact frontend origin in `DEBUGLEAF_CORS_ORIGINS`. See [`web-frontend/src/api/ENDPOINTS.md`](web-frontend/src/api/ENDPOINTS.md) for the route map and remaining backend gaps.
