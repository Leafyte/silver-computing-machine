# Frontend and backend connection map

The frontend calls the existing FastAPI service directly. The web base URL is `VITE_API_URL` (default `http://127.0.0.1:8000`); mobile uses `EXPO_PUBLIC_API_URL`. Browser origins are limited by `DEBUGLEAF_CORS_ORIGINS` in the backend.

## Web dashboard

| Feature | Backend route | Notes |
| --- | --- | --- |
| Health and solver capacity | `GET /api/v1/health` | Used by the dashboard health indicator. |
| Validate scenario | `POST /api/v1/scenarios/validate` | The editor sends the JSON currently in the form. |
| Exact async solve | `POST /api/v1/jobs` | Job is persisted and displayed in Jobs. |
| Warm-start QAOA async solve | `POST /api/v1/jobs/qaoa` | Local Python/Qiskit path; never submits a cloud job. |
| Local QpiAI VQC plan | `POST /api/v1/scenarios/vqc-plan` | Synchronous result; no QCloud job or QpiAI secret is sent from the browser. |
| Job history/detail/cancel | `GET /api/v1/jobs`, `GET /api/v1/jobs/{id}`, `POST /api/v1/jobs/{id}/cancel` | Job list polls while the app is open. |
| Source-scenario risk replay | `POST /api/v1/risk/evaluate` | Replays one deterministic source world; it is not calibrated risk. |
| Recovery replan | `POST /api/v1/recovery/plan` | Backend validates and records the recovery audit. |
| Activate synthetic ETA manifest | `POST /api/v1/eta/plans` | Requires dispatcher/admin bearer token; synthetic data only, separate from live dispatch. |
| Driver report audit | `GET /api/v1/driver/reports?limit=200` | Local mode needs no token; token mode requires a dispatcher/admin bearer token entered on the Activity page. |

## Driver mobile app

| Feature | Backend route | Notes |
| --- | --- | --- |
| One-time pairing | `POST /api/v1/driver/sessions` | Pairing code is configured server-side in `DEBUGLEAF_DRIVER_PAIRING_CODES`; it is single-use. |
| Restore/check session | `GET /api/v1/driver/me` | Opaque token is stored in Expo SecureStore and validated with the backend. |
| Revoke session | `DELETE /api/v1/driver/sessions/current` | Sign-out revokes the server session and deletes local secure storage. |
| Assigned vehicle manifest and event state | `GET /api/v1/driver/state` | Driver session is scoped to its assigned vehicle. |
| Stop arrival/completion and delay reports | `POST /api/v1/driver/reports` | Event IDs are idempotent; stop order and manifest version are validated. |

The current backend does not provide live GPS/telematics, road navigation, consignee contact, proof-of-delivery, shift management, cargo/package details, dock checklist state, or gate clearance. Mobile preview screens are labelled as preview; with an active backend manifest, the Loads page hides sample freight and explains that those services are unavailable.

## Credentials and execution limits

Browser clients never receive QpiAI SDK credentials. QpiAI Cloud and Opt Cloud are not connected to these screens; VQC uses the backend's local QpiAI SDK execution path. Do not put QpiAI credentials in `VITE_*` or `EXPO_PUBLIC_*` environment variables. Dispatcher/admin tokens are entered transiently in the relevant web view and are not persisted by the UI.

For a physical phone, set `EXPO_PUBLIC_API_URL` to the computer's LAN address and configure that web origin/API host appropriately. `127.0.0.1` on the phone points to the phone itself.
