# ArQ web frontend

React 19, TypeScript, and Vite dispatch dashboard.

## Requirements

Use Node.js `^22.22.2`, `^24.15.0`, or `>=26`. This covers the engine requirements of the browser test tooling in the lockfile.

## Commands

```sh
npm ci
npm run dev
npm run typecheck
npm run build
```

The dashboard connects to the local API for health, durable job history, job submission/cancellation, and recovery planning. It uses a bundled tiny scenario for solve requests. Preview plans and research evidence remain local fixtures and are marked as preview. See `../README.md` for integration boundaries and configuration.

Set `VITE_API_URL` when the backend is not at `http://127.0.0.1:8000`.
