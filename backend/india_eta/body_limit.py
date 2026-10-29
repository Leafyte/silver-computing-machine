"""Bound ETA request bytes before FastAPI parses JSON, including chunked bodies."""
import json


class EtaBodyLimit:
    def __init__(self, app, *, max_bytes=256_000):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope.get("path", "").startswith("/api/v1/eta/") or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        length = headers.get(b"content-length")
        if length is not None:
            try:
                size = int(length)
            except ValueError:
                return await self._deny(send)
            if size < 0 or size > self.max_bytes:
                return await self._deny(send)
        chunks, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.max_bytes:
                return await self._deny(send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False
        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()
        await self.app(scope, bounded_receive, send)

    @staticmethod
    async def _deny(send):
        body = json.dumps({"error": {"code": "request_budget_exceeded", "message": "ETA body exceeds the local input budget"}}).encode()
        await send({"type": "http.response.start", "status": 413, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
