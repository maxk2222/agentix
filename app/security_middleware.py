"""Security-focused middleware utilities.

These middlewares are intentionally small, dependency-free, and compatible with
OpenAI-style clients.
"""

from __future__ import annotations

import json
from typing import Awaitable, Callable, Dict, Iterable, Optional, Tuple


class RequestSizeLimitMiddleware:
    """ASGI middleware that rejects requests exceeding a configured byte limit.

    This runs before request parsing so oversized bodies don't consume excessive
    memory/CPU in FastAPI/Pydantic.

    Notes:
    - Uses Content-Length when present.
    - Otherwise counts bytes while streaming the body from the ASGI receive channel.
    - ``path_limits`` maps path prefixes to async resolvers returning a
      per-path byte cap — lets e.g. upload/chat endpoints allow larger bodies
      than the global cap, computed per request so runtime config applies.
    """

    def __init__(
        self,
        app,
        *,
        max_bytes: int = 1_048_576,
        on_block: Optional[Callable] = None,
        exempt_paths: Optional[Iterable[str]] = None,
        path_limits: Optional[Dict[str, Callable[..., Awaitable[int]]]] = None,
    ) -> None:
        self.app = app
        self.max_bytes = int(max_bytes)
        self.on_block = on_block
        self.exempt_paths = {str(path) for path in (exempt_paths or [])}
        self.path_limits = list((path_limits or {}).items())

    async def _resolve_limit(self, scope) -> int:
        path = scope.get("path", "")
        for prefix, resolver in self.path_limits:
            norm = prefix.rstrip("/")
            if path == norm or path.startswith(norm + "/"):
                try:
                    return int(await resolver(scope))
                except Exception:
                    # Fail closed on resolver errors — the global cap still applies.
                    return self.max_bytes
        return self.max_bytes

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or self.max_bytes <= 0:
            return await self.app(scope, receive, send)

        if scope.get("path") in self.exempt_paths:
            return await self.app(scope, receive, send)

        limit = await self._resolve_limit(scope)

        headers = _decode_headers(scope.get("headers") or [])
        content_length = headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > limit:
                    if self.on_block:
                        await self.on_block(scope.get("path", ""), scope.get("method", ""), 413, "Request body too large.")
                    return await _send_json(
                        send,
                        status_code=413,
                        payload={"detail": "Request body too large."},
                    )
            except ValueError:
                # Invalid Content-Length: continue and enforce via streaming counter.
                pass

        response_started = False

        async def send_wrapper(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body") or b""
                received += len(body)
                if received > limit:
                    raise _PayloadTooLarge
            return message

        try:
            return await self.app(scope, limited_receive, send_wrapper)
        except _PayloadTooLarge:
            if response_started:
                # Response already started; cannot send a new one.
                return
            if self.on_block:
                await self.on_block(scope.get("path", ""), scope.get("method", ""), 413, "Request body too large.")
            return await _send_json(
                send,
                status_code=413,
                payload={"detail": "Request body too large."},
            )


class _PayloadTooLarge(Exception):
    pass


def _decode_headers(raw: Iterable[Tuple[bytes, bytes]]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for key, value in raw:
        try:
            headers[key.decode("latin-1").lower()] = value.decode("latin-1")
        except Exception:
            continue
    return headers


async def _send_json(send: Callable, *, status_code: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": int(status_code),
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode("latin-1")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
