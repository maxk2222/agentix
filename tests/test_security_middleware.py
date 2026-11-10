"""Tests for RequestSizeLimitMiddleware path_limits."""
from __future__ import annotations

import json
from typing import Awaitable, Callable

import pytest

from app.security_middleware import RequestSizeLimitMiddleware


def _ok_app(calls: list):
    async def app(scope, receive, send):
        calls.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    return app


def _scope(path: str, content_length: int | None = None):
    headers = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    return {
        "type": "http",
        "path": path,
        "method": "POST",
        "headers": headers,
        "app": None,
    }


async def _receive_empty():
    return {"type": "http.request", "body": b"", "more_body": False}


class _SendCollector:
    def __init__(self):
        self.messages = []

    async def __call__(self, message):
        self.messages.append(message)

    @property
    def status(self):
        for m in self.messages:
            if m.get("type") == "http.response.start":
                return m["status"]
        return None


@pytest.mark.asyncio
async def test_global_cap_blocks_large_body():
    calls = []
    mw = RequestSizeLimitMiddleware(_ok_app(calls), max_bytes=1_000)
    send = _SendCollector()
    await mw(_scope("/any", 2_000), _receive_empty, send)
    assert send.status == 413
    assert calls == []


@pytest.mark.asyncio
async def test_path_limit_allows_larger_body_on_matching_prefix():
    calls = []

    async def resolver(scope) -> int:
        return 10_000

    mw = RequestSizeLimitMiddleware(
        _ok_app(calls), max_bytes=1_000, path_limits={"/chat/attachments": resolver}
    )
    send = _SendCollector()
    await mw(_scope("/chat/attachments", 5_000), _receive_empty, send)
    assert send.status == 200
    assert calls == ["/chat/attachments"]


@pytest.mark.asyncio
async def test_path_limit_prefix_matches_subpaths():
    calls = []

    async def resolver(scope) -> int:
        return 10_000

    mw = RequestSizeLimitMiddleware(
        _ok_app(calls), max_bytes=1_000, path_limits={"/chat/runs": resolver}
    )
    send = _SendCollector()
    await mw(_scope("/chat/runs/conv-1/input", 5_000), _receive_empty, send)
    assert send.status == 200


@pytest.mark.asyncio
async def test_non_matching_path_still_uses_global_cap():
    calls = []

    async def resolver(scope) -> int:
        return 10_000

    mw = RequestSizeLimitMiddleware(
        _ok_app(calls), max_bytes=1_000, path_limits={"/chat/attachments": resolver}
    )
    send = _SendCollector()
    await mw(_scope("/other", 5_000), _receive_empty, send)
    assert send.status == 413


@pytest.mark.asyncio
async def test_resolver_error_fails_closed():
    calls = []

    async def resolver(scope) -> int:
        raise RuntimeError("store down")

    mw = RequestSizeLimitMiddleware(
        _ok_app(calls), max_bytes=1_000, path_limits={"/chat/attachments": resolver}
    )
    send = _SendCollector()
    await mw(_scope("/chat/attachments", 5_000), _receive_empty, send)
    assert send.status == 413


@pytest.mark.asyncio
async def test_streaming_body_enforces_resolved_limit():
    async def resolver(scope) -> int:
        return 10_000

    chunks = [b"x" * 4_000, b"y" * 4_000, b"z" * 4_000]
    idx = {"i": 0}

    async def receive():
        i = idx["i"]
        idx["i"] += 1
        if i < len(chunks):
            return {"type": "http.request", "body": chunks[i], "more_body": True}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def consuming_app(scope, rcv, send):
        while True:
            msg = await rcv()
            if msg.get("type") != "http.request":
                continue
            if not msg.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    mw = RequestSizeLimitMiddleware(
        consuming_app, max_bytes=1_000, path_limits={"/chat/attachments": resolver}
    )
    send = _SendCollector()
    # 12KB > resolved 10KB limit on the attachments path
    await mw(_scope("/chat/attachments"), receive, send)
    assert send.status == 413
