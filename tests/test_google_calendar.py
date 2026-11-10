"""Tests for the google_calendar builtin tool — HTTP fully mocked.

The service-account JWT is generated with a real RSA key via ``cryptography``
but never leaves the process; token + Calendar API calls are faked.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agent.tools.builtin_tools.google_calendar import (
    GoogleCalendarTool,
    _b64url,
    _sign_jwt,
)


@pytest.fixture()
def sa_json() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    return json.dumps({
        "client_email": "svc@proj.iam.gserviceaccount.com",
        "private_key": pem,
        "private_key_id": "kid-1",
    })


CFG_TEMPLATE = {
    "service_account_json": "",
    "calendar_id": "shared@group.calendar.google.com",
    "default_timezone": "Europe/Berlin",
}


def _resp(status=200, payload=None):
    class R:
        def __init__(self):
            self.status = status
        async def json(self, content_type=None):
            return payload if payload is not None else {}
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
    return R()


class _FakeSession:
    """Routes token + API calls to canned responses and records them."""

    calls: list

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def post(self, url, data=None, **kw):
        _FakeSession.calls.append(("POST", url, data))
        assert url == "https://oauth2.googleapis.com/token"
        return _resp(200, {"access_token": "tok-123", "expires_in": 3600})

    def request(self, method, url, **kw):
        _FakeSession.calls.append((method, url, kw.get("params"), kw.get("json")))
        handler = getattr(self, "_handler", None)
        if handler:
            return handler(method, url, kw)
        return _resp(200, {"items": []})


@pytest.fixture()
def api_mock(monkeypatch, sa_json):
    _FakeSession.calls = []
    monkeypatch.setattr(
        "agent.tools.builtin_tools.google_calendar.aiohttp.ClientSession",
        _FakeSession,
    )
    cfg = dict(CFG_TEMPLATE, service_account_json=sa_json)
    return cfg


class TestJwtSigning:
    def test_jwt_structure(self, sa_json):
        import json as _json
        info = _json.loads(sa_json)
        jwt = _sign_jwt(info)
        header_b64, claims_b64, sig_b64 = jwt.split(".")
        header = json.loads(base64_pad(header_b64))
        claims = json.loads(base64_pad(claims_b64))
        assert header["alg"] == "RS256" and header["kid"] == "kid-1"
        assert claims["iss"] == "svc@proj.iam.gserviceaccount.com"
        assert "calendar.events" in claims["scope"]


def base64_pad(segment: str) -> bytes:
    import base64
    pad = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + pad)


class TestActions:
    @pytest.mark.asyncio
    async def test_list(self, api_mock):
        tool = GoogleCalendarTool()

        def handler(method, url, kw):
            assert method == "GET" and url.endswith("/events")
            return _resp(200, {"items": [
                {"id": "ev1", "summary": "Standup",
                 "start": {"dateTime": "2024-01-01T09:00:00+01:00"},
                 "end": {"dateTime": "2024-01-01T09:30:00+01:00"}},
            ]})

        _FakeSession._handler = staticmethod(handler)
        try:
            result = await tool.execute(action="list", context={"tool_config": api_mock})
        finally:
            _FakeSession._handler = None
        assert "Standup" in result and "ev1" in result
        # token call + api call
        assert _FakeSession.calls[0][0] == "POST"
        assert _FakeSession.calls[1][0] == "GET"
        assert "shared%40group.calendar.google.com" in _FakeSession.calls[1][1]

    @pytest.mark.asyncio
    async def test_create(self, api_mock):
        tool = GoogleCalendarTool()

        def handler(method, url, kw):
            assert method == "POST"
            body = kw["json"]
            assert body["summary"] == "Lunch"
            assert body["start"] == {"dateTime": "2024-01-02T12:00:00", "timeZone": "Europe/Berlin"}
            return _resp(200, {"id": "ev9", "summary": "Lunch",
                               "start": {"dateTime": "2024-01-02T12:00:00"},
                               "end": {"dateTime": "2024-01-02T13:00:00"}})

        _FakeSession._handler = staticmethod(handler)
        try:
            result = await tool.execute(
                action="create", summary="Lunch",
                start="2024-01-02T12:00:00", end="2024-01-02T13:00:00",
                context={"tool_config": api_mock},
            )
        finally:
            _FakeSession._handler = None
        assert "created" in result and "ev9" in result

    @pytest.mark.asyncio
    async def test_create_all_day(self, api_mock):
        tool = GoogleCalendarTool()

        def handler(method, url, kw):
            body = kw["json"]
            assert body["start"] == {"date": "2024-03-01"}
            return _resp(200, {"id": "ev10"})

        _FakeSession._handler = staticmethod(handler)
        try:
            await tool.execute(
                action="create", summary="Holiday",
                start="2024-03-01", end="2024-03-02",
                context={"tool_config": api_mock},
            )
        finally:
            _FakeSession._handler = None

    @pytest.mark.asyncio
    async def test_update(self, api_mock):
        tool = GoogleCalendarTool()

        def handler(method, url, kw):
            assert method == "PATCH" and url.endswith("/events/ev1")
            assert kw["json"] == {"summary": "Renamed"}
            return _resp(200, {"id": "ev1", "summary": "Renamed"})

        _FakeSession._handler = staticmethod(handler)
        try:
            result = await tool.execute(
                action="update", event_id="ev1", summary="Renamed",
                context={"tool_config": api_mock},
            )
        finally:
            _FakeSession._handler = None
        assert "updated" in result

    @pytest.mark.asyncio
    async def test_delete(self, api_mock):
        tool = GoogleCalendarTool()

        def handler(method, url, kw):
            assert method == "DELETE"
            return _resp(204)

        _FakeSession._handler = staticmethod(handler)
        try:
            result = await tool.execute(
                action="delete", event_id="ev1", context={"tool_config": api_mock}
            )
        finally:
            _FakeSession._handler = None
        assert "deleted" in result

    @pytest.mark.asyncio
    async def test_unconfigured_returns_error_text(self):
        tool = GoogleCalendarTool()
        result = await tool.execute(action="list", context={"tool_config": {}})
        assert "not configured" in result

    @pytest.mark.asyncio
    async def test_403_surfaces_readable_error(self, api_mock):
        tool = GoogleCalendarTool()

        def handler(method, url, kw):
            return _resp(403, {"error": {"message": "forbidden"}})

        _FakeSession._handler = staticmethod(handler)
        try:
            result = await tool.execute(action="list", context={"tool_config": api_mock})
        finally:
            _FakeSession._handler = None
        assert "Access denied" in result
