"""Google Calendar builtin tool (Service Account, shared calendar).

Authenticates with a service-account JWT (RS256, signed via ``cryptography``)
against ``oauth2.googleapis.com/token`` and calls the Calendar v3 API with
aiohttp — no Google SDK dependency. The service account only sees calendars
explicitly shared with it; the configured ``calendar_id`` scopes all calls.

Credentials live in the per-tool configuration (dashboard → Tools →
google_calendar) and are never logged or returned to the model.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://oauth2.googleapis.com/token"
_API_BASE = "https://www.googleapis.com/calendar/v3"
_SCOPE = "https://www.googleapis.com/auth/calendar.events"
_MAX_RESULTS = 50
_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _sa_info(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Parse the service-account JSON from tool config."""
    raw = cfg.get("service_account_json", "")
    if isinstance(raw, dict):
        info = raw
    else:
        raw = str(raw or "").strip()
        if raw.startswith("base64:"):
            try:
                raw = base64.b64decode(raw[7:]).decode("utf-8")
            except Exception:
                raise RuntimeError("service_account_json is not valid base64.")
        try:
            info = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            raise RuntimeError("service_account_json is not valid JSON.")
    if not info.get("client_email") or not info.get("private_key"):
        raise RuntimeError(
            "Google Calendar is not configured. Set service_account_json "
            "(service-account key JSON) and calendar_id in the tool settings "
            "(dashboard → Tools → google_calendar)."
        )
    return info


def _sign_jwt(info: Dict[str, str]) -> str:
    """Build a signed RS256 service-account JWT (no external JWT lib needed)."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    now = int(time.time())
    header = {"alg": "RS256", "typ": "JWT"}
    key_id = info.get("private_key_id")
    if key_id:
        header["kid"] = key_id
    claims = {
        "iss": info["client_email"],
        "scope": _SCOPE,
        "aud": _TOKEN_URL,
        "iat": now,
        "exp": now + 3600,
    }
    signing_input = (
        _b64url(json.dumps(header, separators=(",", ":")).encode())
        + "."
        + _b64url(json.dumps(claims, separators=(",", ":")).encode())
    )
    key = serialization.load_pem_private_key(
        info["private_key"].encode("utf-8"), password=None
    )
    signature = key.sign(signing_input.encode(), padding.PKCS1v15(), hashes.SHA256())
    return f"{signing_input}.{_b64url(signature)}"


class GoogleCalendarTool:
    """List/create/update/delete events on a shared Google Calendar."""

    name = "google_calendar"
    description = (
        "Manage events on a shared Google Calendar (service-account access). "
        "action='list': upcoming events — 'time_min'/'time_max' (ISO, default now→+14d), "
        "'max_results' (default 20). "
        "action='create': 'summary' + 'start'/'end' (ISO datetime, or 'YYYY-MM-DD' for "
        "all-day) — optional 'description', 'location', 'attendees' (list of emails), "
        "'timezone' (default from tool config). "
        "action='update': 'event_id' + any changed fields. "
        "action='delete': 'event_id'. "
        "Service account + calendar_id are configured in the tool settings "
        "(dashboard → Tools → google_calendar)."
    )

    def __init__(self) -> None:
        self._token: Optional[str] = None
        self._token_exp = 0.0

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["list", "create", "update", "delete"],
                            "description": "Calendar operation.",
                        },
                        "event_id": {"type": "string", "description": "Event id (required for update/delete)."},
                        "summary": {"type": "string", "description": "Event title."},
                        "start": {"type": "string", "description": "Start ISO datetime or YYYY-MM-DD (all-day)."},
                        "end": {"type": "string", "description": "End ISO datetime or YYYY-MM-DD (all-day)."},
                        "timezone": {"type": "string", "description": "IANA timezone, e.g. 'Europe/Berlin'."},
                        "description": {"type": "string", "description": "Event description/body."},
                        "location": {"type": "string", "description": "Event location."},
                        "attendees": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Attendee email addresses.",
                        },
                        "time_min": {"type": "string", "description": "List window start (ISO)."},
                        "time_max": {"type": "string", "description": "List window end (ISO)."},
                        "max_results": {"type": "integer", "description": "Max events for 'list' (default 20, max 50).", "default": 20},
                    },
                    "required": ["action"],
                },
            },
        }

    # -- auth ---------------------------------------------------------------

    async def _access_token(self, cfg: Dict[str, Any], *, force_refresh: bool = False) -> str:
        if self._token and not force_refresh and time.time() < self._token_exp - 120:
            return self._token
        info = _sa_info(cfg)
        assertion = await asyncio.to_thread(_sign_jwt, info)
        form = {
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": assertion,
        }
        async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
            async with session.post(_TOKEN_URL, data=form) as resp:
                data = await resp.json(content_type=None)
                if resp.status != 200:
                    raise RuntimeError(
                        f"Google token request failed (HTTP {resp.status}): "
                        f"{data.get('error_description') or data.get('error') or 'unknown'}"
                    )
        self._token = str(data["access_token"])
        self._token_exp = time.time() + int(data.get("expires_in", 3600))
        return self._token

    async def _api(self, cfg: Dict[str, Any], method: str, path: str,
                   *, params: Optional[Dict[str, Any]] = None,
                   body: Optional[Dict[str, Any]] = None,
                   retry: bool = True) -> Any:
        token = await self._access_token(cfg)
        headers = {"Authorization": f"Bearer {token}"}
        async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
            async with session.request(
                method, f"{_API_BASE}{path}", headers=headers, params=params, json=body
            ) as resp:
                if resp.status == 401 and retry:
                    # Stale cached token — refresh once and retry.
                    await self._access_token(cfg, force_refresh=True)
                    return await self._api(cfg, method, path, params=params, body=body, retry=False)
                if resp.status == 204:
                    return {}
                data = await resp.json(content_type=None)
                if resp.status == 404:
                    raise RuntimeError("Calendar or event not found (404).")
                if resp.status == 403:
                    raise RuntimeError(
                        "Access denied (403) — is the calendar shared with the "
                        "service account and calendar_id correct?"
                    )
                if resp.status >= 400:
                    raise RuntimeError(
                        f"Calendar API error (HTTP {resp.status}): "
                        f"{(data.get('error') or {}).get('message') or data}"
                    )
                return data

    # -- event helpers ------------------------------------------------------

    @staticmethod
    def _valid_iso(value: str) -> bool:
        """Accept ISO 8601 dates/datetimes (including 'Z' UTC suffix)."""
        from datetime import date, datetime
        try:
            if len(value) == 10 and "T" not in value:
                date.fromisoformat(value)
            else:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            return True
        except ValueError:
            return False

    @classmethod
    def _event_time(cls, value: str, tz: str) -> Dict[str, str]:
        value = (value or "").strip()
        if not value:
            raise RuntimeError("start/end datetime is required.")
        if not cls._valid_iso(value):
            raise RuntimeError(
                f"Invalid start/end '{value}' — expected ISO 8601 "
                "(e.g. '2025-03-01T10:00:00' or '2025-03-01' for all-day)."
            )
        # Date-only → all-day event (Calendar API requires date, not dateTime).
        if len(value) == 10 and "T" not in value:
            return {"date": value}
        entry: Dict[str, str] = {"dateTime": value}
        if tz:
            entry["timeZone"] = tz
        return entry

    @staticmethod
    def _format_event(ev: Dict[str, Any]) -> Dict[str, Any]:
        start = ev.get("start") or {}
        end = ev.get("end") or {}
        return {
            "id": ev.get("id"),
            "summary": ev.get("summary") or "(no title)",
            "start": start.get("dateTime") or start.get("date") or "",
            "end": end.get("dateTime") or end.get("date") or "",
            "location": ev.get("location") or "",
            "status": ev.get("status") or "",
            "html_link": ev.get("htmlLink") or "",
        }

    @staticmethod
    def _default_window(tz_name: str) -> tuple:
        from datetime import datetime, timedelta, timezone
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = timezone.utc
        now = datetime.now(tz)
        return now.isoformat(timespec="seconds"), (now + timedelta(days=14)).isoformat(timespec="seconds")

    # -- execute ------------------------------------------------------------

    async def execute(
        self,
        action: str,
        event_id: Optional[str] = None,
        summary: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        timezone: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        time_min: Optional[str] = None,
        time_max: Optional[str] = None,
        max_results: int = 20,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        cfg = (context or {}).get("tool_config", {})
        calendar_id = str(cfg.get("calendar_id", "") or "").strip()
        if not calendar_id:
            return (
                "[Error: google_calendar is not configured — set calendar_id and "
                "service_account_json in the tool settings (dashboard → Tools → "
                "google_calendar) and share the calendar with the service account.]"
            )
        tz = (timezone or "").strip() or str(cfg.get("default_timezone", "") or "Europe/Berlin").strip()
        from urllib.parse import quote
        cal = quote(calendar_id, safe="")
        action = (action or "").strip().lower()

        try:
            if action == "list":
                if not time_min or not time_max:
                    dmin, dmax = self._default_window(tz)
                    time_min = time_min or dmin
                    time_max = time_max or dmax
                if not self._valid_iso(str(time_min)) or not self._valid_iso(str(time_max)):
                    return "[Error: time_min/time_max must be ISO 8601 dates or datetimes.]"
                data = await self._api(
                    cfg, "GET", f"/calendars/{cal}/events",
                    params={
                        "timeMin": time_min,
                        "timeMax": time_max,
                        "maxResults": max(1, min(_MAX_RESULTS, int(max_results or 20))),
                        "singleEvents": "true",
                        "orderBy": "startTime",
                    },
                )
                events = [self._format_event(ev) for ev in data.get("items", [])]
                if not events:
                    return f"No events between {time_min} and {time_max}."
                lines = [f"{len(events)} event(s):"]
                for ev in events:
                    lines.append(
                        f"- id={ev['id']} | {ev['start']} → {ev['end']} | {ev['summary']}"
                        + (f" | {ev['location']}" if ev["location"] else "")
                    )
                return "\n".join(lines)

            if action == "create":
                if not (summary or "").strip():
                    return "[Error: action='create' requires 'summary'.]"
                if not (start and end):
                    return "[Error: action='create' requires 'start' and 'end'.]"
                body: Dict[str, Any] = {
                    "summary": summary.strip(),
                    "start": self._event_time(start, tz),
                    "end": self._event_time(end, tz),
                }
                if description:
                    body["description"] = description
                if location:
                    body["location"] = location
                if attendees:
                    body["attendees"] = [{"email": a.strip()} for a in attendees if a and a.strip()]
                data = await self._api(cfg, "POST", f"/calendars/{cal}/events", body=body)
                ev = self._format_event(data)
                return f"Event created: '{ev['summary']}' | {ev['start']} → {ev['end']} | id={ev['id']}"

            if action == "update":
                if not (event_id or "").strip():
                    return "[Error: action='update' requires 'event_id'.]"
                body = {}
                if (summary or "").strip():
                    body["summary"] = summary.strip()
                if start:
                    body["start"] = self._event_time(start, tz)
                if end:
                    body["end"] = self._event_time(end, tz)
                if description is not None:
                    body["description"] = description
                if location is not None:
                    body["location"] = location
                if attendees:
                    body["attendees"] = [{"email": a.strip()} for a in attendees if a and a.strip()]
                if not body:
                    return "[Error: action='update' needs at least one field to change.]"
                data = await self._api(
                    cfg, "PATCH",
                    f"/calendars/{cal}/events/{quote(event_id.strip(), safe='')}",
                    body=body,
                )
                ev = self._format_event(data)
                return f"Event updated: '{ev['summary']}' | {ev['start']} → {ev['end']} | id={ev['id']}"

            if action == "delete":
                if not (event_id or "").strip():
                    return "[Error: action='delete' requires 'event_id'.]"
                await self._api(
                    cfg, "DELETE",
                    f"/calendars/{cal}/events/{quote(event_id.strip(), safe='')}",
                )
                return f"Event deleted: id={event_id.strip()}"

            return f"[Error: unknown action '{action}' — use list|create|update|delete.]"
        except RuntimeError as exc:
            return f"[Error: {exc}]"
        except Exception as exc:
            logger.warning("google_calendar %s failed: %s", action, exc, exc_info=True)
            return f"[Error: google_calendar request failed: {exc}]"
